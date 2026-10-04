"""Two persistent GPU replicas; globally weighted loss, one clip and optimizer step."""
import copy
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

import torch
from torch import nn

from fall_pipeline.safer import s0a_core as core


def device_for(config):
    if (Path(sys.prefix).name != 'fall_detect' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8' or torch.cuda.device_count() != 2):
        raise RuntimeError('explicitly approved GPU0+1/fall_detect runtime required')
    for device in (0, 1):
        if ('3090' not in torch.cuda.get_device_name(device)
                or torch.cuda.mem_get_info(device)[0] < config['execution']['minimum_gpu_free_gib'] * 1024**3):
            raise RuntimeError('GPU allocation/free-memory gate')
        torch.cuda.set_per_process_memory_fraction(config['execution']['gpu_memory_fraction'], device)
    torch.cuda.set_device(0)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda:0')


def merge_gradients(primary, secondary):
    first = dict(primary.named_parameters())
    second = dict(secondary.named_parameters())
    if first.keys() != second.keys():
        raise RuntimeError('replica parameter mismatch')
    for name, parameter in first.items():
        if not parameter.requires_grad:
            if parameter.grad is not None or second[name].grad is not None:
                raise RuntimeError('frozen gradient leak')
            continue
        if parameter.grad is None or second[name].grad is None:
            raise RuntimeError('missing replica gradient')
        parameter.grad.add_(second[name].grad.to(parameter.device))


class DualStep:
    def __init__(self, head, model):
        if torch.cuda.device_count() != 2:
            raise RuntimeError('exactly two explicitly approved GPUs required')
        self.replica = copy.deepcopy(head).to('cuda:1')
        self.encoder = copy.deepcopy(model).to('cuda:1').eval()
        self.encoder.requires_grad_(False)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='s0b-gpu')
        self.original_model = model
        self.initial_replica_hash = None

    @staticmethod
    def shard(device_index, model, head, x, y, weights, denominator):
        with torch.cuda.device(device_index):
            device = torch.device('cuda', device_index)
            tx = torch.from_numpy(x).to(device)
            truth = torch.from_numpy(y).to(device)
            with torch.no_grad():
                features = core.dense_features(model, tx)
            loss = nn.functional.cross_entropy(head(features).flatten(0, 1), truth.flatten(),
                                               weight=weights, reduction='sum')
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('nonfinite replica loss')
            (loss / denominator).backward()
            torch.cuda.synchronize(device)
            return float(loss.item())

    def __call__(self, head, optimizer, model, x, y, weights, device, training):
        if model is not self.original_model or len(y) > 1024 or len(y) < 2:
            raise RuntimeError('unexpected data-parallel source or effective batch')
        optimizer.zero_grad(set_to_none=True)
        self.replica.zero_grad(set_to_none=True)
        self.replica.train(head.training); self.replica.base.eval()
        with torch.no_grad():
            for target, source in zip(self.replica.parameters(), head.parameters()):
                target.copy_(source.to('cuda:1'))
            for target, source in zip(self.replica.buffers(), head.buffers()):
                target.copy_(source.to('cuda:1'))
        denominator = weights[torch.from_numpy(y).to(device)].sum()
        second_weights = weights.to('cuda:1')
        second_denominator = denominator.to('cuda:1')
        # Make cross-device copies visible before worker-thread default-stream launches.
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        cut = (len(y) + 1) // 2
        tasks = [self.pool.submit(self.shard, 0, model, head, x[:cut], y[:cut], weights, denominator),
                 self.pool.submit(self.shard, 1, self.encoder, self.replica, x[cut:], y[cut:], second_weights, second_denominator)]
        values = [task.result() for task in tasks]
        merge_gradients(head, self.replica)
        parameters = [p for p in head.parameters() if p.requires_grad]
        if not all(bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise RuntimeError('nonfinite merged gradient')
        norm = nn.utils.clip_grad_norm_(parameters, training['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        return sum(values), float(denominator.item()), float(norm.item())

    def close(self):
        self.pool.shutdown(wait=True)
