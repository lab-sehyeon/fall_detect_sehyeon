"""GPU0-only streaming S0-A; full 12 epochs with resumable, locked evaluation."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
from pathlib import Path
import signal
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score

from fall_pipeline.common.integrity import hash_named_tensors
from fall_pipeline.common.eval_safer_legacy_v1_candidates import load_adl_model
from fall_pipeline.safer import v2_controls_reconstruction as io
from fall_pipeline.safer import s0a_core as core

CONFIG = io.ROOT / 'configs/s0a_document_reconstruction_v1.json'
CODE = ['fall_pipeline/safer/run_s0a_reconstruction.py', 'fall_pipeline/safer/s0a_core.py',
        'fall_pipeline/safer/f1_core.py', 'fall_pipeline/safer/v2_controls_reconstruction.py',
        'fall_pipeline/common/integrity.py', 'fall_pipeline/common/eval_safer_legacy_v1_candidates.py',
        'fall_pipeline/common/fall_safer_zeroshot_eval.py', 'fall_pipeline/fu/fall_fu_linear_eval.py',
        'model/DSTE.py', 'model/STTR.py', 'tools.py']
STOP = False


def stop_signal(signum, frame):
    global STOP
    STOP = True


def check_pause(root, config):
    if STOP or (root / 'PAUSE_REQUESTED').exists():
        raise io.PauseRequested('explicit pause requested')
    try:
        io.disk_gate(config)
    except RuntimeError as error:
        raise io.PauseRequested(str(error)) from error


def publish(root, stage, **details):
    state = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **details}
    io.save_json(root / 'status.json', state)
    print(state, flush=True)
    if root.name.endswith('_smoke'):
        return
    status = 'completed' if stage == 'completed' else 'paused' if stage in ('paused', 'failed') else 'in_progress'
    header = f'- 문서 ID: `DOC-20260929-s0a-run-R1`\n- 기준일: 2026-09-29\n- 상태: `{status}`\n'
    internal = '# S0-A 자동 실행 진행\n\n' + header
    internal += f'\nOutput: `{root}`\n\n[계약과 원장](2026-09-29_s0a_reconstruction_internal.md)\n\n'
    internal += '```json\n' + __import__('json').dumps(state, ensure_ascii=False, indent=2) + '\n```\n'
    shared = '# S0-A 실험 진행\n\n' + header
    words = {'preflight': '입력·동결 조건 검사', 'train': '두 상태 분류기 학습', 'val': '검증셋 평가',
             'selection': '검증셋 기반 선택 고정', 'evaluate': '고정 모델 평가', 'audit': '최종 검산',
             'completed': '학습·고정 평가·검산 완료', 'paused': '실험 일시 중단', 'failed': '실험 중단'}
    shared += '\n현재 단계: ' + words[stage] + '.\n\n'
    for key, label in [('epoch', '현재 epoch'), ('windows', '처리 window'), ('total', '전체 window'), ('split', '평가 split')]:
        if key in details:
            shared += f'- {label}: {details[key]}\n'
    shared += '\n[실험 방법과 한계](2026-09-29_s0a_reconstruction_shared.md). 원본 동일성이나 전체 연구 완료를 주장하지 않는다.\n'
    for kind, value in (('internal', internal), ('shared', shared)):
        path = io.ROOT / f'docs/{kind}/2026-09-29_s0a_run_{kind}.md'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(value)
        os.replace(temporary, path)


def contract(config, smoke):
    for name in ('encoder', 'adl_head'):
        io.require(io.sha256_file(io.ROOT / config[name]) == config[name + '_sha256'], name + ' changed')
    return {'config_sha256': io.sha256_file(CONFIG), 'code_sha256': {p: io.sha256_file(io.ROOT / p) for p in CODE},
            'torch': str(torch.__version__), 'numpy': str(np.__version__), 'smoke': smoke,
            'historical_exact_reproduction': False}


def assert_contract(config, locked):
    io.require(contract(config, locked['smoke']) == locked, 'code/config/model asset changed during run')


def make_heads(config, device):
    tr = config['training']
    torch.manual_seed(tr['seed'])
    base = nn.Linear(2048, 16)
    nn.init.normal_(base.weight, mean=0., std=tr['initialization_std'])
    nn.init.zeros_(base.bias)
    heads = {name: copy.deepcopy(base).to(device) for name in tr['candidates']}
    optimizers = {name: torch.optim.SGD(head.parameters(), lr=tr['lr'], momentum=tr['momentum'],
                                      weight_decay=tr['weight_decay']) for name, head in heads.items()}
    return heads, optimizers


def save_state(root, state, heads, optimizers):
    state['heads'] = {name: io.cpu_state(head) for name, head in heads.items()}
    state['optimizers'] = {name: opt.state_dict() for name, opt in optimizers.items()}
    # Alternating slots retain the previously committed checkpoint until new pointer is atomic.
    pointer = root / 'training' / 'progress.json'
    previous = io.read(pointer)['slot'] if pointer.exists() else 1
    slot = 1 - previous
    dest = root / 'training' / f'resume_{slot}.pt'
    io.save_tensor(dest, state)
    io.save_json(pointer, {'slot': slot, 'sha256': io.sha256_file(dest), 'epoch': state['epoch'], 'next': state['next']})


def restore_state(root, locked, heads, optimizers):
    pointer = root / 'training' / 'progress.json'
    if not pointer.exists():
        return {'contract': locked, 'epoch': 1, 'next': 0, 'history': [], 'loss_sum': {n: 0. for n in heads},
                'loss_denom': {n: 0. for n in heads}, 'train_seconds': 0.}
    spec = io.read(pointer)
    path = pointer.parent / f'resume_{spec["slot"]}.pt'
    io.require(io.sha256_file(path) == spec['sha256'], 'resume checkpoint hash')
    state = torch.load(path, map_location='cpu', weights_only=True)
    io.require(state['contract'] == locked and state['epoch'] == spec['epoch'] and state['next'] == spec['next'], 'resume lineage')
    for name in heads:
        heads[name].load_state_dict(state['heads'][name], strict=True)
        optimizers[name].load_state_dict(state['optimizers'][name])
    return state


def metric_instance(config):
    ev = config['evaluation']
    return core.Metrics(ev['transitions'], ev['transition_tolerance_frames'], config['input']['fps'])


def evaluate(root, dest, split, heads, model, device, config, save_logits=False):
    dest.mkdir(parents=True, exist_ok=True)
    metrics = {name: metric_instance(config) for name in heads}
    head_hashes = {name: hash_named_tensors(head.state_dict().items()) for name, head in heads.items()}
    batch = config['training']['batch_size']
    frames = uncovered = 0
    all_truth, all_pred = [], {name: [] for name in heads}
    for seq_number, (row, begin, end) in enumerate(split.sequences(), 1):
        check_pause(root, config)
        uid = row['uid']
        out = dest / uid
        meta_path = out / 'manifest.json'
        _, labels = split.arrays(row['sequence_index'])
        starts = split.starts[begin:end]
        counts = core.coverage(len(labels), starts)
        mask = counts > 0
        # stride8/window64 leaves a contiguous covered prefix; never bridge missing intervals.
        io.require(np.array_equal(np.flatnonzero(mask), np.arange(np.count_nonzero(mask))), 'noncontiguous timeline')
        truth = labels[mask]
        frames += len(truth)
        uncovered += int(np.count_nonzero(~mask))
        if meta_path.exists():
            meta = io.read(meta_path)
            io.require(meta['heads'] == head_hashes and meta['begin'] == begin and meta['end'] == end and meta['save_logits'] == save_logits, 'evaluation resume lineage')
            for name, wanted in meta['payload'].items():
                io.require(io.sha256_file(out / name) == wanted, 'evaluation payload changed')
        else:
            sums = {name: np.zeros((len(labels), 16), np.float64) for name in heads}
            with torch.no_grad():
                for lo in range(begin, end, batch):
                    check_pause(root, config)
                    hi = min(lo + batch, end)
                    x, _ = split.batch(np.arange(lo, hi))
                    feature = core.dense_features(model, torch.from_numpy(x).to(device))
                    for name, head in heads.items():
                        values = head(feature).cpu().numpy()
                        core.add_logits(sums[name], split.starts[lo:hi], values)
            payload = {}
            for name, total in sums.items():
                mean = (total[mask] / counts[mask, None]).astype(np.float32)
                io.require(np.isfinite(mean).all(), 'fused logits finite')
                pred = mean.argmax(1).astype(np.uint8)
                outputs = {name + '_pred.npy': pred}
                if save_logits:
                    outputs[name + '_logits.npy'] = mean
                for filename, array in outputs.items():
                    io.save_array(out / filename, array)
                    payload[filename] = io.sha256_file(out / filename)
            io.save_json(meta_path, {'heads': head_hashes, 'begin': begin, 'end': end, 'covered_frames': len(truth),
                                    'save_logits': save_logits, 'payload': payload})
        all_truth.append(np.asarray(truth))
        for name in heads:
            pred = np.load(out / (name + '_pred.npy'), allow_pickle=False)
            if save_logits:
                logits = np.load(out / (name + '_logits.npy'), mmap_mode='r', allow_pickle=False)
                np.testing.assert_array_equal(logits.argmax(1), pred)
            metrics[name].add(pred, truth)
            all_pred[name].append(pred)
        if seq_number == 1 or seq_number % 5 == 0:
            publish(root, 'evaluate' if save_logits else 'val', split=dest.name, windows=end, total=split.count, sequences=seq_number)
    result = {name: metric.result() for name, metric in metrics.items()}
    truth = np.concatenate(all_truth)
    for name in heads:
        pred = np.concatenate(all_pred[name])
        np.testing.assert_array_equal(confusion_matrix(truth, pred, labels=np.arange(16)), result[name]['confusion'])
        io.require(abs(f1_score(truth, pred, labels=np.arange(16), average='macro', zero_division=0) - result[name]['macro_f1']) < 1e-12, 'independent sklearn F1')
    io.save_json(dest / 'metrics.json', {'candidates': result, 'covered_frames': frames, 'uncovered_frames': uncovered,
                                        'independent_frame_metrics_verified': True})
    return result


def smoke_audit(model, heads, split, device):
    x, _ = split.batch(np.arange(2))
    batch = torch.from_numpy(x).to(device)
    with torch.no_grad():
        actual = core.dense_features(model, batch)
        jt = batch.permute(0, 2, 4, 3, 1).reshape(2, 64, 150)
        js = batch.permute(0, 4, 3, 2, 1).reshape(2, 50, 192)
        temporal, spatial = model.backbone(jt, js)
        contexts = []
        for index in range(2):
            valid = torch.any(js[index] != 0, dim=1)
            contexts.append(spatial[index][valid].mean(0) if valid.any() else torch.zeros(1024, device=device))
        expected = torch.cat((temporal, torch.stack(contexts)[:, None].expand(-1, 64, -1)), -1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        maximum = 0.
        for head in heads.values():
            cpu = nn.Linear(2048, 16)
            cpu.load_state_dict(io.cpu_state(head))
            reference = cpu(actual.cpu()).numpy()
            got = head(actual).cpu().numpy()
            np.testing.assert_allclose(got, reference, atol=1e-4, rtol=1e-4)
            maximum = max(maximum, float(np.max(np.abs(got - reference))))
    io.require(all(p.grad is None and not p.requires_grad for p in model.parameters()), 'frozen gradient leak')
    return {'dense_masked_features_verified': True, 'cpu_head_max_error': maximum, 'frozen_gradients_absent': True}


def train(root, train_data, val_data, heads, optimizers, model, device, config, locked, frozen):
    tr = config['training']
    epochs = config['execution']['smoke_epochs'] if locked['smoke'] else tr['epochs']
    counts = np.asarray(io.read(root / 'source_audit.json')['train_frequency']['counts'], np.float64)
    io.require(np.all(counts > 0), 'all 16 train classes must be present')
    weights = 1 / np.sqrt(counts)
    weights /= weights.mean()
    weights = {'plain': None, 'sqrt': torch.tensor(weights, dtype=torch.float32, device=device)}
    state = restore_state(root, locked, heads, optimizers)
    save_state(root, state, heads, optimizers)
    try:
        while state['epoch'] <= epochs:
            epoch = state['epoch']
            generator = torch.Generator().manual_seed(tr['seed'] + epoch)
            order = torch.randperm(train_data.count, generator=generator).numpy()
            batch = tr['batch_size']
            for begin in range(state['next'], train_data.count, batch):
                check_pause(root, config)
                started = time.monotonic()
                end = min(begin + batch, train_data.count)
                x, y = train_data.batch(order[begin:end])
                truth = torch.from_numpy(y).to(device)
                with torch.no_grad():
                    features = core.dense_features(model, torch.from_numpy(x).to(device))
                for name, head in heads.items():
                    optimizers[name].zero_grad(set_to_none=True)
                    logits = head(features)
                    loss = nn.functional.cross_entropy(logits.flatten(0, 1), truth.flatten(), weight=weights[name])
                    io.require(bool(torch.isfinite(loss)), 'nonfinite loss')
                    loss.backward()
                    io.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()), 'head gradients')
                    optimizers[name].step()
                    denom = float(truth.numel()) if weights[name] is None else float(weights[name][truth].sum().item())
                    state['loss_sum'][name] += float(loss.item()) * denom
                    state['loss_denom'][name] += denom
                state['train_seconds'] += time.monotonic() - started
                state['next'] = end
                if begin == 0 or (end // batch) % config['execution']['checkpoint_batches'] == 0 or end == train_data.count:
                    save_state(root, state, heads, optimizers)
                    publish(root, 'train', epoch=epoch, epochs=epochs, windows=end, total=train_data.count,
                            mean_loss={n: state['loss_sum'][n] / state['loss_denom'][n] for n in heads}, train_seconds=state['train_seconds'],
                            gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
            io.require(hash_named_tensors(model.state_dict().items()) == frozen, 'frozen model changed')
            io.require(all(p.grad is None and not p.requires_grad for p in model.parameters()), 'frozen gradients')
            assert_contract(config, locked)
            epoch_path = root / 'training' / f'epoch_{epoch:02d}.pt'
            current = {'heads': {n: io.cpu_state(h) for n, h in heads.items()}, 'epoch': epoch, 'contract': locked}
            if epoch_path.exists():
                old = torch.load(epoch_path, map_location='cpu', weights_only=True)
                io.require(old['epoch'] == epoch and old['contract'] == locked, 'epoch checkpoint lineage')
                for n in heads:
                    io.require(hash_named_tensors(old['heads'][n].items()) == hash_named_tensors(current['heads'][n].items()), 'epoch checkpoint changed')
            else:
                io.save_tensor(epoch_path, current)
            evaluated = evaluate(root, root / 'validation' / f'epoch_{epoch:02d}', val_data, heads, model, device, config)
            state['history'].append({'epoch': epoch, 'metrics': evaluated, 'checkpoint_sha256': io.sha256_file(epoch_path),
                                     'mean_loss': {n: state['loss_sum'][n] / state['loss_denom'][n] for n in heads}})
            io.save_json(root / 'training' / 'history.json', state['history'])
            state.update(epoch=epoch + 1, next=0, loss_sum={n: 0. for n in heads}, loss_denom={n: 0. for n in heads})
            save_state(root, state, heads, optimizers)
    except (io.PauseRequested, KeyboardInterrupt):
        save_state(root, state, heads, optimizers)
        raise
    return state


def choose(history, names):
    best = None
    for epoch in history:
        for name in names:
            candidate = {'epoch': epoch['epoch'], 'candidate': name, 'metrics': epoch['metrics'][name],
                         'checkpoint_sha256': epoch['checkpoint_sha256']}
            if best is None or core.selection_key(candidate['metrics']) > core.selection_key(best['metrics']):
                best = candidate
    return best


def run(args, config, root):
    locked = contract(config, args.smoke)
    path = root / 'run_contract.json'
    if path.exists():
        io.require(args.resume and io.read(path) == locked, 'nonempty run requires --resume with identical contract')
    else:
        io.require(not any(p.name != 'run.lock' for p in root.iterdir()), 'orphan outputs; preserve and inspect')
        io.save_json(path, locked)
    if (root / 'final_report.json').exists():
        io.require(io.read(root / 'final_report.json')['passed'], 'previous final failed')
        print('Already completed; no retraining.', flush=True)
        return
    check_pause(root, config)
    if not args.smoke:
        smoke_root = Path(str(root) + '_smoke')
        smoke_report = io.read(smoke_root / 'final_report.json')
        smoke_contract = io.read(smoke_root / 'run_contract.json')
        io.require(smoke_report['passed'] and not smoke_report['research_usable'], 'smoke gate')
        io.require({**smoke_contract, 'smoke': False} == locked, 'smoke source/config differs')
    publish(root, 'preflight', holdouts_opened=False)
    train_data = core.verify_split(config, 'train')
    val_data = core.verify_split(config, 'val')
    frequency = train_data.frequencies()
    io.save_json(root / 'source_audit.json', {'passed': True, 'splits': ['train', 'val'], 'holdouts_opened': False,
                                           'train_frequency': frequency, 'val_frequency': val_data.frequencies(),
                                           'parent_sha256': config['data_report_sha256']})
    if args.smoke:
        train_data.count = val_data.count = config['execution']['smoke_windows']
    device = io.device_for(config)
    model = load_adl_model(io.ROOT / config['encoder'], io.ROOT / config['adl_head'], device)
    frozen = hash_named_tensors(model.state_dict().items())
    heads, optimizers = make_heads(config, device)
    state = train(root, train_data, val_data, heads, optimizers, model, device, config, locked, frozen)
    if args.smoke:
        audit = smoke_audit(model, heads, train_data, device)
        io.require(hash_named_tensors(model.state_dict().items()) == frozen, 'smoke frozen model changed')
        report = {'passed': True, 'research_usable': False, 'epochs': 1, 'train_windows': train_data.count,
                  'train_seconds': state['train_seconds'], 'gpu_peak_gib': torch.cuda.max_memory_allocated() / 1024**3,
                  'holdouts_opened': False, 'frozen_model_hash': frozen, 'audit': audit}
        assert_contract(config, locked)
        io.save_json(root / 'final_report.json', report)
        publish(root, 'completed', **report)
        return
    io.require(len(state['history']) == config['training']['epochs'], 'incomplete training')
    chosen = choose(state['history'], config['training']['candidates'])
    selection = {**chosen, 'run_contract_sha256': io.sha256_file(path), 'history_sha256': io.sha256_file(root / 'training/history.json'),
                 'selected_using': 'val only', 'holdouts_opened_at_selection': False}
    select_path = root / 'selection.json'
    if select_path.exists():
        io.require(io.read(select_path) == selection, 'immutable selection differs')
    else:
        io.save_json(select_path, selection)
    selection_sha = io.sha256_file(select_path)
    publish(root, 'selection', epoch=chosen['epoch'], candidate=chosen['candidate'])
    best_path = root / 'training' / f'epoch_{chosen["epoch"]:02d}.pt'
    io.require(io.sha256_file(best_path) == chosen['checkpoint_sha256'], 'selected checkpoint changed')
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    name = chosen['candidate']
    heads[name].load_state_dict(best['heads'][name])
    selected = {name: heads[name]}
    final_metrics = {}
    for split_name in ('val', 'test', 'ood'):
        io.require(io.sha256_file(select_path) == selection_sha, 'selection changed before holdout')
        data = val_data if split_name == 'val' else core.verify_split(config, split_name)
        final_metrics[split_name] = evaluate(root, root / 'evaluation' / split_name, data, selected, model, device, config, True)[name]
        if split_name == 'val':
            io.require(final_metrics['val'] == chosen['metrics'], 'selected val replay differs')
    publish(root, 'audit')
    for split_name in config['input']['split_counts']:
        core.verify_split(config, split_name)
    io.require(hash_named_tensors(model.state_dict().items()) == frozen, 'final frozen model changed')
    assert_contract(config, locked)
    io.require(io.sha256_file(select_path) == selection_sha, 'final selection changed')
    final = {'passed': True, 'research_usable': True, 'historical_exact_reproduction': False,
             'selection_sha256': selection_sha, 'selected_epoch': chosen['epoch'], 'selected_candidate': name,
             'epochs_per_candidate': config['training']['epochs'], 'metrics': final_metrics,
             'frozen_model_hash': frozen, 'frozen_models_unchanged': True, 'all_source_splits_reverified': True,
             'selection_val_exact_replay': True, 'independent_frame_metrics_verified': True,
             'training_seconds': state['train_seconds'], 'smoothing': False, 'automatic_deployment': False}
    io.save_json(root / 'final_report.json', final)
    publish(root, 'completed', epochs=12, selected_epoch=chosen['epoch'], candidate=name,
            final_report_sha256=io.sha256_file(root / 'final_report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = io.read(CONFIG)
    root = io.guard(io.ROOT / (config['output'] + ('_smoke' if args.smoke else '')))
    io.require(root.parent == io.ROOT / 'checkpoint/fall', 'output scope')
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args, config, root)
        except io.PauseRequested as error:
            publish(root, 'paused', reason=str(error))
            raise SystemExit(75)
        except Exception as error:
            publish(root, 'failed', error_type=type(error).__name__, reason=str(error))
            raise


if __name__ == '__main__':
    main()
