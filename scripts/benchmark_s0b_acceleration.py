"""Read-only-source train-sample throughput diagnosis on explicitly approved GPU1."""
import copy
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn

from fall_pipeline.safer import run_s0b_reconstruction_r2 as b

io=b.io


def sync():
    torch.cuda.synchronize()


def main():
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='1','benchmark only approved physical GPU1')
    io.require(torch.cuda.device_count()==1 and '3090' in torch.cuda.get_device_name(0),'GPU1 RTX3090')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.85,0)
    cfg=io.read(b.CONFIG);ac=io.read(io.ROOT/cfg['s0a_config']);source=io.ROOT/cfg['output']
    root=io.ROOT/'checkpoint/fall/S0B_ACCELERATION_BENCH_20260929_R1'
    io.require(not root.exists(),'preserve previous benchmark');root.mkdir()
    pointer=io.read(source/'training/progress.json');path=source/'training'/('resume_'+str(pointer['slot'])+'.pt')
    io.require(io.sha256_file(path)==pointer['sha256'],'source checkpoint raced; retry')
    state=torch.load(path,map_location='cpu',weights_only=True)
    data=b.s.core.verify_split(ac,'train');indices=torch.randperm(data.count,generator=torch.Generator().manual_seed(1)).numpy()[:1024]
    started=time.monotonic();x,y=data.batch(indices);load_seconds=time.monotonic()-started
    device=torch.device('cuda:0');model=b.a.load_adl_model(io.ROOT/ac['encoder'],io.ROOT/ac['adl_head'],device)
    base={key.removeprefix('base.'):value for key,value in state['heads']['tcn'].items() if key.startswith('base.')}
    counts=np.asarray(io.read(source/'source_audit.json')['train_frequency']['counts'],np.float64)
    w=1/np.sqrt(counts);w/=w.mean();weights=torch.tensor(w,dtype=torch.float32,device=device)
    results=[]
    for micro in (512,1024):
        head=b.core.ResidualStateHead(base,cfg['model']).to(device);head.load_state_dict(state['heads']['tcn']);head.train()
        optimizer=torch.optim.AdamW((p for p in head.parameters() if p.requires_grad),lr=cfg['training']['lr'],
                                   betas=tuple(cfg['training']['betas']),eps=cfg['training']['eps'],weight_decay=cfg['training']['weight_decay'])
        optimizer.load_state_dict(copy.deepcopy(state['optimizers']['tcn']))
        torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();times=[]
        for iteration in range(5):
            sync();started=time.monotonic()
            value=b.micro_step(head,optimizer,model,x,y,weights,device,{**cfg['training'],'microbatch_size':micro})
            sync();elapsed=time.monotonic()-started
            if iteration>=2:times.append(elapsed)
            print({'microbatch':micro,'iteration':iteration,'seconds':elapsed,'loss':value[0]/value[1]},flush=True)
        # Separate bounded stages on fixed inputs; these timings include explicit synchronization.
        optimizer.zero_grad(set_to_none=True);sync();started=time.monotonic()
        tx=torch.from_numpy(x[:micro]).to(device);ty=torch.from_numpy(y[:micro]).to(device);sync();h2d=time.monotonic()-started
        started=time.monotonic()
        with torch.no_grad():feature=b.s.core.dense_features(model,tx)
        sync();encoder=time.monotonic()-started
        started=time.monotonic();loss=nn.functional.cross_entropy(head(feature).flatten(0,1),ty.flatten(),weight=weights)
        loss.backward();sync();head_seconds=time.monotonic()-started
        row={'microbatch':micro,'effective_batch':1024,'seconds_median':statistics.median(times),
             'windows_per_second':1024/statistics.median(times),'peak_gib':torch.cuda.max_memory_allocated()/1024**3,
             'stage_batch':micro,'h2d_seconds':h2d,'encoder_seconds':encoder,'head_forward_backward_seconds':head_seconds}
        results.append(row);print(row,flush=True)
        del head,optimizer,feature,tx,ty,loss;torch.cuda.empty_cache()
    io.save_json(root/'report.json',{'passed':True,'research_usable':False,'physical_gpu':1,'global_batch':1024,
                                   'source_checkpoint_sha256':pointer['sha256'],'source_epoch':state['epoch'],'source_next':state['next'],
                                   'input_windows':1024,'cpu_batch_seconds':load_seconds,'results':results,
                                   'precision':'float32 noAMP noTF32','weights_injected_into_training':False})


if __name__=='__main__':main()
