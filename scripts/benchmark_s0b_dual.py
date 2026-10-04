"""Approved GPU0+1 benchmark and real gradient/RNG equivalence checks, train only."""
import copy
import os
import statistics
import time

import numpy as np
import torch

from fall_pipeline.safer import run_s0b_reconstruction_r2 as b
from fall_pipeline.safer.s0b_parallel_step import DualStep

io=b.io


def synchronize():
    for device in (0,1):torch.cuda.synchronize(device)


def main():
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and torch.cuda.device_count()==2,'approved GPU0,1 only')
    for device in (0,1):
        io.require('3090' in torch.cuda.get_device_name(device),'3090 required')
        io.require(torch.cuda.mem_get_info(device)[0]>12*1024**3,'occupied device')
        torch.cuda.set_per_process_memory_fraction(.85,device)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=io.read(b.CONFIG);ac=io.read(io.ROOT/cfg['s0a_config']);source=io.ROOT/cfg['output']
    io.require(io.read(source/'status.json')['stage']=='paused','pause actual training before dual benchmark')
    root=io.ROOT/'checkpoint/fall/S0B_DUAL_BENCH_20260929_R1'
    io.require(not root.exists(),'preserve previous dual benchmark');root.mkdir()
    pointer=io.read(source/'training/progress.json');path=source/'training'/('resume_'+str(pointer['slot'])+'.pt')
    io.require(io.sha256_file(path)==pointer['sha256'],'source checkpoint pin')
    state=torch.load(path,map_location='cpu',weights_only=True)
    data=b.s.core.verify_split(ac,'train');indices=torch.randperm(data.count,generator=torch.Generator().manual_seed(1)).numpy()[:1024]
    x,y=data.batch(indices);device=torch.device('cuda:0')
    model=b.a.load_adl_model(io.ROOT/ac['encoder'],io.ROOT/ac['adl_head'],device)
    base={k.removeprefix('base.'):v for k,v in state['heads']['tcn'].items() if k.startswith('base.')}
    head=b.core.ResidualStateHead(base,cfg['model']).to(device);head.load_state_dict(state['heads']['tcn'])
    counts=np.asarray(io.read(source/'source_audit.json')['train_frequency']['counts'],np.float64)
    w=1/np.sqrt(counts);w/=w.mean();weights=torch.tensor(w,dtype=torch.float32,device=device)
    dual=DualStep(head,model)
    # Real trained head, dropout disabled: check the global weighted-gradient update.
    head.eval();reference=copy.deepcopy(head);opt=torch.optim.SGD(head.parameters(),lr=.001)
    refopt=torch.optim.SGD(reference.parameters(),lr=.001)
    first=b.micro_step(reference,refopt,model,x,y,weights,device,cfg['training'])
    second=dual(head,opt,model,x,y,weights,device,cfg['training']);synchronize()
    error=max((p-q).abs().max().item() for p,q in zip(head.parameters(),reference.parameters()))
    for p,q in zip(head.parameters(),reference.parameters()):torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5)
    io.require(abs(first[0]/first[1]-second[0]/second[1])<1e-5,'loss reduction mismatch')
    del reference,refopt,opt
    head.load_state_dict(state['heads']['tcn']);head.train();head.base.eval()
    optimizer=torch.optim.AdamW((p for p in head.parameters() if p.requires_grad),lr=cfg['training']['lr'],
                               betas=tuple(cfg['training']['betas']),eps=cfg['training']['eps'],weight_decay=cfg['training']['weight_decay'])
    optimizer.load_state_dict(copy.deepcopy(state['optimizers']['tcn']))
    torch.manual_seed(0)
    with torch.cuda.device(1):torch.cuda.manual_seed(1)
    for d in (0,1):torch.cuda.reset_peak_memory_stats(d)
    times=[]
    for iteration in range(6):
        synchronize();started=time.monotonic();value=dual(head,optimizer,model,x,y,weights,device,cfg['training']);synchronize()
        elapsed=time.monotonic()-started
        if iteration>=2:times.append(elapsed)
        print({'iteration':iteration,'seconds':elapsed,'windows_per_second':1024/elapsed,'loss':value[0]/value[1]},flush=True)
    # Both-device dropout RNG and optimizer restoration must replay a full update exactly.
    saved_head=copy.deepcopy(head.state_dict());saved_optimizer=copy.deepcopy(optimizer.state_dict());rng=torch.cuda.get_rng_state_all()
    dual(head,optimizer,model,x,y,weights,device,cfg['training']);expected=copy.deepcopy(head.state_dict())
    head.load_state_dict(saved_head);optimizer.load_state_dict(saved_optimizer);torch.cuda.set_rng_state_all(rng)
    dual(head,optimizer,model,x,y,weights,device,cfg['training'])
    io.require(all(torch.equal(value,expected[k]) for k,value in head.state_dict().items()),'dual dropout resume not exact')
    io.require(all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in dual.encoder.parameters()),'encoder gradient leak')
    frozen=b.s.hash_named_tensors(model.state_dict().items());replica=b.s.hash_named_tensors(dual.encoder.state_dict().items())
    io.require(frozen==replica,'replica frozen encoder changed')
    io.require(io.sha256_file(path)==pointer['sha256'],'parent changed during benchmark')
    report={'passed':True,'research_usable':False,'physical_gpus':[0,1],'effective_batch':1024,'per_device_batch':512,
            'seconds_median':statistics.median(times),'windows_per_second':1024/statistics.median(times),
            'peak_gib':[torch.cuda.max_memory_allocated(d)/1024**3 for d in (0,1)],
            'real_gradient_update_max_error':error,'loss_denominator_global':True,'optimizer_step_once':True,
            'two_device_dropout_resume_exact':True,'frozen_replica_hash':frozen,'source_checkpoint_sha256':pointer['sha256'],
            'source_epoch':state['epoch'],'source_next':state['next'],'weights_injected_into_training':False}
    io.save_json(root/'report.json',report);print(report,flush=True);dual.close()


if __name__=='__main__':main()
