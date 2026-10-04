"""Frozen S0-A plus zero-init TCN, GPU0 streaming, fixed validation gates."""
import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fall_pipeline.safer import state_followup_io as s
from fall_pipeline.safer import state_followup_core as core

io, a = s.io, s.a
CONFIG = io.ROOT / 'configs/s0b_document_reconstruction_v1.json'
CODE = a.CODE + ['fall_pipeline/safer/state_followup_io.py','fall_pipeline/safer/state_followup_core.py',
                 'fall_pipeline/safer/run_s0b_reconstruction.py','configs/s0a_document_reconstruction_v1.json']


def make_contract(config, smoke):
    s.s0a_dependency(config)
    return {'config_sha256':io.sha256_file(CONFIG),'source':s.fingerprint(CODE),
            's0a_report_sha256':config['s0a_report_sha256'],'smoke':smoke,
            'torch':str(torch.__version__),'numpy':str(np.__version__),'historical_exact_reproduction':False}


def frozen_check(model, head, encoder_hash, base_hash):
    io.require(s.hash_named_tensors(model.state_dict().items()) == encoder_hash, 'DSTE/ADL changed')
    io.require(s.hash_named_tensors(head.base.state_dict().items()) == base_hash, 'S0-A base changed')
    io.require(all(not p.requires_grad and p.grad is None for p in list(model.parameters())+list(head.base.parameters())), 'frozen gradient leak')


def commit(root, state, head, optimizer):
    state['cpu_rng'] = torch.get_rng_state()
    state['cuda_rng'] = torch.cuda.get_rng_state()
    a.save_state(root, state, {'tcn':head}, {'tcn':optimizer})


def micro_step(head, optimizer, model, x, y, weights, device, training):
    optimizer.zero_grad(set_to_none=True)
    denom = weights[torch.from_numpy(y).to(device)].sum()
    numerator = 0.
    for begin in range(0,len(y),training['microbatch_size']):
        end = min(begin+training['microbatch_size'],len(y))
        with torch.no_grad():
            feature = s.core.dense_features(model,torch.from_numpy(x[begin:end]).to(device))
        truth = torch.from_numpy(y[begin:end]).to(device)
        loss_sum = nn.functional.cross_entropy(head(feature).flatten(0,1),truth.flatten(),weight=weights,reduction='sum')
        io.require(bool(torch.isfinite(loss_sum)), 'nonfinite TCN loss')
        (loss_sum/denom).backward()
        numerator += float(loss_sum.item())
    parameters = [p for p in head.parameters() if p.requires_grad]
    io.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), 'TCN gradient')
    norm = nn.utils.clip_grad_norm_(parameters,training['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    return numerator, float(denom.item()), float(norm.item())


def run(args, config, root):
    contract = make_contract(config,args.smoke)
    s.lock_contract(root,contract,args.resume)
    if (root/'final_report.json').exists():
        io.require(io.read(root/'final_report.json')['passed'], 'final gate'); return
    s.pause(root,config)
    if not args.smoke:
        smoke_root = Path(str(root)+'_smoke')
        smoke = io.read(smoke_root/'final_report.json')
        io.require(smoke['passed'] and not smoke['research_usable'], 'smoke required')
        io.require({**io.read(smoke_root/'run_contract.json'),'smoke':False}==contract,'smoke contract differs')
    s.publish(root,'s0b','preflight',holdouts_opened=False)
    ac, ar, selected_a, base = s.s0a_dependency(config)
    train_data = s.core.verify_split(ac,'train'); val_data = s.core.verify_split(ac,'val')
    counts = train_data.frequencies()
    io.require(counts == io.read(io.ROOT/config['s0a_root']/'source_audit.json')['train_frequency'],'train frequency differs')
    io.save_json(root/'source_audit.json',{'passed':True,'train_frequency':counts,'splits':['train','val'],'holdouts_opened':False})
    if args.smoke:
        train_data.count = val_data.count = config['execution']['smoke_windows']
    device = io.device_for(config)
    model = a.load_adl_model(io.ROOT/ac['encoder'],io.ROOT/ac['adl_head'],device)
    torch.manual_seed(config['training']['seed'])
    head = core.ResidualStateHead(base,config['model']).to(device)
    encoder_hash = s.hash_named_tensors(model.state_dict().items())
    base_hash = s.hash_named_tensors(head.base.state_dict().items())
    optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad),lr=config['training']['lr'],
                                  betas=tuple(config['training']['betas']),eps=config['training']['eps'],
                                  weight_decay=config['training']['weight_decay'])
    optimizer_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    io.require(not optimizer_ids.intersection(id(p) for p in list(model.parameters())+list(head.base.parameters())), 'optimizer contains frozen params')
    # Initial tensors are exactly zero; first two real windows verify actual forward equality.
    head.eval()
    with torch.no_grad():
        x,_ = train_data.batch(np.arange(2));feature=s.core.dense_features(model,torch.from_numpy(x).to(device))
        io.require(torch.count_nonzero(head.correction(feature)).item()==0,'initial correction nonzero')
        io.require(torch.equal(head(feature),head.base(feature)),'initial dense logits not exact')
    init_path=root/'training/epoch_00.pt'
    if not init_path.exists():
        io.save_tensor(init_path,{'head':io.cpu_state(head),'epoch':0,'contract':contract})
    else:
        init=torch.load(init_path,map_location='cpu',weights_only=True)
        io.require(init['contract']==contract and s.hash_named_tensors(init['head'].items())==s.hash_named_tensors(head.state_dict().items()),'epoch0 changed')
    s.publish(root,'s0b','initial')
    reference=None if args.smoke else io.ROOT/config['s0a_root']/'evaluation/val'
    initial=s.evaluate(root,root/'validation/epoch_00',val_data,head,model,device,ac,config,zero_reference=reference)
    if not args.smoke:
        io.require(initial==ar['metrics']['val'],'full val epoch0 differs from S0-A')
    io.save_json(root/'initial_equivalence.json',{'passed':True,'window_logits_exact':True,'full_val_logits_exact':not args.smoke,
                                              'base_hash':base_hash,'encoder_hash':encoder_hash,'metrics':initial})
    state = a.restore_state(root,contract,{'tcn':head},{'tcn':optimizer})
    if 'cpu_rng' in state:
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
    commit(root,state,head,optimizer)
    raw_counts=np.asarray(counts['counts'],np.float64); w=1/np.sqrt(raw_counts);w/=w.mean()
    weights=torch.tensor(w,dtype=torch.float32,device=device)
    training=config['training'];epochs=config['execution']['smoke_epochs'] if args.smoke else training['epochs']
    try:
        while state['epoch']<=epochs:
            epoch=state['epoch'];head.train();head.base.eval()
            order=torch.randperm(train_data.count,generator=torch.Generator().manual_seed(training['seed']+epoch)).numpy()
            for begin in range(state['next'],train_data.count,training['batch_size']):
                s.pause(root,config); started=time.monotonic();end=min(begin+training['batch_size'],train_data.count)
                x,y=train_data.batch(order[begin:end])
                loss,denom,norm=micro_step(head,optimizer,model,x,y,weights,device,training)
                state['loss_sum']['tcn']+=loss;state['loss_denom']['tcn']+=denom
                state['train_seconds']+=time.monotonic()-started;state['next']=end
                if begin==0 or (end//training['batch_size'])%config['execution']['checkpoint_batches']==0 or end==train_data.count:
                    commit(root,state,head,optimizer)
                    s.publish(root,'s0b','train',epoch=epoch,epochs=epochs,windows=end,total=train_data.count,
                              mean_loss=state['loss_sum']['tcn']/state['loss_denom']['tcn'],gradient_norm=norm,
                              train_seconds=state['train_seconds'],gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3)
            frozen_check(model,head,encoder_hash,base_hash)
            io.require(make_contract(config,args.smoke)==contract,'code/config/dependency changed')
            dest=root/'training'/f'epoch_{epoch:02d}.pt'
            if dest.exists():
                old=torch.load(dest,map_location='cpu',weights_only=True)
                io.require(old['contract']==contract and s.hash_named_tensors(old['head'].items())==s.hash_named_tensors(head.state_dict().items()),'epoch state changed')
            else:
                io.save_tensor(dest,{'head':io.cpu_state(head),'epoch':epoch,'contract':contract})
            metrics=s.evaluate(root,root/'validation'/f'epoch_{epoch:02d}',val_data,head,model,device,ac,config)
            state['history'].append({'epoch':epoch,'metrics':metrics,'checkpoint_sha256':io.sha256_file(dest),
                                     'mean_loss':state['loss_sum']['tcn']/state['loss_denom']['tcn']})
            io.save_json(root/'training/history.json',state['history'])
            state.update(epoch=epoch+1,next=0,loss_sum={'tcn':0.},loss_denom={'tcn':0.})
            commit(root,state,head,optimizer)
    except (io.PauseRequested,KeyboardInterrupt):
        commit(root,state,head,optimizer); raise
    frozen_check(model,head,encoder_hash,base_hash)
    if args.smoke:
        # CPU replay of the trained residual, with dropout disabled, on fixed real features.
        head.eval();cpu=core.ResidualStateHead(base,config['model']).eval();cpu.load_state_dict(io.cpu_state(head))
        with torch.no_grad():
            expected=cpu(feature.cpu()).numpy();actual=head(feature).cpu().numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-4)
        report={'passed':True,'research_usable':False,'epochs':epochs,'effective_batch':training['batch_size'],
                'train_seconds':state['train_seconds'],'initial_equivalence':True,'frozen_models_unchanged':True,
                'cpu_head_max_error':float(np.max(np.abs(actual-expected))),
                'gpu_peak_gib':torch.cuda.max_memory_allocated()/1024**3,'holdouts_opened':False}
        io.require(make_contract(config,True)==contract,'smoke fingerprint changed')
        io.save_json(root/'final_report.json',report);s.publish(root,'s0b','completed',**report);return
    best={'epoch':0,'metrics':initial,'checkpoint_sha256':io.sha256_file(init_path)}
    for row in state['history']:
        if s.core.selection_key(row['metrics'])>s.core.selection_key(best['metrics']):
            best={k:row[k] for k in ('epoch','metrics','checkpoint_sha256')}
    gate=core.adoption_b(best['metrics'],initial,config['evaluation'])
    selection={**best,'gate':gate,'selected_using':'val only','run_contract_sha256':io.sha256_file(root/'run_contract.json'),
               'history_sha256':io.sha256_file(root/'training/history.json'),'holdouts_opened_at_selection':False}
    if (root/'selection.json').exists():
        io.require(io.read(root/'selection.json')==selection,'selection changed')
    else:
        io.save_json(root/'selection.json',selection)
    selection_sha=io.sha256_file(root/'selection.json')
    s.publish(root,'s0b','selection',epoch=best['epoch'],adoption=gate['adoption'])
    selected_path=root/'training'/f'epoch_{best["epoch"]:02d}.pt'
    io.require(io.sha256_file(selected_path)==best['checkpoint_sha256'],'selected head changed')
    head.load_state_dict(torch.load(selected_path,map_location='cpu',weights_only=True)['head'])
    final_metrics={}
    for split in ('val','test','ood'):
        s.pause(root,config)
        io.require(io.sha256_file(root/'selection.json')==selection_sha,'selection changed before holdout')
        data=val_data if split=='val' else s.core.verify_split(ac,split)
        final_metrics[split]=s.evaluate(root,root/'evaluation'/split,data,head,model,device,ac,config,save_logits=True)
        if split=='val':io.require(final_metrics[split]==best['metrics'],'selected val replay differs')
    s.publish(root,'s0b','audit')
    frozen_check(model,head,encoder_hash,base_hash)
    io.require(make_contract(config,False)==contract,'final contract changed')
    for split in ac['input']['split_counts']:s.core.verify_split(ac,split)
    report={'passed':True,'research_usable':True,'historical_exact_reproduction':False,'epochs':epochs,
            'selected_epoch':best['epoch'],'selection_sha256':selection_sha,'gate':gate,'adoption':gate['adoption'],
            'metrics':final_metrics,'initial_full_val_exact':True,'frozen_models_unchanged':True,
            'encoder_hash':encoder_hash,'s0a_base_hash':base_hash,'train_seconds':state['train_seconds'],
            'automatic_deployment':False,'source_splits_reverified':True}
    io.save_json(root/'final_report.json',report)
    s.publish(root,'s0b','completed',epochs=epochs,adoption=gate['adoption'],selected_epoch=best['epoch'],
              final_report_sha256=io.sha256_file(root/'final_report.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=io.read(CONFIG);root=io.guard(io.ROOT/(config['output']+('_smoke' if args.smoke else '')))
    io.require(root.parent==io.ROOT/'checkpoint/fall','output scope');root.mkdir(parents=True,exist_ok=True)
    s.install_signals()
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args,config,root)
        except io.PauseRequested as error:
            s.publish(root,'s0b','paused',reason=str(error));raise SystemExit(75)
        except Exception as error:
            s.publish(root,'s0b','failed',reason=str(error));raise


if __name__=='__main__':main()
