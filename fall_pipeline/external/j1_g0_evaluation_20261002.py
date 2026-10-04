"""Frozen standalone J1+G0 re-inference on validated continuous-video inputs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,fcntl
import numpy as np
from . import rgb_document_io as io
from .le2i_current_evaluation import metrics,rising_edges,match_events

ROOT=io.ROOT
OUTPUT=ROOT/'data/fall_processed/RGB/j1_g0_only_20261002_r1'
CONFIG=ROOT/'configs/j1_g0_only_evaluation_20261002_r1.json'
SOURCES={
    'le2i_cs38':dict(root='data/fall_processed/RGB/omnifall_le2i_continuous_20261002_r1',
        config='configs/omnifall_le2i_continuous_20261002_r1.json',videos=38,positive=22,unit='event',
        audit='data/fall_processed/RGB/omnifall_le2i_continuous_20261002_r1/independent_audit.json'),
    'le2i_127':dict(root='data/fall_processed/RGB/le2i_current_evaluation_20261002_r1',
        config='configs/le2i_current_evaluation_20261002_r1.json',videos=127,positive=96,unit='event',
        audit='data/fall_processed/RGB/le2i_current_evaluation_20261002_r1/independent_audit.json'),
    'urfd70':dict(root='data/fall_processed/RGB/external_urfd_20261002_r1',
        config='configs/external_urfd_20261002_r1.json',videos=70,positive=30,unit='sequence',
        audit='docs/internal/2026-10-02_evidence_audit/urfd_independent_audit.json')}
CODE=['fall_pipeline/external/j1_g0_only.py','fall_pipeline/external/j1_g0_evaluation_20261002.py',
      'configs/active_fall_model.json',
      'scripts/run_j1_g0_evaluation_20261002.py','scripts/audit_j1_g0_evaluation_20261002.py',
      'fall_pipeline/external/rgb_document_io.py','fall_pipeline/external/le2i_current_evaluation.py',
      'fall_pipeline/external/rgb_recovery_common.py','fall_pipeline/fu/fall_fu_linear_eval.py',
      'fall_pipeline/joint/models.py','fall_pipeline/common/integrity.py','model/DSTE.py','tools.py']


def status(stage,**values):
    io.save(OUTPUT/'status.json',dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**values))
    print(stage,values,flush=True)


def budget():
    cfg=io.read(CONFIG);io.safety(cfg)
    io.require(sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())<2*1024**3,'2GiB output budget')


def digest():
    cfg=io.read(CONFIG)
    weights={k:dict(path=v['path'],sha256=io.sha(ROOT/v['path'])) for k,v in cfg['model'].items()}
    io.require(weights==cfg['model'],'selected model changed')
    return dict(config_sha256=io.sha(CONFIG),model=weights,code={p:io.sha(ROOT/p) for p in CODE},
        source_snapshot_sha256=io.sha(OUTPUT/'source_snapshot.json'),plan_sha256=io.sha(OUTPUT/'plan.json'))


def locked():
    io.require(digest()==io.read(OUTPUT/'contract.json'),'J1+G0 contract changed')
    return io.read(CONFIG),io.read(OUTPUT/'plan.json')


def source_snapshot():
    result={}
    for spec in SOURCES.values():
        root=ROOT/spec['root']
        paths=[ROOT/spec['config'],ROOT/spec['audit'],root/'contract.json',root/'plan.json',
               root/'ground_truth.json',root/'evaluation.json']
        for item in io.read(root/'plan.json'):
            dest=root/item['id'];paths.append(dest/'frontend.json')
            if (dest/'frontend.npz').exists():paths.append(dest/'frontend.npz')
            if item.get('mapping'):paths.append(ROOT/item['mapping'])
            if io.stage_done(dest,'frontend')['quality']['passed']:
                for stage in ['lift','inference']:
                    io.require(io.stage_done(dest,stage)['passed'],'source stage invalid')
                    paths.extend([dest/(stage+'.json'),dest/(stage+'.npz')])
        for p in paths:result[str(p.relative_to(ROOT))]=io.sha(p)
    return result


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if CONFIG.exists():locked();return
    from . import omnifall_le2i_continuous_20261002 as cs
    from . import le2i_current_evaluation as legacy
    from . import paper_external_20261002 as urfd
    for fn in [cs.setup,legacy.setup,lambda:urfd.setup('urfd')[0]]:
        old_cfg=fn();io.locked(old_cfg)
    cfg0=io.read(ROOT/SOURCES['le2i_cs38']['config'])
    parent=io.read(ROOT/SOURCES['le2i_cs38']['root']/'contract.json')
    global_cfg=io.read(ROOT/cfg0['inference']['global_config'])
    selected=io.read(ROOT/global_cfg['output_dir']/'selection.json')['best']['G0']
    controls=io.read(ROOT/'configs/safer_v2_controls_document_reconstruction_v1.json')
    model_paths=dict(encoder=controls['encoder'],j1=global_cfg['joint_root']+'/final/final_j1.pt',
        g0=global_cfg['output_dir']+f'/G0/epoch_{selected["epoch"]:03d}.pt')
    io.require(selected['epoch']==5,'G0 source selection changed')
    model={k:dict(path=p,sha256=io.sha(ROOT/p)) for k,p in model_paths.items()}
    io.require(all(v['sha256']==parent['frozen_files'][v['path']] for v in model.values()),'active lineage mismatch')
    plan=[]
    for scope,spec in SOURCES.items():
        root=ROOT/spec['root'];audit=io.read(ROOT/spec['audit'])
        io.require(audit['passed'] and audit['evaluation_sha256']==io.sha(root/'evaluation.json'),'source audit invalid')
        source_cfg=io.read(ROOT/spec['config'])
        for key in ['detector','pose_config','quality','lifting','inference']:
            io.require(source_cfg[key]==cfg0[key],'source model/preprocessing differs: '+key)
        source_contract=io.read(root/'contract.json')
        io.require(all(source_contract['frozen_files'][v['path']]==v['sha256'] for v in model.values()),'source weights mismatch')
        originals=io.read(root/'plan.json');truth={r['id']:r for r in io.read(root/'ground_truth.json')}
        io.require(len(originals)==spec['videos'],'source sample count')
        for item in originals:
            front=io.stage_done(root/item['id'],'frontend')
            plan.append(dict(scope=scope,id=item['id'],frames=item['frames'],unit=spec['unit'],
                source=str((root/item['id']).relative_to(ROOT)),quality=front['quality'],
                classified=front['quality']['passed'],truth=truth[item['id']]))
    io.save(OUTPUT/'plan.json',plan);io.save(OUTPUT/'source_snapshot.json',source_snapshot())
    cfg=dict(experiment_id='J1_G0_ONLY_20261002_R1',output=str(OUTPUT.relative_to(ROOT)),
        model=model,sources=SOURCES,architecture='frozen DSTE backbone -> J1 adapter -> G0 Linear2048x4',
        excluded_active_branches=['ADL60','J1.safer_head','J1.fu_head','G1','G2','global131','global_scaler'],
        inference=dict(window=64,stride=8,append_tail=False,batch=32,fps=25,atol=1e-4,rtol=1e-5,
            classes=['other','fall','lie_down','lying_down']),
        preprocessing='reuse validated whole-video frontend/3D tensors unchanged; no fresh pixel/pose inference',
        evaluation=dict(event='existing D1 rising edge early0.5 late3 seconds',
            sequence='existing any class1 complete-window prediction',quality_rejection='no alarm, retain all inputs',
            retraining=False,threshold_fitting=False,selection=False,old_outputs_used_only_for_regression=True),
        execution=copy.deepcopy(cfg0['execution']))
    cfg['execution']['max_added_gib']=2
    io.save(CONFIG,cfg);io.save(OUTPUT/'contract.json',digest())
    status('prepared',evaluation_entries=len(plan),classified=sum(p['classified'] for p in plan),
        windows=sum((p['frames']-64)//8+1 for p in plan if p['classified']))


def infer():
    import torch
    from .j1_g0_only import load_model
    from fall_pipeline.common.integrity import hash_named_tensors
    cfg,plan=locked();dev=io.device(cfg);model=load_model(ROOT,cfg['model'],dev)
    before=hash_named_tensors(model.state_dict().items())
    io.require(not any(p.requires_grad for p in model.parameters()),'model not frozen')
    runtime=dict(children=list(dict(model.named_children())),parameter_count=sum(p.numel() for p in model.parameters()),
        adapter_parameters=sum(p.numel() for p in model.adapter.parameters()),
        head_parameters=sum(p.numel() for p in model.head.parameters()),
        input_arrays_read=['ntu25','window_starts'],model_before=before,
        optimizer_created=False,global_features_loaded=False,adl_head_loaded=False,
        extra_heads_created=False,model_artifacts=list(cfg['model']),batch=32)
    for n,item in enumerate(plan,1):
        budget();dest=OUTPUT/item['scope']/item['id'];dest.mkdir(parents=True,exist_ok=True)
        if not item['classified']:
            io.save(dest/'rejection.json',dict(classifier_run=False,quality=item['quality']));continue
        if io.stage_done(dest,'inference'):continue
        source=ROOT/item['source']
        with np.load(source/'lift.npz',allow_pickle=False) as z:ntu,starts=z['ntu25'],z['window_starts']
        np.testing.assert_array_equal(starts,np.arange(0,item['frames']-63,8))
        values={k:[] for k in ['pooled','adapted','G0']}
        with torch.inference_mode():
            for first in range(0,len(starts),cfg['inference']['batch']):
                budget();batch=torch.from_numpy(io.windows(ntu,starts[first:first+32])).to(dev)
                out=model(batch)
                io.require(set(out)==set(values),'unexpected branch output')
                for k in values:values[k].append(out[k].cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in values.items()}
        io.require(all(np.isfinite(v).all() for v in arrays.values()),'nonfinite prediction')
        io.npz(dest/'inference.npz',**arrays,window_starts=starts,window_endpoints=starts+63)
        after=hash_named_tensors(model.state_dict().items());io.require(before==after,'model mutation')
        io.save(dest/'inference.json',dict(passed=True,windows=len(starts),payload_sha256=io.sha(dest/'inference.npz'),
            model_before=before,model_after=after,training=False,only_head='G0'))
        status('infer',entry=n,total=len(plan),scope=item['scope'],id=item['id'])
    runtime['model_after']=hash_named_tensors(model.state_dict().items())
    io.require(runtime['model_after']==before,'final model mutation')
    io.save(OUTPUT/'runtime.json',runtime);locked();status('inference_completed')


def binary_metrics(rows):
    from sklearn.metrics import average_precision_score
    y=np.array([r['truth']['label'] for r in rows]);p=np.array([r['prediction'] for r in rows])
    tp=int(sum((y==1)&(p==1)));fp=int(sum((y==0)&(p==1)));fn=int(sum((y==1)&(p==0)));tn=int(sum((y==0)&(p==0)))
    return dict(metrics(tp,fp,fn),tn=tn,accuracy=float(np.mean(y==p)),specificity=tn/(tn+fp),
        ap=float(average_precision_score(y,[r['max_fall_probability'] for r in rows])))


def score():
    cfg,plan=locked();rows=[]
    for item in plan:
        dest=OUTPUT/item['scope']/item['id'];predictions=[];prob=0.;positive=False
        if item['classified']:
            io.require(io.stage_done(dest,'inference'),'missing inference')
            with np.load(dest/'inference.npz') as z:
                logits=z['G0'];predictions=rising_edges(z['window_endpoints'],logits)
                positive=bool((logits.argmax(1)==1).any())
                ex=np.exp(logits.astype(float)-logits.max(1,keepdims=True));prob=float((ex/ex.sum(1,keepdims=True))[:,1].max())
        else:
            io.require(not (dest/'inference.npz').exists(),'rejection bypass')
        row=dict(item,predictions=predictions,prediction=int(positive),max_fall_probability=prob)
        if item['unit']=='event':row['metrics']=match_events(predictions,item['truth']['episodes'])
        rows.append(row)
    evaluations={}
    for scope,spec in SOURCES.items():
        selected=[r for r in rows if r['scope']==scope]
        if spec['unit']=='event':m=metrics(*(sum(r['metrics'][k] for r in selected) for k in ['tp','fp','fn']))
        else:m=binary_metrics(selected)
        evaluations[scope]=dict(unit=spec['unit'],videos=len(selected),positive=spec['positive'],metrics=m,
            classified=sum(r['classified'] for r in selected),
            rejected_positive=sum((len(r['truth']['episodes']) if r['unit']=='event' else r['truth']['label']) for r in selected if not r['classified']))
    result=dict(passed=True,architecture='DSTE + J1 adapter + G0 only',datasets=evaluations,rows=rows,
        no_training=True,no_target_tuning=True,contract_sha256=io.sha(OUTPUT/'contract.json'))
    io.save(OUTPUT/'evaluation.json',result);status('scored_pending_audit',datasets=evaluations)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','score']);a=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare':prepare,'infer':infer,'score':score}[a.stage]()


if __name__=='__main__':main()
