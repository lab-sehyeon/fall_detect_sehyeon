"""Sequential, hash-locked engineering smoke on a preregistered OOPS subset."""
import argparse
from datetime import datetime,timezone
import fcntl
import os
import subprocess
import sys
import numpy as np
from . import rgb_document_io as io

def initialize(config):
    root=io.ROOT/config['output'];root.mkdir(parents=True,exist_ok=True)
    spec=io.contract(config);path=root/'contract.json'
    if path.exists():io.require(io.read(path)==spec,'different code/config: new revision required')
    else:io.save(path,spec)
    source=io.read(io.ROOT/config['alignment']);names=sorted(source)[:config['sample_count']];plan=[]
    for index,name in enumerate(names,1):
        item=source[name];video=io.ROOT/config['video_root']/name
        io.require(video.resolve().is_relative_to(io.ROOT/config['video_root']),'path escape')
        io.require(io.sha(video)==item['source_sha256'],'video hash')
        io.require(io.sha(io.ROOT/item['mapping_file'])==item['mapping_sha256'],'mapping hash')
        plan.append(dict(id=f'oops_{index:04d}',video=str(video.relative_to(io.ROOT)),video_sha256=item['source_sha256'],
                         mapping=item['mapping_file'],mapping_sha256=item['mapping_sha256'],frames=item['canonical_frames']))
    path=root/'plan.json'
    if path.exists():io.require(io.read(path)==plan,'plan changed')
    else:io.save(path,plan)
    return root,plan

def publish(root,stage,**details):
    value=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**details)
    io.save(root/'status.json',value);print(value,flush=True)

def audit(config):
    from fall_pipeline.common.global_reconstruction import raw_channels,causal_arrays,features_from_causal,reference_feature
    from data_gen.safer_v2_geometry import normalize_ntu
    from data_gen.safer_v3_geometry import fuse
    import torch
    from fall_pipeline.joint.models import JointFallModel
    root,plan=io.locked(config);rows=[];glob=io.read(io.ROOT/config['inference']['global_config'])
    selection=io.read(io.ROOT/glob['output_dir']/'selection.json')['best']
    joint=JointFallModel().eval().requires_grad_(False)
    joint.load_state_dict(torch.load(io.ROOT/glob['joint_root']/'final/final_j1.pt',map_location='cpu',weights_only=True))
    with np.load(io.ROOT/glob['output_dir']/'train_scaler.npz',allow_pickle=False) as z:mean,scale=z['mean'],z['scale']
    for item in plan:
        io.safety(config);dest=root/item['id'];front=io.stage_done(dest,'frontend');io.require(front,'frontend missing')
        io.require(io.sha(io.ROOT/item['video'])==item['video_sha256'],'video changed')
        io.require(io.sha(io.ROOT/item['mapping'])==item['mapping_sha256'],'mapping changed')
        with np.load(io.ROOT/item['mapping'],allow_pickle=False) as mapping, np.load(dest/'frontend.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['source_indices'],mapping['source_indices'])
            np.testing.assert_array_equal(f['timestamps'],mapping['canonical_timestamps'])
            io.require((mapping['source_timestamps'][f['source_indices']]<=f['timestamps']).all(),'future frame')
            io.require(io.quality(f['boxes'],f['xy'],f['scores'],config['quality'])==front['quality'],'quality mismatch')
            previous=None
            for j in range(len(f['boxes'])):
                cand=f['candidates'][f['offsets'][j]:f['offsets'][j+1]]
                picked=io.select_track(cand,previous)
                np.testing.assert_array_equal(f['boxes'][j],np.zeros(5,np.float32) if picked is None else picked)
                if picked is not None:previous=picked
                io.require(not f['rescue'][j] or len(cand)==1,'fallback top1')
            a=io.annotation(f)
        row=dict(id=item['id'],frames=front['frames'],quality=front['quality'],inference_run=False)
        if front['quality']['passed']:
            lift=io.stage_done(dest,'lift');infer=io.stage_done(dest,'inference');io.require(lift and infer,'missing derived stages')
            with np.load(dest/'lift.npz',allow_pickle=False) as l, np.load(dest/'inference.npz',allow_pickle=False) as p:
                h,coverage,_=fuse(l['lifting_raw'],l['lifting_starts'],len(l['ntu25']),'triangular',.05,True)
                np.testing.assert_array_equal(l['h36m'],h);np.testing.assert_array_equal(l['coverage'],coverage)
                ntu,_,_=normalize_ntu(h,.5);np.testing.assert_array_equal(l['ntu25'],ntu)
                starts=np.arange(0,len(ntu)-63,8,dtype=np.int64)
                np.testing.assert_array_equal(l['window_starts'],starts);np.testing.assert_array_equal(p['window_starts'],starts)
                channels,mask=raw_channels(a);causal=causal_arrays(channels,mask)
                g=features_from_causal(causal,mask,starts);np.testing.assert_array_equal(l['global131'],g)
                for j,s in enumerate(starts):np.testing.assert_array_equal(g[j],reference_feature(causal,mask,int(s)))
                std=((g.astype(np.float64)-mean)/scale).astype(np.float32);np.testing.assert_array_equal(std,p['global_standardized'])
                with torch.inference_mode():adapted=joint.adapter(torch.from_numpy(p['pooled'])).numpy()
                np.testing.assert_allclose(adapted,p['adapted'],atol=1e-4,rtol=1e-5)
                for name,x in [('G0',p['adapted']),('G1',std),('G2',np.column_stack((p['adapted'],std)))]:
                    head=torch.nn.Linear(x.shape[1],4).eval().requires_grad_(False)
                    state=torch.load(io.ROOT/glob['output_dir']/name/f'epoch_{selection[name]["epoch"]:03d}.pt',map_location='cpu',weights_only=True)
                    head.load_state_dict(state)
                    with torch.inference_mode():pred=head(torch.from_numpy(x)).numpy()
                    np.testing.assert_allclose(pred,p[name],atol=1e-4,rtol=1e-5)
                row.update(inference_run=True,windows=len(starts),cpu_head_max_abs=infer['cpu_head_max_abs'])
        else:
            io.require(not (dest/'inference.npz').exists() and not (dest/'lift.npz').exists(),'low quality bypass')
        rows.append(row)
    io.locked(config)
    report=dict(passed=True,integration_exercised=any(r['inference_run'] for r in rows),videos=len(rows),
                quality_pass=sum(r['quality']['passed'] for r in rows),classified=sum(r['inference_run'] for r in rows),
                windows=sum(r.get('windows',0) for r in rows),rows=rows,training=False,external_performance_evaluated=False,
                recovery_inference=False,historical_exact_reproduction=False,scope='fixed3 engineering smoke; no event metrics',
                plan_sha256=io.sha(root/'plan.json'),contract_sha256=io.sha(root/'contract.json'))
    io.save(root/'final_report.json',report);publish(root,'completed',quality_pass=report['quality_pass'],classified=report['classified'])
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['plan','all','audit'],default='all')
    args=parser.parse_args();config=io.read(io.CONFIG);root=io.ROOT/config['output'];root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            io.safety(config);root,plan=initialize(config)
            if args.stage=='plan':publish(root,'planned',videos=len(plan));return
            if args.stage=='all':
                for stage in ['frontend','lift','infer']:
                    io.safety(config);publish(root,stage)
                    subprocess.run([sys.executable,'-m','fall_pipeline.external.rgb_document_'+stage],cwd=io.ROOT,check=True)
            publish(root,'audit');audit(config)
        except Exception as error:
            publish(root,'paused',reason=f'{type(error).__name__}: {error}');raise

if __name__=='__main__':main()
