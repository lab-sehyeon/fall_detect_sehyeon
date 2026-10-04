"""Frozen reconstructed DSTE/J1/global heads; no optimizer or event evaluation."""
import numpy as np
from . import rgb_document_io as io

def main():
    import torch
    from fall_pipeline.common.eval_safer_legacy_v1_candidates import load_adl_model
    from fall_pipeline.safer.extract_f0b_safer_features import pooled_backbone_features
    from fall_pipeline.joint.models import JointFallModel
    from fall_pipeline.common.integrity import hash_named_tensors
    config=io.read(io.CONFIG);root,plan=io.locked(config);dev=io.device(config)
    cfg=io.read(io.ROOT/config['inference']['global_config'])
    controls=io.read(io.ROOT/'configs/safer_v2_controls_document_reconstruction_v1.json')
    backbone=load_adl_model(io.ROOT/controls['encoder'],io.ROOT/controls['adl_root']/'best_adl_head.pth',dev).eval().requires_grad_(False)
    joint=JointFallModel().eval().requires_grad_(False)
    joint.load_state_dict(torch.load(io.ROOT/cfg['joint_root']/'final/final_j1.pt',map_location='cpu',weights_only=True),strict=True);joint.to(dev)
    selection=io.read(io.ROOT/cfg['output_dir']/'selection.json')['best'];heads={};cpu_heads={}
    for name,dim in [('G0',2048),('G1',131),('G2',2179)]:
        state=torch.load(io.ROOT/cfg['output_dir']/name/f'epoch_{selection[name]["epoch"]:03d}.pt',map_location='cpu',weights_only=True)
        cpu=torch.nn.Linear(dim,4).eval().requires_grad_(False);cpu.load_state_dict(state,strict=True);cpu_heads[name]=cpu
        heads[name]=torch.nn.Linear(dim,4).eval().requires_grad_(False).to(dev);heads[name].load_state_dict(state,strict=True)
    with np.load(io.ROOT/cfg['output_dir']/'train_scaler.npz',allow_pickle=False) as z:mean,scale=z['mean'],z['scale']
    models={'DSTE_ADL':backbone,'J1':joint,**heads};before={k:hash_named_tensors(m.state_dict().items()) for k,m in models.items()}
    for item in plan:
        io.safety(config);dest=root/item['id'];front=io.stage_done(dest,'frontend')
        io.require(front,'frontend missing')
        if not front['quality']['passed']:continue
        io.require(io.stage_done(dest,'lift'),'lifting missing')
        if io.stage_done(dest,'inference'):continue
        with np.load(dest/'lift.npz',allow_pickle=False) as z:ntu,starts,g=z['ntu25'],z['window_starts'],z['global131']
        standardized=((g.astype(np.float64)-mean)/scale).astype(np.float32)
        out={k:[] for k in ['G0','G1','G2','adl','pooled','adapted']};maximum=0.
        with torch.inference_mode():
            for start in range(0,len(starts),config['inference']['batch']):
                io.safety(config);stop=start+config['inference']['batch']
                batch=torch.from_numpy(io.windows(ntu,starts[start:stop])).to(dev)
                t,s=pooled_backbone_features(backbone,batch);pooled=torch.cat((t,s),1);adapted=joint.adapter(pooled)
                globalx=torch.from_numpy(standardized[start:stop]).to(dev)
                for name,x in [('G0',adapted),('G1',globalx),('G2',torch.cat((adapted,globalx),1))]:
                    logits=heads[name](x).cpu().numpy();reference=cpu_heads[name](x.cpu()).numpy()
                    np.testing.assert_allclose(logits,reference,atol=config['inference']['cpu_head_atol'],rtol=config['inference']['cpu_head_rtol'])
                    maximum=max(maximum,float(np.max(np.abs(logits-reference))));out[name].append(logits)
                out['adl'].append(backbone.fc(pooled).cpu().numpy());out['pooled'].append(pooled.cpu().numpy());out['adapted'].append(adapted.cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in out.items()}
        io.require(all(np.isfinite(v).all() for v in arrays.values()),'nonfinite inference')
        io.npz(dest/'inference.npz',**arrays,window_starts=starts,window_endpoints=starts+63,global_standardized=standardized)
        after={k:hash_named_tensors(m.state_dict().items()) for k,m in models.items()};io.require(before==after,'frozen classifier changed')
        io.save(dest/'inference.json',dict(passed=True,payload_sha256=io.sha(dest/'inference.npz'),windows=len(starts),
            cpu_head_max_abs=maximum,models_before=before,models_after=after,
            fall_argmax_windows={k:int((arrays[k].argmax(1)==1).sum()) for k in heads},
            research_performance_claim=False,event_evaluation=False,recovery_inference=False,training=False))
        print('inference',item['id'],len(starts),'CPU check',maximum,flush=True)
    io.locked(config)

if __name__=='__main__':main()
