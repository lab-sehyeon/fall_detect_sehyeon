"""Fresh-process MotionAGFormer V3 lifting; never import the DSTE model here."""
import numpy as np
from . import rgb_document_io as io

def main():
    import torch
    from data_gen import safer_v2_reconstruction as v2
    from data_gen import safer_v3_geometry as geometry
    from fall_pipeline.common.global_reconstruction import raw_channels,causal_arrays,features_from_causal,reference_feature
    from fall_pipeline.common.integrity import hash_named_tensors
    config=io.read(io.CONFIG);root,plan=io.locked(config);dev=io.device(config)
    source=io.read(io.ROOT/config['lifting_source_config']);model=v2.load_model(source,dev)
    before=hash_named_tensors(model.state_dict().items())
    for item in plan:
        io.safety(config);dest=root/item['id'];front=io.stage_done(dest,'frontend')
        io.require(front,'missing frontend')
        if not front['quality']['passed']:
            io.save(dest/'quality_rejection.json',dict(reason='frozen frontend quality gate',quality=front['quality'],classifier_run=False))
            continue
        if io.stage_done(dest,'lift'):continue
        with np.load(dest/'frontend.npz',allow_pickle=False) as saved:
            a=io.annotation(saved)
        values,repairs=v2.model_input(a['keypoint'][0],a['keypoint_score'][0],a['width'],a['height'])
        starts=v2.starts_for(len(values),stride=121);parts=[]
        with torch.inference_mode():
            for begin in range(0,len(starts),4):
                io.safety(config);x=v2.windows_at(values,starts[begin:begin+4])
                normal=model(torch.from_numpy(x).to(dev)).cpu().numpy()
                flip=model(torch.from_numpy(v2.flip_numpy(x)).to(dev)).cpu().numpy()
                y=(normal+v2.flip_numpy(flip))/2;y[:,:,0]=0;parts.append(y)
        raw=np.concatenate(parts);h,cover,divisor=geometry.fuse(raw,starts,len(values),'triangular',.05)
        independent=geometry.fuse(raw,starts,len(values),'triangular',.05,True)
        for x,y in zip((h,cover,divisor),independent):np.testing.assert_array_equal(x,y)
        ntu,trajectory,norm=v2.normalize_ntu(h,source['ntu_proxy']['reference_torso'])
        windows=np.arange(0,len(ntu)-64+1,8,dtype=np.int64)
        channels,mask=raw_channels(a);causal=causal_arrays(channels,mask)
        global131=features_from_causal(causal,mask,windows)
        for i,s in enumerate(windows):np.testing.assert_array_equal(global131[i],reference_feature(causal,mask,int(s)))
        io.require(np.all(h[:,0]==0) and np.all(ntu[:,1]==0),'root invariant')
        io.npz(dest/'lift.npz',h36m=h,ntu25=ntu,root_trajectory=trajectory,lifting_raw=raw,lifting_starts=starts,
               global131=global131,window_starts=windows,channels=channels,valid=mask,coverage=cover)
        after=hash_named_tensors(model.state_dict().items());io.require(before==after,'lifting weights mutated')
        io.save(dest/'lift.json',dict(passed=True,payload_sha256=io.sha(dest/'lift.npz'),normalization=norm,
            frames=len(ntu),windows=len(windows),model_before=before,model_after=after,repairs=repairs,
            global_independent_exact=True,overlap_independent_exact=True,offline=True,labels_used=False))
        print('lift',item['id'],len(ntu),len(windows),'PASS',flush=True)
    io.locked(config)

if __name__=='__main__':main()
