"""Predeclared, label-independent FU skeleton-to-DSTE spot replay."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.common.eval_safer_legacy_v1_candidates import load_adl_model
from fall_pipeline.common.integrity import hash_named_tensors

def main():
    output=ROOT/'docs/internal/2026-10-02_evidence_audit'
    config=io.read(ROOT/'configs/f1_safer_document_reconstruction_v2.json')
    fu=ROOT/io.read(ROOT/'configs/fu_zs_document_reconstruction_v1.json')['fu_root']
    feature=ROOT/io.read(ROOT/'configs/joint_document_reconstruction_v1.json')['fu']['features']
    indices=[0,248,496,744,992]  # Evenly spaced manifest indices, chosen without labels/predictions.
    spec=dict(indices=indices,selection='five evenly spaced indices, no outcome filtering',
        source='preprocessed FU skeleton; official raw ingestion not independently replayed here',
        encoder_sha256=io.sha(ROOT/config['input']['encoder']),feature_sha256=io.sha(feature),
        source_sha256={p:io.sha(fu/p) for p in ['data_joint.npy','num_frame.npy']},
        code_sha256=io.sha(Path(__file__)),atol=1e-4,rtol=1e-5)
    io.save(output/'raw_feature_replay_contract.json',spec)
    dev=io.device(io.read(ROOT/'configs/rgb_document_integration_v1.json'))
    model=load_adl_model(ROOT/config['input']['encoder'],ROOT/config['input']['adl_head'],dev)
    before=hash_named_tensors(model.state_dict().items())
    data=np.load(fu/'data_joint.npy',mmap_mode='r');lengths=np.load(fu/'num_frame.npy')
    results=[]
    with np.load(feature,allow_pickle=False) as saved:
        for idx in indices:
            length=int(lengths[idx]);last=max(0,length-64)
            starts=sorted(set(range(0,last+1,8))|{last})
            rows=[]
            for start in starts:
                x=np.array(data[idx,:,start:min(start+64,length)],copy=True)
                if x.shape[1]<64:x=np.concatenate([x,np.repeat(x[:,-1:],64-x.shape[1],axis=1)],axis=1)
                rows.append(x)
            x=np.stack(rows);t=[];s=[]
            with torch.inference_mode():
                for begin in range(0,len(x),128):
                    batch=torch.from_numpy(x[begin:begin+128]).to(dev);n=len(batch)
                    temporal,spatial=model.backbone(batch.permute(0,2,4,3,1).reshape(n,64,150),batch.permute(0,4,3,2,1).reshape(n,50,192))
                    for j in range(n):
                        width=min(64,length-starts[begin+j])
                        t.append(temporal[j,:width].max(0).values.cpu().numpy())
                        s.append(spatial[j].max(0).values.cpu().numpy())
            errors={}
            for key,parts in [('t',t),('s',s)]:
                value=np.asarray(parts).astype(np.float64).mean(0).astype(np.float32)
                expected=saved[key][idx];np.testing.assert_allclose(value,expected,atol=1e-4,rtol=1e-5)
                errors[key]=float(np.max(np.abs(value-expected)))
            results.append(dict(index=idx,frames=length,windows=len(starts),max_abs_error=errors))
            print(results[-1],flush=True)
    assert hash_named_tensors(model.state_dict().items())==before
    io.save(output/'raw_feature_replay.json',dict(passed=True,rows=results,contract_sha256=io.sha(output/'raw_feature_replay_contract.json'),
        frozen_weights_unchanged=True,limitation='spot-check five of 993 preprocessed clips, not exhaustive raw ingestion audit'))

if __name__=='__main__':main()
