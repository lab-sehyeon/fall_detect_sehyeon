"""Train the revision heads on explicitly supplied pooled feature caches.

This entry point does not construct skeletons or change source splits.
Only training and validation arrays are accepted; no test data are opened.
"""
from pathlib import Path
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fall_pipeline.revision import training as t
from fall_pipeline.joint.models import JointFallModel

HERE = Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--condition', choices=['raw','no_fu','full'], required=True)
    p.add_argument('--seed', type=int, choices=range(5), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda:0'], default='cuda:0')
    a=p.parse_args()
    if a.device.startswith('cuda'):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        assert torch.cuda.device_count()==1
    device=torch.device(a.device)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((HERE/'configs/revision_training.json').read_text())
    jp=protocol['joint']; cp=protocol['classifier']
    jp['training']['seed']=cp['training']['seed']=a.seed
    hashes={}
    def load(split, n, classes):
        data={}
        for key, name, dtype in [('x','features',torch.float32),('y','labels',torch.long)]:
            path=a.cache/f'{split}_{name}.npy'; hashes[path.name]=t.sha256(path)
            data[key]=torch.tensor(np.load(path,allow_pickle=False),dtype=dtype,device=device)
        assert data['x'].shape==(n,2048) and data['y'].shape==(n,)
        assert bool(((data['y']>=0)&(data['y']<classes)).all())
        return data
    train=load('train',609183,4); val=load('val',106680,4)
    fu=load('fu',993,2) if a.condition=='full' else None
    contract=dict(scope='revision feature-level training',seed=a.seed,condition=a.condition,inputs=hashes,
                  torch=torch.__version__,device=a.device,bit_exact_cross_device_claim=False)
    a.output.mkdir(parents=True,exist_ok=True)
    if a.condition!='raw':
        j0=t.run_joint_stage(a.output/'j0','j0',jp,contract,device,train,val,fu,condition=a.condition,fixed_epochs=21)
        j1=t.run_joint_stage(a.output/'j1','j1',jp,contract,device,train,val,fu,condition=a.condition,initial=j0['model'],fixed_epochs=12)
        model=JointFallModel().to(device); model.load_state_dict(j1['model']); model.eval()
        for param in model.parameters(): param.requires_grad_(False)
        train=t.adapt_copy(train,model.adapter); val=t.adapt_copy(val,model.adapter)
        del model
    result=t.fit_classifier_arrays(a.output/'classifier',cp,contract,train,val,device)
    print(json.dumps(dict(completed=True,selected_classifier_epoch=result['epoch'],holdout_used=False)))

if __name__=='__main__': main()
