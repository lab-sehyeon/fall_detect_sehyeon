"""Apply released frozen heads to N x 2048 pooled DSTE features (no pose/backbone inference)."""
from pathlib import Path
import argparse,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fall_pipeline.joint.models import JointFallModel

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--condition',choices=['raw','full_g0','full_j1','no_fu_g0','no_fu_j1'],required=True)
    p.add_argument('--seed',type=int,choices=range(5),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
    family='raw' if a.condition=='raw' else ('full' if a.condition.startswith('full') else 'no_fu')
    state=torch.load(a.weights/f'seed_{a.seed}'/(family+'.pth'),map_location='cpu',weights_only=True)
    x=np.load(a.features,mmap_mode='r');assert x.ndim==2 and x.shape[1]==2048
    if a.condition=='raw':
        model=torch.nn.Linear(2048,4);model.load_state_dict(state);model.eval()
    else:
        joint=JointFallModel();joint.load_state_dict(state['joint']);joint.eval()
        head=torch.nn.Linear(2048,4)
        head.load_state_dict(state['g0_head'] if a.condition.endswith('g0') else joint.safer_head.state_dict());head.eval()
        model=torch.nn.Sequential(joint.adapter,head).eval()
    with torch.inference_mode():
        y=np.concatenate([model(torch.tensor(np.asarray(x[i:i+1024]))).numpy() for i in range(0,len(x),1024)])
    np.save(a.output,y);print(y.shape)

if __name__=='__main__':main()
