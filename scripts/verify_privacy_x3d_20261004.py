"""Verify original UDA checkpoint and untouched official model; no target data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json, sys, hashlib, time
from pathlib import Path
import torch, numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/original_privacy_x3d_20261004'
sys.path.insert(0,str(REPO))
import mmcv
from mmaction.models import build_model
E=ROOT/'docs/internal/2026-10-04_hfd_x3d_external_comparison_evidence'
def load():
    p=REPO/'checkpoints/best_top1_acc_epoch_239.pth'
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64'))]):
        ckpt=torch.load(p,map_location='cpu',weights_only=True)
    cfg=mmcv.Config.fromstring(ckpt['meta']['config'],'.py')
    for part in ['backbone','cls_head','domain_head']:
        cfg.model[part]['pretrained']=str(REPO/cfg.model[part]['pretrained'])
    torch.set_num_threads(4)
    model=build_model(cfg.model,train_cfg=None,test_cfg=cfg.test_cfg)
    model.load_state_dict(ckpt['state_dict'],strict=True)
    model=model.cuda().eval()
    return model,cfg,ckpt
if __name__=='__main__':
    model,cfg,c=load();start=time.monotonic()
    assert all(torch.equal(t.cpu(),model.state_dict()[k].cpu()) for k,t in c['state_dict'].items())
    with torch.no_grad():
        y=model(torch.zeros((1,5,3,16,256,256),device='cuda'),domain_label=torch.ones(1,device='cuda'),return_loss=False)
    assert y.shape==(1,2) and np.isfinite(y).all()
    annotations={}
    for key in ['ann_file_train','ann_file_val','ann_file_test']:
        annotations[key]=[]
        for name in cfg[key].split('&&'):
            p=REPO/name;s=p.read_text();lines=s.splitlines()
            annotations[key].append(dict(file=name,rows=len(lines),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                target_dataset_name_hits={n:sum(n in x.lower() for x in lines) for n in ['le2i','cauca']}))
    p=REPO/'checkpoints/best_top1_acc_epoch_239.pth'
    report=dict(passed=True,checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),strict_tensor_count=len(c['state_dict']),
        every_tensor_exact=True,synthetic_output=y.tolist(),shape=list(y.shape),seconds=time.monotonic()-start,
        device=torch.cuda.get_device_name(0),torch=torch.__version__,mmcv=mmcv.__version__,annotations=annotations,
        source_only=False,training_recipe='Kinetics700 RGB labeled and depth unlabeled UDA',
        ancestry_limitation='Private resume epoch37 checkpoint not supplied; full prior optimization history cannot be independently certified',
        target_inference_executed=False)
    (E/'x3d_load_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
