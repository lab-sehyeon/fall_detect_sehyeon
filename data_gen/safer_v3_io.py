"""Geometry-only IO. Must never import DSTE/model before MotionAGFormer."""
from pathlib import Path
import os
import numpy as np
from data_gen import safer_v2_reconstruction as base

ROOT=base.ROOT
CODE=['data_gen/safer_v3_io.py']
json=base.json
read=base.read
save_json=base.save_json
sha256_file=base.sha256_file
hash_named_tensors=base.hash_named_tensors
require=base.require
disk_gate=base.disk_gate
PauseRequested=base.PauseRequested
pause=base.pause_gate


def guard(path):
    path=Path(path).resolve();require(path.is_relative_to(ROOT),'path outside project')
    return base.guard_path(path)


def save_array(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream: np.save(stream,value,allow_pickle=False)
    os.replace(temp,path)


def payload(path):
    value=np.load(path,mmap_mode='r',allow_pickle=False)
    require(all(np.isfinite(value[i:i+4096]).all() for i in range(0,len(value),4096)),'nonfinite output')
    return {'sha256':sha256_file(path),'bytes':path.stat().st_size,'shape':list(value.shape),'dtype':str(value.dtype)}
