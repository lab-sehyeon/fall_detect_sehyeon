"""Correct original alarm extraction; reuse unchanged R1 pose model predictions.

R1 exact-string print observer missed 'True Fall' and 'True Fall 2'. Every
original true-fall branch draws a 'T Fall...' label. Capture all 68 such branches.
No rule, threshold, input, weights, timestamp or metric changes.
"""
import ast
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import evaluate_original_rthfd_20261004 as base

ROOT=base.ROOT
R1=base.OUT
OUT=ROOT/'data/fall_processed/RGB/original_baselines_20261004_r2'
CONFIG=ROOT/'configs/original_baselines_20261004_r2.json'
base.OUT=OUT


def validate_branches():
    for name in ('Thunder_GMDCSA.ipynb.rules.py','Thunder_URFD.ipynb.rules.py'):
        tree=ast.parse((base.ASSET/name).read_text());branches=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.If) and ast.unparse(node.test)=='fc >= 2':
                labels=[e.value.args[1].value for e in node.body if isinstance(e,ast.Expr)
                    and isinstance(e.value,ast.Call) and isinstance(e.value.func,ast.Attribute)
                    and e.value.func.attr=='putText']
                base.require(len(labels)==1 and labels[0].startswith('T Fall'),'unmapped original branch')
                branches.append(labels[0])
        base.require(len(branches)==68,'original true-fall branches')


def prepare():
    for p,h in base.read(R1/'contract.json')['files'].items():base.require(base.sha(ROOT/p)==h,'R1 contract changed')
    OUT.mkdir(exist_ok=True);validate_branches()
    if (OUT/'contract.json').exists():base.check_contract();return
    for name in ('plan.json','ground_truth.json','own_predictions.json'):
        base.save(OUT/name,base.read(R1/name))
    config=base.read(base.CONFIG)
    config.update(experiment_id='ORIGINAL_BASELINES_20261004_R2',
        supersedes='R1 scores invalid: exact print(Fall) observer missed 20 of 68 original true-fall branches',
        alarm='all original cv2.putText T Fall... declarations per frame, rising edges; independently check all true-fall print variants',
        model_inference='reuse unchanged R1 Thunder v3 raw poses; regenerate every rule output; independent pose re-inference audit',
        revision_reason='Original-code output adapter bug fix, no threshold/rule/target tuning')
    base.save(CONFIG,config)
    files=dict(base.read(R1/'contract.json')['files'])
    paths=[R1/'contract.json',Path(__file__),CONFIG,*[OUT/n for n in ('plan.json','ground_truth.json','own_predictions.json')],
           base.EVIDENCE/'original_alarm_branches.json']
    for item in base.read(OUT/'plan.json'):
        receipt=base.read(R1/item['id']/'rthfd.json')
        base.require(receipt['processed'],'R2 expects completed original pose inference')
        base.require(base.sha(R1/item['id']/'rthfd.npz')==receipt['payload_sha256'],'R1 pose bytes changed')
        paths.extend([R1/item['id']/'rthfd.json',R1/item['id']/'rthfd.npz'])
    files.update({str(p.relative_to(ROOT)):base.sha(p) for p in paths})
    base.save(OUT/'contract.json',dict(files=files,revision='R2 all original alarm declarations',
                                    created=base.datetime.now(base.timezone.utc).isoformat()))


class Drawing:
    FONT_HERSHEY_COMPLEX=3
    def __init__(self):self.alarm=False
    def putText(self,*args):
        if args[1].startswith('T Fall'):self.alarm=True
        return args[0]


def run():
    prepare();base.check_contract();start=time.monotonic();changed=0;frames=0
    code=compile((base.ASSET/'Thunder_GMDCSA.ipynb.rules.py').read_text(),'<original GMDCSA>','exec')
    for number,item in enumerate(base.read(OUT/'plan.json'),1):
        src=R1/item['id'];dest=OUT/item['id'];dest.mkdir(exist_ok=True)
        with np.load(src/'rthfd.npz',allow_pickle=False) as z:payload={k:z[k] for k in z.files}
        drawing=Drawing();printed=[False]
        def emit(*args,**kwargs):
            if args and args[0] in ('Fall','True Fall','True Fall 2'):printed[0]=True
        env=dict(cv2=drawing,frame=None,print=emit,fc=0,flag=0)
        alarms=np.zeros(len(payload['keypoints']),bool)
        for i,pose in enumerate(payload['keypoints']):
            drawing.alarm=False;printed[0]=False;env['kws']=pose[None,None];exec(code,env)
            base.require(drawing.alarm==printed[0],'original printed/displayed fall mismatch')
            base.require(env['fc']==payload['counters'][i] and env['flag']==payload['flags'][i],'rule state changed')
            alarms[i]=drawing.alarm
        changed+=int(np.sum(alarms!=payload['alarms']));payload['alarms']=alarms;frames+=len(alarms)
        tmp=dest/'rthfd.tmp.npz';np.savez_compressed(tmp,**payload);tmp.replace(dest/'rthfd.npz')
        receipt=base.read(src/'rthfd.json')
        receipt.update(payload_sha256=base.sha(dest/'rthfd.npz'),contract_sha256=base.sha(OUT/'contract.json'),
                       model_inference='R1 original poses reused unchanged',original_pose_payload_sha256=base.sha(src/'rthfd.npz'),
                       alarm_extraction_revision='R2 all 68 original true-fall branches')
        base.save(dest/'rthfd.json',receipt)
        if number%20==0:base.status('alarm_replay',completed=number,total=230,frames=frames)
    base.check_contract()
    base.save(OUT/'alarm_replay.json',dict(passed=True,videos=230,frames=frames,changed_alarm_frames=changed,
        seconds=time.monotonic()-start,all68_branches_mapped=True,rules_weights_poses_unchanged=True))
    base.score()


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
