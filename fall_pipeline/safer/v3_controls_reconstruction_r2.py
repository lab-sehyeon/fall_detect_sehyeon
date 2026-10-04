"""Matched V3 controls, reusing frozen V2 trainer/evaluator without file edits."""
from __future__ import annotations
import argparse
import fcntl
from datetime import datetime,timezone
import numpy as np
import torch
from fall_pipeline.safer import v2_controls_reconstruction as io
from data_gen import safer_v3_runtime as runtime
from data_gen import safer_v3_reconstruction_r2 as preprocessing

ROOT=io.ROOT
CONFIG=ROOT/'configs/safer_v3_controls_document_reconstruction_v2.json'
CODE=sorted(set(['fall_pipeline/safer/v3_controls_reconstruction_r2.py','data_gen/safer_v3_runtime.py']+io.CODE+preprocessing.CODE))


def publish(root,stage,**details):
    state={'stage':stage,'time':datetime.now(timezone.utc).isoformat(),**details}
    io.save_json(root/'status.json',state);print(io.json.dumps(state,ensure_ascii=False),flush=True)
    if root.name.endswith('_smoke'): return
    status='completed' if stage=='completed' else 'paused' if stage in ('paused','failed') else 'in_progress'
    head=f'- 문서 ID: `DOC-20260922-safer-v3-controls-run-R2`\n- 기준일: 2026-09-22\n- 상태: `{status}`\n'
    internal=f'# V3 controls 실행 기록\n\n{head}\n```json\n{io.json.dumps(state,indent=2,ensure_ascii=False)}\n```\n\nroot: `{root}`\n'
    shared=f'# V3 matched controls 실행 진행\n\n{head}\n현재 단계: {stage}.\n'
    if 'epoch' in details: shared+=f'완료 epoch: {details["epoch"]}/50.\n'
    shared+='\n[방법](2026-09-22_safer_v3_controls_shared.md)을 따른다. 원본 동등성이나 전체 연구 완료를 뜻하지 않는다.\n'
    for kind,value in (('internal',internal),('shared',shared)):
        preprocessing.v2.atomic_text(ROOT/f'docs/{kind}/2026-09-22_safer_v3_controls_run_{kind}.md',value)


def effective_config():
    spec=io.read(CONFIG);io.require(io.sha256_file(ROOT/spec['base_config'])==spec['base_config_sha256'],'matched V2 config changed')
    config=io.read(ROOT/spec['base_config']);config.update({k:spec[k] for k in ('data_root','output_dir')})
    return spec,config


def make_contract(spec,config,smoke):
    root=ROOT/config['data_root'];report=io.read(root/'final_report.json')
    io.require(report['passed'] and report['research_usable'] and report['geometry_gate']['passed'],'V3 full geometry incomplete')
    io.require(report['contract']==preprocessing.make_contract(io.read(ROOT/spec['preprocessing_config'])),'V3 preprocessing implementation changed')
    return {'config_sha256':io.sha256_file(CONFIG),'preprocessing_final_sha256':io.sha256_file(root/'final_report.json'),
            'source_code':{p:io.sha256_file(ROOT/p) for p in CODE},'base_contract':io.make_contract(config,smoke),
            'historical_exact_reproduction':False,'smoke':smoke}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('preflight','smoke','all'),default='all');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();smoke=args.stage=='smoke';spec,config=effective_config();contract=make_contract(spec,config,smoke)
    root=io.guard(ROOT/(config['output_dir']+('_smoke' if smoke else '')))
    existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    io.disk_gate(config,max(0,(1 if smoke else 12)*1024**3-existing));io.initialize(root,contract,args.resume)
    # Process-local progress callback only. No imported source/config file is changed.
    io.publish=publish;torch.set_num_threads(2)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish(root,'preflight');source_root=ROOT/config['data_root'];io.save_json(root/'source_audit.json',runtime.verify_source(source_root))
            if args.stage=='preflight': return
            device=io.device_for(config);torch.manual_seed(0)
            model=io.load_adl_model(ROOT/config['encoder'],ROOT/config['adl_root']/'best_adl_head.pth',device)
            before=io.hash_named_tensors(model.state_dict().items());cache={}
            for split in ('train','val'):
                cache[split]=io.extract(root,split,runtime.source_split(source_root,split,smoke),model,device,config,before)
            io.require(io.hash_named_tensors(model.state_dict().items())==before,'frozen model changed')
            selection=io.fit_heads(root,cache,io.read(ROOT/config['matched_training_config']),device,smoke,contract)
            lock=io.sha256_file(root/'selection.json')
            for split in ('test','ood'):
                cache[split]=io.extract(root,split,runtime.source_split(source_root,split,smoke),model,device,config,before,lock)
            io.require(io.hash_named_tensors(model.state_dict().items())==before,'frozen model changed')
            io.save_json(root/'invariance.json',{'before':before,'after':before,'passed':True});del model;torch.cuda.empty_cache()
            results=io.evaluate(root,cache,selection,device);publish(root,'audit');io.audit_outputs(root,cache,selection,config)
            io.require(make_contract(spec,config,smoke)==contract,'V3 source/code/config changed')
            io.save_json(root/'final_report.json',{'passed':True,'research_usable':not smoke,'model_frozen':True,'selection':selection,'results':results,
                         'audit_sha256':io.sha256_file(root/'independent_audit.json'),'contract':contract,'historical_exact_reproduction':False})
            publish(root,'completed',research_usable=not smoke)
        except BaseException as error:
            publish(root,'paused' if isinstance(error,io.PauseRequested) else 'failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
