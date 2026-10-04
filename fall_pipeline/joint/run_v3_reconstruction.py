"""V3 lineage of the identical reconstructed J0/J1 comparison protocol."""
import argparse
import fcntl
from datetime import datetime,timezone
import torch
from fall_pipeline.joint import run_reconstruction as base
from fall_pipeline.joint import reconstruction_inputs as inputs
from fall_pipeline.safer import v3_controls_reconstruction as controls

io=base.io
ROOT=io.ROOT
CONFIG=ROOT/'configs/joint_v3_document_reconstruction_v1.json'
CODE=sorted(set(['fall_pipeline/joint/run_v3_reconstruction.py']+base.CODE+controls.CODE))


def config_and_sources():
    spec=io.read(CONFIG)
    io.require(io.sha256_file(ROOT/spec['base_config'])==spec['base_config_sha256'],'joint matched config')
    io.require(io.sha256_file(ROOT/spec['controls_config'])==spec['controls_config_sha256'],'V3 controls config')
    config=io.read(ROOT/spec['base_config']);config['output_dir']=spec['output_dir']
    root=io.guard(ROOT/spec['controls_root']);report=io.read(root/'final_report.json')
    io.require(report['passed'] and report['research_usable'] and report['model_frozen'],'V3 controls incomplete')
    io.require(report['contract']['config_sha256']==spec['controls_config_sha256'],'V3 controls lineage')
    io.require(io.sha256_file(root/'independent_audit.json')==report['audit_sha256'] and io.read(root/'independent_audit.json')['passed'],'V3 controls audit')
    cs,cc=controls.effective_config();io.require(controls.make_contract(cs,cc,False)==report['contract'],'V3 controls source implementation changed')
    sources={}
    for split in ('train','val','test','ood'):
        path=root/'cache'/split/'manifest.json';sources[split]={'path':str(path.parent),'manifest':str(path),'sha256':io.sha256_file(path)}
    prepared={'lineage':'v3','data_root':str(io.guard(ROOT/spec['data_root'])),'lineage_pin':io.sha256_file(root/'final_report.json'),
              'sources':sources,'counts':config['lineages']['v2']['split_counts']}
    return spec,config,prepared


def contract_for(config,prepared,smoke):
    return {'config_sha256':io.sha256_file(CONFIG),'code_sha256':{p:io.sha256_file(ROOT/p) for p in CODE},
            'matched_method_contract':base.contract_for(config,prepared,smoke),'smoke':smoke,'historical_exact_reproduction':False}


def publish(root,stage,**details):
    state={'stage':stage,'time':datetime.now(timezone.utc).isoformat(),'lineage':'v3',**details}
    io.save_json(root/'status.json',state);print(io.json.dumps(state,ensure_ascii=False),flush=True)
    if root.name.endswith('_smoke'): return
    status='completed' if stage=='completed' else 'paused' if stage in ('paused','failed') else 'in_progress'
    head=f'- 문서 ID: `DOC-20260922-joint-v3-run-R1`\n- 기준일: 2026-09-22\n- 상태: `{status}`\n'
    internal=f'# J0/J1 V3 실행 기록\n\n{head}\n```json\n{io.json.dumps(state,indent=2,ensure_ascii=False)}\n```\n\nroot: `{root}`\n'
    shared=f'# J0/J1 V3 진행\n\n{head}\n현재 단계: {stage}.\n'
    if 'epoch' in details: shared+=f'완료 epoch: {details["epoch"]}.\n'
    shared+='\n[방법](2026-09-22_joint_v3_reconstruction_shared.md)을 따른다. 원본과 동일한 수치의 복원은 아니다.\n'
    for kind,value in (('internal',internal),('shared',shared)):
        controls.preprocessing.v2.atomic_text(ROOT/f'docs/{kind}/2026-09-22_joint_v3_run_{kind}.md',value)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('preflight','smoke','all'),default='all');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();spec,config,prepared=config_and_sources();smoke=args.stage=='smoke';contract=contract_for(config,prepared,smoke)
    root=io.guard(ROOT/config['output_dir']/('v3_smoke' if smoke else 'v3'));existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    io.disk_gate(config,max(0,(1 if smoke else 6)*1024**3-existing))
    if root.exists() and any(root.iterdir()): io.require(args.resume and io.read(root/'run_contract.json')==contract,'V3 joint output/contract mismatch')
    else: root.mkdir(parents=True,exist_ok=True);io.save_json(root/'run_contract.json',contract)
    base.publish=publish;torch.set_num_threads(2)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish(root,'source');fu_cpu=inputs.load_fu(config)
            safer_cpu={s:inputs.load_safer(prepared,s,limit=256 if smoke else None) for s in ('train','val')}
            io.save_json(root/'source_audit.json',{'passed':True,'train':len(safer_cpu['train']['y']),'val':len(safer_cpu['val']['y']),'fu':len(fu_cpu['y']),'holdout_arrays_unopened':True})
            if args.stage=='preflight': return
            device=io.device_for(config);fu=inputs.to_device(fu_cpu,device);safer={s:inputs.to_device(v,device) for s,v in safer_cpu.items()}
            base.fit_folds(root,'development',safer,fu,config,device,contract,smoke)
            development=base.aggregate(root,'development',fu,config,smoke);locked=base.locked_safer(root,prepared,config,device,smoke)
            base.fit_folds(root,'nested',safer,fu,config,device,contract,smoke)
            nested=base.aggregate(root,'nested',fu,config,smoke);final=base.final_fit(root,safer,fu,nested,config,device,contract,smoke)
            del fu,safer;torch.cuda.empty_cache();publish(root,'audit');base.audit(root,safer_cpu,fu_cpu,prepared,config,smoke)
            io.require(contract_for(config,prepared,smoke)==contract,'V3 joint source/code/config changed')
            io.save_json(root/'final_report.json',{'passed':True,'research_usable':not smoke,'historical_exact_reproduction':False,
                'development':development,'locked_safer':locked,'nested':nested,'final_fit':final,'audit_sha256':io.sha256_file(root/'independent_audit.json'),
                'contract':contract,'dste_adl_forward_or_optimizer':False})
            publish(root,'completed',research_usable=not smoke)
        except BaseException as error:
            publish(root,'paused' if isinstance(error,io.PauseRequested) else 'failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
