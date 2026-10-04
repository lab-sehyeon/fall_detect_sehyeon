"""CPU-only, read-only replay of completed classifiers against locked V3 R3 OOF."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
from datetime import datetime,timezone
import numpy as np
import torch
from fall_pipeline.benchmarks import fu_classifier_reconstruction as clf
from fall_pipeline.joint import run_reconstruction as joint
from scripts import run_recovery_v3_queue_r3 as queue

io=clf.io
ROOT=io.ROOT
OUTPUT=ROOT/'checkpoint/fall/FU_CLASSIFIER_V3_REFERENCE_20260922_R3'
CLASSIFIERS=ROOT/'checkpoint/fall/FU_CLASSIFIER_DOCUMENT_RECONSTRUCTION_20260922_R1'
CLASSIFIER_SHA='4df2828b7b8012eccd1e998c3d809a5ff6dc8e04a9b75146f02f563542ada6fe'
REFERENCE=queue.JOINT/'v3'


def publish(stage,**details):
    state={'stage':stage,'time':datetime.now(timezone.utc).isoformat(),**details}
    io.save_json(OUTPUT/'status.json',state);print(io.json.dumps(state,ensure_ascii=False),flush=True)
    status='completed' if stage=='completed' else 'paused' if stage in ('failed','paused') else 'in_progress'
    header=f'- 문서 ID: `DOC-20260922-fu-reference-r3-run-R1`\n- 기준일: 2026-09-22\n- 상태: `{status}`\n'
    internal=f'# FU V3 R3 참조 검산\n\n{header}\n```json\n{io.json.dumps(state,indent=2)}\n```\n\nroot: `{OUTPUT}`\n'
    shared=f'# FU 분류기와 V3 공동학습 비교\n\n{header}\n현재 단계: {stage}. 기존 분류기의 선택·학습은 변경하지 않는다.\n'
    shared+='\n[방법과 선행 조건](2026-09-22_v3_valid_support_shared.md)을 따른다.\n'
    for kind,value in (('internal',internal),('shared',shared)):
        path=ROOT/f'docs/{kind}/2026-09-22_fu_reference_r3_run_{kind}.md'
        tmp=path.with_suffix('.tmp');tmp.write_text(value);os.replace(tmp,path)


def check_pause():
    io.pause(OUTPUT)
    if (queue.QUEUE/'PAUSE_REQUESTED').exists(): raise io.PauseRequested('V3 queue pause requested')


def replay_classifiers(root,fu,report):
    results={};payload={}
    for name in report['results']:
        rows=[]
        for k in range(5):
            dest=root/name/f'fold{k}';lock=io.read(dest/'selection.json');outer_report=io.read(dest/'outer.json')
            train,val,outer=clf.core.split_indices(fu['folds'],fu['subjects'],k,True)
            for key,idx in (('train_indices',train),('val_indices',val),('outer_indices',outer)): np.testing.assert_array_equal(lock[key],idx)
            io.require(io.sha256_file(dest/'selection.json')==outer_report['selection_sha256'],'classifier selection changed')
            weight=dest/f'candidate_{lock["candidate"]:03d}.joblib'
            io.require(io.sha256_file(weight)==lock['checkpoint_sha256'],'classifier weight changed')
            model=clf.joblib.load(weight)
            pred,score=clf.outputs(model,fu['x'][outer])
            for key,value in (('pred',pred),('score',score)):
                path=dest/f'outer_{key}.npy'
                io.require(io.sha256_file(path)==outer_report[key+'_sha256'],'classifier outer hash')
                np.testing.assert_array_equal(value,np.load(path,allow_pickle=False))
            rows.extend(zip(outer.tolist(),pred.tolist(),score.tolist()))
        rows.sort(key=lambda x:x[0]);idx=np.array([r[0] for r in rows],np.int64)
        pred=np.array([r[1] for r in rows],np.int64);score=np.array([r[2] for r in rows],np.float64)
        np.testing.assert_array_equal(idx,np.arange(len(fu['y'])))
        for key,value in (('indices',idx),('pred',pred),('score',score)):
            path=root/name/f'oof_{key}.npy';np.testing.assert_array_equal(value,np.load(path,allow_pickle=False));payload[str(path.relative_to(root))]=io.sha256_file(path)
        results[name]=clf.metrics(fu['y'],fu['actions'],pred,score)
        io.require(results[name]==report['results'][name],'classifier OOF metrics changed')
    return results,payload


def replay_joint(source,fu,config,final):
    results={};payload={};max_error=0.
    for stage in ('j0','j1'):
        rows=[]
        for k in range(5):
            dest=source/'nested'/f'fold{k}';outer=clf.core.split_indices(fu['folds'],fu['subjects'],k,True)[2]
            np.testing.assert_array_equal(outer,np.load(dest/'outer_indices.npy',allow_pickle=False))
            record=io.read(dest/'outer_result.json')
            io.require(io.sha256_file(dest/'selection_lock.json')==record['selection_lock_sha256'],'joint outer selection changed')
            path=dest/f'{stage}_outer_logits.npy'
            io.require(io.sha256_file(path)==record['results'][stage]['logits_sha256'],'joint outer hash')
            actual=np.load(path,allow_pickle=False);model=joint.load_selected(dest/stage,stage,config,torch.device('cpu'))
            predicted=clf.core.predict(model,torch.from_numpy(fu['x'][outer]),'fu',config['training']['eval_batch_size'])
            np.testing.assert_allclose(actual,predicted,atol=config['audit']['cpu_prediction_atol'],rtol=config['audit']['cpu_prediction_rtol'])
            max_error=max(max_error,float(np.abs(actual-predicted).max()));rows.extend(zip(outer.tolist(),actual))
        rows.sort(key=lambda x:x[0]);idx=np.array([r[0] for r in rows]);logits=np.stack([r[1] for r in rows])
        np.testing.assert_array_equal(idx,np.arange(len(fu['y'])))
        for key,value in (('indices',idx),('logits',logits)):
            path=source/'nested'/f'{stage}_oof_{key}.npy';np.testing.assert_array_equal(value,np.load(path,allow_pickle=False));payload[str(path.relative_to(source))]=io.sha256_file(path)
        results[stage]=clf.core.binary_metrics(fu['y'],fu['actions'],logits)
        io.require(results[stage]==final['nested'][stage]['metrics'],'joint OOF metrics changed')
    return results,payload,max_error


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait-for-queue',action='store_true');args=parser.parse_args()
    io.require(Path(sys.prefix).name=='fall_detect' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','fall_detect CPU-only required')
    OUTPUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    paths=set(clf.CODE+joint.CODE+['scripts/run_fu_reference_r3.py','configs/fu_classifier_document_reconstruction_v1.json'])
    fingerprint={p:io.sha256_file(ROOT/p) for p in paths};fingerprint.update(queue.fingerprint())
    if (OUTPUT/'run_contract.json').exists(): io.require(io.read(OUTPUT/'run_contract.json')==fingerprint,'reference code changed')
    else: io.save_json(OUTPUT/'run_contract.json',fingerprint)
    with (OUTPUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish('waiting_for_v3' if args.wait_for_queue else 'replay');deadline=time.monotonic()+24*3600
            while args.wait_for_queue:
                check_pause();state=queue.read(queue.QUEUE/'status.json')['stage']
                if state=='completed': break
                io.require(state not in ('failed','paused'),'V3 queue stopped; reference blocked')
                io.require(time.monotonic()<deadline,'V3 reference wait expired');time.sleep(15)
            check_pause()
            io.require(all(io.sha256_file(ROOT/p)==v for p,v in fingerprint.items()),'reference dependency changed')
            queue.verify_report(REFERENCE/'final_report.json')
            io.require(io.sha256_file(REFERENCE/'final_report.json')==queue.read(queue.QUEUE/'v3_joint_full.json')['report_sha256'],'joint queue result changed')
            io.require(io.sha256_file(CLASSIFIERS/'classifier_report.json')==CLASSIFIER_SHA,'classifier report pin')
            local=io.read(CLASSIFIERS/'classifier_report.json');spec=io.read(clf.CONFIG)
            io.require(local['passed'] and local['research_usable'] and local['contract']==clf.make_contract(spec,False),'classifier source contract')
            io.require(io.sha256_file(CLASSIFIERS/'independent_audit.json')==local['audit_sha256'] and io.read(CLASSIFIERS/'independent_audit.json')['passed'],'classifier audit')
            config=io.read(ROOT/spec['source_config']);final=io.read(REFERENCE/'final_report.json')
            io.require(final['contract']['matched_method_contract']['fu']==config['fu'],'different FU feature lineage')
            fu=clf.inputs.load_fu(config);publish('replay')
            own,p1=replay_classifiers(CLASSIFIERS,fu,local);comparison,p2,maximum=replay_joint(REFERENCE,fu,config,final)
            check_pause();io.require(all(io.sha256_file(ROOT/p)==v for p,v in fingerprint.items()),'source changed during replay')
            audit={'passed':True,'classifier_exact_replay':True,'joint_selected_cpu_maxabs':maximum,'oof_exact_once':True,'classifier_payload':p1,'joint_payload':p2}
            io.save_json(OUTPUT/'independent_audit.json',audit)
            result={'passed':True,'research_usable':True,'classifier_results':own,'reference_results':comparison,
                    'classifier_report_sha256':CLASSIFIER_SHA,'reference_report_sha256':io.sha256_file(REFERENCE/'final_report.json'),
                    'audit_sha256':io.sha256_file(OUTPUT/'independent_audit.json'),'contract':fingerprint,'retraining_or_reselection':False,'historical_exact_reproduction':False}
            io.save_json(OUTPUT/'final_report.json',result);publish('completed')
        except BaseException as error:
            publish('paused' if isinstance(error,io.PauseRequested) else 'failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
