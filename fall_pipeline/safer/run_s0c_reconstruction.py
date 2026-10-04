"""44 fixed validation decoder candidates; selected-source holdouts, no training."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from fall_pipeline.safer import state_followup_io as s
from fall_pipeline.safer import state_followup_core as core
from fall_pipeline.safer import run_s0b_reconstruction as b

io=s.io
CONFIG=io.ROOT/'configs/s0c_document_reconstruction_v1.json'
CODE=b.CODE+['fall_pipeline/safer/run_s0b_reconstruction.py','fall_pipeline/safer/run_s0c_reconstruction.py',
             'configs/s0b_document_reconstruction_v1.json']


def sources(config):
    bc=io.read(io.ROOT/config['s0b_config'])
    ac,ar,asel,_=s.s0a_dependency(bc)
    br=io.ROOT/bc['output'];report=io.read(br/'final_report.json')
    io.require(report['passed'] and report['research_usable'],'S0-B integrity incomplete')
    io.require(io.sha256_file(br/'final_report.json')==io.read(br/'status.json')['final_report_sha256'],'S0-B final report pin')
    io.require(b.make_contract(bc,False)==io.read(br/'run_contract.json'),'S0-B contract changed')
    selected=io.read(br/'selection.json')
    io.require(io.sha256_file(br/'selection.json')==report['selection_sha256'],'S0-B selection changed')
    io.require(io.sha256_file(br/'training'/f'epoch_{selected["epoch"]:02d}.pt')==selected['checkpoint_sha256'],'S0-B selected checkpoint changed')
    source={'s0a':{'root':io.ROOT/bc['s0a_root'],'logits':asel['candidate']+'_logits.npy','pred':asel['candidate']+'_pred.npy','report':ar},
            's0b':{'root':br,'logits':'logits.npy','pred':'pred.npy','report':report}}
    return ac,source


def records_for(ac,source,split,smoke=False,config=None):
    data=s.core.verify_split(ac,split);records=[]
    for row,begin,end in data.sequences():
        out=source['root']/'evaluation'/split/row['uid']
        manifest=io.read(out/'manifest.json')
        for name in (source['logits'],source['pred']):
            io.require(io.sha256_file(out/name)==manifest['payload'][name],'source timeline payload changed')
        logits=np.load(out/source['logits'],mmap_mode='r',allow_pickle=False)
        pred=np.load(out/source['pred'],mmap_mode='r',allow_pickle=False)
        _,labels=data.arrays(row['sequence_index'])
        mask=s.core.coverage(len(labels),data.starts[begin:end])>0
        truth=np.array(labels[mask],copy=True)
        io.require(logits.shape==(len(truth),16) and np.isfinite(logits).all(),'source timeline shape')
        np.testing.assert_array_equal(logits.argmax(1),pred)
        if smoke:
            limit=config['execution']['smoke_frames_per_sequence'];logits=logits[:limit];truth=truth[:limit]
        records.append({'uid':row['uid'],'logits':logits,'truth':truth,'source_sha256':manifest['payload'][source['logits']],
                        'label_sha256':io.sha256_file(data.root/row['uid']/'labels.npy')})
        if smoke and len(records)>=config['execution']['smoke_sequences']:break
    return records


def input_signature(records):
    return [{'uid':r['uid'],'source_sha256':r['source_sha256'],'label_sha256':r['label_sha256'],'frames':len(r['truth'])} for r in records]


def candidate_eval(root,dest,records,spec,ac,config):
    signature={'spec':spec,'inputs':input_signature(records)}
    manifest_path=dest/'manifest.json'
    if manifest_path.exists():
        saved=io.read(manifest_path)
        io.require(saved['signature']==signature,'decoder candidate input changed')
        for name,sha in saved['payload'].items():io.require(io.sha256_file(dest/name)==sha,'decoder candidate payload changed')
        # Recompute metric from committed predictions; no inference or altered decoder.
        metric=s.a.metric_instance(ac)
        for record in records:
            metric.add(np.load(dest/(record['uid']+'.npy'),allow_pickle=False),record['truth'])
        io.require(metric.result()==saved['metrics'],'decoder saved metric mismatch')
        return saved['metrics']
    metric=s.a.metric_instance(ac);payload={}
    for record in records:
        s.pause(root,config)
        prediction=core.decode(record['logits'],spec)
        metric.add(prediction,record['truth'])
        name=record['uid']+'.npy';io.save_array(dest/name,prediction);payload[name]=io.sha256_file(dest/name)
        # A fixed prefix must be unaffected by future logits.
        cut=min(128,len(prediction))
        np.testing.assert_array_equal(prediction[:cut],core.decode(record['logits'][:cut],spec))
    result=metric.result()
    io.save_json(manifest_path,{'signature':signature,'payload':payload,'metrics':result,'prefix_invariance_passed':True})
    return result


def latency_report(dest,records,pred_dir,config):
    events=[]
    for record in records:
        pred=np.load(pred_dir/(record['uid']+'.npy'),allow_pickle=False)
        events.extend({'uid':record['uid'],**e} for e in core.latency_events(pred,record['truth'],[(10,12),(6,4),(12,7)],config['latency_diagnostic']))
    summary={}
    for pair in ('10->12','6->4','12->7'):
        rows=[e for e in events if e['pair']==pair];delays=[e['delay_frames'] for e in rows if e['detected']]
        summary[pair]={'events':len(rows),'detected':len(delays),'missed':len(rows)-len(delays),
                       'recall':len(delays)/len(rows) if rows else None,
                       'premature':sum(e['premature'] for e in rows),'boundary_correct':sum(e['boundary_correct'] for e in rows),
                       'delay_median_frames':float(np.median(delays)) if delays else None,
                       'delay_p90_frames':float(np.percentile(delays,90)) if delays else None,
                       'miss_dominant_counts':{str(k):sum(e['miss_dominant_class']==k for e in rows) for k in range(16)}}
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'events.csv';tmp=path.with_suffix('.tmp')
    fields=['uid','pair','boundary','detected','delay_frames','premature','boundary_correct','miss_dominant_class']
    with tmp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(events)
    os.replace(tmp,path)
    io.save_json(dest/'summary.json',{'selection_uses_diagnostic':False,'pairs':summary,'events_csv_sha256':io.sha256_file(path)})
    return summary


def run(args,config,root):
    io.require(Path(sys.prefix).name=='fall_detect','fall_detect required')
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','S0-C is CPU-only')
    torch.set_num_threads(2)
    ac,source=sources(config)
    parent={name:{f:io.sha256_file(item['root']/f) for f in ('final_report.json','selection.json','run_contract.json')}
            for name,item in source.items()}
    contract={'config_sha256':io.sha256_file(CONFIG),'source':s.fingerprint(CODE),'parents':parent,'smoke':args.smoke,
              'historical_exact_reproduction':False,'numpy':str(np.__version__)}
    s.lock_contract(root,contract,args.resume)
    if (root/'final_report.json').exists():
        io.require(io.read(root/'final_report.json')['passed'],'previous final failed');return
    if not args.smoke:
        sr=Path(str(root)+'_smoke')
        io.require(io.read(sr/'final_report.json')['passed'],'S0-C smoke required')
        io.require({**io.read(sr/'run_contract.json'),'smoke':False}==contract,'S0-C smoke contract differs')
    s.publish(root,'s0c','preflight',holdouts_opened=False)
    grid=core.decoder_grid(config)
    io.require(len(grid)==config['grid']['candidates_per_source'] and len(grid)*len(source)==44,'44 candidates required')
    records={name:records_for(ac,source[name],'val',args.smoke,config) for name in config['sources']}
    io.require([r['uid'] for r in records['s0a']]==[r['uid'] for r in records['s0b']],'source sequence order mismatch')
    for first,second in zip(records['s0a'],records['s0b']):np.testing.assert_array_equal(first['truth'],second['truth'])
    rows=[]
    for name in config['sources']:
        baseline=None
        for spec in grid:
            s.pause(root,config)
            dest=root/'validation'/name/spec['name']
            metric=candidate_eval(root,dest,records[name],spec,ac,config)
            if spec['name']=='raw':
                baseline=metric
                if not args.smoke:io.require(metric==source[name]['report']['metrics']['val'],'raw baseline replay changed')
            gate=core.adoption_c(metric,baseline,config['evaluation'])
            rows.append({'source':name,'spec':spec,'metrics':metric,'gate':gate,'manifest_sha256':io.sha256_file(dest/'manifest.json')})
            io.save_json(root/'validation/candidates.json',rows)
            s.publish(root,'s0c','decode',completed_candidates=len(rows),total_candidates=44,source=name,candidate=spec['name'])
    chosen=core.choose_decoder(rows)
    if args.smoke:
        report={'passed':True,'research_usable':False,'candidates':44,'holdouts_opened':False,'prefix_invariance_passed':True}
        io.save_json(root/'final_report.json',report);s.publish(root,'s0c','completed',**report);return
    selection={**chosen,'run_contract_sha256':io.sha256_file(root/'run_contract.json'),
               'candidates_sha256':io.sha256_file(root/'validation/candidates.json'),'selected_using':'val only',
               'holdouts_opened_at_selection':False}
    select_path=root/'selection.json'
    if select_path.exists():io.require(io.read(select_path)==selection,'immutable decoder selection differs')
    else:io.save_json(select_path,selection)
    selection_sha=io.sha256_file(select_path)
    s.publish(root,'s0c','selection',adoption=chosen['adoption'],source=chosen['source'],candidate=chosen['spec']['name'])
    final={};latency={}
    for split in ('val','test','ood'):
        s.pause(root,config);io.require(io.sha256_file(select_path)==selection_sha,'selection changed before holdout')
        current=records[chosen['source']] if split=='val' else records_for(ac,source[chosen['source']],split)
        dest=root/'evaluation'/split
        final[split]=candidate_eval(root,dest,current,chosen['spec'],ac,config)
        if split=='val':io.require(final[split]==chosen['metrics'],'selected val replay differs')
        latency[split]=latency_report(root/'latency'/split,current,dest,config)
        s.publish(root,'s0c','evaluate',split=split)
    s.publish(root,'s0c','audit')
    io.require(s.fingerprint(CODE)==contract['source'] and io.sha256_file(CONFIG)==contract['config_sha256'],'decoder code/config changed')
    for name,item in source.items():
        for filename,sha in parent[name].items():io.require(io.sha256_file(item['root']/filename)==sha,'frozen parent artifact changed')
    # Read back every saved prediction and compare to its deterministic decoder, including holdouts.
    for split in ('val','test','ood'):
        current=records[chosen['source']] if split=='val' else records_for(ac,source[chosen['source']],split)
        for record in current:
            pred=np.load(root/'evaluation'/split/(record['uid']+'.npy'),allow_pickle=False)
            np.testing.assert_array_equal(pred,core.decode(record['logits'],chosen['spec']))
    report={'passed':True,'research_usable':True,'historical_exact_reproduction':False,'candidates':44,
            'selected_source':chosen['source'],'selected_decoder':chosen['spec'],'adoption':chosen['adoption'],
            'eligible_count':chosen['eligible_count'],'selection_sha256':selection_sha,'metrics':final,'latency':latency,
            'frozen_parents_unchanged':True,'selected_prediction_replay_exact':True,'prefix_invariance_passed':True,
            'training':False,'end_to_end_causal':False,'automatic_deployment':False}
    io.save_json(root/'final_report.json',report)
    s.publish(root,'s0c','completed',adoption=chosen['adoption'],final_report_sha256=io.sha256_file(root/'final_report.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=io.read(CONFIG);root=io.guard(io.ROOT/(config['output']+('_smoke' if args.smoke else '')))
    root.mkdir(parents=True,exist_ok=True);s.install_signals()
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args,config,root)
        except io.PauseRequested as error:
            s.publish(root,'s0c','paused',reason=str(error));raise SystemExit(75)
        except Exception as error:
            s.publish(root,'s0c','failed',reason=str(error));raise


if __name__=='__main__':main()
