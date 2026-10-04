"""Train/val-only label-derived contextual recovery targets; never trains a model."""
import argparse
import fcntl
from pathlib import Path
import re
import sys

import numpy as np

from fall_pipeline.safer import state_followup_io as s
from fall_pipeline.safer import state_followup_core as core

io=s.io
CONFIG=io.ROOT/'configs/s0gt0_document_reconstruction_v1.json'
CODE=s.a.CODE+['fall_pipeline/safer/state_followup_io.py','fall_pipeline/safer/state_followup_core.py',
               'fall_pipeline/safer/build_s0gt0_reconstruction.py','configs/s0a_document_reconstruction_v1.json']


def run(args,config,root):
    io.require(Path(sys.prefix).name=='fall_detect','fall_detect required')
    ac=io.read(io.ROOT/config['s0a_config'])
    io.require(s.a.contract(ac,False)==io.read(io.ROOT/ac['output']/'run_contract.json'),'S0-A config/source invariant')
    contract={'config_sha256':io.sha256_file(CONFIG),'source':s.fingerprint(CODE),'data_report_sha256':ac['data_report_sha256'],
              'historical_exact_reproduction':False,'split_payloads_opened':['train','val']}
    s.lock_contract(root,contract,args.resume)
    if (root/'final_report.json').exists():
        io.require(io.read(root/'final_report.json')['passed'],'previous final failed');return
    s.publish(root,'s0gt0','preflight')
    data={split:s.core.verify_split(ac,split) for split in config['splits']}
    subjects={split:sorted({r['subject'] for r in d.rows}) for split,d in data.items()}
    io.require(not set(subjects['train'])&set(subjects['val']),'subject leakage')
    reports={}
    for split,d in data.items():
        report={'sequences':0,'frames':0,'fall_episodes':0,'recovery_episodes':0,'getting_up_hard_negative_segments':0,
                'target_counts':{str(k):0 for k in (-1,0,1,2,3)},'views':{},'payload':{}}
        for row in d.rows:
            s.pause(root,config)
            label_path=d.root/row['uid']/'labels.npy';before=io.sha256_file(label_path)
            labels=np.load(label_path,allow_pickle=False)
            target,episodes=core.contextual_targets(labels)
            np.testing.assert_array_equal(target,core.reference_contextual(labels))
            values,starts,ends=s.core.segments(labels)
            negative=sum(int(v==7 and np.all(target[start:end]==0)) for v,start,end in zip(values,starts,ends))
            match=re.search(r'_d(\d+)$',row['source_name'])
            io.require(match is not None,'unknown camera-view convention')
            view=match.group(1);recovery=sum(e['recovery_start'] is not None for e in episodes)
            stats=report['views'].setdefault(view,{'sequences':0,'fall_episodes':0,'recovery_episodes':0,'recovering_frames':0})
            stats['sequences']+=1;stats['fall_episodes']+=len(episodes);stats['recovery_episodes']+=recovery
            stats['recovering_frames']+=int(np.count_nonzero(target==3))
            out=root/split/row['uid']
            if (out/'manifest.json').exists():
                saved=io.read(out/'manifest.json')
                io.require(saved['source_label_sha256']==before,'GT0 source changed')
                for name,digest in saved['payload'].items():io.require(io.sha256_file(out/name)==digest,'GT0 output changed')
                np.testing.assert_array_equal(np.load(out/'targets.npy',allow_pickle=False),target)
                io.require(io.read(out/'episodes.json')==episodes,'GT0 episode roundtrip')
            else:
                io.save_array(out/'targets.npy',target);io.save_json(out/'episodes.json',episodes)
                np.testing.assert_array_equal(np.load(out/'targets.npy',allow_pickle=False),target)
                io.require(io.read(out/'episodes.json')==episodes,'new GT0 episode roundtrip')
                io.save_json(out/'manifest.json',{'uid':row['uid'],'source_label_sha256':before,'frames':len(labels),
                                                'payload':{name:io.sha256_file(out/name) for name in ('targets.npy','episodes.json')}})
            io.require(io.sha256_file(label_path)==before,'original labels changed')
            report['payload'][row['uid']]=io.sha256_file(out/'manifest.json')
            report['sequences']+=1;report['frames']+=len(labels);report['fall_episodes']+=len(episodes)
            report['recovery_episodes']+=recovery;report['getting_up_hard_negative_segments']+=negative
            for k in report['target_counts']:report['target_counts'][k]+=int(np.count_nonzero(target==int(k)))
            if report['sequences']==1 or report['sequences']%20==0:
                s.publish(root,'s0gt0','targets',split=split,sequences=report['sequences'],total_sequences=len(d.rows))
        io.require(sum(report['target_counts'].values())==report['frames'],'target frame totals')
        io.require(all(v['recovery_episodes']>0 for v in report['views'].values()),'view lacks recovery support')
        reports[split]=report
    io.require(sum(r['sequences'] for r in reports.values())==371,'GT0 train/val sequence count')
    io.require(s.fingerprint(CODE)==contract['source'] and io.sha256_file(CONFIG)==contract['config_sha256'],'GT0 code/config changed')
    for split in config['splits']:s.core.verify_split(ac,split)
    final={'passed':True,'research_usable':True,'historical_exact_reproduction':False,'splits':reports,'subjects':subjects,
           'test_ood_payloads_opened':False,'label_only':True,'source_unchanged':True,'independent_reference_exact':True,
           'target_roundtrip_exact':True,'train_val_subjects_disjoint':True,'online_inference_uses_future_labels':False}
    io.save_json(root/'final_report.json',final)
    s.publish(root,'s0gt0','completed',sequences=371,final_report_sha256=io.sha256_file(root/'final_report.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=io.read(CONFIG);root=io.guard(io.ROOT/config['output']);root.mkdir(parents=True,exist_ok=True)
    s.install_signals()
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args,config,root)
        except io.PauseRequested as error:
            s.publish(root,'s0gt0','paused',reason=str(error));raise SystemExit(75)
        except Exception as error:
            s.publish(root,'s0gt0','failed',reason=str(error));raise


if __name__=='__main__':main()
