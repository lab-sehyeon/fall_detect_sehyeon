"""Nested classifiers on the fixed FU DSTE TS cache, with locked V3 replay later."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import warnings
from datetime import datetime,timezone
import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix,f1_score,average_precision_score
from fall_pipeline.safer import v2_controls_reconstruction as io
from fall_pipeline.joint import reconstruction_inputs as inputs
from fall_pipeline.joint import reconstruction_core as core

ROOT=io.ROOT
CONFIG=ROOT/'configs/fu_classifier_document_reconstruction_v1.json'
CODE=['fall_pipeline/benchmarks/fu_classifier_reconstruction.py','fall_pipeline/joint/reconstruction_inputs.py',
      'fall_pipeline/joint/reconstruction_core.py']+io.CODE
CLASSES={'logistic':LogisticRegression,'linear_svm':LinearSVC,'rbf_svm':SVC,'random_forest':RandomForestClassifier}


def publish(root,stage,**details):
    state={'stage':stage,'time':datetime.now(timezone.utc).isoformat(),**details};io.save_json(root/'status.json',state)
    print(io.json.dumps(state,ensure_ascii=False),flush=True)
    if root.name.endswith('_smoke'): return
    status='completed' if stage=='completed' else 'paused' if stage in ('failed','paused') else 'in_progress'
    header=f'- 문서 ID: `DOC-20260922-fu-classifier-run-R1`\n- 기준일: 2026-09-22\n- 상태: `{status}`\n'
    internal=f'# FU classifier 실행 기록\n\n{header}\n```json\n{io.json.dumps(state,indent=2,ensure_ascii=False)}\n```\n\nroot: `{root}`\n'
    shared=f'# FU classifier 비교 진행\n\n{header}\n현재 단계: {stage}.\n'
    if 'model' in details: shared+=f'분류기: {details["model"]}, fold: {details.get("fold", "")}.\n'
    shared+='\n[방법](2026-09-22_fu_classifier_reconstruction_shared.md)을 따른다. V3 reference 대기는 분류기 자체의 검증 완료와 구분한다.\n'
    for kind,value in (('internal',internal),('shared',shared)):
        path=ROOT/f'docs/{kind}/2026-09-22_fu_classifier_run_{kind}.md';temp=path.with_suffix('.tmp');temp.write_text(value);os.replace(temp,path)


def make_model(name,params,config):
    return Pipeline([('scaler',StandardScaler()),('classifier',CLASSES[name](**params,random_state=config['seed']))])


def fit(model,x,y):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always');model.fit(x,y)
    if any(issubclass(w.category,ConvergenceWarning) for w in seen): raise RuntimeError('candidate did not converge; preserve failure, no outer evaluation')
    return [str(w.message) for w in seen]


def outputs(model,x):
    prediction=model.predict(x).astype(np.int64)
    score=model.decision_function(x) if hasattr(model,'decision_function') else model.predict_proba(x)[:,1]
    score=np.asarray(score,np.float64)
    io.require(prediction.shape==score.shape==(len(x),) and np.isfinite(score).all(),'prediction shape/finite')
    return prediction,score


def metrics(y,actions,pred,score):
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel());lying=actions==4
    return {'count':len(y),'f1':float(f1_score(y,pred,zero_division=0)),'auprc':float(average_precision_score(y,score)),
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'lying_fp':int(np.sum(pred[lying]==1)),
            'lying_count':int(lying.sum()),'lying_fpr':float(np.mean(pred[lying]==1)) if lying.any() else None}


def selected_index(rows): return max(range(len(rows)),key=lambda i:(rows[i]['metrics']['f1'],rows[i]['metrics']['auprc']))


def model_save(path,model):
    temp=path.with_suffix('.tmp');joblib.dump(model,temp,compress=0);os.replace(temp,path)


def train(root,fu,config,contract,smoke):
    names=list(config['grids']);folds=[0] if smoke else range(5)
    for name in names:
        for k in folds:
            io.pause(root);train_idx,val_idx,outer=core.split_indices(fu['folds'],fu['subjects'],k,True)
            dest=root/name/f'fold{k}';dest.mkdir(parents=True,exist_ok=True);history=[]
            grid=list(ParameterGrid(config['grids'][name]));grid=grid[:1] if smoke else grid
            for i,params in enumerate(grid):
                io.pause(root);path=dest/f'candidate_{i:03d}.json';weight=dest/f'candidate_{i:03d}.joblib'
                if path.exists():
                    row=io.read(path);io.require(row['params']==params and row['contract']==contract and io.sha256_file(weight)==row['checkpoint_sha256'],'candidate resume lineage')
                    for key in ('pred','score'): io.require(io.sha256_file(dest/f'{key}_{i:03d}.npy')==row[key+'_sha256'],'candidate payload changed')
                else:
                    model=make_model(name,params,config);notes=fit(model,fu['x'][train_idx],fu['y'][train_idx])
                    pred,score=outputs(model,fu['x'][val_idx]);model_save(weight,model)
                    io.save_array(dest/f'pred_{i:03d}.npy',pred);io.save_array(dest/f'score_{i:03d}.npy',score)
                    row={'params':params,'metrics':metrics(fu['y'][val_idx],fu['actions'][val_idx],pred,score),
                         'checkpoint_sha256':io.sha256_file(weight),'pred_sha256':io.sha256_file(dest/f'pred_{i:03d}.npy'),
                         'score_sha256':io.sha256_file(dest/f'score_{i:03d}.npy'),'contract':contract,'warnings':notes}
                    io.save_json(path,row)
                history.append(row)
            best=selected_index(history)
            lock={'candidate':best,'params':history[best]['params'],'checkpoint_sha256':history[best]['checkpoint_sha256'],
                  'train_indices':train_idx.tolist(),'val_indices':val_idx.tolist(),'outer_indices':outer.tolist(),
                  'history':history,'contract':contract,'outer_unread_before_selection':True}
            if (dest/'selection.json').exists(): io.require(io.read(dest/'selection.json')==lock,'selection changed')
            else: io.save_json(dest/'selection.json',lock)
            weight=dest/f'candidate_{best:03d}.joblib';io.require(io.sha256_file(weight)==lock['checkpoint_sha256'],'own selected checkpoint hash')
            model=joblib.load(weight)  # Only this run's own hash-verified file, never an external pickle.
            pred,score=outputs(model,fu['x'][outer]);io.save_array(dest/'outer_pred.npy',pred);io.save_array(dest/'outer_score.npy',score)
            io.save_json(dest/'outer.json',{'selection_sha256':io.sha256_file(dest/'selection.json'),
                'pred_sha256':io.sha256_file(dest/'outer_pred.npy'),'score_sha256':io.sha256_file(dest/'outer_score.npy'),
                'metrics':metrics(fu['y'][outer],fu['actions'][outer],pred,score)})
            publish(root,'nested',model=name,fold=k,candidates=len(grid));io.disk_gate(config)


def audit(root,fu,config,smoke):
    results={};folds=[0] if smoke else range(5)
    for name in config['grids']:
        ids=[];preds=[];scores=[]
        for k in folds:
            io.pause(root);dest=root/name/f'fold{k}';lock=io.read(dest/'selection.json');outer_report=io.read(dest/'outer.json')
            train_idx,val_idx,outer=core.split_indices(fu['folds'],fu['subjects'],k,True)
            for key,value in (('train_indices',train_idx),('val_indices',val_idx),('outer_indices',outer)): np.testing.assert_array_equal(lock[key],value)
            rows=lock['history'];io.require(selected_index(rows)==lock['candidate'],'independent candidate selection')
            for i,row in enumerate(rows):
                for key in ('pred','score'): io.require(io.sha256_file(dest/f'{key}_{i:03d}.npy')==row[key+'_sha256'],'inner payload hash')
                pred=np.load(dest/f'pred_{i:03d}.npy',allow_pickle=False);score=np.load(dest/f'score_{i:03d}.npy',allow_pickle=False)
                io.require(metrics(fu['y'][val_idx],fu['actions'][val_idx],pred,score)==row['metrics'],'inner metrics')
                io.require(io.sha256_file(dest/f'candidate_{i:03d}.joblib')==row['checkpoint_sha256'],'candidate model hash')
            model=joblib.load(dest/f'candidate_{lock["candidate"]:03d}.joblib')
            fresh=make_model(name,lock['params'],config);fit(fresh,fu['x'][train_idx],fu['y'][train_idx])
            np.testing.assert_array_equal(model['scaler'].mean_,fu['x'][train_idx].astype(np.float64).mean(0))
            for idx,pred_path,score_path in ((val_idx,dest/f'pred_{lock["candidate"]:03d}.npy',dest/f'score_{lock["candidate"]:03d}.npy'),
                                           (outer,dest/'outer_pred.npy',dest/'outer_score.npy')):
                old=outputs(model,fu['x'][idx]);replay=outputs(fresh,fu['x'][idx]);saved=(np.load(pred_path,allow_pickle=False),np.load(score_path,allow_pickle=False))
                for expected,loaded,retrained in zip(saved,old,replay): np.testing.assert_array_equal(expected,loaded);np.testing.assert_array_equal(expected,retrained)
            io.require(io.sha256_file(dest/'selection.json')==outer_report['selection_sha256'],'outer selection lock')
            for key in ('pred','score'): io.require(io.sha256_file(dest/f'outer_{key}.npy')==outer_report[key+'_sha256'],'outer payload')
            pred,score=outputs(model,fu['x'][outer]);io.require(metrics(fu['y'][outer],fu['actions'][outer],pred,score)==outer_report['metrics'],'outer metrics')
            ids.extend(outer.tolist());preds.extend(pred.tolist());scores.extend(score.tolist())
        order=np.argsort(ids);idx=np.array(ids,np.int64)[order];pred=np.array(preds,np.int64)[order];score=np.array(scores,np.float64)[order]
        io.require(len(set(idx))==len(idx) and (smoke or np.array_equal(idx,np.arange(len(fu['y'])))),'OOF exact once')
        io.save_array(root/name/'oof_indices.npy',idx);io.save_array(root/name/'oof_pred.npy',pred);io.save_array(root/name/'oof_score.npy',score)
        results[name]=metrics(fu['y'][idx],fu['actions'][idx],pred,score)
    report={'passed':True,'deterministic_refit_exact':True,'inner_selection_verified':True,'scaler_train_only':True,'oof_exact_once':True,'results':results}
    io.save_json(root/'independent_audit.json',report);return report


def reference(root,fu,config):
    source=ROOT/config['reference_root'];final=io.read(source/'final_report.json')
    io.require(final['passed'] and final['research_usable'],'V3 reference incomplete')
    io.require(final['contract']['matched_method_contract']['fu']==io.read(ROOT/config['source_config'])['fu'],'V3 reference FU feature lineage differs')
    io.require(io.sha256_file(source/'independent_audit.json')==final['audit_sha256'] and io.read(source/'independent_audit.json')['passed'],'V3 reference audit')
    comparison={}
    for phase in ('j0','j1'):
        ids=np.load(source/'nested'/f'{phase}_oof_indices.npy',allow_pickle=False);logits=np.load(source/'nested'/f'{phase}_oof_logits.npy',allow_pickle=False)
        np.testing.assert_array_equal(ids,np.arange(993))
        m=core.binary_metrics(fu['y'],fu['actions'],logits);io.require(m==final['nested'][phase]['metrics'],'V3 nested exact replay')
        comparison[phase]=m
    local=io.read(root/'classifier_report.json');io.require(local['passed'] and local['research_usable'],'classifier full incomplete')
    io.require(io.sha256_file(root/'independent_audit.json')==local['audit_sha256'],'classifier audit changed')
    result={**local,'reference_pending':False,'reference':comparison,'reference_final_sha256':io.sha256_file(source/'final_report.json')}
    io.save_json(root/'final_report.json',result);return result


def make_contract(config,smoke):
    io.require(io.sha256_file(ROOT/config['source_config'])==config['source_config_sha256'],'FU source configuration changed')
    defaults={name:[make_model(name,p,config)['classifier'].get_params(deep=False) for p in ParameterGrid(grid)] for name,grid in config['grids'].items()}
    return {'config_sha256':io.sha256_file(CONFIG),'code_sha256':{p:io.sha256_file(ROOT/p) for p in CODE},
            'source_config_sha256':config['source_config_sha256'],'sklearn':sklearn.__version__,'joblib':joblib.__version__,
            'effective_estimator_parameters':defaults,'smoke':smoke,'historical_exact_reproduction':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('train','reference','all'),default='train');parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=io.read(CONFIG);contract=make_contract(config,args.smoke)
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only classifier runner')
    root=io.guard(ROOT/(config['output_dir']+('_smoke' if args.smoke else '')));existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    io.disk_gate(config,max(0,config['execution']['output_budget_gib']*1024**3-existing));io.initialize(root,contract,args.resume)
    import torch
    torch.set_num_threads(2)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish(root,'source');fu=inputs.load_fu(io.read(ROOT/config['source_config']))
            if args.stage!='reference':
                train(root,fu,config,contract,args.smoke);publish(root,'audit');result=audit(root,fu,config,args.smoke)
                io.require(make_contract(config,args.smoke)==contract,'classifier code/config changed')
                io.save_json(root/'classifier_report.json',{'passed':True,'research_usable':not args.smoke,'results':result['results'],
                    'audit_sha256':io.sha256_file(root/'independent_audit.json'),'contract':contract,'reference_pending':True,'historical_exact_reproduction':False})
            if args.stage in ('reference','all'):
                io.require(not args.smoke,'reference comparison requires full classifiers');reference(root,fu,config);publish(root,'completed')
            else: publish(root,'classifiers_completed_reference_pending',research_usable=not args.smoke)
        except BaseException as error:
            publish(root,'paused' if isinstance(error,io.PauseRequested) else 'failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
