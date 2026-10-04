"""Official HFD C3D extraction and source SVM reproduction; frozen external test."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import ast, argparse, hashlib, json, sys, time, types, contextlib, io
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2, numpy as np, joblib
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from fall_pipeline.external.le2i_current_evaluation import match_events, metrics
E=ROOT/'docs/internal/2026-10-04_hfd_x3d_external_comparison_evidence'
OUT=ROOT/'data/fall_processed/RGB/hfd_reproduction_20261004_r1'
OLD=ROOT/'data/fall_processed/RGB/original_baselines_20261004_r2'
CFG=ROOT/'configs/hfd_reproduction_20261004_r1.json'
NB=ROOT/'third_party/original_hfd_3dcnn_20261004/HFD_3D_CNN_EA_GitHub_.ipynb'
WEIGHTS=ROOT/'data/source_archives/OriginalCheckpointSearch_20261004_r1/c3d_sports1m_author_linked.h5'
def read(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_rows():return sorted([x for x in read(E/'hfd_source_manifest.json')['files'] if 'label'in x],key=lambda x:x['notebook_expected']['order'])
def check_contract():
    for p,h in read(OUT/'contract.json')['files'].items():assert sha(ROOT/p)==h,p
def official_namespace():
    sys.path.insert(0,str(ROOT/'data/source_archives/OriginalBaselines_20261004_r1/runtime'))
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8);tf.config.threading.set_inter_op_parallelism_threads(1)
    ns=dict(frame_n=16,Sequential=tf.keras.Sequential,Model=tf.keras.Model,os=os,cv2=cv2,np=np,cams=['cam1'])
    ns.update({k:getattr(tf.keras.layers,k) for k in ['Conv3D','MaxPooling3D','ZeroPadding3D','Flatten','Dense','Dropout']})
    selected=[]
    for c in read(NB)['cells']:
        s=''.join(c['source'])
        if 'def create_C3D_model' in s or 'def getFeatureExtractor' in s:
            selected.extend(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef))
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(NB),'exec'),ns)
    return ns
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'contract.json').exists(),'Already frozen'
    rows=source_rows();assert len(rows)==32 and sum(r['chunks'] for r in rows)==360
    assert all(r['frames']==r['notebook_expected']['frames'] for r in rows)
    for n in ['plan.json','ground_truth.json','own_predictions.json']:save(OUT/n,read(OLD/n))
    config=dict(experiment='HFD_OFFICIAL_CODE_REPRODUCTION_20261004_R1',author_commit='b74c13524979302641c1e55fc8582939daa58800',
      provenance='Official code reproduction; newly fitted SVM, not author final checkpoint',source='GMDCSA original 32 / GMDCSA24 Subject 1 only',
      source_order='Exact video order embedded in author extraction stdout; feature CSV read order historically unrecorded',
      source_fps='Published files average rates 29.23..29.78; notebook prints30. FPS is not used in feature extraction: no resampling',
      input=dict(color='OpenCV BGR',resize=[112,112],resize_interpolation='cv2.INTER_LINEAR',dtype='float32',range=[0,255],normalization='none',clip=16,stride=16,tail='discard fewer than16 frames',feature='fc6 4096'),
      svm=dict(kernel='linear',C=1,probability=True),numpy_seed=33,
      validation=dict(type='StratifiedShuffleSplit',n_splits=5,train_size=.7,random_state=33,unit='chunk; video overlap allowed by official code'),
      final_export='After reporting all5 source folds, refit fixed SVC on all360 source chunks; export newly fitted classifier. This all-source export is an external-test extension, not present in original notebook.',
      inference=dict(decision='SVC.predict; margin threshold0; no probability threshold tuning',event='positive-chunk rising edges, time of last actual frame of the positive chunk; initial previous class0',video='any positive chunk',decoder='OpenCV native sequential frames, pixel hashes checked against frozen common traces',failure='retain video in denominator with no alarm'),
      metric=dict(early=.5,late=3.,matching='existing ordered greedy one-to-one common evaluator'),
      target_training=False,target_adaptation=False,target_threshold_tuning=False,source_reference='paper F1 98.08%; notebook stored F1 99.1443%; not guaranteed bit-exact historical reproduction')
    save(CFG,config)
    files=[Path(__file__),NB,WEIGHTS,CFG,E/'hfd_source_manifest.json',ROOT/'fall_pipeline/external/le2i_current_evaluation.py']
    files += [ROOT/r['path'] for r in rows]+[OUT/n for n in ['plan.json','ground_truth.json','own_predictions.json']]
    for item in read(OUT/'plan.json'):files.extend([ROOT/item['video'],ROOT/item['trace']])
    save(OUT/'contract.json',dict(created=datetime.now(timezone.utc).isoformat(),files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('frozen',len(files),'files',flush=True)
def train():
    check_contract();assert not (OUT/'svm.joblib').exists(),'Do not overwrite trained model'
    start=time.monotonic();rows=source_rows();base=OUT/'source_links'
    for r in rows:
        p=base/('Fall' if r['label'] else 'ADL')/'cam1'/Path(r['path']).name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.symlink_to(ROOT/r['path'])
    order={a:[Path(r['path']).name for r in rows if r['label']==int(a=='Fall')] for a in ['Fall','ADL']}
    ns=official_namespace()
    # Freeze otherwise unspecified filesystem traversal using the notebook's output order.
    ns['os']=types.SimpleNamespace(path=os.path,listdir=lambda p:order[Path(p).parent.name])
    X,y=ns['extractFeatures'](str(WEIGHTS),str(base),'',False)
    assert X.shape==(360,4096) and np.isfinite(X).all() and np.bincount(y).tolist()==[201,159]
    np.savez_compressed(OUT/'source_features.npz',X=X,y=y)
    groups=np.concatenate([np.repeat(i,r['chunks']) for i,r in enumerate(rows)])
    np.random.seed(33);clf=svm.SVC(kernel='linear',C=1,probability=True)
    cv=StratifiedShuffleSplit(n_splits=5,train_size=.7,random_state=33);folds=[]
    for k,(tr,te) in enumerate(cv.split(X,y),1):
        clf.fit(X[tr],y[tr]);pred=clf.predict(X[te]);cm=confusion_matrix(y[te],pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
        fold=dict(fold=k,train_indices=tr.tolist(),test_indices=te.tolist(),predictions=pred.tolist(),
          confusion=cm.tolist(),accuracy=float((tp+tn)/len(te)),fall=metrics(tp,fp,fn),specificity=tn/(tn+fp),
          videos_on_both_sides=len(set(groups[tr])&set(groups[te])))
        folds.append(fold);print('source fold',k,fold['fall'],flush=True)
    validation=dict(folds=folds,mean_accuracy=float(np.mean([r['accuracy'] for r in folds])),mean_fall_f1=float(np.mean([r['fall']['f1'] for r in folds])),
      source_video_count=32,source_chunks=360,feature_sha256=sha(OUT/'source_features.npz'),seconds=time.monotonic()-start)
    save(OUT/'source_validation.json',validation)
    # Fixed predeclared export rule, no target feedback or source fold selection.
    clf.fit(X,y);joblib.dump(clf,OUT/'svm.joblib')
    save(OUT/'model_receipt.json',dict(svm_sha256=sha(OUT/'svm.joblib'),contract_sha256=sha(OUT/'contract.json'),feature_sha256=sha(OUT/'source_features.npz'),
         created=datetime.now(timezone.utc).isoformat(),classes=clf.classes_.tolist(),support_vectors=list(clf.support_vectors_.shape)))
    check_contract();print('source done',validation['mean_fall_f1'],flush=True)
def evaluate():
    check_contract();receipt=read(OUT/'model_receipt.json');assert receipt['svm_sha256']==sha(OUT/'svm.joblib')
    clf=joblib.load(OUT/'svm.joblib');ns=official_namespace();extractor=ns['getFeatureExtractor'](str(WEIGHTS),'fc6',False)
    start=time.monotonic();rows=[];gts={r['id']:r for r in read(OUT/'ground_truth.json')}
    for i,item in enumerate(read(OUT/'plan.json'),1):
        dest=OUT/item['id'];dest.mkdir(exist_ok=True)
        if (dest/'prediction.json').exists():
            r=read(dest/'prediction.json');assert r['contract_sha256']==sha(OUT/'contract.json');assert r['payload_sha256']==sha(dest/'features.npz');rows.append(r);continue
        r=dict(id=item['id'],scope=item['scope'],path=item['path'],model='hfd_reproduction',processed=False,predictions=[],video_prediction=0,contract_sha256=sha(OUT/'contract.json'))
        cap=None
        try:
            trace=read(ROOT/item['trace']); cap=cv2.VideoCapture(str(ROOT/item['video']));frames=[];n=0
            while True:
                ret,bgr=cap.read()
                if not ret:break
                assert n<len(trace['pts']) and hashlib.sha256(bgr.tobytes()).hexdigest()==trace['bgr_sha256'][n],(item['id'],n)
                frames.append(cv2.resize(bgr,(112,112)));n+=1
            cap.release();assert n==item['source_frames']==len(trace['pts'])
            nchunk=n//16;assert nchunk>0
            X=np.concatenate([extractor.predict(np.asarray([frames[k*16:(k+1)*16]],dtype=np.float32),verbose=0) for k in range(nchunk)])
            assert np.isfinite(X).all();pred=clf.predict(X);decision=clf.decision_function(X)
            end=np.arange(15,nchunk*16,16);times=(np.array(trace['pts'],np.int64)[end]-trace['start_pts'])*trace['time_base_num']/trace['time_base_den']
            rising=(pred==1)&np.r_[True,pred[:-1]==0]
            alarms=[dict(frame=int(f),time=float(t)) for f,t in zip(end[rising],times[rising])]
            np.savez_compressed(dest/'features.npz',features=X,predictions=pred,decision=decision,end_frames=end,times=times)
            r.update(processed=True,predictions=alarms,video_prediction=int(pred.any()),frames=n,chunks=nchunk,tail=n%16,payload_sha256=sha(dest/'features.npz'))
        except Exception as e:
            r.update(error=repr(e));print('FAILED',item['id'],repr(e),flush=True)
        finally:
            if cap is not None:cap.release()
        r['event']=match_events(r['predictions'],gts[item['id']]['episodes']);save(dest/'prediction.json',r);rows.append(r)
        if i%10==0:print('external',i,'/230',round(time.monotonic()-start,1),'sec',flush=True)
    save(OUT/'predictions.json',rows);check_contract();score()
def score():
    rows=read(OUT/'predictions.json')+read(OUT/'own_predictions.json');gts={r['id']:r for r in read(OUT/'ground_truth.json')};summary={}
    for model in ['own','hfd_reproduction']:
        summary[model]={}
        for scope in ['le2i130','cauca100']:
            subset=[r for r in rows if r['model']==model and r['scope']==scope];tp=fp=fn=vtp=vfp=vfn=vtn=0
            for r in subset:
                e=match_events(r['predictions'],gts[r['id']]['episodes']);tp+=e['tp'];fp+=e['fp'];fn+=e['fn']
                gt=bool(gts[r['id']]['episodes']);p=bool(r['video_prediction']);vtp+=int(gt and p);vfp+=int(not gt and p);vfn+=int(gt and not p);vtn+=int(not gt and not p)
            summary[model][scope]=dict(videos=len(subset),processed=sum(r['processed'] for r in subset),event=metrics(tp,fp,fn),video=dict(**metrics(vtp,vfp,vfn),tn=vtn,accuracy=(vtp+vtn)/len(subset)))
    save(OUT/'evaluation.json',dict(summaries=summary,rows=rows));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','train','evaluate','score']);a=p.parse_args();globals()[a.action]()
