"""Full regression, independent CPU math and source-specific metric audit."""
from pathlib import Path
import csv,os,sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import j1_g0_evaluation_20261002 as run
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external.rgb_recovery_common import decode_recovery_endpoints,D1_DECOUPLED
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,average_precision_score


def cpu_replay(pooled,joint,head):
    x=torch.from_numpy(pooled)
    z=F.layer_norm(x,(2048,),joint['adapter.norm.weight'],joint['adapter.norm.bias'],eps=1e-5)
    z=F.gelu(F.linear(z,joint['adapter.down.weight'],joint['adapter.down.bias']))
    adapted=x+F.linear(z,joint['adapter.up.weight'],joint['adapter.up.bias'])
    logits=F.linear(adapted,head['weight'],head['bias'])
    return adapted.numpy(),logits.numpy()


def metric_values(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def check_metrics(actual,expected,tolerance=1e-12):
    for k,v in actual.items():io.require(abs(v-expected[k])<tolerance,'metric mismatch: '+k)


def independent_truth(scope,item,source_item,official):
    if scope=='urfd70':return int(item['id'].startswith('fall-'))
    if scope=='le2i_cs38':
        return sorted((float(r['start']),float(r['end'])) for r in official if r['path']==source_item['path'] and int(r['label'])==1)
    gt=item['truth'];path=ROOT/gt['annotation']
    io.require(io.sha(path)==gt['annotation_sha256'],'original annotation changed')
    a,b=map(int,path.read_text().splitlines()[:2])
    return [] if a==b==0 else [(a/source_item['source_fps'],b/source_item['source_fps'])]


def main():
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit required');torch.set_num_threads(2)
    cfg,plan=run.locked();result=io.read(run.OUTPUT/'evaluation.json')
    io.require(run.source_snapshot()==io.read(run.OUTPUT/'source_snapshot.json'),'source artifacts changed')
    runtime=io.read(run.OUTPUT/'runtime.json')
    io.require(set(runtime['children'])=={'backbone','adapter','head'},'extra active model branch')
    io.require(runtime['model_before']==runtime['model_after'],'model mutation')
    io.require(runtime['input_arrays_read']==['ntu25','window_starts'] and not runtime['global_features_loaded']
        and not runtime['adl_head_loaded'] and not runtime['extra_heads_created'],'unwanted inference input')
    joint=torch.load(ROOT/cfg['model']['j1']['path'],map_location='cpu',weights_only=True)
    head=torch.load(ROOT/cfg['model']['g0']['path'],map_location='cpu',weights_only=True)
    official=list(csv.DictReader((ROOT/'data/source_archives/OmniFall/le2i_cs_20261002_r1/metadata/labels/le2i.csv').read_text().splitlines()))
    sources={s:{r['id']:r for r in io.read(ROOT/spec['root']/'plan.json')} for s,spec in run.SOURCES.items()}
    previous={s:{r['id']:r for r in io.read(ROOT/spec['root']/'evaluation.json')['rows']} for s,spec in run.SOURCES.items()}
    totals={s:np.zeros(3,dtype=np.int64) for s in sources};ys=[];ps=[];probs=[]
    maxima={k:0. for k in ['pooled_regression','adapted_regression','G0_regression','cpu_adapter','cpu_G0']}
    classified=0;windows=0;rows=[]
    io.require(len(plan)==len(result['rows'])==235,'full scope')
    for n,(item,reported) in enumerate(zip(plan,result['rows']),1):
        run.budget();scope=item['scope'];sid=item['id'];source_item=sources[scope][sid]
        io.require((scope,sid)==(reported['scope'],reported['id']),'row order')
        source=ROOT/item['source'];dest=run.OUTPUT/scope/sid
        front=io.stage_done(source,'frontend')
        if source_item.get('preprocessing_rejection'):
            io.require(scope=='urfd70' and sid=='adl-37' and not item['classified'],'timestamp exception')
            quality=False
        else:
            with np.load(source/'frontend.npz') as f,np.load(ROOT/source_item['mapping']) as m:
                np.testing.assert_array_equal(f['source_indices'],m['source_indices'])
                np.testing.assert_array_equal(f['timestamps'],m['canonical_timestamps'])
                b,x,s=f['boxes'],f['xy'],f['scores']
                bv=np.isfinite(b).all(1)&(b[:,2]>b[:,0])&(b[:,3]>b[:,1])&(b[:,4]>0)
                pv=bv&np.isfinite(x).all((1,2))&np.isfinite(s).all(1)&(s>0).any(1)
                quality=bool(bv.mean()>=.8 and pv.mean()>=.8 and np.median(np.where(pv,np.minimum(s[:,11],s[:,12]),0))>=.3 and len(b)>=64)
        io.require(quality==item['classified']==front['quality']['passed'],'quality regression')
        times=[];prediction=0;probability=0.
        if quality:
            classified+=1;meta=io.stage_done(dest,'inference')
            io.require(meta['passed'] and meta['model_before']==meta['model_after']==runtime['model_before'],'frozen inference')
            with np.load(dest/'inference.npz') as z,np.load(source/'inference.npz') as old:
                io.require(set(z.files)=={'pooled','adapted','G0','window_starts','window_endpoints'},'extra prediction branch')
                starts=np.arange(0,item['frames']-63,8)
                np.testing.assert_array_equal(z['window_starts'],starts)
                np.testing.assert_array_equal(z['window_endpoints'],starts+63);windows+=len(starts)
                for key in ['pooled','adapted','G0']:
                    np.testing.assert_allclose(z[key],old[key],atol=1e-4,rtol=1e-5)
                    maxima[key+'_regression']=max(maxima[key+'_regression'],float(np.max(np.abs(z[key]-old[key]))))
                np.testing.assert_array_equal(z['G0'].argmax(1),old['G0'].argmax(1))
                with torch.inference_mode():adapted,logits=cpu_replay(z['pooled'],joint,head)
                for actual,key,outkey in [(adapted,'adapted','cpu_adapter'),(logits,'G0','cpu_G0')]:
                    np.testing.assert_allclose(actual,z[key],atol=1e-4,rtol=1e-5)
                    maxima[outkey]=max(maxima[outkey],float(np.max(np.abs(actual-z[key]))))
                np.testing.assert_array_equal(logits.argmax(1),z['G0'].argmax(1))
                decoded=decode_recovery_endpoints(starts+63,z['G0'],np.zeros(len(starts)),fall_alert_mode=D1_DECOUPLED)
                times=[r['time'] for r in decoded['events'] if r['type']=='fall_trigger']
                prediction=int(any(np.argmax(v)==1 for v in z['G0']))
                probability=float(max(1/sum(np.exp(v.astype(float)-float(v[1]))) for v in z['G0']))
        else:
            io.require(not (dest/'inference.npz').exists(),'quality bypass')
            io.require(io.read(dest/'rejection.json')['classifier_run'] is False,'rejection missing')
        io.require(times==[r['time'] for r in reported['predictions']],'independent D1 mismatch')
        io.require(prediction==reported['prediction'] and abs(probability-reported['max_fall_probability'])<1e-12,'sequence decision mismatch')
        gt=independent_truth(scope,item,source_item,official)
        oldrow=previous[scope][sid]
        if scope=='urfd70':
            io.require(gt==item['truth']['label']==oldrow['label'],'label mismatch')
            io.require(prediction==oldrow['prediction'],'old/new sequence decision mismatch')
            ys.append(gt);ps.append(prediction);probs.append(probability)
        else:
            io.require(len(gt)<=1,'independent matching assumes one event')
            stored=[(e['fall_start'],e['fall_end']) for e in item['truth']['episodes']]
            io.require(len(stored)==len(gt),'GT count')
            if gt:np.testing.assert_allclose(stored,gt,atol=5e-6,rtol=0)
            tp=int(bool(gt) and any(gt[0][0]-.5<=t<=gt[0][1]+3 for t in times))
            counts=(tp,len(times)-tp,len(gt)-tp);totals[scope]+=counts
            check_metrics(metric_values(*counts),reported['metrics'])
            check_metrics(metric_values(*counts),oldrow['heads']['G0'])
            io.require(times==[r['time'] for r in oldrow['heads']['G0']['predictions']],'old/new event sequence differs')
        rows.append(dict(scope=scope,id=sid,classified=quality,passed=True))
        if n%25==0:print('audit',n,len(plan),flush=True)
    verified={}
    for scope in ['le2i_cs38','le2i_127']:
        m=metric_values(*map(int,totals[scope]));check_metrics(m,result['datasets'][scope]['metrics']);verified[scope]=m
    tn,fp,fn,tp=map(int,confusion_matrix(ys,ps,labels=[0,1]).ravel())
    precision,recall,f1,_=precision_recall_fscore_support(ys,ps,average='binary',zero_division=0)
    m=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=float(precision),recall=float(recall),f1=float(f1),
        accuracy=float(np.mean(np.array(ys)==ps)),specificity=tn/(tn+fp),ap=float(average_precision_score(ys,probs)))
    check_metrics(m,result['datasets']['urfd70']['metrics']);verified['urfd70']=m
    run.locked()
    audit=dict(passed=True,entries=235,classified_entries=classified,windows=windows,
        datasets=verified,all_old_G0_argmax_identical=True,all_old_decisions_identical=True,
        independent_cpu_argmax_identical=True,source_artifacts_unchanged=True,max_abs=maxima,
        no_extra_model_branches=True,rows=rows,evaluation_sha256=io.sha(run.OUTPUT/'evaluation.json'),
        audit_code_sha256=io.sha(Path(__file__)))
    io.save(run.OUTPUT/'independent_audit.json',audit);run.status('completed',datasets=verified,windows=windows)


if __name__=='__main__':main()
