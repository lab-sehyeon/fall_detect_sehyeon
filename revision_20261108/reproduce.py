"""Regenerate quantitative revision tables and ROC/confusion figures from released predictions."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def binary(q):
    y=q.ground_truth.to_numpy(dtype=int);p=q.prediction.to_numpy(dtype=int)
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
    score=q.margin.fillna(q.margin.min()-1).to_numpy()
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn),auc=roc_auc_score(y,score))

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=HERE/'generated');args=a.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(HERE/'results/external_predictions.csv');rows=[]
    for (condition,seed,ds),q in d.groupby(['condition','seed','dataset'],sort=False):
        assert len(q)==(100 if ds=='cauca100' else 130) and not q.id.duplicated().any()
        rows.append(dict(condition=condition,seed=seed,dataset=ds,**binary(q)))
    raw=pd.DataFrame(rows);raw.to_csv(out/'external_per_run.csv',index=False)
    stats=raw.groupby(['condition','dataset'])[['precision','recall','f1','auc']].agg(['mean','std'])
    stats.to_csv(out/'external_mean_sd.csv')
    pairs=[('no_fu_g0','raw'),('full_g0','raw'),('full_g0','no_fu_g0'),('full_j1','no_fu_j1'),('full_g0','full_j1'),('no_fu_g0','no_fu_j1')]
    contrasts=[]
    for a,b in pairs:
        for ds in ['cauca100','le2i130']:
            aa=raw[(raw.condition==a)&(raw.dataset==ds)].set_index('seed').f1
            bb=raw[(raw.condition==b)&(raw.dataset==ds)].set_index('seed').f1
            delta=(aa-bb).sort_index()*100
            contrasts.append(dict(contrast=a+' - '+b,dataset=ds,mean_pp=delta.mean(),sample_sd_pp=delta.std(ddof=1),positive_seeds=int((delta>0).sum())))
    pd.DataFrame(contrasts).to_csv(out/'paired_differences.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,['cauca100','le2i130']):
        for condition,color in [('raw','#307ac6'),('full_g0','#df4c68')]:
            for seed in range(5):
                q=d[(d.condition==condition)&(d.seed==seed)&(d.dataset==ds)]
                fpr,tpr,_=roc_curve(q.ground_truth,q.margin.fillna(q.margin.min()-1))
                ax.plot(fpr,tpr,color=color,alpha=.5,label=condition if seed==0 else None)
        ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title=ds);ax.legend()
    fig.tight_layout();fig.savefig(out/'roc.pdf');plt.close(fig)
    source=json.loads((HERE/'results/source_confusions.json').read_text());classrows=[]
    for r in source:
        cm=np.array(r['confusion']);tp=cm.diagonal();support=cm.sum(1);pred=cm.sum(0)
        for k,name in enumerate(['other','fall','lie_down','lying_down']):
            classrows.append(dict(condition=r['condition'],seed=r['seed'],split=r['split'],class_name=name,support=int(support[k]),
                                  precision=tp[k]/pred[k] if pred[k] else 0.,recall=tp[k]/support[k],f1=2*tp[k]/(pred[k]+support[k])))
    c=pd.DataFrame(classrows);c.to_csv(out/'source_per_class.csv',index=False)
    c.groupby(['condition','split','class_name'])[['precision','recall','f1']].agg(['mean','std']).to_csv(out/'source_mean_sd.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,['test','ood']):
        matrices=[np.array(r['confusion']) for r in source if r['condition']=='full_g0' and r['split']==split]
        norm=np.array([m/m.sum(1,keepdims=True)*100 for m in matrices]);m=norm.mean(0)
        ax.imshow(m,vmin=0,vmax=100,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{m[i,j]:.1f}',ha='center',va='center',color='white' if m[i,j]>50 else 'black')
        ax.set(title=split,xlabel='Prediction',ylabel='Ground truth',xticks=range(4),yticks=range(4))
    fig.tight_layout();fig.savefig(out/'confusion.pdf');plt.close(fig)
    # Verify against the fixed author export rather than only printing regenerated numbers.
    expected=pd.read_csv(HERE/'results/expected_external.csv')
    check=raw.merge(expected,on=['condition','seed','dataset'],suffixes=('','_expected'),validate='one_to_one')
    assert len(check)==len(expected)==len(raw)
    for col in ['precision','recall','f1','auc']:np.testing.assert_allclose(check[col],check[col+'_expected'],rtol=0,atol=1e-12)
    (out/'validation.json').write_text(json.dumps(dict(passed=True,external_runs=len(raw),source_confusions=len(source),new_inference=False),indent=2)+'\n')
    print('PASS: external metrics, paired differences, source class metrics, ROC and confusion figures')

if __name__=='__main__':main()
