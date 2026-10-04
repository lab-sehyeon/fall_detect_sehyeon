"""Actual URFD RGB/pose/posterior examples, including missing case categories."""
from pathlib import Path
import sys,os,json,zipfile,re
os.environ['MPLCONFIGDIR']='/tmp/foundskel-evidence-mpl'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fall_pipeline.external import rgb_document_io as io

def main():
    root=ROOT/'data/fall_processed/RGB/external_urfd_20261002_r1';out=ROOT/'docs/images/paper_evidence_20261002'
    report=io.read(root/'evaluation.json');plan={p['id']:p for p in io.read(root/'plan.json')}
    audit=io.read(ROOT/'docs/internal/2026-10-02_evidence_audit/urfd_independent_audit.json');reasons={r['id']:r['reason'] for r in audit['rows']}
    categories=[('TP',1,1),('TN',0,0),('FP',0,1),('FN',1,0),('SHORT',None,None),('QUALITY',None,None)]
    fig,axes=plt.subplots(6,2,figsize=(14,18),gridspec_kw={'width_ratios':[1,1.6]})
    selected=[];bones=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for row,(category,label,pred) in enumerate(categories):
        candidates=[r for r in report['rows'] if (r['quality_passed'] and r['label']==label and r['prediction']==pred) if category in ['TP','TN','FP','FN']] if category in ['TP','TN','FP','FN'] else [r for r in report['rows'] if reasons[r['id']]==('shorter_than64' if category=='SHORT' else 'pose_quality')]
        left,right=axes[row]
        if not candidates:
            for ax in [left,right]:ax.axis('off')
            left.text(.1,.5,f'{category}: no observed classified sequence\nNo example fabricated.',fontsize=13)
            selected.append(dict(category=category,available=False));continue
        case=sorted(candidates,key=lambda r:r['id'])[0];sid=case['id'];item=plan[sid];dest=root/sid
        with np.load(dest/'frontend.npz') as f:
            if case['quality_passed']:
                with np.load(dest/'inference.npz') as z:
                    logits=z['G0'].astype(float);exp=np.exp(logits-logits.max(1,keepdims=True));p=exp[:,1]/exp.sum(1)
                    starts=z['window_starts'];winning=logits.argmax(1)==1
                k=int(np.flatnonzero(winning)[0]) if winning.any() else int(p.argmax());center=int(starts[k]+32)
                t=(starts+32)/25
                right.plot(t,p,color='#008b9b',marker='o',markersize=3,label='P(fall)')
                right.scatter(t[winning],p[winning],color='#d97120',s=14,label='fall argmax')
                right.axvline(center/25,color='#ba315b',linestyle='--',label='shown frame')
                right.set(xlabel='Time (s), window center',ylabel='Fall probability',ylim=(-.02,1.02));right.grid(alpha=.25);right.legend(fontsize=8,loc='upper right')
                right.set_title('Sequence GT: '+('fall' if case['label'] else 'ADL')+' | any-window prediction: '+('fall' if case['prediction'] else 'no alarm'),fontsize=11)
            else:
                center=len(f['xy'])//2;right.axis('off')
                reason=reasons[sid]
                text=(f"Only {item['frames']} canonical frames; minimum 64.\nNo padding; classifier not run.\nRecorded as no alarm in full 70-sequence evaluation." if reason=='shorter_than64' else f"Frozen pose-quality gate failed.\nBBox coverage: {case['quality']['bbox_coverage']:.3f}\nPose coverage: {case['quality']['pose_coverage']:.3f}\nMedian pelvis confidence: {case['quality']['pelvis_median']:.3f}\nClassifier not run; recorded as no alarm.")
                right.text(.03,.72,text,fontsize=12,va='top',linespacing=1.7)
            source=int(f['source_indices'][center]);xy=f['xy'][center];sc=f['scores'][center]
        member=item['png_members'][source]
        with zipfile.ZipFile(ROOT/item['archive']) as z:bgr=cv2.imdecode(np.frombuffer(z.read(member),np.uint8),cv2.IMREAD_COLOR)
        assert bgr is not None
        left.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        for a,b in bones:
            if min(sc[a],sc[b])>0:left.plot(xy[[a,b],0],xy[[a,b],1],color='#00e5ff',lw=1.2)
        left.scatter(xy[sc>0,0],xy[sc>0,1],s=5,color='#ffed45')
        left.set_title(f'{category} sequence | {sid} | source frame {source+1}',fontsize=11);left.axis('off')
        selected.append(dict(category=category,available=True,id=sid,sequence_label=case['label'],prediction=case['prediction'],
            canonical_frame=center,source_frame_index=source,archive_sha256=item['archive_sha256'],png_member=member,
            frontend_sha256=io.sha(dest/'frontend.npz'),quality_passed=case['quality_passed']))
    fig.suptitle('Observed URFD results and input failures',fontsize=20,y=.996)
    fig.tight_layout(rect=(0,.035,1,.98),h_pad=2)
    fig.text(.02,.008,'Actual RGB and frozen ViTPose-B outputs. TP/TN/FP/FN refer to sequence labels; no event-time GT is drawn.\nFirst lexical classified sequence per case. First alarm window shown; otherwise maximum fall probability. Offline processing.',fontsize=10)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'urfd_cases.{ext}',dpi=150)
    io.save(ROOT/'docs/internal/2026-10-02_evidence_audit/urfd_case_provenance.json',dict(passed=True,cases=selected,
        selection='lexically first classified sequence per TP/TN/FP/FN; first lexical short and pose rejection; no fabricated missing cases',
        frame_selection='first winning window center, else maximum-fall window center, else midpoint for rejection',
        dataset='URFD camera0',evaluation_sha256=io.sha(root/'evaluation.json')))
    print(json.dumps(selected,indent=2))

if __name__=='__main__':main()
