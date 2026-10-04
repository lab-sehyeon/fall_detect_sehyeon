"""Paper figures from fixed cached evidence: no new model inference.

Three main figures and twenty single-column supplementary case figures.
PDF/SVG retain vector text and lines; source frames are embedded raster images.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import sys
import zipfile

import make_fall_demo_20261004 as old
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

ROOT=old.ROOT
OUT=ROOT/'docs/images/paper_demo_20261004'
EVIDENCE=ROOT/'docs/internal/2026-10-04_paper_demo_evidence'
NAMES=dict(old.NAMES,own='Ours')
SHORT={'own':'Ours','usdrl_ntu60':'USDRL+NTU60','hfd_reproduction':'HFD*',
       'privacy_x3d_uda_rgb':'X3D-UDA','flash':'FLASH*'}
BLUE='#0072b2'; ORANGE='#d55e00'; INK='#161616'; GRAY='#686868'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':9,
    'legend.fontsize':8.5,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'axes.edgecolor':'#777777','axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'paper-demo-20261004'})
CATALOG=[]
RENDER_CHECKS=[]


def axis(fig,x,y,w,h,**kw):
    fw,fh=fig.get_size_inches()
    return fig.add_axes([x/fw,y/fh,w/fw,h/fh],**kw)


def text(fig,x,y,s,size=9,**kw):
    fw,fh=fig.get_size_inches()
    return fig.text(x/fw,y/fh,s,fontsize=size,va='top',color=kw.pop('color',INK),**kw)


def image_axis(fig,x,y,w,h,im):
    ax=axis(fig,x,y,w,h);ax.imshow(im);ax.axis('off')
    return ax


def class_code(row):
    truth=bool(row['episodes']);pred=bool(row['video_prediction'])
    return {(True,True):'TP',(True,False):'FN',(False,True):'FP',(False,False):'TN'}[(truth,pred)]


def short_id(ident):
    return ident.removeprefix('le2i__').removeprefix('cauca__').replace('__','/')


def source_times(item):
    t=old.read(ROOT/item['trace'])
    return (np.asarray(t['pts'])-t['start_pts'])*t['time_base_num']/t['time_base_den']


def frame_indices(episodes,times,count=3):
    """Three or four illustrative points, independent of model output."""
    assert count in (3,4)
    if episodes:
        a,b=episodes[0]['fall_start'],episodes[0]['fall_end']
        wanted=[max(0,a-.4),a+.3*(b-a),b+.25]
        if count==4:wanted.insert(2,a+.65*(b-a))
    else:
        wanted=np.linspace(.15*times[-1],.85*times[-1],count)
    return [int(np.argmin(abs(times-min(t,times[-1])))) for t in wanted]


def load_series(row,item):
    """Extract actual output positions; never invent a continuous state trace."""
    m=row['model'];d=old.BASE/old.FOLDERS[m]/row['id']
    t=source_times(item);result=dict(source_times=t,processed=row['processed'],
                                   alarms=np.array([p['time'] for p in row.get('predictions',[])]))
    if m=='privacy_x3d_uda_rgb':
        p=old.softmax(row['logits']);assert int(p[1]>p[0])==row['video_prediction']
        assert abs(p[1]-row['fall_probability'])<1e-6
        return dict(result,probabilities=p,temporal=False)
    if not row['processed']:
        receipt=old.read(d/'pose.json')
        assert m=='flash' and 'no pose in entire video' in receipt['error']
        assert receipt['frames']==len(t)
        return dict(result,temporal=False,input_failure=True)
    if m in ('own','usdrl_ntu60'):
        if m=='own':d=old.BASE/f"{row['scope']}_20261003_r1"/row['id']
        z=old.arrays(d/'inference.npz',d/'inference.json')
        logits=z['G0' if m=='own' else 'logits'];p=old.softmax(logits);c=1 if m=='own' else 42
        times=z['window_endpoints']/25;positive=logits.argmax(1)==c
        result.update(score=p[:,c],competitor=np.delete(p,c,axis=1).max(1))
    elif m=='hfd_reproduction':
        z=old.arrays(d/'features.npz',d/'prediction.json')
        times=z['times'];positive=z['predictions'].astype(bool)
        np.testing.assert_array_equal(positive,z['decision']>0)
        np.testing.assert_allclose(times,t[z['end_frames']],rtol=0,atol=1e-12)
        result['score']=z['decision']
    else:
        z=old.arrays(d/'prediction.npz',d/'prediction.json')
        times=z['timestamps'];positive=z['alarms']
        np.testing.assert_array_equal(positive,(z['logits']>0)&z['detected'])
        np.testing.assert_allclose(times,t,rtol=0,atol=1e-12)
        result.update(score=z['logits'],detected=z['detected'])
    assert int(positive.any())==row['video_prediction']
    np.testing.assert_allclose(old.edges(positive,times),result['alarms'],rtol=0,atol=1e-9)
    return dict(result,times=times,positive=positive,temporal=True)


def verified_stills(row,item,count=3):
    times=source_times(item);indices=frame_indices(row['episodes'],times,count)
    return times,indices,old.decoded_frames(item,indices)


def save(fig,stem,caption,kind,book,metadata):
    # Automatic locators retain out-of-view tick Text objects even though the
    # axis does not draw them. Keep in-range ticks before auditing page bounds.
    for ax in fig.axes:
        if not ax.axison or ax.name=='3d':continue
        lo,hi=sorted(ax.get_xlim());ax.set_xticks([v for v in ax.get_xticks() if lo-1e-9<=v<=hi+1e-9])
        lo,hi=sorted(ax.get_ylim());ax.set_yticks([v for v in ax.get_yticks() if lo-1e-9<=v<=hi+1e-9])
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[a for a in fig.findobj(matplotlib.text.Text) if a.get_visible() and a.get_text()]
    assert texts and min(a.get_fontsize() for a in texts)>=8.5
    for a in texts:
        assert not any(token in a.get_text() for token in ('J1','G0','ourmodel'))
        bbox=a.get_window_extent(renderer)
        assert bbox.x0>=-1 and bbox.y0>=-1 and bbox.x1<=fig.bbox.width+1 and bbox.y1<=fig.bbox.height+1,(stem,a.get_text(),bbox.bounds,fig.bbox.bounds)
    dimensions=fig.get_size_inches().tolist()
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{stem}.{ext}',dpi=600)
    if book is not None:book.savefig(fig,dpi=600)
    svg=(OUT/f'{stem}.svg').read_text()
    assert '<text' in svg and '<path' in svg and '<image' in svg
    RENDER_CHECKS.append(dict(figure=stem,width_inches=dimensions[0],height_inches=dimensions[1],
                             minimum_font_pt=min(a.get_fontsize() for a in texts),visible_text_count=len(texts),
                             all_text_within_canvas=True,vector_text_paths_and_raster_frames=True))
    CATALOG.append(dict(stem=stem,kind=kind,caption=caption,**metadata))
    plt.close(fig)
    print('Rendered',stem,flush=True)


def pipeline(case,item,xmanifest,book):
    d=old.prepare_case(case,item,xmanifest);end=d['endpoint'];k=d['selected']
    fig=plt.figure(figsize=(7.16,2.45),dpi=160)
    xs=[.04,1.45,2.87,4.17,5.60]; widths=[1.19,1.19,1.06,1.16,1.50]
    heads=['RGB frame','2D pose','3D skeleton','Ours','Prediction']
    for x,w,h in zip(xs,widths,heads):text(fig,x+w/2,2.33,h,10,ha='center')
    im=d['images'][d['image_index']]
    image_axis(fig,xs[0]+.075,.74,1.04,1.20,im)
    ax=image_axis(fig,xs[1]+.075,.74,1.04,1.20,im)
    old.pose_overlay(ax,d['front']['xy'][end],d['front']['scores'][end],old.COCO_EDGES,d['front']['boxes'][end])
    ax=axis(fig,xs[2],.65,widths[2],1.35,projection='3d');q=d['lift']['ntu25'][end]
    for a,b in old.NTU_EDGES:ax.plot(q[[a,b],0],q[[a,b],1],q[[a,b],2],color=BLUE,lw=1.2)
    ax.scatter(*q.T,s=8,c=ORANGE,depthshade=False);ax.set_axis_off();ax.view_init(elev=18,azim=65)
    center=(q.max(0)+q.min(0))/2;span=max(np.ptp(q,axis=0).max(),.01)*.58
    ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span)
    ax.set_box_aspect((1,1,1))
    fig.add_artist(Rectangle((4.10/7.16,.54/2.45),1.30/7.16,1.56/2.45,transform=fig.transFigure,
                             facecolor='none',edgecolor='#aaaaaa',linewidth=.6))
    for y,name in [(1.66,'Frozen\nbackbone'),(1.13,'Residual\nadapter'),(.60,'4-class\nclassifier')]:
        ax=axis(fig,xs[3],y,widths[3],.36);ax.axis('off')
        ax.add_patch(Rectangle((.01,.02),.98,.96,facecolor='#f4f4f4',edgecolor='#777',linewidth=.6))
        ax.text(.5,.5,name,ha='center',va='center',fontsize=9)
    for y in (1.64,1.11):
        fig.add_artist(FancyArrowPatch((4.75/7.16,y/2.45),(4.75/7.16,(y-.13)/2.45),
                                      transform=fig.transFigure,arrowstyle='-|>',mutation_scale=8,lw=.6,color=INK))
    p=d['probs'][k]
    ax=axis(fig,6.23,.78,.83,1.17)
    names=['Other','Fall','Posture','Transition']
    ax.barh(range(4),p,color=[GRAY,BLUE,GRAY,GRAY],height=.48)
    ax.set_yticks(range(4),names,fontsize=8.5);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xticks([0,.5,1])
    ax.tick_params(axis='y',length=0,pad=2);ax.set_xlabel('Score',labelpad=2,fontsize=8.5)
    for i,v in enumerate(p):ax.text(.98,i,f'{v:.2f}',ha='right',va='center',fontsize=8.5,
                                  bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=0))
    for i in range(4):
        x0=xs[i]+widths[i]+.025;x1=xs[i+1]-.04
        if i==3:x1=5.60
        fig.add_artist(FancyArrowPatch((x0/7.16,1.32/2.45),(x1/7.16,1.32/2.45),
                                      transform=fig.transFigure,arrowstyle='-|>',mutation_scale=9,lw=.7,color=INK))
    text(fig,.64,.48,f'Le2i, t = {d["example_time"]:.2f} s',8.5,ha='center')
    text(fig,2.04,.48,'YOLOv8x + ViTPose',8.5,ha='center')
    text(fig,3.40,.48,'Proxy NTU25',8.5,ha='center')
    text(fig,4.75,.37,'64 frames',8.5,ha='center')
    text(fig,6.37,.34,'Argmax: Fall',9,ha='center',color=BLUE)
    caption=("Illustration of the RGB-to-fall inference pathway using a real Le2i sequence (Coffee_room_01_001). "
             "A frame at the end of the 64-frame window (2.84 s on the 25 Hz evaluation timeline) is shown alongside "
             "its stored person detection, ViTPose landmarks and MotionAGFormer-derived, normalized proxy NTU25 skeleton. "
             "The frozen skeleton backbone, residual adapter and four-class classifier operate on the full window, not on "
             "the displayed frame alone. Bars show the saved four-class softmax scores; the predicted class is selected by "
             "argmax. Posture denotes the lying posture class and Transition denotes the lie-down transition class. "
             "Pose overlays omit joints with display confidence below 0.2; this does not change model inputs or predictions. "
             "The skeleton is normalized, not a metric world-coordinate measurement.")
    save(fig,'fig1_inference_pathway',caption,'main',book,dict(case_id=case['id'],displayed_source_frame=d['image_index'],
         source_time=float(d['source_times'][d['image_index']]),window_end_time=d['example_time'],scores=p.tolist()))


def comparison_panel(fig,row,item,rows,base,letter):
    times,indices,images=verified_stills(row,item,4)
    text(fig,.04,base+2.83,f'({letter}) {short_id(row["id"])}',10)
    text(fig,7.10,base+2.83,f'Ours: {class_code(row)}',9,ha='right')
    for j,idx in enumerate(indices):
        x=1.15+j*1.41
        image_axis(fig,x,base+1.78,1.06,.80,images[idx])
        text(fig,x+.53,base+1.71,f'{times[idx]:.2f} s',8.5,ha='center')
    ax=axis(fig,1.29,base+.39,4.78,1.03)
    labels=['GT']+[SHORT[m] for m in old.FOLDERS]
    ax.set_yticks(range(6),labels);ax.set_ylim(5.55,-.6);ax.set_xlim(0,times[-1]);ax.tick_params(axis='y',length=0,pad=5)
    ax.set_xlabel('Time (s)',labelpad=1,fontsize=9);ax.spines['left'].set_visible(False)
    ax.set_xticks(np.linspace(0,times[-1],5));ax.set_xticklabels([f'{v:g}' if abs(v-round(v))<.01 else f'{v:.1f}' for v in np.linspace(0,times[-1],5)])
    for j in range(1,6):ax.axhline(j,color='#dddddd',lw=.55,zorder=0)
    for ep in row['episodes']:ax.plot([ep['fall_start'],ep['fall_end']],[0,0],color=INK,lw=5,solid_capstyle='butt')
    text(fig,6.39,base+1.59,'Video',8.5,ha='center')
    text(fig,6.96,base+1.59,'E-FP',8.5,ha='center')
    public=[]
    for j,m in enumerate(old.FOLDERS,1):
        r=rows[m];z=load_series(r,item);c=BLUE if m=='own' else GRAY
        if z.get('input_failure'):
            ax.text(times[-1]/2,j,'No valid pose',ha='center',va='center',fontsize=8.5,color=GRAY)
        elif not z['temporal']:
            ax.text(times[-1]/2,j,'Video-level output only',ha='center',va='center',fontsize=8.5,color=GRAY,
                    bbox=dict(facecolor='white',edgecolor='none',pad=0))
        else:
            pp=z['times'][z['positive']]
            ax.plot(pp,np.full(len(pp),j),'|',color=c,markersize=6,markeredgewidth=1)
            ax.plot(z['alarms'],np.full(len(z['alarms']),j),'^',markerfacecolor='white',markeredgecolor=ORANGE,
                    markeredgewidth=.7,markersize=4,linestyle='none',zorder=4)
        ax.text(1.067,j,class_code(r),transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=9,fontweight='bold' if m=='own' else 'normal')
        efp=str(r['event']['fp']) if 'event' in r else '—'
        ax.text(1.186,j,efp,transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=9)
        public.append(dict(model=NAMES[m],model_key=m,case=class_code(r),processed=r['processed'],
                           video_prediction=r['video_prediction'],event={k:r['event'][k] for k in ('tp','fp','fn')} if 'event' in r else None,
                           alarm_times=z['alarms'].tolist() if z['temporal'] else None,
                           positive_output_times=z['times'][z['positive']].tolist() if z['temporal'] else None))
    return dict(id=row['id'],gt=row['episodes'],frames=indices,frame_times=[float(times[i]) for i in indices],models=public)


def comparison(scope,owncases,by_model,plan,book):
    fig=plt.figure(figsize=(7.16,6.30),dpi=160)
    panels=[]
    for row,base,letter in zip(owncases,[3.22,.23],['a','b']):
        rr={m:by_model[m][row['id']] for m in old.FOLDERS}
        panels.append(comparison_panel(fig,row,plan[row['id']],rr,base,letter))
    legend=[Line2D([0],[0],color=INK,lw=4,label='GT fall'),
            Line2D([0],[0],color=BLUE,marker='|',ls='none',markersize=7,label='Fall-positive output'),
            Line2D([0],[0],color=ORANGE,marker='^',mfc='white',ls='none',markersize=4,label='Alarm onset')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.002),ncol=3,frameon=False,handlelength=1.2,columnspacing=1.2)
    dataset=old.SCOPES[scope]
    caption=(f"Qualitative comparison on two {dataset} videos shared across all five models: "
             f"(a) {short_id(owncases[0]['id'])}, a video-level true positive for Ours, and "
             f"(b) {short_id(owncases[1]['id'])}, a video-level false negative for Ours. "
             "The examples were selected post hoc, conditional on the outcome of Ours, from the previously fixed case list; "
             "they are illustrative and not representative performance estimates. Identical video frames and ground truth "
             "are used for all models. The GT bar marks the annotated fall interval. Vertical ticks mark fall-positive "
             "outputs at their stored timestamps: window endpoints for Ours and USDRL+NTU60, clip endpoints for HFD, and "
             "valid-pose frame times for FLASH. Tick densities differ because temporal sampling differs and must not be "
             "interpreted as comparable confidence. Open triangles mark alarm onsets. Empty processed rows indicate no "
             "fall-positive outputs, not missing data. X3D-UDA produces a video-level decision only; no temporal output is "
             "imputed. The Video column reports video-level TP/FN, whereas E-FP reports unmatched event alarms under the "
             "fixed matching interval from GT onset minus 0.5 s to GT end plus 3 s. A video TP does not guarantee correct "
             "event timing. *HFD and FLASH use project-retrained classifiers/models; USDRL+NTU60 uses the official backbone "
             "with a project-trained NTU60 head. All results are from fixed offline evaluations, without new target tuning.")
    if scope=='cauca100':
        caption+=' In panel (a), HFD is a video TP but misses the annotated event under the timing rule (event TP=0, FP=1, FN=1).'
    else:
        caption+=' In panel (a), FLASH has one matched event and one extra event alarm despite being a video TP.'
    stem='fig2_le2i_comparison' if scope=='le2i130' else 'fig3_cauca_comparison'
    save(fig,stem,caption,'main',book,dict(dataset=dataset,panels=panels))


def score_panel(fig,row,item):
    m=row['model'];s=load_series(row,item)
    ax=axis(fig,.56,.66,2.78,1.17)
    if not s['processed']:
        ax.axis('off');ax.text(.5,.6,'No pose in any frame\nClassifier not executed',ha='center',va='center',transform=ax.transAxes,fontsize=10)
        ax.text(.5,.19,f'0 / {len(s["source_times"])} detected frames',ha='center',va='center',transform=ax.transAxes,fontsize=9)
    elif m=='privacy_x3d_uda_rgb':
        p=s['probabilities'];ax.barh([0,1],p,color=[GRAY,BLUE],height=.42)
        ax.set_yticks([0,1],['Non-fall','Fall'],fontsize=8.5);ax.tick_params(axis='y',length=0,pad=2)
        ax.set_xlim(0,1);ax.set_xlabel('Video softmax score',labelpad=2)
        for j,v in enumerate(p):ax.text(.98,j,f'{v:.3f}',ha='right',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=0))
    else:
        for ep in row['episodes']:ax.axvspan(ep['fall_start'],ep['fall_end'],facecolor='#eeeeee',edgecolor='#bbbbbb',lw=.5,hatch='///',alpha=.5)
        t,y=s['times'],s['score'];ax.plot(t,y,color=BLUE,lw=.9)
        if 'competitor' in s:
            ax.plot(t,s['competitor'],color=GRAY,ls='--',lw=.8);ax.set_ylim(-.03,1.03);ax.set_yticks([0,.5,1]);ax.set_ylabel('Score',labelpad=2)
        else:
            ax.axhline(0,color=GRAY,lw=.7,ls='--');ax.set_ylabel('Margin' if m=='hfd_reproduction' else 'Logit',labelpad=2)
        ax.scatter(t[s['positive']],y[s['positive']],s=7,c=BLUE,zorder=3)
        for onset in s['alarms']:ax.axvline(onset,color=ORANGE,lw=.6,alpha=.8)
        ax.set_xlim(0,s['source_times'][-1]);ax.set_xlabel('Time (s)',labelpad=2)
        ax.grid(axis='y',lw=.4,alpha=.25)
        for a in ax.get_xticklabels()+ax.get_yticklabels():a.set_fontsize(8.5)
    return s


def supplementary(row,item,number,book):
    times,indices,images=verified_stills(row,item)
    fig=plt.figure(figsize=(3.5,3.45),dpi=160)
    title=f'{old.SCOPES[row["scope"]]} · {SHORT[row["model"]]} · {class_code(row)}'
    text(fig,.07,3.37,title,10,fontweight='bold')
    text(fig,.07,3.15,short_id(row['id']),8.5)
    for j,idx in enumerate(indices):
        x=.035+j*1.16
        image_axis(fig,x,2.35,1.11,.68,images[idx])
        text(fig,x+.555,2.30,f'{times[idx]:.2f} s',8.5,ha='center')
    label={'own':'Window scores','usdrl_ntu60':'Window scores','hfd_reproduction':'Clip decision margins',
           'privacy_x3d_uda_rgb':'Video classification only','flash':'Frame logits and pose gating'}[row['model']]
    if not row['processed']:label='Input failure (FN)'
    text(fig,.07,2.05,label,9)
    s=score_panel(fig,row,item)
    if row['episodes']:
        e=row['episodes'][0];truth=f'GT fall: {e["fall_start"]:.2f}–{e["fall_end"]:.2f} s'
    else:truth='GT: non-fall video'
    text(fig,.07,.29,truth,8.5)
    event=row.get('event')
    if event:text(fig,3.40,.29,f'E: {event["tp"]}/{event["fp"]}/{event["fn"]}',8.5,ha='right')
    note='Video TP/FN/FP; event counts = TP/FP/FN.'
    if row['model']=='privacy_x3d_uda_rgb':note='No temporal predictions are available.'
    elif not row['processed']:note='No classifier score was produced.'
    text(fig,.07,.145,note,8.5)
    cap=(f"{old.SCOPES[row['scope']]} qualitative example for {NAMES[row['model']]} "
         f"({short_id(row['id'])}; video-level {class_code(row)}). "
         "The three frames are taken from the actual evaluated video. "
         "This example belongs to the previously fixed, model-specific post-hoc success/failure selection and is not a "
         "representative sample or a paired comparison with the other supplementary examples. ")
    if row['model']=='privacy_x3d_uda_rgb':
        cap+='Bars show the softmax of the five-clip averaged logits. No temporal confidence or alarm time is available. '
    elif not row['processed']:
        cap+=f"The pose extractor failed on all {len(times)} frames. The classifier was not executed; the empty alarm list is counted as a false negative. "
    else:
        cap+='The hatched interval denotes ground-truth fall; orange vertical lines denote alarm onsets. '
        if 'competitor' in s:
            cap+='The blue curve is the fall softmax score, and the dashed gray curve is the maximum non-fall score. Decisions use multiclass argmax, not a fixed 0.5 probability threshold. '
        else:
            cap+='The blue curve shows the SVM margin or FLASH logit, not a calibrated probability; the dashed line is the fixed zero boundary. '
        cap+='Blue points mark fall-positive outputs; connecting segments are display aids, not additional predictions. '
        if row['model']=='flash':cap+='Missing-pose frames are gated off even if their logits are positive. '
        cap+='E reports event TP/FP/FN under the fixed matching rule; a video TP may include extra event alarms. '
    if row['model']=='flash' and row['scope']=='le2i130' and row['case']=='FP':
        cap+='This non-fall false positive replaces a fall-video false negative because none exists for FLASH in this Le2i evaluation. '
    if row['model'] in ('hfd_reproduction','flash'):cap+='*This is the project-retrained configuration, not the author\'s final classifier weights. '
    if row['model']=='usdrl_ntu60':cap+='The official USDRL backbone is paired with a project-trained NTU60 classifier; A043 is the fall class. '
    cap+='Scores and timing come from stored offline evaluation outputs, without retraining or retuning.'
    stem=f'supp_{number:02d}_{row["scope"]}_{row["model"]}_{class_code(row)}'
    save(fig,stem,cap,'supplementary',book,dict(id=row['id'],model=NAMES[row['model']],model_key=row['model'],
        dataset=old.SCOPES[row['scope']],case=class_code(row),processed=row['processed'],
        frames=indices,frame_times=[float(times[i]) for i in indices],gt=row['episodes'],
        video_prediction=row['video_prediction'],event_counts={k:event[k] for k in ('tp','fp','fn')} if event else None))


def package():
    (OUT/'figure_metadata.json').write_text(json.dumps(CATALOG,indent=2,ensure_ascii=False)+'\n')
    caption_text='\n\n'.join(r['stem']+'\n'+r['caption'] for r in CATALOG)
    (OUT/'captions.txt').write_text(caption_text+'\n')
    latex=[r'\usepackage{graphicx}',r'% Use main PDFs at two-column width and supplementary PDFs at column width.',
           r'% Copy this file and the PDFs into the same project folder; adjust paths if required.']
    for r in CATALOG:
        env='figure*' if r['kind']=='main' else 'figure';width=r'\textwidth' if env=='figure*' else r'\columnwidth'
        cap=r['caption'].replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
        latex += [f'\\begin{{{env}}}[t]',r'\centering',f'\\includegraphics[width={width}]{{{r["stem"]}.pdf}}',
                  f'\\caption{{{cap}}}',f'\\label{{fig:{r["stem"]}}}',f'\\end{{{env}}}','']
    (OUT/'latex_figures.tex').write_text('\n'.join(latex)+'\n')
    intro='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper figures for fall recognition</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#222}
figure{margin:40px 0}img{max-width:100%;height:auto;border:1px solid #ddd}figcaption{margin-top:12px}a{color:#0072b2}h2{font-size:20px}</style>
<h1>Paper figures for fall recognition</h1><p>Three main figures and twenty supplementary cases. Main figures are designed for a 182 mm two-column width;
supplementary figures use an 88.9 mm single-column width. All labels are at least 8.5 pt at those sizes.
PDF/SVG keep text and lines as vector objects; PNG exports use 600 dpi. Source RGB images retain their original information content.</p>
<p>Examples are selected post hoc. Main comparisons reuse the fixed Ours TP/FN videos for all models; supplementary examples are selected separately by model.
Model weights, thresholds, video lists and numerical results have not changed. Identifiable RGB images remain subject to dataset terms and publication permissions;
these materials have not been uploaded or approved by a publisher.</p>
<p><a href="paper_figures.pdf">All figures PDF</a> · <a href="captions.txt">English captions</a> · <a href="latex_figures.tex">LaTeX snippets</a> ·
<a href="figure_metadata.json">Case metadata</a></p>'''
    cards=[]
    for r in CATALOG:
        s=r['stem'];caption=html.escape(r['caption'])
        cards.append(f'<figure><h2>{html.escape(s)}</h2><a href="{s}.pdf"><img src="{s}.png" alt="{caption}" loading="lazy"></a>'
                     f'<figcaption>{caption}<br><a href="{s}.pdf">PDF</a> · <a href="{s}.svg">SVG</a> · <a href="{s}.png">PNG</a></figcaption></figure>')
    (OUT/'index.html').write_text(intro+''.join(cards)+'</html>')
    artifacts={p.name:old.sha(p) for p in OUT.iterdir() if p.suffix in ('.png','.pdf','.svg','.json','.txt','.tex','.html')}
    with zipfile.ZipFile(OUT/'paper_demo_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(artifacts):z.write(OUT/name,arcname=name)
    return artifacts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--main-only',action='store_true');args=ap.parse_args()
    old.cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True);EVIDENCE.mkdir(parents=True,exist_ok=True)
    old.remember(Path(__file__));old.remember(Path(old.__file__))
    results,plan=old.load_evaluations();cases=old.select_cases(results)
    frozen=old.read(ROOT/'docs/internal/2026-10-04_fall_demo_cases_evidence/selection.json')
    assert frozen['cases']==cases
    selected=[c for c in cases if c['model']=='own'];assert len(selected)==4
    contract=dict(main_comparison_cases=[c['id'] for c in selected],supplementary_cases=[{k:c[k] for k in ('id','model','case')} for c in cases],
                  main_selection='Reuse frozen Ours TP/FN cases; conditional on Ours and post hoc. Same videos for all five models.',
                  no_new_training=True,no_new_inference=True,no_threshold_change=True)
    cp=EVIDENCE/'selection.json'
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:cp.write_text(json.dumps(contract,indent=2)+'\n')
    by_model={m:{r['id']:r for r in rows} for m,rows in results.items()}
    xm={r['id']:r for r in old.read(old.BASE/old.FOLDERS['privacy_x3d_uda_rgb']/'input_manifest.json')}
    with PdfPages(OUT/'paper_figures.pdf',metadata={'Title':'Evidence-based fall recognition figures','Author':'','Creator':'Matplotlib'}) as book:
        pipeline(selected[0],plan[selected[0]['id']],xm,book)
        for scope in old.SCOPES:
            comparison(scope,[c for c in selected if c['scope']==scope],by_model,plan,book)
        if not args.main_only:
            for number,row in enumerate(cases,1):supplementary(row,plan[row['id']],number,book)
    if args.main_only:
        print('Main figure preview complete; full package not built.',flush=True);return
    assert len(CATALOG)==23
    artifacts=package()
    for p,h in old.SOURCES.items():assert old.sha(ROOT/p)==h,'Source changed: '+p
    report=dict(passed=True,main_figures=3,supplementary_figures=20,unchanged_experiment_inputs=True,
                no_training=True,no_new_inference=True,no_gpu=True,selected_cases=contract,
                render_checks=RENDER_CHECKS,source_hashes=old.SOURCES,artifact_hashes=artifacts,
                release_checks_pending=['Target venue final formatting','Dataset image reproduction rights and identifiable-image publication requirements'])
    (EVIDENCE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE: 3 main figures + 20 supplementary cases; PDF/SVG/600-dpi PNG; vector labels verified.',flush=True)


if __name__=='__main__':main()
