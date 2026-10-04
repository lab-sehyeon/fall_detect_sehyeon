"""Compose 20 qualitative cases from audited predictions; never run a model.

CPU only. Actual pixels, pose arrays and scores are read without modification.
PNG/PDF/HTML are presentation artifacts, not a new performance evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import textwrap
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fall_demo_mpl_20261004')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/fall_processed/RGB'
OUT = ROOT / 'docs/images/fall_demo_20261004'
EVIDENCE = ROOT / 'docs/internal/2026-10-04_fall_demo_cases_evidence'
FOLDERS = {
    'own': 'flash_external_20261004_r1',
    'usdrl_ntu60': 'usdrl_external_20261004_r1',
    'hfd_reproduction': 'hfd_reproduction_20261004_r1',
    'privacy_x3d_uda_rgb': 'privacy_x3d_external_20261004_r1',
    'flash': 'flash_external_20261004_r1',
}
NAMES = {'own': 'ourmodel', 'usdrl_ntu60': 'USDRL + NTU60 head',
         'hfd_reproduction': 'HFD', 'privacy_x3d_uda_rgb': 'Privacy X3D-UDA', 'flash': 'FLASH'}
QUALIFIERS = {
    'own': 'Project model | frozen skeleton backbone + residual adapter + 4-class classifier',
    'usdrl_ntu60': 'Official DSTE backbone + project-trained NTU60 head | A043 = falling | 60-class baseline',
    'hfd_reproduction': 'Official C3D feature extractor + project-retrained linear SVM | not author final SVM',
    'privacy_x3d_uda_rgb': 'Author UDA checkpoint | RGB input | five-clip video classification, no temporal localization',
    'flash': 'Official HyperMamba architecture, project-retrained weights | adapted MediaPipe RGB frontend',
}
SCOPES = {'le2i130': 'Le2i', 'cauca100': 'CAUCAFall'}
BLUE, INK, GRAY, GREEN, RED, ORANGE = '#1566b5', '#132b46', '#56677a', '#07836e', '#ba3e46', '#d98318'
COCO_EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
              (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
MP_EDGES = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
            (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
            (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
            (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
            (24,26),(26,28),(28,30),(30,32),(28,32)]
NTU_EDGES = [(0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(6,7),(7,21),(6,22),
             (20,8),(8,9),(9,10),(10,11),(11,23),(10,24),(0,12),(12,13),(13,14),
             (14,15),(0,16),(16,17),(17,18),(18,19)]
SOURCES: dict[str,str] = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def remember(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None:
        assert digest == expected, f'Changed evidence: {path}'
    key = str(path.relative_to(ROOT))
    if key in SOURCES:
        assert SOURCES[key] == digest
    SOURCES[key] = digest
    return path


def read(path):
    return json.loads(remember(path).read_text())


def arrays(path, receipt=None):
    if receipt:
        remember(path, read(receipt)['payload_sha256'])
    else:
        remember(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    y = np.exp(x - x.max(axis=-1, keepdims=True))
    return y / y.sum(axis=-1, keepdims=True)


def edges(flags, times):
    flags = np.asarray(flags, bool)
    return np.asarray(times)[flags & ~np.r_[False, flags[:-1]]]


def load_evaluations():
    ground = {r['id']: r for r in read(BASE / FOLDERS['hfd_reproduction'] / 'ground_truth.json')}
    plan = {r['id']: r for r in read(BASE / FOLDERS['hfd_reproduction'] / 'plan.json')}
    results = {}
    for m, folder in FOLDERS.items():
        d = BASE / folder
        audit, evaluation = read(d/'independent_audit.json'), read(d/'evaluation.json')
        assert audit['passed']
        if 'evaluation_sha256' in audit:
            remember(d/'evaluation.json', audit['evaluation_sha256'])
        rows = [dict(r, episodes=ground[r['id']]['episodes']) for r in evaluation['rows'] if r['model'] == m]
        assert len(rows) == len({r['id'] for r in rows}) == 230
        results[m] = rows
        if m == 'hfd_reproduction':
            # HFD's historical audit has no evaluation hash; rederive all binary counts.
            for scope in SCOPES:
                rr = [r for r in rows if r['scope'] == scope]
                a = np.array([bool(r['episodes']) for r in rr]); b = np.array([bool(r['video_prediction']) for r in rr])
                actual = dict(tp=int((a&b).sum()), fp=int((~a&b).sum()), fn=int((a&~b).sum()), tn=int((~a&~b).sum()))
                assert all(audit['summaries'][m][scope]['video'][k] == v for k,v in actual.items())
    # Own rows copied into the FLASH comparison must still equal the original evaluations.
    for scope in SCOPES:
        d = BASE / f'{scope}_20261003_r1'
        ev, audit = read(d/'evaluation.json'), read(d/'independent_audit.json')
        assert audit['passed']; remember(d/'evaluation.json', audit['evaluation_sha256'])
        original = {r['id']: r for r in ev['rows'] if r['scope'] == scope and r.get('model','own') == 'own'}
        for r in results['own']:
            if r['scope'] != scope:
                continue
            for k in ('episodes','video_prediction','predictions','event'):
                assert r[k] == original[r['id']][k], (r['id'], k)
    return results, plan


def select_cases(results):
    """Post-hoc deterministic selection, independent for each model/dataset.

    Success: event-matched TP (where timing exists), minimum extra alarms then ID.
    Failure: FN first, processed before rejected, then ID; FLASH Le2i uses FP.
    No confidence, visual appearance or target metric is optimized.
    """
    cases = []
    for m, allrows in results.items():
        for scope in SCOPES:
            rr = [r for r in allrows if r['scope'] == scope]
            tp = [r for r in rr if r['episodes'] and r['video_prediction']]
            eligible = [r for r in tp if m == 'privacy_x3d_uda_rgb' or r['event']['tp'] >= 1]
            success = min(eligible, key=lambda r: (r.get('event',{}).get('fp',0), r['id']))
            fn = [r for r in rr if r['episodes'] and not r['video_prediction']]
            fp = [r for r in rr if not r['episodes'] and r['video_prediction']]
            if not fn:
                assert (m, scope) == ('flash','le2i130')
            failure = min(fn or fp, key=lambda r: (not r['processed'], r['id']))
            for kind, r in [('success',success), ('failure',failure)]:
                label = 'TP' if kind == 'success' else ('FN' if r['episodes'] else 'FP')
                cases.append(dict(r, kind=kind, case=label, pool=dict(videos=len(rr), video_tp=len(tp),
                             eligible_success=len(eligible), video_fn=len(fn), processed_fn=sum(x['processed'] for x in fn), video_fp=len(fp))))
    assert len(cases) == 20
    return cases


def frame_targets(episodes, duration, alarms):
    if episodes:
        a,b = episodes[0]['fall_start'], episodes[0]['fall_end']
        return np.clip([a-.6, (a+b)/2, b, max(b+.8, alarms[0] if alarms else 0)], 0, duration)
    return np.linspace(.12*duration,.88*duration,4)


def decoded_frames(item, indices):
    """Sequential decoder; every displayed source frame is matched to its prior BGR hash."""
    remember(ROOT/item['video'], item['video_sha256'])
    trace = read(ROOT/item['trace']); remember(ROOT/item['trace'], item['trace_sha256'])
    indices = set(map(int, indices)); assert indices and min(indices) >= 0 and max(indices) < item['source_frames']
    cap = cv2.VideoCapture(str(ROOT/item['video'])); assert cap.isOpened()
    found = {}
    try:
        for i in range(max(indices)+1):
            ok,bgr = cap.read(); assert ok, (item['id'],i)
            if i in indices:
                assert hashlib.sha256(bgr.tobytes()).hexdigest() == trace['bgr_sha256'][i], (item['id'],i)
                found[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return found


def prepare_case(case, item, xmanifest):
    m, ident = case['model'], case['id']
    trace = read(ROOT/item['trace'])
    source_times = (np.array(trace['pts'])-trace['start_pts'])*trace['time_base_num']/trace['time_base_den']
    assert len(source_times) == item['source_frames'] and np.all(np.diff(source_times)>0)
    d = BASE / FOLDERS[m] / ident
    info = dict(source_times=source_times, alarms=[p['time'] for p in case.get('predictions',[])])
    example_time = (case['episodes'][0]['fall_start']+case['episodes'][0]['fall_end'])/2 if case['episodes'] else source_times[-1]/2
    if m in ('own','usdrl_ntu60'):
        parent = BASE / f"{case['scope']}_20261003_r1" / ident
        front, lift = arrays(parent/'frontend.npz',parent/'frontend.json'), arrays(parent/'lift.npz',parent/'lift.json')
        infer = arrays((parent if m=='own' else d)/'inference.npz', (parent if m=='own' else d)/'inference.json')
        logits = infer['G0' if m=='own' else 'logits']; fall = 1 if m=='own' else 42
        p = softmax(logits); times = infer['window_endpoints']/25
        positive = logits.argmax(1) == fall
        if case['kind']=='success':
            focus = min(case['event']['matches'], key=lambda q:q['prediction']['time'])['prediction']['time']
            k = int(np.argmin(np.abs(times-focus)))
        else:
            k = int(p[:,fall].argmax())
        end = int(infer['window_endpoints'][k]); image_index = int(front['source_indices'][end])
        info.update(front=front,lift=lift,infer=infer,logits=logits,probs=p,times=times,positive=positive,
                    scores=p[:,fall],competitor=np.max(np.delete(p,fall,axis=1),axis=1),fall_class=fall,
                    selected=k,endpoint=end,image_index=image_index,example_time=float(times[k]))
    elif m == 'hfd_reproduction':
        pred = read(d/'prediction.json'); z = arrays(d/'features.npz',d/'prediction.json')
        assert pred['video_prediction'] == case['video_prediction'] and pred['predictions'] == case['predictions']
        np.testing.assert_array_equal(z['predictions'],z['decision']>0)
        assert np.isfinite(z['features']).all() and z['features'].shape[1]==4096
        np.testing.assert_allclose(z['times'],source_times[z['end_frames']],atol=1e-12,rtol=0)
        if case['kind']=='success':
            k = int(np.argmin(np.abs(z['times']-case['event']['matches'][0]['prediction']['time'])))
        else:
            k = int(z['decision'].argmax())
        info.update(features=z,times=z['times'],scores=z['decision'],positive=z['predictions'].astype(bool),
                    selected=k,image_index=int(z['end_frames'][k]),example_time=float(z['times'][k]),tail=pred['tail'])
    elif m == 'privacy_x3d_uda_rgb':
        a=xmanifest[ident]; sample=np.array(a['sample_indices_1based']).reshape(5,16)-1
        assert a['input_shape']==[5,3,16,256,256]
        assert np.all((sample>=0)&(sample<len(source_times)))
        p=softmax(case['logits']); assert abs(p[1]-case['fall_probability'])<1e-6
        assert int(case['logits'][1]>case['logits'][0])==case['video_prediction']
        info.update(sample=sample,probs=p,manifest=a,image_index=int(sample[2,8]),example_time=float(source_times[sample[2,8]]))
    elif case['processed']:
        pose=arrays(d/'pose.npz',d/'pose.json'); pred=arrays(d/'prediction.npz',d/'prediction.json')
        np.testing.assert_array_equal(pred['alarms'],(pred['logits']>0)&pred['detected'])
        np.testing.assert_array_equal(pose['detected'],pred['detected'])
        np.testing.assert_allclose(pred['timestamps'],source_times,atol=1e-12,rtol=0)
        focus = case['event']['matches'][0]['prediction']['time'] if case['kind']=='success' else info['alarms'][0]
        k=int(np.argmin(np.abs(source_times-focus)))
        info.update(pose=pose,pred=pred,times=pred['timestamps'],scores=pred['logits'],positive=pred['alarms'],
                    selected=k,image_index=k,example_time=float(source_times[k]))
    else:
        pose, pred = read(d/'pose.json'), read(d/'prediction.json')
        assert 'no pose in entire video' in pose['error'] and not pred['processed']
        assert pose['frames']==len(source_times) and not (d/'prediction.npz').exists()
        info.update(failure_receipt=pose,image_index=int(np.argmin(abs(source_times-example_time))),example_time=example_time)
    if 'positive' in info:
        assert int(info['positive'].any())==case['video_prediction']
        np.testing.assert_allclose(edges(info['positive'],info['times']),info['alarms'],rtol=0,atol=1e-9)
    wanted=frame_targets(case['episodes'],source_times[-1],info['alarms'])
    frame_indices=[int(np.argmin(abs(source_times-t))) for t in wanted]
    needed=set(frame_indices+[info['image_index']])
    if 'sample' in info:
        needed.update(info['sample'][:,8].tolist())
    images=decoded_frames(item,needed)
    if 'manifest' in info:
        pngs={r['source_index']:r for r in info['manifest']['pngs']}
        for i in info['sample'][:,8]:
            row=pngs[int(i)]; p=remember(ROOT/row['path'],row['sha256'])
            with Image.open(p) as im:
                np.testing.assert_array_equal(np.array(im.convert('RGB')),images[int(i)])
    info.update(images=images,frame_indices=frame_indices)
    return info


plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,
                     'axes.labelcolor':GRAY,'xtick.color':GRAY,'ytick.color':GRAY,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#d6dfe8',
                     'savefig.facecolor':'#f6f8fb'})


def label(fig,x,y,s,size=11,color=INK,weight='normal',maxw=None,**kw):
    artist=fig.text(x,y,s,fontsize=size,color=color,fontweight=weight,va='top',**kw)
    if maxw is not None:
        width=artist.get_window_extent(fig.canvas.get_renderer()).width
        if width>maxw*fig.bbox.width:
            artist.set_fontsize(size*maxw*fig.bbox.width/width)
    return artist


def box(fig,x,y,w,h,color='white',edge='#e0e7ee'):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.009',
                    transform=fig.transFigure,facecolor=color,edgecolor=edge,linewidth=.8,zorder=-1)
    fig.patches.append(p)


def panel_axes(fig,x,y,w,h,projection=None):
    ax=fig.add_axes([x,y,w,h],projection=projection)
    ax.set_facecolor('white')
    return ax


def photo(fig,rect,im):
    ax=panel_axes(fig,*rect);ax.imshow(im);ax.axis('off');return ax


def panel_title(fig,i,title,subtitle):
    x=.032+i*.24
    box(fig,x,.336,.225,.222)
    label(fig,x+.01,.549,title,12,weight='bold',maxw=.205)
    label(fig,x+.01,.521,subtitle,8.6,GRAY,maxw=.205)
    if i<3: label(fig,x+.229,.435,'›',20,GRAY,weight='bold')
    return x


def pose_overlay(ax,points,scores,edges_,box_=None):
    # Confidence filtering is display only; no saved points or model inputs are changed.
    valid=np.isfinite(points).all(1)&(scores>=.2)
    for a,b in edges_:
        if valid[a] and valid[b]:
            ax.plot(points[[a,b],0],points[[a,b],1],color='#00e0f0',lw=2,solid_capstyle='round')
    ax.scatter(points[valid,0],points[valid,1],s=9,c='#ffe766',edgecolors=INK,linewidths=.2,zorder=4)
    if box_ is not None and box_[4]>0:
        x,y,r,b=box_[:4];ax.plot([x,r,r,x,x],[y,y,b,b,y],color='#72ff7d',lw=1)


def scalar_bars(ax,names,vals,colors,xlim=None):
    ax.barh(np.arange(len(names)),vals,color=colors,height=.52)
    ax.set_yticks(np.arange(len(names)),names,fontsize=8)
    ax.invert_yaxis();ax.tick_params(axis='y',length=0)
    if xlim: ax.set_xlim(*xlim)
    for i,v in enumerate(vals):
        ax.text(.99,i,f'{v:.3f}',ha='right',va='center',fontsize=8,transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor='white',alpha=.78,edgecolor='none',pad=.5))
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)


def render_process(fig,case,d):
    m=case['model'];im=d['images'][d['image_index']];t=d['example_time']
    if m in ('own','usdrl_ntu60'):
        end=d['endpoint'];k=d['selected']
        x=panel_title(fig,0,'Detection + 2D pose',f'YOLOv8x / ViTPose | window end {t:.2f} s')
        ax=photo(fig,(x+.009,.357,.206,.15),im)
        pose_overlay(ax,d['front']['xy'][end],d['front']['scores'][end],COCO_EDGES,d['front']['boxes'][end])
        label(fig,x+.01,.352,'Stored keypoints; display confidence ≥ 0.2',7.6,GRAY)
        x=panel_title(fig,1,'3D lift + NTU25 formatting','MotionAGFormer → proxy NTU25 → 64 frames')
        ax=panel_axes(fig,x+.012,.35,.201,.161,projection='3d')
        q=d['lift']['ntu25'][end]
        for a,b in NTU_EDGES: ax.plot(q[[a,b],0],q[[a,b],1],q[[a,b],2],color=BLUE,lw=1.5)
        ax.scatter(*q.T,s=9,c=ORANGE,depthshade=False)
        ax.view_init(elev=18,azim=65);ax.set_box_aspect((1,1,1));ax.set_axis_off()
        center=(q.max(0)+q.min(0))/2; span=max(np.ptp(q,axis=0).max(),.01)*.58
        ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span)
        label(fig,x+.01,.352,'Stored normalized skeleton; not metric world pose',7.6,GRAY)
        x=panel_title(fig,2,'ourmodel representation' if m=='own' else 'DSTE representation','2048-d vector → 64 display bins (mean per 32)')
        vectors=[d['infer']['pooled'][k]]
        names=['Backbone' if m=='own' else 'DSTE']
        if m=='own': vectors.append(d['infer']['adapted'][k]);names.append('Adapted')
        v=np.stack(vectors).reshape(len(vectors),64,32).mean(-1)
        ax=panel_axes(fig,x+.043,.405,.161,.077);ax.imshow(v,aspect='auto',cmap='viridis',interpolation='nearest')
        ax.set_yticks(range(len(names)),names,fontsize=7);ax.set_xticks([0,31,63],['1','32','64'],fontsize=7)
        label(fig,x+.01,.378,'Feature summary only — not saliency or attention.',8,GRAY)
        label(fig,x+.01,.355,'64-frame windows at 25 Hz; stride 8 frames.',8,GRAY)
        x=panel_title(fig,3,'ourmodel classifier' if m=='own' else 'NTU60 head','Softmax scores for the displayed window')
        p=d['probs'][k]
        if m=='own': ix=list(range(4)); names=['other','fall','lie_down','lying_down']
        else:
            ix=list(np.argsort(-p)[:3]);ix=list(dict.fromkeys(ix+[42]));names=[f'A{i+1:03d}'+(' fall' if i==42 else '') for i in ix]
        ax=panel_axes(fig,x+.065,.377,.145,.121)
        scalar_bars(ax,names,p[ix],[RED if i==d['fall_class'] else BLUE for i in ix],(0,1))
        label(fig,x+.01,.352,'Decision uses argmax, not a 0.5 fall threshold.',7.8,GRAY)
    elif m=='hfd_reproduction':
        k=d['selected'];end=int(d['features']['end_frames'][k]);start=end-15
        x=panel_title(fig,0,'16-frame RGB clip',f'Native frames {start}–{end} | ending {t:.2f} s')
        resized=cv2.resize(im,(112,112),interpolation=cv2.INTER_LINEAR)
        photo(fig,(x+.048,.365,.128,.14),resized)
        label(fig,x+.01,.352,'112 × 112 display; model channels are BGR, 0–255.',7.5,GRAY)
        x=panel_title(fig,1,'C3D fc6 features','4096-d vector → 64 display bins (mean per 64)')
        v=d['features']['features'][k].reshape(64,64).mean(1).reshape(8,8)
        ax=panel_axes(fig,x+.055,.371,.12,.131);ax.imshow(v,cmap='viridis',interpolation='nearest');ax.axis('off')
        label(fig,x+.01,.352,'Stored feature summary, not an attention map.',7.8,GRAY)
        x=panel_title(fig,2,'Linear SVM',f'Stored decision margin = {d["scores"][k]:+.3f}')
        ax=panel_axes(fig,x+.039,.392,.166,.088);v=float(d['scores'][k])
        ax.barh([0],[v],color=RED if v>0 else BLUE);ax.axvline(0,color=INK,lw=1)
        ax.set_xlim(-max(1,abs(v)*1.3),max(1,abs(v)*1.3));ax.set_yticks([])
        ax.set_xlabel('Margin (not a probability)',fontsize=8)
        label(fig,x+.01,.355,'Positive margin → fall clip.',8,GRAY)
        x=panel_title(fig,3,'Video decision','A video is positive if any clip is classified as fall')
        label(fig,x+.02,.48,'FALL' if case['video_prediction'] else 'NO FALL ALERT',20,RED if case['video_prediction'] else BLUE,weight='bold')
        label(fig,x+.02,.424,f'{int(d["positive"].sum())} / {len(d["positive"])} positive clips',12)
        label(fig,x+.02,.381,f'Non-overlapping clips; {d["tail"]} tail frames discarded.',8.3,GRAY)
    elif m=='privacy_x3d_uda_rgb':
        x=panel_title(fig,0,'Five sampled RGB clips','Middle sampled frame from each actual input clip')
        for j,idx in enumerate(d['sample'][:,8]):
            xx=x+.009+(j%3)*.071; yy=.44 if j<3 else .356
            photo(fig,(xx,yy,.067,.06),d['images'][int(idx)])
            label(fig,xx,yy-.001,f'C{j+1}: {d["source_times"][idx]:.2f}s',6.5,GRAY)
        x=panel_title(fig,1,'X3D RGB input','5 clips × 16 frames × 256 × 256')
        photo(fig,(x+.064,.38,.105,.112),cv2.resize(im,(256,256)))
        label(fig,x+.01,.377,'Frame interval 5; RGB mean/std normalization.',8,GRAY)
        label(fig,x+.01,.355,'No pose estimator is used by this model.',8,GRAY)
        x=panel_title(fig,2,'Average clip logits','Author test rule: average_clips = score')
        ax=panel_axes(fig,x+.057,.39,.153,.103)
        scalar_bars(ax,['non-fall','fall'],case['logits'],[BLUE,RED]);ax.axvline(0,color=GRAY,lw=.6)
        label(fig,x+.01,.362,'Only the aggregate logits were retained.',8,GRAY)
        x=panel_title(fig,3,'Video-level classifier','Softmax applied after logit averaging')
        ax=panel_axes(fig,x+.057,.395,.153,.103)
        scalar_bars(ax,['non-fall','fall'],d['probs'],[BLUE,RED],(0,1))
        label(fig,x+.01,.367,'Fall if averaged fall logit > non-fall logit.',8,GRAY)
        label(fig,x+.01,.345,'No frame-level prediction or detection timestamp.',7.5,GRAY)
    elif case['processed']:
        k=d['selected'];p=d['pose']
        x=panel_title(fig,0,'MediaPipe pose',f'33 landmarks | source frame {k} | {t:.2f} s')
        ax=photo(fig,(x+.009,.359,.206,.145),im)
        xy=p['xyz'][k,:,:2]*np.array([im.shape[1],im.shape[0]])
        pose_overlay(ax,xy,p['visibility'][k],MP_EDGES)
        label(fig,x+.01,.352,'Actual saved pose; visibility ≥ 0.2 for display.',7.7,GRAY)
        x=panel_title(fig,1,'XYZ pose sequence','MediaPipe image-normalized coordinates (not metres)')
        ax=panel_axes(fig,x+.045,.395,.162,.097)
        ax.imshow(p['xyz'][k].T,aspect='auto',cmap='coolwarm',interpolation='nearest');ax.set_yticks([0,1,2],['x','y','z'],fontsize=8)
        ax.set_xticks([0,16,32],['0','16','32'],fontsize=8);ax.set_xlabel('Landmark index',fontsize=8)
        label(fig,x+.01,.355,'Then: source scaler + fixed graph + 100-frame blocks.',7.5,GRAY)
        x=panel_title(fig,2,'HyperMamba output','Stored per-frame logit; offline 100-frame context')
        label(fig,x+.02,.478,f'{d["scores"][k]:+.3f}',25,BLUE,weight='bold')
        label(fig,x+.02,.424,'Positive logit → candidate fall',11)
        label(fig,x+.02,.381,'Future frames may affect output inside each block.',8,GRAY)
        x=panel_title(fig,3,'Valid-pose gate + alarm','Positive logit AND detected pose; rising edges = alarms')
        label(fig,x+.02,.48,'POSE VALID' if p['detected'][k] else 'POSE MISSING',15,GREEN if p['detected'][k] else RED,weight='bold')
        label(fig,x+.02,.433,'FALL at this frame' if d['positive'][k] else 'No alert at this frame',14,RED if d['positive'][k] else BLUE)
        label(fig,x+.02,.38,f'{len(d["alarms"])} total video alarms; no refractory period.',8.3,GRAY)
    else:
        x=panel_title(fig,0,'MediaPipe pose extraction','The frame below is real; no landmarks were returned')
        photo(fig,(x+.009,.37,.206,.133),im)
        label(fig,x+.02,.365,'NO POSE DETECTED',11,RED,weight='bold')
        x=panel_title(fig,1,'No valid pose sequence','Recorded input failure across the complete video')
        label(fig,x+.02,.468,f'0 / {len(d["source_times"])} frames',22,RED,weight='bold')
        label(fig,x+.02,.41,'No pose tensor saved.',12)
        x=panel_title(fig,2,'Classifier not executed','No valid input; no prediction logits were produced')
        label(fig,x+.02,.468,'NOT RUN',24,GRAY,weight='bold')
        label(fig,x+.02,.404,'Do not interpret this as a negative model score.',8.3,GRAY)
        x=panel_title(fig,3,'Evaluation fallback','The video stays in the evaluation denominator')
        label(fig,x+.02,.478,'EMPTY ALARM LIST',18,RED,weight='bold')
        label(fig,x+.02,.425,'Fall GT + no alarm → FN',12)
        label(fig,x+.02,.381,'Input failure, not a valid-pose classifier mistake.',8.2,GRAY)


def timeline(fig,case,d):
    ax=panel_axes(fig,.073,.107,.535,.155)
    duration=d['source_times'][-1];gt=case['episodes'];m=case['model']
    for e in gt: ax.axvspan(e['fall_start'],e['fall_end'],color=GREEN,alpha=.14,label='GT fall interval')
    if m=='privacy_x3d_uda_rgb':
        for j,indices in enumerate(d['sample']):
            times=d['source_times'][indices]
            ax.scatter(times,np.full(16,5-j),marker='|',s=55,c=BLUE)
        ax.set_yticks(range(1,6),['Clip 5','Clip 4','Clip 3','Clip 2','Clip 1'],fontsize=8)
        ax.set_ylim(.5,5.7);ax.set_title('Sampling positions only — no temporal scores or alarms',loc='left',fontsize=10,pad=8)
    elif not case['processed']:
        ax.text(.5,.52,'No classifier trace exists for this video.\nThe cached receipt reports no pose in any frame.',ha='center',va='center',transform=ax.transAxes,fontsize=12,color=GRAY)
        ax.set_yticks([]);ax.set_ylim(0,1);ax.set_title('Input failure retained as an evaluation miss',loc='left',fontsize=10,pad=8)
    else:
        x,y=d['times'],d['scores']
        title={'own':'Window-end softmax scores (ourmodel)','usdrl_ntu60':'Window-end softmax scores (60 classes)',
               'hfd_reproduction':'SVM margin at the end of each 16-frame clip','flash':'Per-frame logit with valid-pose gating'}[m]
        ax.set_title(title,loc='left',fontsize=10,pad=8)
        ax.plot(x,y,color=BLUE,lw=1.35,label='Fall score' if m in ('own','usdrl_ntu60') else 'Stored margin / logit')
        if m in ('own','usdrl_ntu60'):
            ax.plot(x,d['competitor'],color=GRAY,lw=.8,alpha=.65,label='Largest non-fall score')
            ax.set_ylim(-.04,1.06);ax.set_ylabel('Softmax score',fontsize=9)
        else:
            ax.axhline(0,color=GRAY,ls='--',lw=1,label='Decision boundary = 0')
            ax.set_ylabel('Margin' if m=='hfd_reproduction' else 'Logit',fontsize=9)
        ax.scatter(x[d['positive']],y[d['positive']],s=12,c=ORANGE,zorder=4,label='Fall-positive output')
        if m=='flash':
            missing=~d['pred']['detected']
            if missing.any(): ax.scatter(x[missing],y[missing],s=12,c=GRAY,marker='x',label='Pose missing (gated off)')
        for j,t in enumerate(d['alarms']): ax.axvline(t,color=RED,alpha=.5,lw=.8,label='Alarm onset' if j==0 else None)
        k=d['selected'];ax.scatter([x[k]],[y[k]],s=60,facecolors='none',edgecolors=INK,linewidths=1.2,zorder=5)
    ax.set_xlim(0,duration);ax.set_xlabel('Source video time (seconds)',fontsize=9)
    ax.grid(axis='y',alpha=.15);ax.tick_params(labelsize=8)
    handles,labels=ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,fontsize=6.7,frameon=False)


def summary_lines(case,d):
    m=case['model'];event=case.get('event');lines=[]
    if case['episodes']:
        e=case['episodes'][0];lines.append(f'GT fall interval: {e["fall_start"]:.2f}–{e["fall_end"]:.2f} s')
    else: lines.append('Ground truth: no fall anywhere in this video')
    if m=='privacy_x3d_uda_rgb':
        lines += [f'Video fall score: {d["probs"][1]:.4f}',
                  'One decision for the full video; timing unavailable.']
    elif not case['processed']:
        lines += [f'No pose detected in all {len(d["source_times"])} frames.',
                  'Classifier skipped; no negative score was generated.']
    else:
        if m in ('own','usdrl_ntu60'):
            k=d['selected'];top=int(d['logits'][k].argmax())
            name=['other','fall','lie_down','lying_down'][top] if m=='own' else f'A{top+1:03d}'
            lines += [f'Positive windows: {int(d["positive"].sum())} / {len(d["positive"])}',
                      f'Shown window: {d["example_time"]:.2f} s, top class = {name}']
            if case['kind']=='failure': lines.append(f'Even peak fall score is {d["scores"].max():.4f}; no fall argmax.')
        elif m=='hfd_reproduction':
            lines += [f'Positive clips: {int(d["positive"].sum())} / {len(d["positive"])}',
                      f'Shown clip end: {d["example_time"]:.2f} s; margin {d["scores"][d["selected"]]:+.3f}']
        else:
            lines += [f'Positive valid frames: {int(d["positive"].sum())} / {len(d["positive"])}',
                      f'Total alarm onsets: {len(d["alarms"])}']
        lines.append(f'Event matching: TP {event["tp"]} / extra alarms {event["fp"]} / FN {event["fn"]}')
    if m=='flash' and case['scope']=='le2i130':
        lines.append('Video TP does not mean clean event detection.' if case['kind']=='success' else 'FP substitute: this model had no Le2i fall-video FN.')
    return lines


def render_case(case,d,number):
    fig=plt.figure(figsize=(16,10.5),dpi=200,facecolor='#f6f8fb')
    good=case['kind']=='success';color=GREEN if good else RED
    label(fig,.031,.967,f'{SCOPES[case["scope"]]}  /  {NAMES[case["model"]]}',25,weight='bold')
    box(fig,.742,.908,.223,.065,color='#e0f3ed' if good else '#fbe8e9',edge='none')
    badge='SUCCESS · VIDEO TP' if good else ('FAILURE · FALSE ALARM' if case['case']=='FP' else 'FAILURE · MISSED FALL')
    label(fig,.758,.951,badge,12,color,weight='bold')
    label(fig,.758,.923,'Input failure (no pose)' if not case['processed'] else 'Full-video classification',9,color)
    label(fig,.032,.917,QUALIFIERS[case['model']],10,GRAY)
    label(fig,.032,.89,f'CASE {number:02d}   {case["id"]}   |   Ground truth: '+('FALL' if case['episodes'] else 'NON-FALL')+'   →   Video output: '+('FALL' if case['video_prediction'] else 'NO FALL ALERT'),10,weight='bold')
    label(fig,.032,.855,'01   ACTUAL VIDEO FRAMES',11,BLUE,weight='bold')
    phases=['Before GT fall','Within GT fall','GT fall end','Later frame'] if case['episodes'] else ['Non-fall video']*4
    for j,idx in enumerate(d['frame_indices']):
        x=.032+j*.24;box(fig,x,.622,.225,.21)
        photo(fig,(x+.004,.65,.217,.177),d['images'][idx])
        label(fig,x+.008,.645,f'{phases[j]}  |  {d["source_times"][idx]:.2f} s  |  frame {idx}',8.4,GRAY)
    label(fig,.032,.597,'02   MODEL PATH AND STORED INTERMEDIATES',11,BLUE,weight='bold')
    render_process(fig,case,d)
    label(fig,.032,.310,'03   DECISION EVIDENCE',11,BLUE,weight='bold')
    box(fig,.032,.033,.595,.259)
    timeline(fig,case,d)
    box(fig,.651,.046,.314,.246,color='#edf6f2' if good else '#fdf0f0',edge='none')
    label(fig,.666,.28,'CASE INTERPRETATION',12,color,weight='bold')
    y=.246
    for s in summary_lines(case,d):
        lines=textwrap.wrap(s,width=57,break_long_words=False)
        for line in lines:
            label(fig,.666,y,line,9.3,INK); y-=.022
    if case['model']!='privacy_x3d_uda_rgb':
        label(fig,.666,.087,'Event match tolerance: GT start − 0.5 s to GT end + 3 s.\nTiming is offline output alignment, not online latency.',7.7,GRAY)
    else:
        label(fig,.666,.087,'No per-frame confidence or attention map is inferred.\nClip averaging can hide temporal differences.',7.7,GRAY)
    label(fig,.032,.020,'Actual pixels + cached outputs  •  Post-hoc illustrative cases, not representative samples  •  Frame indices start at 0  •  Scores are not calibrated risk estimates',7.8,GRAY)
    filename=f'{number:02d}_{case["scope"]}_{case["model"]}_{case["kind"]}_{case["case"]}.png'
    for artist in fig.findobj(matplotlib.text.Text):
        assert not any(token in artist.get_text() for token in ('J1','G0')), artist.get_text()
    fig.savefig(OUT/filename,dpi=200)
    plt.close(fig)
    return filename


def public_case(case,d,filename):
    return dict(model=NAMES[case['model']],model_key=case['model'],dataset=SCOPES[case['scope']],scope=case['scope'],
                id=case['id'],kind=case['kind'],case=case['case'],processed=case['processed'],
                ground_truth=int(bool(case['episodes'])),video_prediction=case['video_prediction'],
                gt_intervals=case['episodes'],alarm_seconds=d['alarms'] if case['model']!='privacy_x3d_uda_rgb' else None,
                event_counts={k:case['event'][k] for k in ('tp','fp','fn')} if 'event' in case else None,
                source_frame_indices=d['frame_indices'],source_frame_seconds=[float(d['source_times'][i]) for i in d['frame_indices']],
                selected_intermediate_time=d['example_time'],png=filename,
                interpretation=summary_lines(case,d),selection_pool=case['pool'])


def make_collection(rows):
    # A PDF for slide review, and a local HTML gallery that needs no external services.
    pages=[Image.open(OUT/r['png']).convert('RGB') for r in rows]
    pages[0].save(OUT/'fall_demo_20_cases.pdf',save_all=True,append_images=pages[1:],resolution=200.0)
    width,height=1600,1400
    sheet=Image.new('RGB',(width,height),'#f6f8fb');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',27)
    draw.text((20,14),'Fall recognition demo — 20 audited case illustrations',font=title,fill=INK)
    for i,(r,im) in enumerate(zip(rows,pages)):
        col,row=i%4,i//4;x=col*400+12;y=row*268+55
        im.thumbnail((376,247));sheet.paste(im,(x,y))
        draw.text((x,y+248),f'{i+1:02d}  {r["dataset"]} / {r["case"]}',font=font,fill=GREEN if r['kind']=='success' else RED)
    sheet.save(OUT/'overview.png')
    for p in pages:p.close()
    cards=[]
    for r in rows:
        esc=html.escape
        cards.append(f'<article><h2>{esc(r["dataset"])} · {esc(r["model"])} · {esc(r["case"])}</h2>'
                     f'<a href="{esc(r["png"])}"><img loading="lazy" src="{esc(r["png"])}" alt="{esc(r["id"])} {r["case"]}"></a>'
                     '<p>'+esc(' '.join(r['interpretation']))+'</p></article>')
    gallery='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Le2i and CAUCAFall model demo cases</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f6f8fb;color:#132b46;margin:2rem auto;max-width:1700px;padding:0 20px}
h1{font-size:2rem}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{background:white;padding:18px;border-radius:12px}
article h2{font-size:18px}img{width:100%;height:auto}a{color:#1566b5}p{max-width:1100px}@media(max-width:950px){main{grid-template-columns:1fr}}
</style><h1>Le2i and CAUCAFall model demo cases</h1>
<p>Five models, two datasets, one video-level success and one failure per pair. Click a figure for its full-resolution PNG.
Frames and intermediates come from saved evaluations; no images or predictions were generated by a model for this gallery.</p>
<p>Examples were selected after evaluation, independently for each model. They are not representative samples or a new performance benchmark.
TP = fall video classified as fall; FN = missed fall video; FP = non-fall video classified as fall.
FLASH Le2i failure is an approved FP substitute (no fall-video FN exists); FLASH CAUCA failure is a pose-input failure, not a valid-input classification error.
Video TP can coexist with extra event alarms. X3D has video scores only.</p>
<p>Local research demo: dataset use and redistribution conditions still apply. RGB frames are not anonymized. Nothing has been publicly uploaded.</p>
<p><a href="fall_demo_20_cases.pdf">20-page PDF</a> · <a href="overview.png">Overview</a> · <a href="cases.csv">Case table</a> · <a href="cases.json">Detailed case metadata</a></p><main>'''+''.join(cards)+'</main></html>'
    (OUT/'index.html').write_text(gallery)
    (OUT/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    keys=['model','dataset','id','kind','case','processed','ground_truth','video_prediction','png']
    with (OUT/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True);EVIDENCE.mkdir(parents=True,exist_ok=True)
    remember(Path(__file__))
    results,plan=load_evaluations();cases=select_cases(results)
    selection=dict(post_hoc=True,no_training=True,no_inference=True,no_threshold_changes=True,
                   rule=select_cases.__doc__,cases=cases)
    selection_path=EVIDENCE/'selection.json'
    if selection_path.exists(): assert json.loads(selection_path.read_text())==selection, 'Selection changed'
    else: selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    xm={r['id']:r for r in read(BASE/FOLDERS['privacy_x3d_uda_rgb']/'input_manifest.json')}
    rows=[]
    for number,c in enumerate(cases,1):
        if args.preview and number not in (1,2,9,13,17,20):continue
        print(f'Preparing {number:02d}/20 {c["model"]} {c["id"]} {c["case"]}',flush=True)
        d=prepare_case(c,plan[c['id']],xm)
        filename=render_case(c,d,number);rows.append(public_case(c,d,filename))
        print('Rendered',filename,flush=True)
    if args.preview:
        print('Preview complete; collection not created.',flush=True);return
    assert len(rows)==20
    make_collection(rows)
    artifacts={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.png','.pdf','.html','.json','.csv')}
    sizes=[]
    for r in rows:
        with Image.open(OUT/r['png']) as im:
            assert im.size==(3200,2100);im.verify();sizes.append([r['png'],3200,2100])
    for p,h in SOURCES.items():assert sha(ROOT/p)==h,'Evidence changed during rendering: '+p
    validation=dict(passed=True,cases=len(rows),models=len(FOLDERS),datasets=len(SCOPES),
                    exact_displayed_source_pixels_verified=True,cached_decisions_recomputed=True,
                    source_files_unchanged=True,generative_images=False,training=False,inference=False,gpu_used=False,
                    image_dimensions=sizes,source_hashes=SOURCES,artifact_hashes=artifacts,
                    limitations=['Post-hoc qualitative examples, not a new or representative evaluation.',
                                 'Offline time alignment is not measured online latency.',
                                 'X3D has no temporal score; FLASH CAUCA failure has no classifier output.',
                                 'RGB frames are not anonymized; dataset redistribution rights are not established.'])
    (EVIDENCE/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'fall_demo_20_cases.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(artifacts):z.write(OUT/name,arcname=name)
    print('COMPLETE',json.dumps({k:validation[k] for k in ('passed','cases','models','datasets','gpu_used')}),flush=True)


if __name__=='__main__':main()
