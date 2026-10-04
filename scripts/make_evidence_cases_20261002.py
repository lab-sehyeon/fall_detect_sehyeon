"""Actual SAFER OOD window examples: original RGB, used 3D pose and saved scores."""
from pathlib import Path
import hashlib,json,os
os.environ['MPLCONFIGDIR']='/tmp/foundskel-evidence-mpl'
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/fall_processed/SAFER-Activities/v3_document_reconstruction_20260922_r3/sequence_cache'
CACHE=ROOT/'checkpoint/fall/F0_CONTROLS_SAFER_V3_DOCUMENT_RECONSTRUCTION_20260922_R3/cache/ood'
GLOBAL=ROOT/'checkpoint/fall/GLOBAL_MOTION_DOCUMENT_RECONSTRUCTION_20260928_R1'
OUT=ROOT/'docs/images/paper_evidence_20261002'
INTERNAL=ROOT/'docs/internal/2026-10-02_evidence_audit'
EDGES=[(0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(6,7),(20,8),(8,9),(9,10),(10,11),
       (0,12),(12,13),(13,14),(14,15),(0,16),(16,17),(17,18),(18,19)]


def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()


def main():
 import sys
 sys.path.insert(0,str(ROOT))
 from fall_pipeline.joint.models import JointFallModel
 OUT.mkdir(exist_ok=True);INTERNAL.mkdir(exist_ok=True)
 torch.set_num_threads(1);cv2.setNumThreads(1)
 seqs=read(BASE/'ood/sequences.json')
 videos={p.stem.lower():p for p in (ROOT/'data/source_archives/SAFER-Activities/raw').rglob('*.mp4')}
 y=np.load(CACHE/'center_derived_labels.npy');ids=np.load(CACHE/'sequence_index.npy');starts=np.load(CACHE/'window_start.npy')
 logits=np.load(GLOBAL/'evaluation/ood/G0_logits.npy');pred=logits.argmax(1)
 prob=torch.softmax(torch.from_numpy(logits),1).numpy()[:,1]
 t=np.load(CACHE/'temporal_features.npy',mmap_mode='r');s=np.load(CACHE/'spatial_features.npy',mmap_mode='r')
 model=JointFallModel().eval().requires_grad_(False)
 jp=ROOT/'checkpoint/fall/JOINT_V3_DOCUMENT_RECONSTRUCTION_20260922_R3/v3/final/final_j1.pt'
 model.load_state_dict(torch.load(jp,map_location='cpu',weights_only=True))
 selected=read(GLOBAL/'selection.json')['best']['G0'];hp=GLOBAL/f'G0/epoch_{selected["epoch"]:03d}.pt'
 head=torch.nn.Linear(2048,4).eval().requires_grad_(False);head.load_state_dict(torch.load(hp,map_location='cpu',weights_only=True))
 masks={'TP':(y==1)&(pred==1),'TN':(y!=1)&(pred!=1),'FP':(y!=1)&(pred==1),'FN':(y==1)&(pred!=1)}
 rows=[];used=set();fig=plt.figure(figsize=(17,13));grid=fig.add_gridspec(4,3,width_ratios=[1.05,.85,1.8],hspace=.55,wspace=.3)
 classnames=['other','fall','lie_down','lying_down']
 for rownum,(case,mask) in enumerate(masks.items()):
  candidates=[int(i) for i in np.flatnonzero(mask) if seqs[int(ids[i])]['source_name'].lower() in videos]
  candidates.sort(key=lambda i:(seqs[int(ids[i])]['source_name'],int(starts[i])))
  # First eligible frame in lexical source/time order; prefer a new source, without confidence ranking.
  fresh=[i for i in candidates if int(ids[i]) not in used]
  if not candidates:raise RuntimeError('No real '+case+' candidate; do not synthesize')
  ix=(fresh or candidates)[0];sid=int(ids[ix]);used.add(sid);seq=seqs[sid];center=int(starts[ix])+32
  video=videos[seq['source_name'].lower()];cap=cv2.VideoCapture(str(video));assert cap.isOpened()
  fps=cap.get(cv2.CAP_PROP_FPS);assert abs(fps-25)<.001
  cap.set(cv2.CAP_PROP_POS_FRAMES,center);ok,bgr=cap.read();cap.release();assert ok
  source=BASE/seq['uid'];pose=np.load(source/'ntu25.npy',mmap_mode='r');coarse=np.load(source/'labels.npy',mmap_mode='r')
  labels=np.asarray([0]*10+[1,2,3]+[0]*3,dtype=np.int64)[coarse]
  assert len(pose)==seq['total_frames'] and int(labels[center])==int(y[ix])
  with torch.inference_mode():
   feature=torch.from_numpy(np.concatenate((t[ix:ix+1],s[ix:ix+1]),axis=1))
   replay=head(model.adapter(feature)).numpy()[0]
  np.testing.assert_allclose(replay,logits[ix],atol=1e-4,rtol=1e-5)
  assert replay.argmax()==pred[ix]
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  ax=fig.add_subplot(grid[rownum,0]);ax.imshow(rgb);ax.axis('off')
  ax.set_title(f'{case} window  |  GT: {classnames[y[ix]]}\nPrediction: {classnames[pred[ix]]}',fontsize=11,loc='left')
  ax=fig.add_subplot(grid[rownum,1],projection='3d');q=np.array(pose[center]);q=q[:,[0,2,1]];q[:,2]*=-1
  for a,b in EDGES:ax.plot(*q[[a,b]].T,color='#267c92',linewidth=2)
  ax.scatter(*q.T,s=8,color='#142c3e');ax.set_title('3D input at center frame',fontsize=11)
  midpoint=(q.max(0)+q.min(0))/2;radius=max(float(np.ptp(q,axis=0).max())*.6,.2)
  ax.set_xlim(midpoint[0]-radius,midpoint[0]+radius);ax.set_ylim(midpoint[1]-radius,midpoint[1]+radius);ax.set_zlim(midpoint[2]-radius,midpoint[2]+radius)
  ax.set_xlabel('x');ax.set_ylabel('z');ax.set_zlabel('−y');ax.view_init(12,-65);ax.set_box_aspect([1,1,1]);ax.tick_params(labelsize=6)
  ax=fig.add_subplot(grid[rownum,2]);near=(ids==sid)&(np.abs(starts+32-center)<=200);times=(starts[near]+32)/25
  ax.plot(times,prob[near],color='#087f8c',label='P(fall)',linewidth=1.7)
  ax.scatter(times[pred[near]==1],prob[near][pred[near]==1],s=9,color='#d46b25',label='fall argmax')
  lo=max(0,center-200);hi=min(len(labels),center+201);segment=np.asarray(labels[lo:hi])==1
  boundaries=np.flatnonzero(np.diff(np.r_[False,segment,False]));first=True
  for a,b in boundaries.reshape(-1,2):ax.axvspan((lo+a)/25,(lo+b)/25,color='#b9c3cc',alpha=.5,label='GT fall' if first else None);first=False
  ax.axvline(center/25,color='#ad3862',linestyle='--',label='shown center')
  ax.set_ylim(-.04,1.04);ax.set_xlabel('Time (s), window center');ax.set_ylabel('Fall probability');ax.grid(alpha=.2)
  ax.set_title(f'{seq["source_name"]}  |  frame {center}  |  P(fall)={prob[ix]:.3f}',fontsize=10)
  if rownum==0:ax.legend(fontsize=8,loc='upper right',ncol=2)
  rows.append(dict(case=case,unit='window (64 frames, center label)',sample_index=ix,sequence_index=sid,
    source=seq['source_name'],window_start=int(starts[ix]),center_frame=center,gt=int(y[ix]),pred=int(pred[ix]),
    probability=float(prob[ix]),video=str(video.relative_to(ROOT)),video_sha256=sha(video),
    pose_sha256=sha(source/'ntu25.npy'),labels_sha256=sha(source/'labels.npy'),cpu_replay_max_abs=float(np.abs(replay-logits[ix]).max())))
 fig.suptitle('Observed SAFER OOD cases: TP, TN, FP and FN',fontsize=19,y=.985)
 fig.text(.05,.018,'Actual RGB frames, supplied-pose 3D inputs and frozen J1 + G0 scores. Window examples, not event-level or YOLO end-to-end results.\nGray: fall ground truth. Orange: fall wins 4-class argmax. No probability threshold is used. Offline processing.',fontsize=10)
 fig.subplots_adjust(top=.92,bottom=.08)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'safer_ood_cases.{ext}',dpi=150)
 plt.close(fig)
 report=dict(passed=True,selection='first lexical source/start per actual case, preferring distinct source; no confidence or aesthetics ranking',
  data='SAFER OOD provided pose pathway',head='final J1 adapter + separately trained G0',head_sha256=sha(hp),j1_sha256=sha(jp),
  logits_sha256=sha(GLOBAL/'evaluation/ood/G0_logits.npy'),cases=rows,
  limitation='illustrative post-hoc window examples, not proof of population performance or new RGB frontend evaluation')
 (INTERNAL/'case_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
