"""Prepare lossless frames/configs for untouched author test.py; do not infer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,copy,pprint
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/original_privacy_x3d_20261004'
sys.path.insert(0,str(REPO))
import mmcv
from mmaction.datasets.pipelines import Compose
OUT=ROOT/'data/fall_processed/RGB/privacy_x3d_external_20261004_r1'
OLD=ROOT/'data/fall_processed/RGB/original_baselines_20261004_r2'
E=ROOT/'docs/internal/2026-10-04_hfd_x3d_external_comparison_evidence'
def read(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'contract.json').exists()
    meta=read(E/'x3d_checkpoint_meta.json');cfg=mmcv.Config.fromstring(meta['config'],'.py')
    sample=Compose([cfg.test_pipeline[0]])
    plan=read(OLD/'plan.json');gts={x['id']:x for x in read(OLD/'ground_truth.json')};manifest=[]
    for no,item in enumerate(plan,1):
        dest=OUT/'frames'/item['id'];dest.mkdir(parents=True,exist_ok=True)
        trace=read(ROOT/item['trace']);n=item['source_frames'];info=sample(dict(total_frames=n,start_index=1))
        indices=info['frame_inds'].tolist();wanted=set(indices);framefiles=[];cap=cv2.VideoCapture(str(ROOT/item['video']));count=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert count<len(trace['pts']) and hashlib.sha256(bgr.tobytes()).hexdigest()==trace['bgr_sha256'][count],(item['id'],count)
            count+=1
            if count in wanted:
                p=dest/f'img_{count:04}.png'
                if not p.exists():assert cv2.imwrite(str(p),bgr,[cv2.IMWRITE_PNG_COMPRESSION,1])
                assert np.array_equal(cv2.imread(str(p)),bgr)
                framefiles.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),source_index=count-1,pixel_sha256=trace['bgr_sha256'][count-1]))
        cap.release();assert count==n and len(framefiles)==len(wanted)
        row=dict(id=item['id'],scope=item['scope'],source_frames=n,frame_dir=str(dest.relative_to(ROOT)),sample_indices_1based=indices,
          label=int(bool(gts[item['id']]['episodes'])),domain_label=1,pngs=framefiles)
        # Exercise the exact official RawFrameDecode and later transforms, without a model.
        pipe=Compose(cfg.test_pipeline)
        data=pipe(dict(total_frames=n,start_index=1,frame_dir=str(dest),filename_tmpl='img_{:04}.png',modality='RGB',label=row['label'],domain_label=1))
        assert tuple(data['imgs'].shape)==(5,3,16,256,256)
        row['input_shape']=list(data['imgs'].shape);row['input_tensor_sha256']=hashlib.sha256(data['imgs'].numpy().tobytes()).hexdigest()
        manifest.append(row)
        if no%20==0:print('prepared',no,'/230',flush=True)
    save(OUT/'input_manifest.json',manifest)
    for name in ['plan.json','ground_truth.json','own_predictions.json']:save(OUT/name,read(OLD/name))
    finish()
def finish():
    manifest=read(OUT/'input_manifest.json')
    cfg=mmcv.Config.fromstring(read(E/'x3d_checkpoint_meta.json')['config'],'.py')
    configs=[]
    for scope in ['le2i130','cauca100']:
        rows=[r for r in manifest if r['scope']==scope];ann=OUT/(scope+'_annotation.txt');lab=OUT/(scope+'_labels.txt')
        ann.write_text(''.join(f'{ROOT/r["frame_dir"]} {r["source_frames"]} {r["label"]} 1\n' for r in rows));lab.write_text(''.join(str(r['label'])+'\n' for r in rows))
        c=copy.deepcopy(cfg);c.data.test.ann_file=str(ann);c.data.test.data_prefix='';c.gpu_ids=[0];c.data.workers_per_gpu=0;c.data.test_dataloader=dict(videos_per_gpu=1,workers_per_gpu=0)
        c.eval_config={};c.data.test.test_mode=True;c.opencv_num_threads=0;c.omp_num_threads=4;c.cudnn_benchmark=False
        # The author's tools/test.py disables pretrained initialization itself.
        file=ROOT/'configs'/('privacy_x3d_external_20261004_r1_'+scope+'.py')
        # Old MMCV passes removed YAPF verify kwarg. Serialize plain config values
        # without changing either shared package; verify round-trip equality.
        file.write_text(''.join(k+' = '+pprint.pformat(v,width=110,sort_dicts=False)+'\n' for k,v in c._cfg_dict.items()))
        assert dict(mmcv.Config.fromfile(str(file)))==dict(c)
        configs.append(file)
    # Draft: UDA on RGB needs the pending user's scope confirmation before execution.
    files=[Path(__file__),E/'x3d_load_verification.json',REPO/'checkpoints/best_top1_acc_epoch_239.pth']+configs
    files+=list(REPO.rglob('*.py'))+[OUT/n for n in ['input_manifest.json','plan.json','ground_truth.json','own_predictions.json']]
    files+=list(OUT.glob('*_annotation.txt'))+list(OUT.glob('*_labels.txt'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    for r in manifest:
        hashes.update({p['path']:p['sha256'] for p in r['pngs']})
    save(OUT/'prepared_contract.json',dict(status='prepared; requires approval to use UDA checkpoint as RGB baseline',
      evaluation='Original author tools/test.py and own_scipt/evaluation/evaluation.py. Whole-video5-clip average logits, softmax fall>0.5, no event timestamp.',
      scope='Le2i130 CAUCA100 same video labels; original UDA checkpoint, RGB input, not source-only RGB checkpoint',
      target_training=False,target_adaptation=False,target_threshold_tuning=False,files=hashes))
    print('prepared all230; inference not executed',flush=True)
if __name__=='__main__':
    if (OUT/'input_manifest.json').exists():finish()
    else:prepare()
