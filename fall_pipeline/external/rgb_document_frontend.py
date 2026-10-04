"""Fixed C2 detector and official ViTPose inference, label-blind RGB inputs."""
import os
import numpy as np
from . import rgb_document_io as io

def main():
    import cv2,torch
    config=io.read(io.CONFIG);root,plan=io.locked(config);dev=io.device(config)
    cv2.setNumThreads(1)
    os.environ['YOLO_CONFIG_DIR']=str(root/'yolo_runtime')
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib_runtime')
    from ultralytics import YOLO
    from mmcv import Config
    from mmpose.models import build_posenet
    from mmpose.apis import inference_top_down_pose_model
    from mmpose.datasets import DatasetInfo
    from fall_pipeline.common.integrity import hash_named_tensors
    detector=YOLO(str(io.ROOT/config['asset_root']/'yolov8x.pt'))
    # Initialize/fuse the official inference graph before freezing its integrity hash.
    detector.predict(np.zeros((64,64,3),np.uint8),device=0,verbose=False,save=False,imgsz=640)
    detector.model.eval().requires_grad_(False)
    cfg=Config.fromfile(str(io.ROOT/config['pose_config']));pose=build_posenet(cfg.model)
    ckpt=torch.load(io.ROOT/config['asset_root']/'vitpose-b-multi-coco.pth',map_location='cpu',weights_only=True)
    state=ckpt.get('state_dict',ckpt);state={k.removeprefix('module.'):v for k,v in state.items()}
    io.require(all(torch.isfinite(v).all() for v in state.values()),'ViTPose nonfinite weights')
    pose.load_state_dict(state,strict=True);del ckpt,state
    pose.cfg=cfg;pose.eval().requires_grad_(False).to(dev)
    dataset_info=DatasetInfo(cfg.data.test.dataset_info)
    before={k:hash_named_tensors(m.state_dict().items()) for k,m in [('detector',detector.model),('pose',pose)]}
    settings=config['detector']
    def detect(image,conf,size):
        with torch.inference_mode():
            result=detector.predict(image,conf=conf,imgsz=size,iou=settings['iou'],classes=[0],
                                    max_det=settings['max_det'],device=0,verbose=False,save=False,
                                    augment=False,half=False)[0]
        return np.column_stack((result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy())).astype(np.float32)
    for item in plan:
        io.safety(config);dest=root/item['id'];dest.mkdir(exist_ok=True)
        if io.stage_done(dest,'frontend'):continue
        video=io.ROOT/item['video'];mapping=io.ROOT/item['mapping']
        io.require(io.sha(video)==item['video_sha256'] and io.sha(mapping)==item['mapping_sha256'],'input changed')
        with np.load(mapping,allow_pickle=False) as z:
            indices=z['source_indices'];times=z['canonical_timestamps'];source_pts=z['source_timestamps']
        n=len(indices);xy=np.zeros((n,17,2),np.float32);scores=np.zeros((n,17),np.float32)
        boxes=np.zeros((n,5),np.float32);rescue=np.zeros(n,bool);candidates=[];offsets=[0]
        cap=cv2.VideoCapture(str(video));io.require(cap.isOpened(),'video open')
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        previous=None;source_index=-1;frame=None
        try:
            for i,wanted in enumerate(indices):
                io.safety(config)
                while source_index<int(wanted):
                    ok,frame=cap.read();source_index+=1;io.require(ok,'early decode end')
                    io.require(abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-source_pts[source_index])<1e-6,'decoded PTS changed')
                io.require(source_pts[wanted]<=times[i],'future frame')
                base=detect(frame,settings['base_conf'],settings['base_imgsz'])
                cand,used=io.cascade(base,lambda:detect(frame,settings['fallback_conf'],settings['fallback_imgsz']))
                candidates.append(cand);offsets.append(offsets[-1]+len(cand));rescue[i]=used
                chosen=io.select_track(cand,previous)
                if chosen is not None:
                    boxes[i]=chosen;previous=chosen
                    with torch.inference_mode():
                        result,_=inference_top_down_pose_model(pose,frame,[{'bbox':chosen}],bbox_thr=None,
                            format='xyxy',dataset='TopDownCocoDataset',dataset_info=dataset_info)
                    io.require(len(result)==1,'dominant pose count')
                    keypoints=result[0]['keypoints'];io.require(keypoints.shape==(17,3) and np.isfinite(keypoints).all(),'pose shape/finite')
                    xy[i]=keypoints[:,:2];scores[i]=keypoints[:,2]
                if i==0 or (i+1)%50==0 or i+1==n:print('frontend',item['id'],i+1,n,flush=True)
        finally:cap.release()
        q=io.quality(boxes,xy,scores,config['quality'])
        io.npz(dest/'frontend.npz',xy=xy,scores=scores,boxes=boxes,rescue=rescue,
               candidates=np.concatenate(candidates),offsets=np.asarray(offsets,np.int64),
               width=np.array(width),height=np.array(height),source_indices=indices,timestamps=times)
        after={k:hash_named_tensors(m.state_dict().items()) for k,m in [('detector',detector.model),('pose',pose)]}
        io.require(before==after,'frontend weights mutated')
        io.save(dest/'frontend.json',dict(passed=True,quality=q,frames=n,width=width,height=height,
            payload_sha256=io.sha(dest/'frontend.npz'),models_before=before,models_after=after,
            fallback_frames=int(rescue.sum()),labels_used=False,training=False))
        print(item['id'],'quality',q,flush=True)
    io.locked(config)

if __name__=='__main__':main()
