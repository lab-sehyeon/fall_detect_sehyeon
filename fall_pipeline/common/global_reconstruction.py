"""Preregistered global131 R1. Full-sequence causal masks precede window slicing."""
import numpy as np


def raw_channels(annotation, cutoff=.2):
    xy = np.asarray(annotation['keypoint'][0], np.float64)
    score = np.asarray(annotation['keypoint_score'][0], np.float64)
    box = np.asarray(annotation['bboxes'], np.float64)
    n = int(annotation['total_frames'])
    if xy.shape != (n,17,2) or score.shape != (n,17) or box.shape != (n,4):
        raise ValueError('raw source shape')
    size = np.array([annotation['width'],annotation['height']], np.float64)
    if not (np.isfinite(size).all() and (size>0).all()): raise ValueError('image size')
    bv = np.isfinite(box).all(1) & (box[:,2:]>0).all(1)
    jv = np.isfinite(xy).all(2) & np.isfinite(score) & (score>=cutoff) & (xy!=0).any(2)
    pv = bv & jv[:,11] & jv[:,12]; sv = bv & jv[:,5] & jv[:,6]
    pelvis = (xy[:,11]+xy[:,12])/2/size
    shoulder = (xy[:,5]+xy[:,6])/2/size
    torso = shoulder-pelvis; length = np.sqrt((torso**2).sum(1))
    tv = pv & sv & np.isfinite(length) & (length>1e-8)
    finite_score = np.isfinite(score) & (score>=0)
    confidence = np.where(finite_score,score,0).sum(1)/np.maximum(1,finite_score.sum(1))
    cv = bv & (finite_score & (score>0)).any(1)
    values = np.column_stack((pelvis,(box[:,:2]+box[:,2:]/2)/size,box[:,2:]/size,
                              shoulder,torso,length,confidence))
    valid = np.column_stack((pv,pv,bv,bv,bv,bv,sv,sv,tv,tv,tv,cv))
    valid &= np.isfinite(values)
    return np.where(valid,values,0).astype(np.float32), valid


def causal_arrays(values, valid, fps=25):
    values, valid = np.asarray(values,np.float32),np.asarray(valid,bool)
    if values.ndim!=2 or values.shape[1]!=12 or values.shape!=valid.shape:
        raise ValueError('channels must be [T,12]')
    total=np.zeros(values.shape,np.float64);count=np.zeros(values.shape,np.int64)
    for lag in range(3):
        if lag>=len(values): continue
        src=slice(None,-lag) if lag else slice(None)
        total[lag:]+=np.where(valid[src],values[src],0)
        count[lag:]+=valid[src]
    smask=count>0
    smooth=np.divide(total,count,out=np.zeros_like(total),where=smask).astype(np.float32)
    velocity=np.zeros_like(smooth);vmask=np.zeros_like(smask)
    velocity[1:]=(smooth[1:]-smooth[:-1])*fps
    vmask[1:]=smask[1:]&smask[:-1];velocity[~vmask]=0
    acceleration=np.zeros((len(values),1),np.float32);amask=np.zeros((len(values),1),bool)
    acceleration[1:,0]=(velocity[1:,1]-velocity[:-1,1])*fps
    amask[1:,0]=vmask[1:,1]&vmask[:-1,1];acceleration[~amask]=0
    return smooth,smask,velocity,vmask,acceleration,amask


def masked_stats(values, mask, last=True):
    # [windows,64,channels], float64 accumulation; masked population moments.
    v=np.asarray(values,np.float64);m=np.asarray(mask,bool);count=m.sum(1)
    clean=np.where(m,v,0);mean=clean.sum(1)/np.maximum(count,1)
    variance=np.where(m,(v-mean[:,None,:])**2,0).sum(1)/np.maximum(count,1)
    lo=np.where(m,v,np.inf).min(1);hi=np.where(m,v,-np.inf).max(1)
    lo=np.where(count>0,lo,0);hi=np.where(count>0,hi,0)
    parts=[mean,np.sqrt(variance),lo,hi]
    if last: parts.append(np.where(m[:,-1],v[:,-1],0))
    return np.stack(parts,axis=-1).reshape(len(v),-1).astype(np.float32)


def features_from_causal(causal, raw_mask, starts):
    starts=np.asarray(starts,np.int64)
    if np.any(starts<0) or np.any(starts+64>len(raw_mask)): raise ValueError('window outside source')
    if not len(starts): return np.empty((0,131),np.float32)
    index=starts[:,None]+np.arange(64)
    smooth,smask,velocity,vmask,acceleration,amask=causal
    m=np.asarray(raw_mask,bool)[index];coverage=m.mean(1)
    quality=np.column_stack((m.mean((1,2)),m.all(2).mean(1),m[:,-1].mean(1),
                             coverage.min(1),coverage.mean(1),coverage.max(1),np.full(len(m),64)))
    result=np.column_stack((masked_stats(smooth[index],smask[index]),
                            masked_stats(velocity[index],vmask[index]),
                            masked_stats(acceleration[index],amask[index],last=False),quality)).astype(np.float32)
    if result.shape!=(len(starts),131) or not np.isfinite(result).all(): raise ValueError('feature invalid')
    return result


def reference_feature(causal, raw_mask, start):
    result=[]
    for values,mask,endpoint in ((causal[0],causal[1],True),(causal[2],causal[3],True),
                                 (causal[4],causal[5],False)):
        for channel in range(values.shape[1]):
            v=values[start:start+64,channel].astype(np.float64);m=mask[start:start+64,channel]
            keep=v[m]
            result.extend([keep.mean(),keep.std(),keep.min(),keep.max()] if len(keep) else [0]*4)
            if endpoint: result.append(v[-1] if m[-1] else 0)
    m=raw_mask[start:start+64];coverage=m.mean(0)
    result.extend([m.mean(),m.all(1).mean(),m[-1].mean(),coverage.min(),coverage.mean(),coverage.max(),64])
    return np.array(result,np.float32)
