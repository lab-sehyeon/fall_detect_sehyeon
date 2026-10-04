"""Preregistered geometry-only reconstruction metrics; no label/classifier inputs."""
from __future__ import annotations
import hashlib
import numpy as np
from data_gen.safer_v2_geometry import coco_xy_to_h36m

BONES=np.array([(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
                (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)],np.int64)


def pilot_records(records,config):
    p=config['pilot']
    candidates=[a for a in records if a['split']=='train' and p['min_frames']<=int(a['total_frames'])<=p['max_frames']]
    candidates.sort(key=lambda a:hashlib.sha256((str(p['seed'])+'|'+a['frame_dir']).encode()).hexdigest())
    if len(candidates)<p['count']: raise ValueError('not enough train-only pilot sequences')
    return candidates[:p['count']]


def weights(kind,width=243,floor=.05):
    if kind=='uniform': return np.ones(width,np.float64)
    if kind=='triangular': value=1-np.abs(np.arange(width)-(width-1)/2)/max((width-1)/2,1)
    elif kind=='hann': value=np.hanning(width)
    else: raise ValueError('unknown weighting')
    return np.maximum(value,floor)


def fuse(predictions,starts,frames,kind,floor=.05,independent=False):
    p=np.asarray(predictions);starts=np.asarray(starts)
    if p.shape!=(len(starts),243,17,3) or not np.isfinite(p).all() or not len(starts): raise ValueError('prediction shape')
    total=np.zeros((frames,17,3),np.float64);divisor=np.zeros(frames,np.float64);cover=np.zeros(frames,np.int64)
    w=weights(kind,floor=floor)
    if independent:
        index=starts[:,None]+np.arange(243);valid=(index>=0)&(index<frames)
        wi=np.broadcast_to(w,index.shape)[valid]
        np.add.at(total,index[valid],p[valid]*wi[:,None,None]);np.add.at(divisor,index[valid],wi)
        cover=np.bincount(index[valid],minlength=frames)
    else:
        for start,values in zip(starts,p):
            if start<0 or start>=frames: raise ValueError('invalid start')
            end=min(int(start)+243,frames);n=end-int(start)
            total[start:end]+=values[:n]*w[:n,None,None];divisor[start:end]+=w[:n];cover[start:end]+=1
    if np.any(divisor<=0): raise ValueError('uncovered timeline')
    return (total/divisor[:,None,None]).astype(np.float32),cover,divisor


def boundaries(starts,frames,width=243):
    values=np.unique(np.concatenate((starts,np.asarray(starts)+width)))
    return values[(values>0)&(values<frames)].astype(np.int64)


def speed(values):
    return np.linalg.norm(np.diff(values-values[:,:1],axis=0)[:,1:],axis=-1).mean(1)


def jump_ratios(values,points,radius=5,epsilon=1e-8):
    velocities=speed(values);result=[]
    for point in points:
        i=int(point)-1
        neighbors=np.concatenate((velocities[max(0,i-radius):i],velocities[i+1:min(len(velocities),i+radius+1)]))
        if not len(neighbors): raise ValueError('no local reference')
        result.append(velocities[i]/max(float(np.median(neighbors)),epsilon))
    return np.asarray(result,np.float64)


def bone_lengths(values): return np.linalg.norm(values[:,BONES[:,1]]-values[:,BONES[:,0]],axis=-1)


def measure(h36m,coco_xy,starts,config):
    h=np.asarray(h36m,np.float64);xy=coco_xy_to_h36m(coco_xy).astype(np.float64);g=config['geometry'];eps=g['epsilon']
    if h.shape!=(len(xy),17,3) or not np.isfinite(h).all(): raise ValueError('geometry input')
    x=h[:,:,:2]-h[:,:1,:2];y=xy-xy[:,:1];x=x[:,1:];y=y[:,1:]
    denom=(x*x).sum((1,2));scale=np.maximum((x*y).sum((1,2))/np.maximum(denom,eps),0)
    diagonal=np.linalg.norm(np.ptp(xy,axis=1),axis=-1)
    nme=np.linalg.norm(x*scale[:,None,None]-y,axis=-1).mean(1)/np.maximum(diagonal,eps)
    lengths=bone_lengths(h);cv=lengths.std(0)/np.maximum(lengths.mean(0),eps)
    velocity=speed(h)/max(float(np.median(lengths.mean(1))),eps)
    points=boundaries(starts,len(h));j3=jump_ratios(h,points,g['local_radius'],eps);j2=jump_ratios(xy,points,g['local_radius'],eps)
    artifacts=(j3>3)&(j2<2)
    return {'nme':nme,'cv':cv,'velocity':velocity,'seam_points':points,'j3':j3,'j2':j2,
            'seam_artifacts':int(artifacts.sum()),'seam_count':len(points),'root_exact_zero':bool(np.all(h[:,0]==0)),
            'degenerate_projection_frames':int(np.sum((denom<=eps)|(diagonal<=eps))),
            'degenerate_bones':int(np.sum(lengths.mean(0)<=eps))}


def aggregate(measurements):
    total=sum(m['seam_count'] for m in measurements);artifacts=sum(m['seam_artifacts'] for m in measurements)
    out={'seam_count':total,'seam_artifacts':artifacts,'seam_rate':artifacts/total if total else None,
         'root_exact_zero':all(m['root_exact_zero'] for m in measurements),
         'degenerate_projection_frames':sum(m['degenerate_projection_frames'] for m in measurements),
         'degenerate_bones':sum(m['degenerate_bones'] for m in measurements)}
    for key,outkey,q in (('nme','nme_p95',95),('cv','bone_cv_p95',95),('velocity','speed_p99',99)):
        values=np.concatenate([m[key] for m in measurements]);out[outkey]=float(np.percentile(values,q,method='linear'))
    if not all(np.isfinite(out[k]) for k in ('nme_p95','bone_cv_p95','speed_p99')): raise ValueError('nonfinite metric')
    return out


def gates(candidate,baseline,config):
    p=config['pilot']['gate'];eps=config['geometry']['epsilon']
    ratio={k:candidate[k]/max(baseline[k],eps) for k in ('nme_p95','bone_cv_p95','speed_p99')}
    reduction=1-candidate['seam_rate']/baseline['seam_rate'] if baseline['seam_rate'] and candidate['seam_rate'] is not None else None
    checks={'seam':reduction is not None and reduction>=p['seam_reduction_min'],
            'root':candidate['root_exact_zero'] and baseline['root_exact_zero'],
            'nondegenerate':candidate['degenerate_projection_frames']==baseline['degenerate_projection_frames']==0 and candidate['degenerate_bones']==baseline['degenerate_bones']==0}
    for key,value in ratio.items(): checks[key]=value<=p[key+'_ratio_max']
    return {'passed':all(checks.values()),'checks':checks,'seam_reduction':reduction,'ratios':ratio}


def overlap_diagnostics(predictions,starts):
    collected={'shape_rms':[],'bone_scale_ratio':[],'shoulder_angle_degrees':[],'spine_angle_degrees':[]}
    for a,b,sa,sb in zip(predictions[:-1],predictions[1:],starts[:-1],starts[1:]):
        width=int(sa)+243-int(sb)
        if width<=0: continue
        a=a[-width:].astype(np.float64);b=b[:width].astype(np.float64);a=a-a[:,:1];b=b-b[:,:1]
        collected['shape_rms'].extend(np.sqrt(np.mean((a-b)**2,axis=(1,2))).tolist())
        collected['bone_scale_ratio'].extend((bone_lengths(a).mean(1)/np.maximum(bone_lengths(b).mean(1),1e-8)).tolist())
        for name,i,j in (('shoulder_angle_degrees',14,11),('spine_angle_degrees',8,0)):
            va=a[:,i]-a[:,j];vb=b[:,i]-b[:,j];den=np.linalg.norm(va,axis=-1)*np.linalg.norm(vb,axis=-1)
            angle=np.degrees(np.arccos(np.clip(np.sum(va*vb,axis=-1)/np.maximum(den,1e-8),-1,1)))
            collected[name].extend(angle.tolist())
    return {k:{'count':len(v),'p50':float(np.percentile(v,50)) if v else None,'p95':float(np.percentile(v,95)) if v else None} for k,v in collected.items()}
