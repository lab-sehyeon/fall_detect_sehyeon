"""Post-hoc R3 source-only 2D evidence mask; unchanged full-timeline 3D tests."""
from __future__ import annotations
import numpy as np
from data_gen import safer_v3_geometry as original

pilot_records=original.pilot_records
weights=original.weights
fuse=original.fuse
overlap_diagnostics=original.overlap_diagnostics


def source_support(coco_xy,epsilon):
    xy=original.coco_xy_to_h36m(coco_xy).astype(np.float64)
    if not np.isfinite(coco_xy).all() or not np.isfinite(xy).all():
        raise ValueError('nonfinite source coordinates are not masked')
    return np.linalg.norm(np.ptp(xy,axis=1),axis=-1)>epsilon


def seam_support(valid,points,radius):
    return np.array([bool(valid[max(0,int(b)-1-radius):min(len(valid),int(b)+radius+1)].all())
                     for b in points],dtype=bool)


def measure(h36m,coco_xy,starts,config):
    valid=source_support(coco_xy,config['geometry']['epsilon'])
    m=original.measure(h36m,coco_xy,starts,config)
    eligible=seam_support(valid,m['seam_points'],config['geometry']['local_radius'])
    x=np.asarray(h36m,np.float64)[:,:,:2];x=x-x[:,:1]
    projection=(x[:,1:]*x[:,1:]).sum((1,2))
    m.update(source_frames=len(valid),source_valid_frames=int(valid.sum()),
             source_invalid_frames=int((~valid).sum()),all_seam_count=len(eligible),
             unsupported_seam_count=int((~eligible).sum()),
             degenerate_projection_frames=int((projection<=config['geometry']['epsilon']).sum()))
    m['nme']=m['nme'][valid]
    for key in ('seam_points','j3','j2'): m[key]=m[key][eligible]
    m['seam_count']=int(eligible.sum())
    m['seam_artifacts']=int(((m['j3']>3)&(m['j2']<2)).sum())
    return m


def aggregate(measurements):
    if not measurements or not sum(len(m['nme']) for m in measurements) or not sum(m['seam_count'] for m in measurements):
        raise ValueError('no eligible source-supported geometry evidence')
    out=original.aggregate(measurements)
    fields=('source_frames','source_valid_frames','source_invalid_frames','all_seam_count',
            'seam_count','unsupported_seam_count')
    out.update({key:sum(m[key] for m in measurements) for key in fields})
    out['source_valid_fraction']=out['source_valid_frames']/out['source_frames']
    out['seam_supported_fraction']=out['seam_count']/out['all_seam_count']
    out['sequence_support']=[{key:m[key] for key in fields} for m in measurements]
    out['sequences_without_valid_frames']=sum(m['source_valid_frames']==0 for m in measurements)
    out['sequences_without_supported_seams']=sum(m['seam_count']==0 for m in measurements)
    return out


def gates(candidate,baseline,config):
    result=original.gates(candidate,baseline,config)
    same_frames=all(candidate[k]==baseline[k] for k in ('source_frames','source_valid_frames','source_invalid_frames'))
    same_sequences=[(m['source_frames'],m['source_valid_frames']) for m in candidate['sequence_support']]==[
        (m['source_frames'],m['source_valid_frames']) for m in baseline['sequence_support']]
    result['checks']['source_support_identical']=same_frames and same_sequences
    result['checks']['nonempty_evidence']=all(m['source_valid_frames']>0 and m['seam_count']>0 for m in (candidate,baseline))
    result['passed']=all(result['checks'].values())
    return result
