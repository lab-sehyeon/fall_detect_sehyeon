"""Access the unmodified official decoder and independently trace its pixels."""
from pathlib import Path
import sys,importlib.util,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1'
sys.path.insert(0,str(ASSET/'runtime'))
DECODE=ASSET/'package_source/src/omnifall/_decode.py'
spec=importlib.util.spec_from_file_location('_official_omnifall_decode',DECODE)
official=importlib.util.module_from_spec(spec);sys.modules[spec.name]=official;spec.loader.exec_module(official)

def source_trace(path):
    import av
    with av.open(str(path)) as c:
        s=c.streams.video[0];s.codec_context.thread_count=2
        frames=[dict(pts=int(f.pts),rgb_sha256=hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()) for f in c.decode(s) if f.pts is not None]
        assert frames and all(b['pts']>a['pts'] for a,b in zip(frames,frames[1:]))
        return dict(time_base_num=s.time_base.numerator,time_base_den=s.time_base.denominator,
                    start_pts=0 if s.start_time is None else int(s.start_time),frames=frames,
                    width=s.codec_context.width,height=s.codec_context.height,fps=float(s.average_rate),
                    duration=None if s.duration is None else float(s.duration*s.time_base))

def sampling_trace(trace,start,end):
    # Independent nearest-PTS selection; on an exact distance tie choose the later frame.
    wanted=[start+i*(end-start)/63 for i in range(64)]
    tb=trace['time_base_num']/trace['time_base_den'];pts=np.array([r['pts'] for r in trace['frames']])
    target=np.array([round(t/tb)+trace['start_pts'] for t in wanted]);hi=np.searchsorted(pts,target)
    hi=hi.clip(0,len(pts)-1);lo=(hi-1).clip(0,len(pts)-1)
    indices=np.where(np.abs(pts[lo]-target)<np.abs(pts[hi]-target),lo,hi)
    times=(pts[indices]-trace['start_pts'])*tb
    return dict(requested_seconds=wanted,source_indices=indices.tolist(),source_pts=pts[indices].tolist(),
        actual_seconds=times.tolist(),unique_frames=int(len(set(indices))),
        max_boundary_excursion_seconds=float(max(0,start-times.min(),times.max()-end)),
        pixel_sha256=[trace['frames'][i]['rgb_sha256'] for i in indices])

def decode(path,start,end,expected=None):
    frames=official.decode_segment(str(path),start=start,end=end,num_frames=64,target_fps=25,sampling='uniform')
    assert frames.shape[0]==64 and frames.shape[-1]==3 and frames.dtype==np.uint8
    if expected is not None:
        assert [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]==expected['pixel_sha256'],'official decoder differs from independent nearest-PTS trace'
    return frames
