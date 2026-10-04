"""CPU-only checks for paper figure selection and output interpretation."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import make_paper_demo_20261004 as paper


def test_outcome_codes():
    for gt,pred,code in [(1,1,'TP'),(1,0,'FN'),(0,1,'FP'),(0,0,'TN')]:
        assert paper.class_code(dict(episodes=[{}] if gt else [],video_prediction=pred))==code


def test_frame_selection():
    times=np.arange(0,3,.05)
    for n in (3,4):
        for gt in [[],[dict(fall_start=0,fall_end=2.9)],[dict(fall_start=1,fall_end=2)]]:
            indices=paper.frame_indices(gt,times,n)
            assert len(indices)==n and min(indices)>=0 and max(indices)<len(times)
            assert indices==sorted(indices)


def test_paper_model_label():
    assert paper.NAMES['own']==paper.SHORT['own']=='Ours'
    assert all('J1' not in s and 'G0' not in s for s in [*paper.NAMES.values(),*paper.SHORT.values()])


def test_same_video_comparisons():
    rows,plan=paper.old.load_evaluations();cases=paper.old.select_cases(rows)
    own=[c for c in cases if c['model']=='own'];assert len(own)==4
    for case in own:
        for m,rr in rows.items():
            row=next(r for r in rr if r['id']==case['id'])
            assert row['episodes']==case['episodes']
            s=paper.load_series(row,plan[row['id']])
            if m=='privacy_x3d_uda_rgb':assert not s['temporal'] and 'times' not in s
            else:assert s['temporal'] and int(s['positive'].any())==row['video_prediction']


def test_video_tp_can_miss_event():
    rows,plan=paper.old.load_evaluations()
    row=next(r for r in rows['hfd_reproduction'] if r['id']=='cauca__backwards__FallBackwardsS1')
    assert paper.class_code(row)=='TP'
    assert [row['event'][k] for k in ('tp','fp','fn')]==[0,1,1]
    s=paper.load_series(row,plan[row['id']]);np.testing.assert_allclose(s['alarms'],[.75])


def test_pose_failure_is_not_negative_score():
    rows,plan=paper.old.load_evaluations()
    row=next(r for r in rows['flash'] if r['id']=='cauca__forward__FallForwardS8')
    s=paper.load_series(row,plan[row['id']])
    assert s['input_failure'] and not s['processed'] and 'score' not in s and 'times' not in s
