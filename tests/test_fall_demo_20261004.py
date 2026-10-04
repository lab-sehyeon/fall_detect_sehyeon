"""Rendering helpers and selection policy, without training or GPUs."""
import importlib.util
from pathlib import Path
import numpy as np

SPEC=importlib.util.spec_from_file_location('fall_demo',Path(__file__).resolve().parents[1]/'scripts/make_fall_demo_20261004.py')
demo=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(demo)


def test_stable_softmax():
    np.testing.assert_allclose(demo.softmax([[1000,1000],[-1000,-1000]]),[[.5,.5],[.5,.5]])


def test_public_display_name():
    assert demo.NAMES['own']=='ourmodel'
    for value in [*demo.NAMES.values(),*demo.QUALIFIERS.values()]:
        assert 'J1' not in value and 'G0' not in value


def test_edges_and_gating():
    t=np.arange(7)*.04
    np.testing.assert_allclose(demo.edges([False,True,True,False,True,False,False],t),[.04,.16])
    np.testing.assert_allclose(demo.edges([True,True,False,False,False,False,False],t),[0])
    assert not len(demo.edges([False]*7,t))


def test_frame_selection_is_bounded():
    a=demo.frame_targets([{'fall_start':.2,'fall_end':.8}],1.,[3.])
    np.testing.assert_allclose(a,[0,.5,.8,1.])
    assert np.all(np.diff(demo.frame_targets([],10,[]))>0)


def test_audited_case_selection():
    results,_=demo.load_evaluations();cases=demo.select_cases(results)
    assert len(cases)==20
    assert len({(r['model'],r['scope'],r['kind']) for r in cases})==20
    success=[r for r in cases if r['kind']=='success']
    assert len(success)==10 and all(r['case']=='TP' and r['processed'] for r in success)
    assert all(r['event']['tp']>=1 for r in success if 'event' in r)
    fp=[r for r in cases if r['case']=='FP'];assert len(fp)==1
    assert fp[0]['model']=='flash' and fp[0]['scope']=='le2i130'
    rejected=[r for r in cases if not r['processed']];assert len(rejected)==1
    assert rejected[0]['id']=='cauca__forward__FallForwardS8' and rejected[0]['case']=='FN'
    for r in cases:
        if r['kind']=='failure' and r['model']!='flash': assert r['processed'] and r['case']=='FN'
