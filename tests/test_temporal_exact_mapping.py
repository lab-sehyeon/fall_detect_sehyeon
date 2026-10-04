"""Regression coverage for coincident PTS boundaries and variable source rates."""
from fractions import Fraction
import numpy as np
from fall_pipeline.external.temporal_continuous_20261002 import exact_indices


def test_equal_timestamp_selects_current_frame():
    pts=np.arange(100)*40
    actual=exact_indices(pts,1,1000,0,100)
    np.testing.assert_array_equal(actual,np.arange(100))
    floating=np.searchsorted(pts.astype(float)*.001,np.arange(100)/25,side='right')-1
    assert floating[35]==34 and actual[35]==35


def test_rational_mapping_matches_fraction_oracle():
    for num,den,step,origin in [(1,1000,50,17),(1,24000,1000,500),(1001,30000,1,7),(1,1000,40,0)]:
        pts=origin+np.arange(150)*step
        actual=exact_indices(pts,num,den,origin,100)
        expected=[max(i for i,p in enumerate(pts) if Fraction(int(p)-origin)*Fraction(num,den)<=Fraction(j,25)) for j in range(100)]
        np.testing.assert_array_equal(actual,expected)


def test_truly_future_frame_is_not_admitted_by_tolerance():
    pts=np.array([0,1400000000001],np.int64)
    actual=exact_indices(pts,1,10**12,0,36)
    assert actual[35]==0
