import pytest
import pandas as pd

# Import directly from cuts_tdrive_same_setup.py
from cuts_tdrive_same_setup import (
    Segment,
    point_to_segment_distance,
    segments_intersect,
    segment_distance,
    segment_interval_overlap
)

def test_point_to_segment_distance():
    assert point_to_segment_distance(5.0, 5.0, 0.0, 0.0, 10.0, 10.0) == 0.0
    assert point_to_segment_distance(5.0, 10.0, 0.0, 0.0, 10.0, 0.0) == 10.0
    assert point_to_segment_distance(-3.0, 4.0, 0.0, 0.0, 10.0, 0.0) == 5.0

def test_segments_intersect():
    t0 = pd.Timestamp('2008-02-02 13:00')
    t1 = pd.Timestamp('2008-02-02 13:10')
    a = Segment(1, t0, t1, 0.0, 0.0, 10.0, 10.0, 0.0)
    b = Segment(2, t0, t1, 0.0, 10.0, 10.0, 0.0, 0.0)
    c = Segment(3, t0, t1, 0.0, 15.0, 10.0, 15.0, 0.0)
    
    assert segments_intersect(a, b) is True
    assert segments_intersect(a, c) is False

def test_segment_distance():
    t0 = pd.Timestamp('2008-02-02 13:00')
    t1 = pd.Timestamp('2008-02-02 13:10')
    a = Segment(1, t0, t1, 0.0, 0.0, 10.0, 0.0, 0.0)
    b = Segment(2, t0, t1, 0.0, 10.0, 10.0, 10.0, 0.0)
    c = Segment(3, t0, t1, 5.0, -5.0, 5.0, 5.0, 0.0)
    
    assert segment_distance(a, b) == 10.0
    assert segment_distance(a, c) == 0.0

def test_segment_interval_overlap():
    window_start = pd.Timestamp('2008-02-02 13:00')
    window_end = pd.Timestamp('2008-02-02 18:00')
    
    a = Segment(1, pd.Timestamp('2008-02-02 12:00'), pd.Timestamp('2008-02-02 14:00'), 0, 0, 0, 0, 0)
    b = Segment(2, pd.Timestamp('2008-02-02 13:30'), pd.Timestamp('2008-02-02 15:00'), 0, 0, 0, 0, 0)
    c = Segment(3, pd.Timestamp('2008-02-02 19:00'), pd.Timestamp('2008-02-02 20:00'), 0, 0, 0, 0, 0)
    
    assert segment_interval_overlap(a, b, window_start, window_end) is True
    assert segment_interval_overlap(b, c, window_start, window_end) is False