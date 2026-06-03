"""Tests for homography coplanar guards on overlap detection."""

import numpy as np

from app.pipeline.overlap import _homography_is_coplanar


def test_identity_homography_accepted():
    H = np.eye(3, dtype=np.float64)
    assert _homography_is_coplanar(H) is True


def test_large_rotation_rejected():
    import math

    angle = math.radians(60)
    c, s = math.cos(angle), math.sin(angle)
    H = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    assert _homography_is_coplanar(H) is False
