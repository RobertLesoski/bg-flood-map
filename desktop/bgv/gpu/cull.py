"""Shared, allocation-light culling helpers (numpy only).

`frustum_planes(vp)`  5 normalised planes (left, right, bottom, top, near) of a reversed-Z / infinite-far view-projection
                      matrix (row-vector convention of bgv.gpu.camera: clip = vp @ [x y z 1]).  `app.frustum()` caches the
                      result per frame so the terrain, road and building layers do not each recompute it.
"""
from __future__ import annotations

import numpy as np


def frustum_planes(vp: np.ndarray) -> np.ndarray:
    vp = np.asarray(vp, np.float64)
    r0, r1, r2, r3 = vp[0], vp[1], vp[2], vp[3]
    p = np.empty((5, 4), np.float64)
    np.add(r3, r0, out=p[0])
    np.subtract(r3, r0, out=p[1])
    np.add(r3, r1, out=p[2])
    np.subtract(r3, r1, out=p[3])
    np.subtract(r3, r2, out=p[4])
    n = np.sqrt(p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1] + p[:, 2] * p[:, 2])
    p /= n[:, None]
    return p


def planes_for(app, vp=None) -> np.ndarray:
    """Frustum planes of `vp` (default `app.view_proj`), cached on the app per view-projection object."""
    if vp is None:
        vp = app.view_proj
    fn = getattr(app, "frustum", None)
    if fn is not None and vp is getattr(app, "view_proj", None):
        return fn()
    return frustum_planes(vp)
