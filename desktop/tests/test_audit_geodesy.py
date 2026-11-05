"""Audit regression: polygon area must integrate edges (linear in lng/lat) exactly, not by a single chord per edge."""
import numpy as np

from bgv.analysis import geodesy


def test_area_independent_of_edge_densification():
    rng = np.random.default_rng(3)
    for _ in range(20):
        lng = -86.44 + rng.random(5) * 0.2
        lat = 36.9 + rng.random(5) * 0.2
        o = np.argsort(np.arctan2(lat - lat.mean(), lng - lng.mean()))
        lng, lat = lng[o], lat[o]
        t = np.linspace(0, 1, 400, endpoint=False)
        dl = (lng[:, None] + (np.roll(lng, -1) - lng)[:, None] * t).ravel()
        dt = (lat[:, None] + (np.roll(lat, -1) - lat)[:, None] * t).ravel()
        a, b = geodesy.polygon_area_m2(lng, lat), geodesy.polygon_area_m2(dl, dt)
        assert abs(a - b) / b < 2e-5


def test_acre_conversion():
    assert geodesy.fmt_area(4046.8564224, True)[0].startswith("1.00 acre")
