"""Audit regression: a building's depth series on the tiles must equal the building card (all-touched footprint max)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")
GJ = ROOT / "layers" / "buildings.geojson"
pytestmark = pytest.mark.skipif(not (GJ.is_file() and (ROOT / "layers/cog/county/100yr/cog_meta.json").is_file()),
                                reason="county tiles / buildings absent")


def test_series_tiles_samples_match_card():
    from bgv.data.tiles import TileService
    from bgv.analysis.depth_source import TileDepthSource
    from bgv.analysis import series
    svc = TileService(ROOT, None, {"tiles_enabled": True})
    svc.domain_key = "county"
    ts = svc.tileset("100yr")
    svc._build_index(lambda: str(GJ), ts.grid_signature())
    assert svc.index_state == "ready"
    cm = svc.building_depths("100yr", 12, block=True)
    wet = np.flatnonzero(cm >= 2)
    rng = np.random.default_rng(0)
    stub = SimpleNamespace(index=SimpleNamespace(pixels_of=lambda i: np.zeros(0, np.int64)))
    src = TileDepthSource(ROOT / "layers" / "cog" / "county")
    for i in rng.choice(wet, 40, replace=False):
        t = series.building_target(stub, int(i), None, "x", tile_index=svc.index, tile_sig=svc.index_sig)
        lng, lat = t.arrays("tiles")
        d = src.depth_many(lng, lat, "100yr", 12.0, method="nearest").max()
        assert abs(d - cm[i] / 100.0) < 1e-6, (i, d, cm[i])
