"""Audit regression: the flood timeline's flooded-building count must use the building card's all-touched footprint cells
(count of buildings whose footprint max >= 10 cm), not corner+centroid samples (which under-counted by 15-50 %)."""
import pytest

from bgv import config


def _root():
    try:
        r = config.data_root()
    except Exception:
        return None
    return r if r and (r / "layers" / "cog" / "county" / "ev6" / "cog_meta.json").is_file() else None


@pytest.mark.skipif(_root() is None, reason="county tiles absent")
def test_timeline_flooded_equals_card_statistic():
    from bgv import domain as dm
    from bgv.data import buildings as bmod, roads as rmod
    from bgv.data.tiles import TileService
    from bgv.analysis.depth_source import make_depth_source
    from bgv.analysis import timeline
    root = _root()
    dom = dm.resolve("county", root, {})
    src = make_depth_source(root=root, domain=dom)
    frame = dom.open_storm("100yr").frame
    bp, rp, _ = dom.scene_paths(frame)
    B, R = bmod.load_buildings(root, frame, geojson=bp), rmod.load_roads(root, frame, geojson=rp)
    svc = TileService(root, None, {"tiles_enabled": True})
    svc.domain_key = "county"
    svc._build_index(lambda: str(bp), svc.tileset("ev6").grid_signature())
    r = src.tiles.storm("ev6").raster(3)
    bpts, rpts = timeline.prepare_points(r, frame, B, R, svc.index)
    got = timeline.tile_hour_stats(r, frame, bpts, rpts)["flooded"]
    want = int((svc.building_depths("ev6", 3, block=True) >= 10).sum())
    assert got == want
