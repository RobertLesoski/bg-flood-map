"""Native tiles on a real app (offscreen, real data): the depth API on App, the probe's tile fields, the building card rows.  Skipped without a GPU or without the tiles."""
from __future__ import annotations

import time

import numpy as np
import pytest

from data_helpers import real_root

pytestmark = pytest.mark.gpu
wgpu = pytest.importorskip("wgpu")
W, H = 1280, 720


@pytest.fixture(scope="module")
def env():
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import tools
    from bgv.gpu.context import GpuContext
    from bgv.ui import bridge
    root = real_root()
    if not (root / "layers" / "cog" / "county" / "100yr" / "cog_meta.json").is_file():
        pytest.skip("no solver tiles for 100yr")
    try:
        ctx = GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")
    canvas = OffscreenRenderCanvas(size=(W, H), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        storm = "100yr"
        msaa = 4
        quiet = True
        exag = None
        sun = None
        hour = 14.0
        camera = "0,0,3000,10,50"
        at = "-86.4436,36.9903"
        ripple = None
        haze = None
    app = tools.make_app(A, ctx, fmt)
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=False)
    assert bridge.ensure_hub(app) and app._stats_hub.wait_ready(180.0)
    canvas.set_physical_size(W, H)
    canvas.request_draw(app.draw_frame)

    class Env:
        pass
    e = Env()
    e.app, e.canvas = app, canvas
    yield e
    app.shutdown()


def test_api_point_building_profile_and_info(env):
    app = env.app
    info = app.tile_info()
    assert info is not None and info.scope == "county" and 14 in info.hours and "verification pending" in info.label
    lng, lat = -86.4305, 37.0040                                   # the Barren River: wet in the 100-year storm
    t = app.depth_at(lng, lat, hour="peak", source="tiles")
    g = app.depth_at(lng, lat, hour="peak", source="grid")
    assert t.source == "tiles" and g.source == "grid" and t.status in ("depth", "dry", "nodata")
    assert g.label.startswith("flow-grid") and t.label.startswith("solver depth")
    auto = app.depth_at(lng, lat)                                   # displayed storm and hour, tiles preferred
    assert auto.source == "tiles" and auto.hour == 14
    prof = app.depth_profile(np.linspace(-86.45, -86.42, 40), np.full(40, 37.004), hour=14)
    assert prof.shape == (40,) and np.isfinite(prof).any()
    assert app.depth_at(lng, lat, storm="nope", source="tiles").status == "unavailable"


def test_building_depth_waits_for_the_index_then_matches_the_statistics_scale(env):
    from bgv.ui import bridge
    app = env.app
    svc = app.tiles
    b = bridge.buildings(app)
    assert b is not None
    first = app.building_depth(0, source="tiles")
    assert first.status in ("unavailable", "depth", "dry", "nodata")
    svc.start_index(lambda: app.domain.buildings_geojson, "100yr")
    t0 = time.time()
    while svc.index_state == "building" and time.time() - t0 < 120:
        time.sleep(0.2)
    assert svc.index_state == "ready", svc.index_msg
    cm = svc.building_depths("100yr", "peak", block=True)
    assert cm is not None and len(cm) == b.n and (cm >= 10).sum() > 100
    wet = int(np.flatnonzero(cm >= 100)[0])
    smp = app.building_depth(wet, hour="peak", source="tiles")
    assert smp.status == "depth" and smp.cm == cm[wet]
    g = app.building_depth(wet, hour="peak", source="grid")
    assert g.source == "grid" and g.m is not None and g.m > 0
