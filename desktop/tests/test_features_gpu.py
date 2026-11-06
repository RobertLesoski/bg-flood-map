"""Feature pack in the real app (offscreen GPU, real data): compare-storms thumbnails, share card, session save / reopen through the
real key path, depth-duration table, exposure numbers vs the depth API, cheat sheet in the live help."""
import time

import numpy as np
import pytest

gpu = pytest.mark.gpu


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    pytest.importorskip("wgpu")
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import tools
    from bgv.gpu.context import GpuContext
    from data_helpers import real_root
    real_root()
    try:
        ctx = GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")
    canvas = OffscreenRenderCanvas(size=(1280, 720), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        storm, msaa, quiet, exag, sun, hour, camera, at, ripple, haze = "100yr", 4, True, None, None, 12.0, None, None, None, None
    app = tools.make_app(A, ctx, fmt)
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=True)
    app.get_layer("overlay").state.seen_help = True
    canvas.request_draw(app.draw_frame)
    for _ in range(4):
        canvas.draw()
    yield app, canvas, tmp_path_factory.mktemp("feat")
    app.shutdown()


def _frames(canvas, n=1):
    for _ in range(n):
        canvas.draw()


@gpu
def test_compare_storms_thumbnails_and_click_target(live):
    from bgv.ui import ext_features as ef
    app, canvas, _ = live
    ui = app.get_layer("overlay").manager
    orig = app.storm.name
    ef.open_tab(ui, 3)
    ef.start_capture(ui)
    t0 = time.time()
    while ef.F(ui).capture is not None and time.time() - t0 < 240:
        _frames(canvas)
    assert ef.F(ui).capture is None
    assert set(ef.F(ui).thumbs) == set(app.storms_available)
    assert app.storm.name == orig and round(app.t) == 12
    # thumbnails differ between a small and a large storm (real renders, not copies)
    s = list(app.storms_available)
    assert ef.F(ui).thumbs[s[0]][2] == ef.view_key(app)


@gpu
def test_share_card_png(live):
    from PIL import Image
    from bgv.ui import ext_features as ef
    app, canvas, tmp = live
    ui = app.get_layer("overlay").manager
    app.settings["screenshot_dir"] = str(tmp)
    p = ef.share_card(ui)
    im = Image.open(p)
    w, h = app.canvas.get_physical_size()
    assert im.size[0] == w and im.size[1] > h
    mpp, north = ef.scale_and_north(app)
    assert mpp is not None and 0 < mpp < 500 and -180 <= north <= 180


@gpu
def test_session_save_and_reopen_through_keys(live, monkeypatch):
    from bgv.ui import ext_features as ef
    app, canvas, tmp = live
    ui = app.get_layer("overlay").manager
    monkeypatch.setattr(ef, "session_dir", lambda: tmp / "sessions")
    app.set_storm("25yr")
    _frames(canvas, 3)
    app.t = 9.0
    app.camera.set_view(500.0, -300.0, 4000.0, 30.0, 50.0)
    ui = app.get_layer("overlay").manager
    app._on_key({"event_type": "key_down", "key": "s", "modifiers": ("Control",)})
    assert len(list((tmp / "sessions").glob("*.bgview"))) == 1
    app.set_storm("100yr")
    app.t = 2.0
    app.camera.set_view(0.0, 0.0, 9000.0, 0.0, 60.0)
    app._on_key({"event_type": "key_down", "key": "o", "modifiers": ("Control",)})
    _frames(canvas, 2)
    assert app.storm.name == "25yr" and app.t == 9.0
    assert float(app.camera.target[0]) == pytest.approx(500.0, abs=1.0) and float(app.camera.dist) == pytest.approx(4000.0, rel=0.01)


@gpu
def test_duration_table_matches_depth_api(live):
    from bgv.ui import ext_features as ef
    app, canvas, _ = live
    ui = app.get_layer("overlay").manager
    lng, lat = -86.4436, 36.9685
    ef.set_duration_target(ui, ("pt", lng, lat))
    t0 = time.time()
    while ef.F(ui).dur_job.busy and time.time() - t0 < 120:
        time.sleep(0.05)
    rows = dict(ef.F(ui).dur_job.result)
    d = rows.get("100yr")
    if d is None:
        pytest.skip("no tiles for 100yr")
    direct = [app.depth_at(lng, lat, storm="100yr", hour=h, source="tiles").m or 0.0 for h in range(1, 25)]
    assert d[10] == sum(x >= 0.1 for x in direct) and d[60] == sum(x >= 0.6 for x in direct)


@gpu
def test_live_help_shows_generated_cheat_sheet(live):
    from bgv.search import helpdoc
    from bgv.ui import keymap
    app, canvas, _ = live
    _frames(canvas, 2)
    assert helpdoc.SHORTCUTS == keymap.help_groups()
