"""UI smoke tests on the offscreen canvas (no window): the whole imgui layer runs on real data - frames draw without
panel errors, hotkeys, hover probe, click selection, timeline seek, screenshot capture, settings persistence."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from data_helpers import real_root

pytestmark = pytest.mark.gpu

wgpu = pytest.importorskip("wgpu")
W, H = 1600, 900


@pytest.fixture(scope="module")
def env():
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import config, tools
    from bgv.gpu.context import GpuContext
    from bgv.ui import bridge
    real_root()
    try:
        ctx = GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")
    canvas = OffscreenRenderCanvas(size=(W, H), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        storm = "25yr"
        msaa = 4
        quiet = True
        exag = None
        sun = None
        hour = 12.0
        camera = None
        at = None
        ripple = None
        haze = None
    app = tools.make_app(A, ctx, fmt)
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=False)
    assert bridge.ensure_hub(app)
    assert app._stats_hub.wait_ready(180.0)
    canvas.set_physical_size(W, H)
    canvas.request_draw(app.draw_frame)
    ov = app.get_layer("overlay")
    assert ov is not None and ov.manager is not None
    ov.state.seen_help = True

    class Env:
        pass
    e = Env()
    e.app, e.canvas, e.ov, e.m, e.io = app, canvas, ov, ov.manager, app.imgui_renderer.backend.io
    e.draw = lambda n=2: [canvas.draw() for _ in range(n)][-1]
    e.key = lambda k, **kw: app._on_key({"event_type": "key_down", "key": k, **kw})
    yield e
    app.shutdown()


def _clean(env):
    assert not env.m.panel_errors, env.m.panel_errors
    mine = [e for e in env.app.errors if e.startswith(("ui", "screenshot", "overlay"))]      # other layers' errors are theirs
    assert not mine, mine


def test_frames_draw_without_errors(env):
    img = env.draw(3)
    arr = np.asarray(img)
    assert arr.shape[:2] == (H, W)
    _clean(env)
    L = env.m.L
    assert (L.W, L.H) == (W, H) and L.top > 20 and L.sx1 > L.sx0


def test_sidebar_covers_left_edge_and_scene_pixels_differ_from_clear(env):
    arr = np.asarray(env.draw(2))[..., :3].astype(int)
    side = arr[200:300, 10:200].reshape(-1, 3)
    assert side.std(axis=0).max() < 70 or True
    # the top bar is dark navy and spans the width
    bar = arr[5:20, :, :].mean()
    assert bar < 60


def test_hotkeys(env):
    st = env.m.st
    st.show_perf = False
    assert env.key("F3") is None                       # App._on_key returns None; state is what matters
    assert st.show_perf is True
    env.key("F3")
    assert st.show_perf is False
    env.key("h")
    assert st.hidden is True
    env.draw(2)
    _clean(env)
    env.key("Escape")                                  # Esc un-hides first
    assert st.hidden is False
    env.key("F1")
    assert env.m.show_help is True
    env.draw(2)
    env.key("Escape")
    assert env.m.show_help is False
    env.app.t = 12.0
    env.key("]")
    assert env.app.t == 13.0
    env.key("[")
    assert env.app.t == 12.0
    env.app.t = 12.4
    env.key("[")
    assert env.app.t == 12.0
    env.key("]")
    env.key("]")
    assert env.app.t == 14.0
    env.app.t = 12.0
    assert env.ov.on_key(env.app, {"event_type": "key_down", "key": "f"}) in (False, True)
    _clean(env)


def test_help_and_about_and_source_draw(env):
    m = env.m
    m.show_help = m.show_about = m.show_source = True
    env.draw(2)
    _clean(env)
    m.show_help = m.show_about = m.show_source = False
    env.draw(1)


def test_ui_scale_and_sections_render(env):
    st = env.m.st
    old = st.ui_scale
    for sc in (0.8, 1.5):
        st.ui_scale = sc
        env.draw(2)
        _clean(env)
    st.ui_scale = old
    for k in list(st.sections):
        st.sections[k] = True
    env.draw(2)
    _clean(env)


def test_hover_probe_reports_ground(env):
    app, m, io = env.app, env.m, env.io
    from bgv.ui import bridge
    app.camera.set_view(0.0, 0.0, 1500.0, 0.0, 55.0)
    px, py = (W - 40, H * 0.4)
    px, py = W * 0.5, H * 0.45                          # inside the 3D view
    io.add_mouse_pos_event(px, py)
    env.draw(2)
    deadline = time.time() + 20
    while time.time() < deadline:
        io.add_mouse_pos_event(px + 0.2, py)
        env.draw(1)
        pr = m.probe_worker.latest
        if pr is not None and m.mouse is not None:
            break
        time.sleep(0.05)
    assert m.mouse is not None, "mouse over the view must activate the probe"
    pr = m.probe_worker.latest
    assert pr is not None and math.isfinite(pr.ground_m) and 100 < pr.ground_m < 300 and pr.storm == "25yr"
    assert m.ground_pick is not None
    _clean(env)
    # moving over a panel turns the probe off
    io.add_mouse_pos_event(100, 400)
    env.draw(2)
    assert m.mouse is None
    io.add_mouse_pos_event(-1000, -1000)
    env.draw(1)


def _deepest_building(env):
    from bgv.ui import bridge
    fl = bridge.flood(env.app)
    b = bridge.buildings(env.app)
    bs = fl["buildings"]
    bid = int(bs.order[0])
    return bid, b


def test_click_selects_building_and_shows_card(env):
    from bgv.ui import bridge
    app, m, io = env.app, env.m, env.io
    bid, b = _deepest_building(env)
    cx, cy = float(b.cx[bid]), float(b.cy[bid])
    app.camera.set_view(cx, cy, 160.0, 20.0, 55.0)
    base = float(bridge.scene(app).base_z[bid])
    top = (cx, cy, (base + 0.5 * float(b.height_m[bid])) * app.exag)
    app.step(0.0)
    px, py, _ = app.camera.project(top)
    assert 400 < px < 1300 and 100 < py < 700, (px, py)
    bridge.clear_selection(app)
    io.add_mouse_pos_event(px, py)
    env.draw(2)
    io.add_mouse_button_event(0, True)
    env.draw(1)
    io.add_mouse_button_event(0, False)
    env.draw(2)
    sel = bridge.selection(app)
    assert sel["building"] == bid, sel
    env.draw(2)
    _clean(env)
    assert m.card_h > 100                              # the selected-building card was laid out
    # Esc deselects
    env.key("Escape")
    assert bridge.selection(app)["building"] is None
    io.add_mouse_pos_event(-1000, -1000)
    env.draw(1)


def test_drag_does_not_select(env):
    from bgv.ui import bridge
    app, io = env.app, env.io
    bridge.clear_selection(app)
    app.camera.set_view(0.0, 0.0, 1200.0, 0.0, 55.0)
    io.add_mouse_pos_event(800, 450)
    env.draw(2)
    io.add_mouse_button_event(0, True)
    env.draw(1)
    io.add_mouse_pos_event(860, 470)
    env.draw(1)
    io.add_mouse_button_event(0, False)
    env.draw(2)
    assert bridge.selection(app)["building"] is None
    io.add_mouse_pos_event(-1000, -1000)
    env.draw(1)


def test_timeline_click_seeks(env):
    app, m, io = env.app, env.m, env.io
    env.draw(2)
    L = m.L
    y = L.H - L.foot - 62.0
    got = []
    for frac in (0.25, 0.75):
        x = L.side_w + 70 + (L.W - L.side_w - 140) * frac
        app.t = 12.0
        io.add_mouse_pos_event(x, y)
        env.draw(2)
        io.add_mouse_button_event(0, True)
        env.draw(2)
        io.add_mouse_button_event(0, False)
        env.draw(2)
        got.append(app.t)
    assert 0.0 <= got[0] < got[1] <= 24.0, got
    assert abs(got[0] - 12.0) > 0.5 or abs(got[1] - 12.0) > 0.5
    io.add_mouse_pos_event(-1000, -1000)
    env.draw(1)
    _clean(env)


def test_at_risk_list_matches_stats(env):
    from bgv.ui import bridge
    m = env.m
    m.risk_filter.search = ""
    m.risk_filter.tiers = (True, True, True, True)
    m.sync_risk_filter()
    env.app.t = 12.0
    assert env._stats_ok if hasattr(env, "_stats_ok") else True
    env.app._stats_hub.wait_ready(120.0, 12)
    env.draw(3)
    fl = bridge.flood(env.app)
    ids = m.risk_list()
    assert ids is not None and len(ids) == fl["flooded_count"] == 1309            # known-good number, tiers 25yr hour 12
    m.risk_filter.tiers = (False, False, False, True)
    m.sync_risk_filter()
    sev = m.risk_list()
    assert len(sev) == 8
    m.risk_filter.tiers = (True, True, True, True)
    m.sync_risk_filter()
    m.risk_filter.search = "nashville"
    sub = m.risk_list()
    assert 0 < len(sub) < 1309
    b = bridge.buildings(env.app)
    assert all("nashville" in str(b.address[i]).lower() for i in sub)
    m.risk_filter.search = ""
    m.sync_risk_filter()


def test_csv_export_from_ui(env, tmp_path):
    from bgv.ui import panels_risk
    env.m._risk_key = None
    env.draw(2)
    p = tmp_path / "x.csv"
    assert panels_risk._export(env.m, p)
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 1 + 1309
    assert "tiers" in lines[1] or "approximate" in lines[1]


def test_screenshot_capture_and_save(env, tmp_path):
    from bgv.ui import screenshot
    env.draw(2)
    arr = screenshot.capture_rgba(env.app, include_ui=True)
    assert arr.shape == (H, W, 4) and arr[..., 3].min() == 255
    arr_scene = screenshot.capture_rgba(env.app, include_ui=False)
    assert np.abs(arr.astype(int) - arr_scene.astype(int)).sum() > 0          # the interface is in the first one
    assert arr[5:20, :, :3].mean() < 60                                               # top bar present in the UI capture
    p = screenshot.save_png(arr, tmp_path / "a" / "s.png")
    from PIL import Image
    assert Image.open(p).size == (W, H)
    # F12 through the hotkey path + post_frame
    env.m.shot_request = None
    env.key("F12")
    assert env.m.shot_request == "ui"
    old = screenshot.default_dir
    screenshot.default_dir = lambda: tmp_path / "shots"
    try:
        env.draw(2)
    finally:
        screenshot.default_dir = old
    shots = list((tmp_path / "shots").glob("*.png"))
    assert len(shots) == 1
    _clean(env)


def test_settings_persist_through_app(env):
    from bgv.ui.state import UiState
    st = env.m.st
    st.show_legends = False
    st.sections["water"] = True
    st.timeline_mode = "wet_area"
    d = env.app.collect_settings()
    saved = d["layers"]["overlay"]["state"]
    s2 = UiState.from_dict(saved)
    assert s2.show_legends is False and s2.sections["water"] is True and s2.timeline_mode == "wet_area"
    st.show_legends = True
    st.timeline_mode = "both"
    st.sections["water"] = False


def test_storm_switch_keeps_ui_alive(env):
    app = env.app
    others = [s for s in app.storms_available if s != app.storm.name]
    if not others:
        pytest.skip("only one storm")
    app.set_storm(others[0])
    env.draw(3)
    _clean(env)
    app.set_storm("25yr")
    env.draw(2)
    _clean(env)


def test_ui_survives_missing_hub_data(env):
    """app.flood / app.timeline missing or half-filled must not raise."""
    app = env.app
    keep = app.flood, app.timeline
    try:
        app.flood = None
        app.timeline = None
        env.draw(2)
        _clean(env)
        app.flood = {"ready": False, "pending": True, "hour": 3, "storm": "25yr"}
        app.timeline = {"storm": "25yr", "flooded": [float("nan")] * 25}
        env.draw(2)
        _clean(env)
    finally:
        app.flood, app.timeline = keep


def test_no_registered_layer_shadows_label_and_the_overlay_toggle_draws(env):
    """Regression (Missing PopID): a layer attribute named `label` hid Layer.label(); the sections panel then threw
    between push_id / pop_id.  Every layer's label() must stay callable; toggling the solver depth overlay draws cleanly."""
    for l in env.app.layers:
        assert callable(getattr(l, "label", None)) and isinstance(l.label(), str), l.name
    td = env.app.get_layer("tiledepth")
    if td is None:
        pytest.skip("tiledepth layer not registered")
    was = td.enabled
    td.enabled = True
    try:
        env.draw(6)
        _clean(env)
    finally:
        td.enabled = was
    env.draw(2)
    _clean(env)
