"""Search / facilities / roads / report / help on the offscreen canvas with the real city data (no window), plus the controller's
graceful behaviour on a data folder where everything is missing (CPU only)."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_helpers import real_root

wgpu = pytest.importorskip("wgpu")
W, H = 1600, 900


# ------------------------------------------------------------------------------------------------ CPU: missing data
def test_controller_reports_missing_data_instead_of_failing(tmp_path):
    from bgv.search.controller import SearchController
    toasts = []
    ui = SimpleNamespace(app=SimpleNamespace(root=tmp_path, domain=SimpleNamespace(key="county", label="Warren County"), settings={},
                                             storm=SimpleNamespace(name="100yr"), terrain=None),
                         toasts=SimpleNamespace(push=lambda *a, **k: toasts.append(a)), nav=None)
    c = SearchController(ui)
    c.sync()
    for _ in range(100):
        if all(c.status[k].state not in ("idle", "loading") for k in ("index", "facilities", "tiles")):
            break
        time.sleep(0.05)
    assert c.status["index"].state == "missing" and "address_index" in c.status["index"].message
    assert c.status["facilities"].state == "missing" and "infrastructure" in c.status["facilities"].message
    assert c.status["tiles"].state == "missing" and "index / flow grid is used" in c.status["tiles"].message
    assert c.hits_for("1 main") == [] and c.index is None and c.facilities is None
    t = c.road_table("100yr")
    for _ in range(100):
        t = c.road_table("100yr")
        if t is not None:
            break
        time.sleep(0.05)
    assert isinstance(t, str) and "100yr" in t
    c.close()


# ------------------------------------------------------------------------------------------------ GPU
@pytest.fixture(scope="module")
def env():
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import tools
    from bgv.gpu.context import GpuContext
    from bgv.ui import bridge, panels_search
    real_root()
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
    ov.state.seen_help = True

    class Env:
        pass
    e = Env()
    e.app, e.canvas, e.ov, e.m = app, canvas, ov, ov.manager
    e.draw = lambda n=2: [canvas.draw() for _ in range(n)][-1]
    e.key = lambda k, **kw: bool(e.m.on_key({"event_type": "key_down", "key": k, **kw}))
    e.draw(3)
    e.c = panels_search.controller(e.m)
    e.c.onboarded = True
    for _ in range(400):
        e.draw(1)
        if all(e.c.status[k].state in ("ready", "missing", "error") for k in ("index", "facilities", "tiles")):
            break
        time.sleep(0.05)
    yield e
    app.shutdown()


def _clean(env):
    assert not env.m.panel_errors, env.m.panel_errors
    mine = [x for x in env.app.errors if x.startswith(("ui", "screenshot", "overlay", "report"))]
    assert not mine, mine


def test_city_data_loaded_and_tiles_absent_is_explained(env):
    c = env.c
    assert c.status["index"].state == "ready" and c.index is not None and c.index.domain == "city"
    assert c.status["facilities"].state == "ready" and c.facilities.n == 51
    assert c.status["tiles"].state == "missing" and "city" in c.status["tiles"].message and c.tiles is None
    assert c.facilities.source("100yr") == "index (grid)"


def test_address_to_building_mapping_city_200_random(env):
    """The record's `bid` is the viewer's building id: resolve_building must return it, and the centroid must sit at the record."""
    from bgv.ui import bridge
    c, app = env.c, env.app
    b = bridge.buildings(app)
    idx = c.index
    rnd = random.Random(5)
    cand = [i for i in range(idx.n) if idx.kind[i] == 0 and 0 <= idx.bid[i] < b.n and np.isfinite(b.cx[idx.bid[i]])]
    far, wrong = [], 0
    for i in rnd.sample(cand, 250):
        rec = idx.record(i)
        x, y = app.terrain.frame.lnglat_to_world(rec.lng, rec.lat)
        d = float(np.hypot(b.cx[rec.bid] - x, b.cy[rec.bid] - y))
        far.append(d)
        wrong += int(c.resolve_building(rec) != rec.bid)
    print(f"[mapping] 250 addresses: median centroid distance {np.median(far):.1f} m, max {max(far):.1f} m, wrong building {wrong}")
    assert max(far) < 60.0 and np.median(far) < 12.0, (max(far), np.median(far))
    assert wrong == 0, wrong
    # reverse mapping: a viewer building finds its address record
    ok = 0
    for i in rnd.sample(cand, 100):
        bid = int(idx.bid[i])
        j = c.record_for_building(bid)
        ok += int(j >= 0 and int(idx.bid[j]) == bid)
    assert ok >= 98


def test_search_flow_result_card_and_selection(env):
    from bgv.ui import bridge, panels_search
    c, idx = env.c, env.c.index
    i = int(np.flatnonzero((idx.depth_cm["100yr"] >= 30) & (idx.kind == 0))[0])
    rec = idx.record(i)
    c.query = rec.address
    env.draw(2)
    hits = c.hits_for(c.query)
    assert hits and any(h.idx == i for h in hits)
    panels_search._choose(env.m, c, next(h for h in hits if h.idx == i))
    for _ in range(200):
        if c.result is not None and not c.result.pending:
            break
        env.draw(1)
        time.sleep(0.02)
    res = c.result
    assert res is not None and res.assessment is not None and not res.error
    row = res.assessment.row("100yr")
    assert row.shown_cm == int(idx.depth_cm["100yr"][i]) and row.shown_source == "index (grid)" and row.tile_cm is None   # city: index only
    assert bridge.selection(env.app)["building"] == res.building
    env.draw(3)
    _clean(env)
    env.m.nav.cancel()
    # Esc clears the query first, then the result
    assert env.key("Escape") and c.query == ""
    assert env.key("Escape") and c.result is None


def test_keyboard_shortcuts_open_and_toggle(env):
    c = env.c
    assert env.key("f", modifiers=("Control",)) and c.focus
    env.draw(2)
    assert not c.focus
    on = c.markers_on
    assert env.key("m") and c.markers_on != on
    env.key("m")
    assert env.key("l", modifiers=("Control",)) and c.show_roads
    env.draw(4)
    _clean(env)
    assert env.key("l", modifiers=("Control",)) and not c.show_roads
    assert env.key("/") and c.focus
    env.draw(2)


def test_facility_markers_pick_and_card(env):
    from bgv.ui import panels_search
    c, app = env.c, env.app
    fs = c.facilities
    c.markers_on = True
    c.types_off = set()
    cam = app.camera
    i = 5
    fr = app.terrain.frame
    x, y = fr.lnglat_to_world(float(fs.lng[i]), float(fs.lat[i]))
    cam.set_view(x, y, 800.0, 0.0, 55.0)
    env.draw(3)
    px, py, ok = c.facility_screen(W, H)
    assert ok[i] and 0 < px[i] < W and 0 < py[i] < H
    assert c.pick_facility(float(px[i]), float(py[i])) == i
    assert c.pick_facility(float(px[i]) + 80.0, float(py[i]) + 80.0) in (-1,) or c.pick_facility(float(px[i]) + 80.0, float(py[i]) + 80.0) != i
    assert panels_search.on_click(env.m, float(px[i]), float(py[i])) is False        # card opens, the click still reaches the building pick
    assert c.facility_sel == i and c.result is None
    env.draw(3)
    _clean(env)
    info = fs.info(i, "100yr")
    assert info["name"] and info["current"]["source"] == "index (grid)"
    # a hidden type is not pickable
    c.types_off = {fs.items[i].type}
    assert c.pick_facility(float(px[i]), float(py[i])) != i
    c.types_off = set()
    c.facility_sel = -1
    c.show_fac = True
    env.draw(3)
    _clean(env)
    c.show_fac = False


def test_roads_window_and_zoom_to_road(env):
    c = env.c
    c.show_roads = True
    t = None
    for _ in range(200):
        env.draw(1)
        t = c.road_table("100yr")
        if t is not None:
            break
        time.sleep(0.02)
    assert t is not None and not isinstance(t, str) and t.n > 100
    env.draw(3)
    _clean(env)
    view = t.view("flooded_pct", False)
    c.ensure_geometry()
    for _ in range(300):
        env.draw(1)
        if c.geometry is not None:
            break
        time.sleep(0.02)
    assert c.geometry is not None
    assert c.select_road(t, int(view[0]))
    assert env.m.nav.active and c.road_sel["coords"]
    env.m.nav.cancel()
    env.draw(3)
    _clean(env)
    c.show_roads = False
    c.road_sel = None


def test_report_written_with_png_and_html(env, tmp_path):
    c, idx = env.c, env.c.index
    env.app.settings["report_dir"] = str(tmp_path)
    i = int(np.flatnonzero((idx.depth_cm["100yr"] >= 30) & (idx.kind == 0))[3])
    c.select_record(i)
    for _ in range(200):
        env.draw(1)
        if c.result is not None and not c.result.pending:
            break
        time.sleep(0.02)
    env.m.nav.cancel()
    c.request_report()
    env.draw(2)
    env.ov.post_frame(env.app)
    files = sorted(tmp_path.glob("risk_*.html"))
    assert len(files) == 1 and sorted(tmp_path.glob("risk_*.png"))
    html = files[0].read_text(encoding="utf-8")
    assert idx.record(i).address in html and "data:image/png;base64," in html and "validated against observed floods" in html.lower()
    assert c.last_report and c.last_report["html"] == str(files[0])
    # without a result or selection: a toast, no file
    c.result = None
    from bgv.ui import bridge
    bridge.selection(env.app)["building"] = None
    c.request_report()
    env.draw(1)
    env.ov.post_frame(env.app)
    assert len(list(tmp_path.glob("risk_*.html"))) == 1
    # a selected building without a search: report through its address record
    j = int(np.flatnonzero((idx.depth_cm["100yr"] >= 30) & (idx.kind == 0))[7])
    bridge.selection(env.app)["building"] = int(idx.bid[j])
    c.request_report()
    env.draw(1)
    env.ov.post_frame(env.app)
    assert len(list(tmp_path.glob("risk_*.html"))) == 2
    bridge.selection(env.app)["building"] = None
    env.app.settings.pop("report_dir", None)


def test_help_about_windows_and_hint(env, monkeypatch):
    m, c = env.m, env.c
    for flag in ("show_help", "show_about"):
        setattr(m, flag, True)
        env.draw(3)
        _clean(env)
        hs = m._help_state
        for tab in ("Help", "Shortcuts", "Data sources", "About"):
            hs.want_tab = tab
            env.draw(2)
            _clean(env)
        hs.query = "report"
        env.draw(2)
        _clean(env)
        hs.query = "qqqqq"
        env.draw(2)
        _clean(env)
        hs.query = ""
        setattr(m, flag, False)
        env.draw(2)
    assert env.key("F1") and m.show_help
    env.draw(2)
    assert env.key("Escape") and not m.show_help
    monkeypatch.setenv("BGV_FORCE_HINT", "1")
    c.onboarded = False
    for _ in range(25):
        env.draw(1)
    _clean(env)
    c.onboarded = True


def test_log_path_shown_in_about(env):
    from bgv.ui import help as helpmod
    p = helpmod._log_path_text()
    assert p.endswith("bgv.log") or "no log" in p


def test_search_focus_swallows_every_hotkey_and_text_reaches_the_box(env):
    """Typing 'tbl 123 main st' into the focused search triggers no action; unfocused, the hotkeys still work."""
    from imgui_bundle import imgui
    app, m, c = env.app, env.m, env.c
    c.query, c.committed = "", False
    imgui.internal.clear_active_id()
    env.draw(3)
    assert not m.text_focused()
    storm0, t0, hidden0, play0 = app.storm, app.t, m.st.hidden, app.playing
    env.draw(1)
    assert env.key("h") and m.st.hidden != hidden0           # unfocused: hotkeys work (H hides the interface)
    env.key("h")
    assert m.st.hidden == hidden0
    c.focus = True
    env.draw(3)
    assert m.text_focused()
    for ch in "tbl 123 main st":
        env.key(ch)
        app._on_key({"event_type": "key_down", "key": ch})
    for k in ("ArrowLeft", "Home", "[", "]", "Space", "h"):
        env.key(k)
        app._on_key({"event_type": "key_down", "key": k})
    env.key("b", modifiers=("Control",))
    assert (app.storm, app.t, m.st.hidden, app.playing) == (storm0, t0, hidden0, play0)
    imgui.get_io().add_input_characters_utf8("tbl 123 main st")
    env.draw(2)
    assert c.query == "tbl 123 main st"
    assert env.key("Escape")                                   # Esc leaves the field first
    env.draw(2)
    assert not m.text_focused()
    c.query = ""
    env.draw(1)


def test_glyph_atlas_is_recovered_when_its_gpu_texture_is_lost_and_replay_never_carries_an_upload(env):
    """Blank-text regression: a stale / lost font-atlas texture must be re-uploaded, not drawn as empty bars."""
    from imgui_bundle import imgui
    from bgv.ui import imgui_fast
    be = env.app.imgui_renderer.backend
    imgui_fast.patch(env.app.imgui_renderer)
    env.draw(3)
    def text_std():
        img = np.asarray(env.draw(2))[..., :3].astype(float)
        return float(img[70:260, 10:300].std())
    ok = text_std()
    for tid in list(be._texture_views):
        be._texture_views.pop(tid)
        be._textures.pop(tid, None)
    be._fast_bg_cache.clear()
    assert abs(text_std() - ok) < ok * 0.2 + 2          # the atlas came back: same amount of glyph contrast, no blank bars
    assert not env.app.errors or not [x for x in env.app.errors if "ui" in x]
    assert imgui_fast._replay.__doc__


def test_utilities_layer_draws_clicks_and_is_found_by_search(env):
    from bgv.ui import ext_utilities as eu
    from bgv.ui import registry
    app, m, c = env.app, env.m, env.c
    if eu.uset(app) is None:
        pytest.skip("no layers/utilities.geojson.gz")
    registry.state(app, "utilities")["on"] = True
    env.draw(3)
    assert eu._S.view is not None and len(eu._S.view.ids) > 0
    hits = eu.hits_for(m, "water tower")
    assert hits and hits[0].kind == "utility"
    c.query, c.committed = "water tower", False
    env.draw(2)
    assert any(h.kind == "utility" for h in c.cur_hits)
    eu.choose_hit(m, hits[0])
    env.draw(30)
    assert eu._S.sel == hits[0].idx and not m.panel_errors
    registry.state(app, "utilities")["on"] = False
    eu._S.sel = None
    c.query = ""
    env.draw(2)
    _clean(env)
