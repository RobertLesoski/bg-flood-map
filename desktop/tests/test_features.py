"""Feature pack (bgv/ui/ext_features.py and its data modules) + the UX audit checks (keymap, tool exclusivity, stale state)."""
import json
import types
from pathlib import Path

import numpy as np
import pytest

from bgv.data import duration, exposure, routing, session
from bgv.ui import cvd, keymap, registry, sharecard


# ------------------------------------------------------------------------------------------------ keymap / cheat sheet
def _load():
    import bgv.ui.ext_features  # noqa: F401  (registers tools / hotkeys / declarations)


def test_no_hotkey_conflicts_across_all_tool_sets():
    _load()
    assert keymap.conflicts() == []


def test_conflict_detector_finds_a_real_clash():
    b = keymap.all_bindings() + [keymap.Binding("t", (), "someone", "global", "x")]
    pairs = keymap.conflicts(b)
    assert any({p[0].owner, p[1].owner} == {"cinematic", "someone"} for p in pairs)
    # a registry hotkey on a built-in camera key is a conflict, a different modifier is not
    assert keymap.conflicts(keymap.BUILTIN + [keymap.Binding("w", (), "registry", "global", "x")])
    assert not keymap.conflicts(keymap.BUILTIN + [keymap.Binding("w", ("Control",), "registry", "global", "x")])


def test_every_registry_key_is_in_the_table_and_the_cheat_sheet_is_generated_from_it():
    _load()
    keys = {(b.key, b.mods) for b in keymap.all_bindings()}
    for h in registry.HOTKEYS:
        assert (h.key.lower() if len(h.key) == 1 else h.key, keymap.norm_mods(h.mods)) in keys
    for t in registry.TOOLS.values():
        if t.shortcut:
            assert (t.shortcut.lower(), ()) in keys
    from bgv.search import helpdoc
    keymap.install_help()
    assert helpdoc.SHORTCUTS == keymap.help_groups()
    flat = {k for _, rows in helpdoc.SHORTCUTS for k, _ in rows}
    for must in ("G", "U", "X", "V", "Ctrl+S", "Ctrl+O", "F9", "F6", "T", "L", "C", "M", "Ctrl+L", "Ctrl+P", "Ctrl+B", "1 .. 9"):
        assert must in flat, must


def test_declared_builtin_keys_exist_in_the_handler_sources():
    """The static table must not drift: each declared key of a key-hook owner appears in that handler's code."""
    src = {o: (Path(__file__).resolve().parents[1] / "bgv" / "ui" / f).read_text(encoding="utf-8")
           for o, f in (("search", "panels_search.py"), ("cinematic", "panels_cinematic.py"), ("manager", "manager.py"))}
    for b in keymap.BUILTIN:
        if b.owner in src and len(b.key) == 1 and b.key.isalpha():
            assert f'"{b.key}"' in src[b.owner], (b.owner, b.key)


# ------------------------------------------------------------------------------------------------ tool exclusivity / stale state
class _FakeUI:
    def __init__(self, app):
        from bgv.ui.manager import UiManager
        self.app, self.tool, self.toasts = app, None, types.SimpleNamespace(push=lambda *a, **k: None)
        self.activate_tool = types.MethodType(UiManager.activate_tool, self)
        self.toggle_tool = types.MethodType(UiManager.toggle_tool, self)
        self.st = types.SimpleNamespace()
        self.error_logged = set()


def _fake_app(domain="city", storm="100yr"):
    an = types.SimpleNamespace(tool="none")
    app = types.SimpleNamespace(domain=types.SimpleNamespace(key=domain), storm=types.SimpleNamespace(name=storm),
                                settings={}, loading=False, storms_available=[storm], playing=False, t=12.0)
    app.get_layer = lambda n: an if n == "analysis" else None
    app.report_error = lambda *a: None
    return app, an


def _tick(ui, monkeypatch):
    from bgv.ui import ext_features as ef
    monkeypatch.setattr(ef, "apply_access", lambda ui: None)
    monkeypatch.setattr(ef.keymap, "install_help", lambda: None)
    monkeypatch.setattr(ef.imgui, "is_key_pressed", lambda *a: False)
    ef.arbiter(ui)


def test_only_one_click_tool_at_a_time(monkeypatch):
    _load()
    app, an = _fake_app()
    ui = _FakeUI(app)
    _tick(ui, monkeypatch)
    ui.activate_tool("route")                    # registry tool on, then the analysis measure tool is switched on
    an.tool = "measure_line"
    _tick(ui, monkeypatch)
    assert ui.tool is None and an.tool == "measure_line"
    ui.activate_tool("duration")                 # registry tool switched on while the analysis tool runs: analysis leaves
    _tick(ui, monkeypatch)
    assert ui.tool is not None and ui.tool.name == "duration" and an.tool == "none"
    ui.activate_tool(None)


def test_domain_switch_clears_route_tool_and_old_ids(monkeypatch):
    _load()
    from bgv.ui import ext_features as ef
    app, an = _fake_app()
    ui = _FakeUI(app)
    _tick(ui, monkeypatch)
    f = ef.F(ui)
    ui.activate_tool("route")
    f.route_pts, f.dur_target, f.expo_ids = [(-86.4, 36.9)], ("bld", 5), np.arange(3)
    app.domain = types.SimpleNamespace(key="county")
    _tick(ui, monkeypatch)
    assert ui.tool is None and f.route_pts == [] and f.dur_target is None and f.expo_ids is None and f.graph is None


# ------------------------------------------------------------------------------------------------ routing
def _grid_graph():
    # a 3x3 lattice 100 m apart; the middle row's centre edge is the short way from W to E
    feats = []
    d = 0.0009          # ~100 m in latitude / ~80 m in longitude at 37 N
    for r in range(3):
        feats.append({"geometry": {"type": "LineString", "coordinates": [[-86.4 + c * d, 37.0 + r * d] for c in range(3)]}})
    for c in range(3):
        feats.append({"geometry": {"type": "LineString", "coordinates": [[-86.4 + c * d, 37.0 + r * d] for r in range(3)]}})
    return routing.graph_from_features(feats), d


def test_route_detours_around_deep_water_and_reports_lengths():
    g, d = _grid_graph()
    a, _ = routing.nearest_node(g, -86.4, 37.0 + d)
    b, _ = routing.nearest_node(g, -86.4 + 2 * d, 37.0 + d)
    normal = routing.shortest(g, a, b)
    depth = np.zeros(g.m)
    centre, _ = routing.nearest_node(g, -86.4 + d, 37.0 + d)
    depth[(g.ea == centre) | (g.eb == centre)] = 0.5            # flood the centre junction's four edges
    dry = routing.shortest(g, a, b, routing.blocked_edges(g, depth, 0.3))
    assert normal.found and dry.found
    assert dry.length_m == pytest.approx(normal.length_m + 2 * d * np.pi / 180 * 6371008.8, rel=0.02)
    assert centre not in dry.nodes and centre in normal.nodes
    assert routing.path_max_depth(g, normal.nodes, depth) == pytest.approx(0.5)
    g.bridge = np.ones(g.m, bool)                               # bridges are never blocked by the water under them
    assert not routing.blocked_edges(g, depth, 0.3).any()


@pytest.mark.real
def test_route_on_real_county_data():
    from data_helpers import real_root
    from bgv import config
    from bgv.data import tiles
    root = real_root()
    pbf = root / "kentucky-latest.osm.pbf"
    if not pbf.is_file():
        pytest.skip("no PBF")
    bbox = (-86.759981, 36.610125, -86.100111, 37.229991)
    g = routing.load_osm_graph(pbf, bbox, config.cache_dir() / "routing")
    assert g.n > 200_000 and g.bridge.sum() > 100
    svc = tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {})
    if svc.tileset("100yr") is None:
        pytest.skip("no county tiles")
    c = routing.DepthCache(g, svc)
    r = routing.plan(g, c, (-86.4436, 36.9685), (-86.40, 36.99), "100yr", 12, 0.3)
    assert r.normal.found and r.dry.found
    assert r.dry.length_m >= r.normal.length_m - 1e-6
    assert r.dry_max_depth_m < 0.3 <= r.normal_max_depth_m      # recorded 2026-11-05: 6.41 km dry vs 6.16 km, normal crosses 106 cm
    # the drawn route's depth agrees with an independent point sample of the tiles along it
    e = c.get("100yr", 12)
    worst = int(np.argmax(np.where(g.bridge, 0, np.nan_to_num(e))))
    s = svc.depth_lnglat("100yr", float(g.s_lng[g.s_edge == worst][0]), float(g.s_lat[g.s_edge == worst][0]), 12)
    assert s.status in ("depth", "dry", "nodata")


# ------------------------------------------------------------------------------------------------ exposure
def test_exposure_filter_and_exports(tmp_path):
    cm = np.array([5, 15, 40, 70, 150, -1])
    lu = np.array(["R", "", "C", "R", "R", "R"])
    price = np.array([1e5, np.nan, 3e5, 2e5, 5e5, 1e5])
    f = exposure.ExposureFilter()
    assert list(exposure.apply(cm, lu, price, f)) == [4, 3, 2, 1]
    f.land_uses = ("R",)
    assert list(exposure.apply(cm, lu, price, f)) == [4, 3]
    f.min_value, f.max_value = 150000, 400000
    assert list(exposure.apply(cm, lu, price, f)) == [3]
    f2 = exposure.ExposureFilter(land_uses=(exposure.UNKNOWN,))
    assert list(exposure.apply(cm, lu, price, f2)) == [1]
    b = types.SimpleNamespace(cx=np.zeros(6), cy=np.zeros(6), address=np.array(["a", "b", "c", "d", "e", "f"]), land_use=lu,
                              sale_price=price)
    fr = types.SimpleNamespace(world_to_lnglat=lambda X, Y: (-86.4, 37.0))
    rows = exposure.rows([4, 1], b, cm, fr)
    p = exposure.write_csv(tmp_path / "x.csv", rows, {"storm": "100yr"})
    back = exposure.read_csv(p)
    assert [r["address"] for r in back] == ["e", "b"] and back[0]["tier"] == "severe" and back[1]["land_use"] == "unknown"
    gj = json.loads(exposure.write_geojson(tmp_path / "x.geojson", rows, {"storm": "100yr"}).read_text())
    assert gj["features"][0]["properties"]["depth_cm"] == 150 and gj["metadata"]["storm"] == "100yr"


# ------------------------------------------------------------------------------------------------ depth-duration
def test_hours_above_counts_snapshots_after_the_start():
    s = np.zeros(25)
    s[0] = 1.0                      # start state is not an hour of flooding
    s[5:10] = 0.2
    s[7:9] = 0.7
    assert duration.hours_above(s) == {10: 5, 30: 2, 60: 2}


@pytest.mark.real
def test_duration_matches_the_tiles_at_a_flooded_point():
    from data_helpers import real_root
    from bgv.data import tiles
    root = real_root()
    svc = tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {})
    if svc.tileset("100yr") is None:
        pytest.skip("no county tiles")
    lng, lat = -86.4436, 36.9685
    ser = duration.series_point(svc, "100yr", lng, lat)
    # independent: every hour read directly
    direct = [svc.depth_lnglat("100yr", lng, lat, h).m or 0.0 for h in range(1, 25)]
    assert duration.hours_above(ser)[10] == sum(d >= 0.1 for d in direct)


@pytest.mark.real
def test_hours_flooded_grid_agrees_with_point_series():
    from data_helpers import real_root
    from bgv.data import tiles
    from bgv.geo import GridFrame, lnglat_to_merc
    root = real_root()
    svc = tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {})
    if svc.tileset("100yr") is None:
        pytest.skip("no county tiles")
    x, y = lnglat_to_merc(-86.4436, 36.9685)
    fr = GridFrame(5, 5, 20.0, 16.0, x - 50, y - 50, x + 50, y + 50)      # 5x5 cells around the point, centre cell on it
    g = duration.hours_flooded_grid(svc, "100yr", fr)
    ser = duration.series_point(svc, "100yr", -86.4436, 36.9685)
    assert int(g[2, 2]) == duration.hours_above(ser)[10]


def test_heat_layer_texture_encoding():
    from bgv.gpu.layers import hoursflooded
    t = hoursflooded.to_rgba(np.array([[0, 1, 24, 255]], np.uint8))
    assert list(t[0, :, 0]) == [0, 255, 255, 0] and list(t[0, :, 3]) == [255, 255, 255, 0] and t[0, 2, 2] == 255


# ------------------------------------------------------------------------------------------------ session / share card / colours
def test_session_round_trip(tmp_path):
    class Cam:
        target = np.array([10.0, 20.0, 0.0]); dist, yaw, pitch = 3000.0, 30.0, 50.0
        def set_view(self, *v):
            self.v = v
    lay = types.SimpleNamespace(name="water", enabled=True)
    app = types.SimpleNamespace(camera=Cam(), domain=types.SimpleNamespace(key="county"), storm=types.SimpleNamespace(name="25yr"),
                                t=7.0, exag=2.0, layers=[lay], settings={"ext": {"features": {"high_contrast": True}}},
                                storms_available=["25yr", "100yr"])
    app.get_layer = lambda n: lay if n == "water" else None
    app.set_storm = lambda s: setattr(app, "storm", types.SimpleNamespace(name=s))
    d = session.collect(app)
    p = session.save(tmp_path / "a", d)
    assert p.suffix == ".bgview"
    app.t, lay.enabled = 0.0, False
    app.set_storm("100yr")
    missed = session.apply(app, None, session.load(p))
    assert missed == [] and app.storm.name == "25yr" and app.t == 7.0 and lay.enabled and app.camera.v == (10.0, 20.0, 3000.0, 30.0, 50.0)
    with pytest.raises(ValueError):
        session.apply(app, None, {"format": "other"})


def test_share_card_has_header_footer_and_scale():
    rgb = np.full((300, 400, 3), 90, np.uint8)
    card = sharecard.compose(rgb, {"storm": "100yr", "hour": 12, "legend": [("Shallow", (77, 171, 247))], "metres_per_px": 10.0,
                                   "north_deg": 30.0})
    assert card.size[0] == 400 and card.size[1] > 300
    a = np.asarray(card)
    assert (a[: card.size[1] - 300] != 90).any()
    assert sharecard.nice_length(1234) == 1000 and sharecard.nice_length(260) == 200 and sharecard.nice_length(7) == 5


def test_new_legends_are_colour_blind_safe():
    from bgv.ui import ext_features as ef
    from bgv.gpu.layers import flowpaths
    assert cvd.safe([ef.ROUTE_DRY, ef.ROUTE_NORMAL])
    assert cvd.safe(list(ef.DUR_COLOURS))
    ramp = [flowpaths.ramp_colour("hours", t) for t in (0.0, 0.33, 0.66, 1.0)]
    assert cvd.safe(ramp, min_de=8.0)
    assert not cvd.safe([(0.85, 0.35, 0.3), (0.45, 0.55, 0.25)])   # a red / green pair (deutan dE 4.3) fails (the check has teeth)
