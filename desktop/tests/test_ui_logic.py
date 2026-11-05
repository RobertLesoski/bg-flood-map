"""UI logic tests (pure CPU, no imgui frame): units, compass, scale bar, at-risk filtering, CSV export, settings round trip,
layout, timeline series, theme ramps, probe sampling."""
from __future__ import annotations

import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest

from bgv.ui import atrisk, theme, units
from bgv.ui.manager import compute_layout
from bgv.ui.state import SECTIONS, UiState
from bgv.ui.timeline import chart_series, meta_series, value_at


# ------------------------------------------------------------------------------------------------ units
def test_unit_conversions():
    assert units.m_to_ft(1.0) == pytest.approx(3.28084, rel=1e-5)
    assert units.ft_to_m(units.m_to_ft(2.5)) == pytest.approx(2.5)
    assert units.ms_to_mph(1.0) == pytest.approx(2.23694, rel=1e-5)
    assert units.mph_to_ms(units.ms_to_mph(3.3)) == pytest.approx(3.3)
    assert units.fmt_area_km2(1.0, True).startswith("0.39 mi")


@pytest.mark.parametrize("u,v,deg", [(0, 1, 0), (1, 0, 90), (0, -1, 180), (-1, 0, 270), (1, 1, 45), (-1, -1, 225)])
def test_bearing_is_compass_of_motion(u, v, deg):
    assert units.bearing_deg(u, v) == pytest.approx(deg)


def test_bearing_zero_vector_and_cardinals():
    assert units.bearing_deg(0.0, 0.0) == 0.0
    assert units.cardinal(0) == "N" and units.cardinal(359) == "N" and units.cardinal(45) == "NE"
    assert units.cardinal(90) == "E" and units.cardinal(202.5) == "SSW" and units.cardinal(270, 8) == "W"
    assert units.cardinal(11.2) == "N" and units.cardinal(11.3) == "NNE"


def test_screen_arrow_follows_camera_yaw():
    # looking north: east-moving water points right on screen
    dx, dy = units.screen_arrow(1.0, 0.0, 0.0)
    assert dx == pytest.approx(1.0) and dy == pytest.approx(0.0, abs=1e-9)
    # looking east: east-moving water flows away from the camera (up the screen), north is to the left
    dx, dy = units.screen_arrow(1.0, 0.0, 90.0)
    assert dx == pytest.approx(0.0, abs=1e-9) and dy == pytest.approx(-1.0)
    dx, dy = units.screen_arrow(0.0, 1.0, 90.0)
    assert dx == pytest.approx(-1.0) and dy == pytest.approx(0.0, abs=1e-9)
    # looking south: north-moving water comes toward the camera (down the screen)
    dx, dy = units.screen_arrow(0.0, 1.0, 180.0)
    assert dx == pytest.approx(0.0, abs=1e-9) and dy == pytest.approx(1.0)
    dx, dy = units.screen_arrow(0.0, 1.0, 0.0)           # north-moving, camera looks north: up on screen
    assert dy == pytest.approx(-1.0)
    assert units.screen_arrow(0.0, 0.0, 33.0) == (0.0, 0.0)


def test_north_arrow_rotation():
    assert units.north_angle(0.0) == 0.0
    assert units.north_angle(90.0) == 270.0          # camera looks east -> north is to the left (270 deg clockwise)
    assert units.north_angle(360.0 + 10.0) == pytest.approx(350.0)


def test_hour_labels():
    assert units.hour_label(12.0) == "12:00" and units.hour_label(12.5) == "12:30" and units.hour_label(0) == "00:00"
    assert units.hour_label(24.0) == "24:00" and units.hour_label(7.25) == "07:15"
    assert units.snapshot_hour(12.5) == 13 and units.snapshot_hour(12.49) == 12 and units.snapshot_hour(30) == 24
    assert units.hour_of_24(12.0) == "Hour 12 of 24"


def test_depth_text_honesty_for_tiers():
    assert units.depth_text(0.456, approx=False) == "0.46 m (1.5 ft)"
    assert units.depth_text(0.456, approx=True) == "~0.46 m (~1.5 ft)"
    # the severe tier of a tiers preview is open ended: never print a precise depth for it
    assert units.depth_text(5.9, tier=4, approx=True).startswith(">= 1.2 m")
    assert units.depth_short(5.9, tier=4, approx=True) == ">=1.2 m"
    assert units.depth_short(5.9, tier=4, approx=False) == "5.90 m"       # solver rasters: the number is real
    assert units.depth_short(0.0) == "dry"
    assert units.tier_range_text(1) == "0.1-0.3 m" and units.tier_range_text(4) == ">= 1.2 m"


def test_speed_and_money_text():
    assert units.speed_text(1.0) == "1.00 m/s (2.2 mph)"
    assert units.fmt_money(float("nan")) == "-" and units.fmt_money(0) == "-"
    assert units.fmt_money(175000) == "$175,000" and units.fmt_money(5.75e6) == "$5.75M"
    assert units.land_use_label("R") == "Residential (R)" and units.land_use_label("M") == "code M" and units.land_use_label("") == "-"


def test_scale_bar_lengths_are_round_and_fit():
    for target in (3.0, 17.0, 130.0, 777.0, 4200.0, 90000.0):
        for imp in (False, True):
            length, label = units.nice_length(target, imp)
            assert 0 < length <= target * 1.0001
            assert length > target * 0.19                  # 1-2-5 steps never lose more than a factor 2.5
            assert label.split()[0].replace(".", "").isdigit()
    assert units.nice_length(130.0)[1] == "100 m"
    assert units.nice_length(2600.0)[1] == "2 km"
    assert units.nice_length(130.0 * 0.3048 * 5, True)[1].endswith("ft")
    assert units.nice_length(5000.0, True)[1].endswith("mi")
    mpp = units.metres_per_pixel(1000.0, math.radians(45), 1000)
    assert mpp == pytest.approx(2 * 1000 * math.tan(math.radians(22.5)) / 1000)


# ------------------------------------------------------------------------------------------------ at-risk
def _fake_data(n=12):
    depth = np.array([0.0, 0.05, 0.12, 0.25, 0.35, 0.5, 0.7, 1.0, 1.3, 2.5, 0.2, 0.9], np.float32)[:n]
    tier = np.searchsorted(np.array([0.1, 0.3, 0.6, 1.2]), depth, side="right").astype(np.uint8)
    flooded = depth >= 0.1
    ids = np.flatnonzero(flooded)
    order = ids[np.argsort(-depth[ids], kind="stable")].astype(np.int32)
    stats = SimpleNamespace(depth_m=depth, tier=tier, flooded=flooded, order=order, speed_ms=np.linspace(0, 1, n).astype(np.float32),
                            flooded_count=int(flooded.sum()))
    addr = np.array(["1 MAIN STREET", "2 MAIN STREET", "10 OAK AVENUE", "5 RIVER ROAD", "7 BARREN DRIVE", "9 MAIN STREET",
                     "12 ELM COURT", "14 ELM COURT", "20 CHERRY LANE", "22 CHERRY LANE", "3 PINE WAY", ""])[:n]
    land = np.array(["R", "R", "C", "R", "V", "I", "R", "M", "R", "C", "P", "R"])[:n]
    b = SimpleNamespace(n=n, address=addr, land_use=land, sale_price=np.arange(n, dtype=np.float32) * 1000 + 5000,
                        height_m=np.full(n, 6.0, np.float32), area_m2=np.full(n, 120.0, np.float32),
                        cx=np.linspace(-500, 500, n).astype(np.float32), cy=np.linspace(-300, 300, n).astype(np.float32))
    return stats, b


def test_filter_default_is_deepest_first_and_only_flooded():
    stats, b = _fake_data()
    ids = atrisk.filter_ids(stats, b, atrisk.AtRiskFilter())
    d = stats.depth_m[ids]
    assert list(ids) == [9, 8, 7, 11, 6, 5, 4, 3, 10, 2]          # 0 and 1 are dry (< 0.1 m)
    assert np.all(np.diff(d) <= 0)


def test_filter_by_tier():
    stats, b = _fake_data()
    only_severe = atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(tiers=(False, False, False, True)))
    assert list(only_severe) == [8, 9] or list(only_severe) == [9, 8]
    assert set(stats.tier[only_severe]) == {4}
    shallow_mod = atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(tiers=(True, True, False, False)))
    assert set(stats.tier[shallow_mod]) <= {1, 2}
    none = atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(tiers=(False, False, False, False)))
    assert len(none) == 0


def test_filter_search_matches_address_and_land_use_all_words():
    stats, b = _fake_data()
    f = atrisk.AtRiskFilter(search="elm")
    assert sorted(atrisk.filter_ids(stats, b, f)) == [6, 7]
    assert sorted(atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(search="  CHERRY   lane "))) == [8, 9]
    assert list(atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(search="elm court 14"))) == [7]
    assert len(atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(search="nothing here"))) == 0
    # land use is searchable too ('c' alone also matches street names, so combine with a word)
    assert sorted(atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(search="cherry c"))) == [8, 9]
    f2 = atrisk.AtRiskFilter(min_depth_m=0.9)
    assert sorted(atrisk.filter_ids(stats, b, f2)) == [7, 8, 9, 11]


def test_filter_handles_missing_data_and_limit():
    stats, b = _fake_data()
    assert len(atrisk.filter_ids(None, b, atrisk.AtRiskFilter())) == 0
    assert len(atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(), limit=3)) == 3
    empty = SimpleNamespace(depth_m=np.zeros(3, np.float32), tier=np.zeros(3, np.uint8), order=np.zeros(0, np.int32))
    assert len(atrisk.filter_ids(empty, b, atrisk.AtRiskFilter())) == 0


def test_filter_key_changes_with_every_field():
    a = atrisk.AtRiskFilter()
    assert a.key() == atrisk.AtRiskFilter().key()
    assert a.key() != atrisk.AtRiskFilter(search="x").key()
    assert a.key() != atrisk.AtRiskFilter(tiers=(True, True, True, False)).key()
    assert atrisk.AtRiskFilter(search=" Elm ").key() == atrisk.AtRiskFilter(search="elm").key()


def test_csv_export_roundtrip(tmp_path):
    stats, b = _fake_data()
    ids = atrisk.filter_ids(stats, b, atrisk.AtRiskFilter(tiers=(False, True, True, True)))
    frame = SimpleNamespace(world_to_lnglat=lambda x, y: (-86.44 + x * 1e-5, 36.99 + y * 1e-5))
    p = tmp_path / "sub" / "out.csv"
    n = atrisk.export_csv(p, ids, b, stats, storm="25yr", hour=12, source="tiers", frame=frame)
    assert n == len(ids) and p.is_file() and not list(tmp_path.glob("**/*.tmp"))
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                      # BOM so Excel reads UTF-8
    rows = list(csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == n
    assert list(rows[0].keys()) == atrisk.CSV_COLUMNS
    assert [int(r["building_id"]) for r in rows] == list(ids)
    top = rows[0]
    assert top["rank"] == "1" and top["tier"] == "Severe" and top["depth_note"] == ">=1.2 m"
    assert top["storm"] == "25yr" and top["hour"] == "12" and "approximate" in top["depth_source"]
    assert float(top["depth_ft"]) == pytest.approx(float(top["depth_m"]) * 3.28084, abs=0.01)
    assert top["address"] == "22 CHERRY LANE" and top["land_use"] == "C"
    assert -87 < float(top["lng"]) < -86 and 36 < float(top["lat"]) < 37
    # rasters source: precise number, no tilde / bound
    atrisk.export_csv(p, ids, b, stats, storm="25yr", hour=12, source="rasters", frame=None)
    rows = list(csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines()))
    assert rows[0]["depth_note"] == "2.50 m" and rows[0]["lng"] == "" and "solver" in rows[0]["depth_source"]


def test_csv_export_empty_list(tmp_path):
    stats, b = _fake_data()
    p = tmp_path / "e.csv"
    assert atrisk.export_csv(p, np.zeros(0, np.int32), b, stats, storm="5yr", hour=3, source="rasters") == 0
    assert p.read_text(encoding="utf-8-sig").strip().split(",")[0] == "rank"


def test_export_names_and_dirs():
    name = atrisk.default_export_name("25yr", 7)
    assert name.startswith("at_risk_25yr_h07_") and name.endswith(".csv")
    assert atrisk.default_export_dir().name == "BG_Flood_Exports"


def test_row_for_formats():
    stats, b = _fake_data()
    r = atrisk.row_for(11, 4, stats, b)
    assert r.address.startswith("Building #") and r.tier == 3 and r.rank == 4          # empty address -> placeholder
    r2 = atrisk.row_for(2, 1, stats, b)
    assert r2.address == "10 OAK AVENUE" and r2.depth_m == pytest.approx(0.12)


# ------------------------------------------------------------------------------------------------ settings
def test_state_defaults_roundtrip():
    s = UiState()
    s.show_perf = True
    s.sections["stats"] = False
    s.ui_scale = 1.25
    s.timeline_mode = "flooded"
    s.risk_tiers = [True, False, True, False]
    s.risk_search = "main"
    s.imperial = True
    s.quality_preset = "Ultra"
    s.hidden = True                                   # not persisted
    d = s.to_dict()
    assert "hidden" not in d and "fullscreen" not in d
    import json
    s2 = UiState.from_dict(json.loads(json.dumps(d)))
    assert s2.to_dict() == d
    assert s2.show_perf and not s2.sections["stats"] and s2.ui_scale == 1.25 and s2.timeline_mode == "flooded"
    assert s2.risk_tiers == [True, False, True, False] and s2.risk_search == "main" and s2.imperial
    assert s2.quality_preset == "Ultra" and not s2.hidden


def test_state_ignores_garbage():
    s = UiState.from_dict({"ui_scale": "huge", "sections": {"stats": 0, "bogus": 1}, "timeline_mode": "nope",
                           "risk_tiers": [1, 2], "show_perf": "yes", "quality_preset": 5, "sidebar_w": 99999,
                           "timeline_h": float("nan"), "unknown": 1})
    assert s.ui_scale == 1.0 and s.timeline_mode == "both" and s.risk_tiers == [True] * 4
    assert s.sections["stats"] is False and "bogus" not in s.sections and set(s.sections) == set(SECTIONS)
    assert s.show_perf is True and s.quality_preset == "High" and s.sidebar_w == 520.0 and s.timeline_h == 176.0
    assert UiState.from_dict(None).to_dict() == UiState().to_dict()
    assert UiState.from_dict("x").to_dict() == UiState().to_dict()


def test_state_clamps_scale():
    assert UiState.from_dict({"ui_scale": 0.1}).ui_scale == 0.75
    assert UiState.from_dict({"ui_scale": 9}).ui_scale == 2.0


def test_state_survives_settings_file_roundtrip(tmp_path, monkeypatch):
    from bgv import config
    monkeypatch.setattr(config, "settings_path", lambda: tmp_path / "user_settings.json")
    s = UiState()
    s.show_atrisk = False
    s.sections["layers"] = False
    config.save_settings({"layers": {"overlay": {"enabled": True, "state": s.to_dict()}}})
    back = config.load_settings()["layers"]["overlay"]["state"]
    s2 = UiState.from_dict(back)
    assert s2.show_atrisk is False and s2.sections["layers"] is False


# ------------------------------------------------------------------------------------------------ layout
def test_layout_fits_minimum_window():
    st = UiState()
    L = compute_layout(1024, 600, st, 32.0, 1.0, True)
    assert L.side_w <= 1024 * 0.32 + 1e-6
    assert L.sx1 > L.sx0 + 300 and L.sy1 > L.sy0 + 200          # a usable 3D view remains at 1024x600
    assert L.tl_h <= 600 * 0.32 + 1e-6
    assert L.sy1 + L.foot + L.tl_h == pytest.approx(600)
    big = compute_layout(1920, 1080, st, 32.0, 1.0, True)
    assert big.right_on and big.sx1 == pytest.approx(1920 - big.right_w - 2 * big.margin)
    assert big.side_w == pytest.approx(340.0)


def test_layout_hides_right_panels_when_narrow_or_off():
    st = UiState()
    assert compute_layout(1400, 800, st, 32.0, 1.0, True).right_on
    narrow = compute_layout(760, 600, st, 32.0, 1.0, True)
    assert not narrow.right_on and narrow.sx1 == 760
    off = compute_layout(1920, 1080, st, 32.0, 1.0, False)
    assert not off.right_on and off.sx1 == 1920
    st.show_sidebar = False
    st.show_timeline = False
    L = compute_layout(1600, 900, st, 32.0, 1.0, True)
    assert L.side_w == 0 and L.tl_h == 0 and L.sx0 == 0 and L.sy1 == 900 - L.foot


def test_layout_scales_with_ui_scale():
    st = UiState()
    a = compute_layout(2560, 1440, st, 40.0, 1.5, True)
    assert a.side_w == pytest.approx(340 * 1.5) and a.foot == pytest.approx(26 * 1.5)


# ------------------------------------------------------------------------------------------------ timeline series
def test_meta_series_and_chart_series():
    meta = {"hours": [2, 3], "hourly": [{"hour": 2, "wet_area_km2": 0.5, "max_depth_m": 0.3},
                                       {"hour": 3, "wet_area_km2": 1.5, "max_depth_m": 0.9}]}
    wet, mx = meta_series(meta)
    assert wet[2] == 0.5 and wet[3] == 1.5 and wet[0] == 0.0 and wet[10] == 0.0 and mx[3] == 0.9
    s = chart_series(None, meta)
    assert not s.has_flooded and s.done == 0 and not s.complete and s.wet_area[3] == 1.5
    tl = {"flooded": [float("nan")] * 25, "wet_area_km2": [float("nan")] * 25, "max_depth_m": [float("nan")] * 25}
    tl["flooded"][2], tl["flooded"][3] = 10.0, 30.0
    tl["wet_area_km2"][3] = 1.6                                   # hub value wins over the meta value
    s = chart_series(tl, meta)
    assert s.has_flooded and s.done == 2 and s.wet_area[3] == 1.6 and s.wet_area[2] == 0.5
    assert value_at(s.flooded, 2.5) == pytest.approx(20.0)
    assert math.isnan(value_at(s.flooded, 3.5))
    assert value_at(s.flooded, 3.0) == 30.0 and math.isnan(value_at(s.flooded, 0.0))
    full = chart_series({"flooded": list(range(25))}, None)
    assert full.complete


# ------------------------------------------------------------------------------------------------ theme
def test_depth_ramp_matches_wgsl():
    """The legend ramp is parsed from the water shader source: endpoints equal the WGSL constants, darkens with depth."""
    import re
    from bgv.gpu.layers import terrain
    r0 = theme.depth_ramp(0.0)
    assert theme.RAMP_FROM_WGSL
    src = terrain.WATER_WGSL
    body = src[src.index("fn depth_ramp"):]
    cols = re.findall(r"let c(\d) = vec3<f32>\(([\d.]+), ([\d.]+), ([\d.]+)\);", body)
    c0 = tuple(float(v) for v in cols[0][1:])
    c4 = tuple(float(v) for v in cols[4][1:])
    assert r0 == pytest.approx(c0)
    assert theme.depth_ramp(50.0) == pytest.approx(c4)
    lum = [sum(theme.depth_ramp(d)) for d in np.linspace(0, 6, 30)]
    assert all(a >= b - 1e-9 for a, b in zip(lum, lum[1:]))
    s = theme.depth_ramp_srgb(0.0)
    assert s[0] > r0[0] and s[3] == 1.0                           # sRGB encoding brightens linear values


def test_tier_colors_are_web_colors():
    assert theme.TIER_RGB[0][:3] == pytest.approx(theme.hexrgb("#4dabf7")[:3])
    assert theme.TIER_RGB[3][:3] == pytest.approx(theme.hexrgb("#e74c3c")[:3])
    assert theme.col32((1, 0, 0, 1)) == 0xFF0000FF and theme.col32((0, 0, 1, 0.5)) == 0x80FF0000
    assert theme.speed_ramp(-1) == theme.SPEED_STOPS[0][1] and theme.speed_ramp(99) == theme.SPEED_STOPS[-1][1]


# ------------------------------------------------------------------------------------------------ probe
class _FakeStorm:
    name = "t"

    def sample_t(self, t, x, y):
        return {"depth": 0.8 if x > 0 else 0.0, "u": 1.0, "v": 1.0, "speed": math.sqrt(2.0), "valid": True}


def test_probe_sampling_and_dry_cleanup():
    from bgv.ui.probe import ProbeWorker, sample_probe
    sampler = SimpleNamespace(sample=lambda x, y: 150.0)
    r = sample_probe(_FakeStorm(), 12.0, 5.0, 0.0, sampler)
    assert r.depth_m == 0.8 and r.bearing == pytest.approx(45.0) and r.ground_m == 150.0 and r.valid
    dry = sample_probe(_FakeStorm(), 12.0, -5.0, 0.0, sampler)
    assert dry.depth_m == 0.0 and dry.speed_ms == 0.0 and dry.bearing == 0.0         # no phantom velocity on dry ground
    w = ProbeWorker()
    try:
        w.request(_FakeStorm(), 12.0, 5.0, 0.0, sampler, key=("a",))
        import time
        for _ in range(100):
            if w.latest is not None:
                break
            time.sleep(0.01)
        assert w.latest is not None and w.latest.key == ("a",) and w.error is None
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ camera navigation
def test_camera_flight_reaches_target_and_takes_short_way_round():
    from bgv.gpu.camera import OrbitCamera
    from bgv.ui.cameranav import CameraNav, building_view
    cam = OrbitCamera()
    cam.set_view(0, 0, 5000, 350.0, 50.0)
    nav = CameraNav()
    nav.fly(cam, 100.0, 200.0, 300.0, 10.0, 45.0, seconds=1.0)
    yaws = []
    for _ in range(80):
        nav.update(cam, 0.02)
        yaws.append(cam.yaw)
    assert not nav.active
    assert (cam.target[0], cam.target[1]) == pytest.approx((100.0, 200.0), abs=1e-6) and cam.dist == pytest.approx(300.0, rel=1e-6)
    assert cam.yaw == pytest.approx(10.0, abs=1e-6) and cam.pitch == pytest.approx(45.0)
    # 350 -> 10 degrees goes through 0 (20 degrees), never through 180
    assert all(y >= 349.0 or y <= 11.0 for y in yaws)
    x, y, d, yaw, pitch = building_view(5.0, 6.0, 12.0, cam)
    assert 90 <= d <= 220 and pitch == 52.0


def test_presets_resolve():
    from bgv.geo import GridFrame
    from bgv.ui.cameranav import PRESETS, preset_view
    f = GridFrame(2947, 3368, 6.4, 5.1128, -9633395.2, 4425075.2, -9614534.4, 4446630.4)
    for name in PRESETS:
        x, y, dist, yaw, pitch = preset_view(f, name)
        assert abs(x) < f.size_m[0] / 2 and abs(y) < f.size_m[1] / 2 and 500 < dist < 20000
    assert preset_view(f, "Overview")[:2] == (0.0, 0.0)


# ------------------------------------------------------------------------------------------------ narrow windows
def test_layout_keeps_a_usable_scene_with_right_panels_at_1024():
    st = UiState()
    L = compute_layout(1024, 600, st, 32.0, 1.0, True)
    assert L.right_on and L.sx1 - L.sx0 >= 340.0           # the card / list fit beside a usable 3D view
    assert L.sx1 + L.margin * 2 + L.right_w == pytest.approx(1024)


def test_atrisk_list_is_opt_in_on_narrow_windows():
    from bgv.ui.manager import UiManager
    m = UiManager(None, UiState())
    try:
        m.s = 1.0
        m.L = compute_layout(1920, 1080, m.st, 32.0, 1.0, True)
        assert not m.is_narrow() and m.atrisk_shown()
        m.set_atrisk(False)
        assert not m.atrisk_shown() and m.st.show_atrisk is False
        m.set_atrisk(True)
        m.L = compute_layout(1024, 600, m.st, 32.0, 1.0, True)
        assert m.is_narrow()
        assert not m.atrisk_shown()                            # hidden by default when narrow ...
        m.set_atrisk(True)
        assert m.atrisk_shown()                                 # ... until the user asks for it
        m.set_atrisk(False)
        assert not m.atrisk_shown() and m.st.show_atrisk is True   # the saved preference is untouched
        m.s = 1.5                                               # 1500 logical px at 150 % interface size is narrow too
        m.L = compute_layout(1500, 900, m.st, 32.0, 1.5, True)
        assert m.is_narrow()
    finally:
        m.probe_worker.close()


def test_default_storm_and_hour_ignore_stored_session_values():
    from bgv import domain as dm
    stale = {"storm": "100yr", "domain_views": {"county": {"storm": "50yr"}}, "t": 24.0}
    assert dm.pick_storm(stale, "county", ["100yr", "ev6"]) == "ev6"
    assert dm.pick_storm(stale, "city", ["25yr", "100yr"]) == "25yr"               # the city has no ev6 run
    assert dm.pick_storm(dict(stale, restore_last=True), "county", ["50yr", "ev6"]) == "50yr"
    assert dm.pick_storm(stale, "county", ["100yr", "ev6"], explicit="100yr") == "100yr"
    assert dm.DEFAULT_HOUR == 1.0


def test_hour_note_labels_hours_before_the_first_solver_hour():
    from types import SimpleNamespace as NS
    from bgv.ui import units
    app = NS(t=1.0, storm=NS(hours=[2, 3, 4]))
    assert "first solver hour is 2" in units.hour_note(app)
    app.t = 5.0
    assert units.hour_note(app) == ""
