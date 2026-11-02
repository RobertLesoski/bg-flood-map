"""Glue between the UI and the other agents' modules (scene data, stats hub, selection).  Everything degrades gracefully:
a missing module / attribute returns None / False, nothing here raises into the render loop.

    flood(app) -> dict | None            app.flood (current snapshot statistics)
    timeline(app) -> dict | None         app.timeline
    scene(app) -> SceneData | None       the shared scene, only when `state == 'ready'`  (never blocks)
    buildings(app) -> Buildings | None
    ensure_hub(app)                      install the stats hub once (idempotent)
    selection(app) -> dict               app.selection (created when missing)
    pick_building(app, px, py) -> int | None
    select_building(app, id | None, fly=False)
    depth_series(app, i) -> list[float] | None   building depth at hours 0..24 from the hub's cached snapshots
"""
from __future__ import annotations

import math

import numpy as np

N_HOURS = 25


def flood(app):
    f = getattr(app, "flood", None)
    return f if isinstance(f, dict) else None


def timeline(app):
    t = getattr(app, "timeline", None)
    return t if isinstance(t, dict) else None


def hub(app):
    return getattr(app, "_stats_hub", None)


def ensure_hub(app) -> bool:
    if hub(app) is not None:
        return True
    try:
        from ..gpu import stats_hub
        stats_hub.get_hub(app)
        return True
    except Exception:
        return False


def scene_state(app):
    """(state, message, frac) of the scene loader, or ('missing', '', 0)."""
    sc = getattr(app, "_scene_data", None)
    if sc is None:
        return ("missing", "", 0.0)
    return (sc.state, sc.message, float(sc.frac))


def scene(app):
    sc = getattr(app, "_scene_data", None)
    if sc is not None and getattr(sc, "state", "") == "ready":
        return sc
    return None


def buildings(app):
    sc = scene(app)
    return sc.buildings if sc is not None else None


def selection_module():
    try:
        from ..gpu import selection as m
        return m
    except Exception:
        return None


def selection(app) -> dict:
    m = selection_module()
    if m is not None:
        try:
            return m.ensure(app)
        except Exception:
            pass
    s = getattr(app, "selection", None)
    if not isinstance(s, dict):
        s = {"building": None, "road": None, "hover_building": None, "point": None}
        app.selection = s
    s.setdefault("building", None)
    s.setdefault("road", None)
    return s


def hover_building(app, px: float, py: float):
    """Building under a window pixel (logical px) or None (also stored in selection['hover_building'])."""
    m = selection_module()
    if m is not None:
        try:
            r = m.hover_at_pixel(app, px, py)
            return None if r is None else int(r)
        except Exception:
            return None
    return _fallback_pick(app, px, py)


def _fallback_pick(app, px: float, py: float):
    """Without the selection module: ground pick -> nearest building centroid inside its footprint radius."""
    sc = scene(app)
    if sc is None or sc.pick is None:
        return None
    p = app.camera.pick(px, py)
    if p is None:
        return None
    pk = sc.pick
    try:
        d, k = pk.bld_tree.query([p[0], p[1]])
        bid = int(pk.bld_ids[k])
        if d <= float(pk.bld_radius[bid]) * 0.9 + 2.0:
            return bid
    except Exception:
        return None
    return None


def click_select(app, px: float, py: float):
    """Click on the 3D view: select the building (or road) under the pixel, clear the selection on a miss."""
    m = selection_module()
    if m is not None:
        try:
            m.select_at_pixel(app, px, py)
            return
        except Exception:
            pass
    s = selection(app)
    s["building"] = _fallback_pick(app, px, py)


def clear_selection(app):
    m = selection_module()
    s = selection(app)
    if m is not None:
        try:
            m.clear_selection(app)
            return
        except Exception:
            pass
    s["building"] = s["road"] = None


def select_building(app, bid, fly: bool = False) -> None:
    m = selection_module()
    if bid is not None and fly and fly_to_building(app, int(bid)):
        return
    s = selection(app)
    if m is not None:
        m.select_building(app, bid)
    else:
        s["building"] = None if bid is None else int(bid)


def fly_to_building(app, bid: int) -> bool:
    """The scene module's flight (selects the building too); the UI's own camera navigator otherwise."""
    m = selection_module()
    if m is not None:
        try:
            if m.fly_to_building(app, int(bid)) is not None:
                return True
        except Exception:
            pass
    ov = app.get_layer("overlay") if hasattr(app, "get_layer") else None
    mgr = getattr(ov, "manager", None)
    b = buildings(app)
    if mgr is None or b is None:
        return False
    from .cameranav import building_view
    cx, cy = float(b.cx[bid]), float(b.cy[bid])
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return False
    selection(app)["building"] = int(bid)
    mgr.nav.fly(app.camera, *building_view(cx, cy, float(b.height_m[bid]), app.camera), seconds=1.0)
    return True


def describe_road(app, rid: int):
    m = selection_module()
    if m is None:
        return None
    try:
        return m.describe_road(app, int(rid))
    except Exception:
        return None


def depth_series(app, i: int):
    """Depth (m) of building i at hours 0..24 (NaN where the hour is not computed yet) from the hub's cache."""
    h = hub(app)
    st = getattr(app, "storm", None)
    if h is None or st is None:
        return None
    out = [float("nan")] * N_HOURS
    got = 0
    for hr in range(N_HOURS):
        snap = h.get_snapshot(st.name, hr)
        if snap is not None:
            out[hr] = float(snap.buildings.depth_m[i])
            got += 1
    return out if got else None
