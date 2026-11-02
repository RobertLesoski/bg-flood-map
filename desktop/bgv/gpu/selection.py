"""Selection + camera flights for buildings and roads.

State (created by `ensure(app)`, which stats_hub.get_hub does for you):
    app.selection = {"building": int | None, "road": int | None, "hover_building": int | None, "point": (x, y, z) | None}
BuildingsLayer highlights `selection["building"]` (emissive pulse + outline) and `selection["hover_building"]` (soft).

API for the UI / input code (all main thread; pixels are the LOGICAL window pixels of the pointer events, i.e. the frame
the camera works in - `camera.ray(px, py)`):
    select_at_pixel(app, px, py)        -> app.selection   click handling: picks a building or a road (building wins),
                                        clears the selection when the click hits neither.  Typical use in the pointer-up
                                        handler when the pointer did not move (a drag is a pan):
                                            if abs(x - x0) < 4 and abs(y - y0) < 4 and not imgui_wants_mouse:
                                                selection.select_at_pixel(app, x, y)
    hover_at_pixel(app, px, py)         -> building id | None   cheap variant for hover highlighting (sets "hover_building")
    pick_at_pixel(app, px, py)          -> {"building", "road", "point"} without touching app.selection
    select_building(app, id) / select_road(app, id) / clear_selection(app)
    describe_building(app, id) -> dict  address, land use, height, footprint area, sale price, depth/tier/speed at the
                                        displayed snapshot (hour of app.flood), rank among flooded buildings
    describe_road(app, id) -> dict      class, length, flooded share at the displayed snapshot
    fly_to_building(app, id, dist=120.0, pitch=50.0, yaw=None, duration=None)
                                        smooth camera transition (also selects the building); advanced every frame by the
                                        stats hook (`update_flight`), cancelled by mouse drags / held keys.  `step_flight(app, dt)`
                                        advances it manually (tests, headless tools).

Picking is exact for the 3D shape: the camera ray is marched in 1-10 m steps; every sample inside a building footprint
(roof triangles of the extruded mesh, so holes and L shapes are exact) whose height is inside [base, base + height] is a hit and the
nearest one wins.  If the ray hits no prism, the ground point under the cursor (camera.pick) is matched against the
footprints with a tolerance of max(3 m, 0.6 % of the distance) and then against the road ribbons (half width + 2 m).
Typical cost 2-8 ms (cKDTree candidate search + vectorised point-in-triangle tests).
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..data.roads import ROAD_CLASSES
from .scene_data import get_scene

R_SMALL = 40.0


def ensure(app):
    if getattr(app, "selection", None) is None:
        app.selection = {"building": None, "road": None, "hover_building": None, "point": None}
    return app.selection


# ------------------------------------------------------------------------------------------------ geometry helpers
def _pts_in_tris(px, py, tri):
    """bool per (point, triangle) row: point inside the triangle (tri float64[N,6])."""
    ax, ay, bx, by, cx, cy = (tri[:, i] for i in range(6))
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _dist_pt_tri(px, py, tri):
    """distance from point to each triangle (0 inside)."""
    inside = _pts_in_tris(px, py, tri)
    d = np.full(tri.shape[0], np.inf)
    for k in range(3):
        x0, y0 = tri[:, 2 * k], tri[:, 2 * k + 1]
        x1, y1 = tri[:, (2 * k + 2) % 6], tri[:, (2 * k + 3) % 6]
        d = np.minimum(d, _dist_pt_seg(px, py, x0, y0, x1, y1))
    return np.where(inside, 0.0, d)


def _dist_pt_seg(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - x0) * dx + (py - y0) * dy) / np.where(l2 > 0, l2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


RADIUS_CLASSES = (40.0, 120.0, 1e9)


def _trees(pick):
    """[(tree, ids, class radius)]: buildings split by bounding radius so a ball query stays cheap for the many small ones."""
    tr = getattr(pick, "_split", None)
    if tr is None:
        from scipy.spatial import cKDTree
        tr = []
        if pick.bld_tree is not None:
            rad = pick.bld_radius[pick.bld_ids]
            pts = pick.bld_tree.data
            lo = -1.0
            for hi in RADIUS_CLASSES:
                m = (rad > lo) & (rad <= hi)
                if m.any():
                    tr.append((cKDTree(pts[m]), pick.bld_ids[m], float(rad[m].max())))
                lo = hi
        pick._split = tr
    return tr


def _footprint_tris(pick, ids):
    """roof triangles of the buildings `ids` -> (tri float64[N,6], owner index into ids int[N])"""
    off = pick.roof_off
    cnt = off[ids + 1] - off[ids]
    total = int(cnt.sum())
    if total == 0:
        return np.zeros((0, 6)), np.zeros(0, np.int64)
    owner = np.repeat(np.arange(ids.size), cnt)
    start = np.repeat(off[ids], cnt)
    within = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return pick.roof_tri[start + within].astype(np.float64), owner


def _candidates(pick, xy: np.ndarray, extra: float = 0.0):
    """(sample index, building id) pairs: buildings whose bounding circle (+ extra metres) contains each sample point."""
    si, bi = [], []
    for tree, ids, r in _trees(pick):
        for k, l in enumerate(tree.query_ball_point(xy, r + extra)):
            if l:
                si.append(np.full(len(l), k, np.int64))
                bi.append(ids[np.asarray(l, np.int64)])
    if not si:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(si), np.concatenate(bi).astype(np.int64)


def _ray_samples(o, d, t_end, zmax):
    d = d / np.linalg.norm(d)
    t0 = 0.0
    if d[2] < -1e-6 and o[2] > zmax:
        t0 = (o[2] - zmax) / -d[2]
    elif d[2] >= -1e-6 and o[2] > zmax:
        return np.zeros(0), np.zeros((0, 3))
    if t_end <= t0:
        return np.zeros(0), np.zeros((0, 3))
    # step grows with distance: 1.5 m at the camera ... ~10 m at 5 km
    n_est = int(min(6000, 2.0 * (t_end - t0) / 2.0 + 64))
    ts = []
    t = t0
    while t < t_end and len(ts) < n_est:
        ts.append(t)
        t += 1.5 + 0.002 * t
    ts = np.asarray(ts)
    return ts, o[None, :] + ts[:, None] * d[None, :]


# ------------------------------------------------------------------------------------------------ picking
def pick_building_ray(app, px, py, ground="auto"):
    """Nearest building prism along the camera ray -> (id, point xyz, t) or None.  `ground` = camera.pick result if known."""
    sc = get_scene(app)
    if not sc.ready:
        return None
    b, pick, E = sc.buildings, sc.pick, app.exag
    hs = _height_scale(app)
    o, d = app.camera.ray(px, py)
    g = app.camera.pick(px, py) if isinstance(ground, str) else ground
    t_end = g[3] + 30.0 if g is not None else 25000.0
    zmax = float(np.max(sc.base_z) * E + 60.0 * hs + 5.0)
    ts, P = _ray_samples(o, d, t_end, zmax)
    if ts.size == 0:
        return None
    si, bi = _candidates(pick, P[:, :2])
    if si.size == 0:
        return None
    # height window: sample z inside [base - skirt, base + height]
    base = sc.base_z[bi].astype(np.float64) * E
    z = P[si, 2]
    ok = (z >= base - (0.8 + 0.35 * E)) & (z <= base + b.height_m[bi].astype(np.float64) * hs + 0.3)
    si, bi = si[ok], bi[ok]
    if si.size == 0:
        return None
    # point in footprint (roof triangles)
    uniq, inv = np.unique(bi, return_inverse=True)
    tri, owner = _footprint_tris(pick, uniq)
    if tri.shape[0] == 0:
        return None
    # pairs (sample, building) x triangles of that building
    cnt = (pick.roof_off[uniq + 1] - pick.roof_off[uniq])[inv]
    pair_of = np.repeat(np.arange(si.size), cnt)
    first = np.cumsum(np.r_[0, (pick.roof_off[uniq + 1] - pick.roof_off[uniq])])[:-1][inv]
    within = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    trow = np.repeat(first, cnt) + within
    inside = _pts_in_tris(P[si[pair_of], 0], P[si[pair_of], 1], tri[trow])
    hit_pairs = np.unique(pair_of[inside])
    if hit_pairs.size == 0:
        return None
    best = hit_pairs[np.argmin(ts[si[hit_pairs]])]
    k = int(si[best])
    return int(bi[best]), (float(P[k, 0]), float(P[k, 1]), float(P[k, 2])), float(ts[k])


def _height_scale(app) -> float:
    l = app.get_layer("buildings")
    return float(getattr(l, "height_scale", 1.0)) if l is not None else 1.0


def building_near_point(app, x, y, tol=3.0):
    """Building whose footprint is within `tol` metres of the ground point (x, y) -> (id, distance) or None."""
    sc = get_scene(app)
    if not sc.ready:
        return None
    pick = sc.pick
    si, bi = _candidates(pick, np.array([[x, y]]), tol)
    if bi.size == 0:
        return None
    uniq = np.unique(bi)
    tri, owner = _footprint_tris(pick, uniq)
    if tri.shape[0] == 0:
        return None
    dist = _dist_pt_tri(x, y, tri)
    per = np.full(uniq.size, np.inf)
    np.minimum.at(per, owner, dist)
    k = int(np.argmin(per))
    return (int(uniq[k]), float(per[k])) if per[k] <= tol else None


def road_near_point(app, x, y, tol=2.0):
    """Road (feature id) whose ribbon is within `tol` metres of (x, y) -> (road id, distance) or None."""
    sc = get_scene(app)
    if not sc.ready or sc.pick.seg_tree is None:
        return None
    r, pick = sc.roads, sc.pick
    rad = pick.seg_len_max * 0.5 + 12.0 + tol
    idx = np.asarray(pick.seg_tree.query_ball_point([x, y], rad), np.int64)
    if idx.size == 0:
        return None
    d = _dist_pt_seg(x, y, r.seg_x0[idx].astype(np.float64), r.seg_y0[idx].astype(np.float64),
                     r.seg_x1[idx].astype(np.float64), r.seg_y1[idx].astype(np.float64))
    d = d - pick.seg_half[idx]
    k = int(np.argmin(d))
    if d[k] <= tol:
        return int(r.seg_road[idx[k]]), float(max(d[k], 0.0))
    return None


def pick_at_pixel(app, px, py) -> dict:
    """{"building": id|None, "road": id|None, "point": (x, y, z)|None} for a window pixel (no side effects)."""
    out = {"building": None, "road": None, "point": None}
    g = app.camera.pick(px, py)
    if g is not None:
        out["point"] = (g[0], g[1], g[2])
    hit = pick_building_ray(app, px, py, g)
    if hit is not None:
        out["building"] = hit[0]
        out["point"] = hit[1]
        return out
    if g is not None:
        tol = max(3.0, 0.006 * g[3])
        nb = building_near_point(app, g[0], g[1], tol)
        if nb is not None:
            out["building"] = nb[0]
            return out
        nr = road_near_point(app, g[0], g[1], max(2.0, 0.003 * g[3]))
        if nr is not None:
            out["road"] = nr[0]
    return out


def select_at_pixel(app, px, py) -> dict:
    """Click handling.  Sets and returns app.selection (building wins over road; a miss clears)."""
    sel = ensure(app)
    p = pick_at_pixel(app, px, py)
    sel["building"], sel["road"], sel["point"] = p["building"], p["road"], p["point"]
    return sel


def hover_at_pixel(app, px, py):
    sel = ensure(app)
    hit = pick_building_ray(app, px, py)
    sel["hover_building"] = None if hit is None else hit[0]
    return sel["hover_building"]


def select_building(app, bid):
    sel = ensure(app)
    sel["building"] = None if bid is None else int(bid)
    sel["road"] = None
    return sel


def select_road(app, rid):
    sel = ensure(app)
    sel["road"] = None if rid is None else int(rid)
    sel["building"] = None
    return sel


def clear_selection(app):
    sel = ensure(app)
    sel.update(building=None, road=None, hover_building=None, point=None)
    return sel


# ------------------------------------------------------------------------------------------------ info
def describe_building(app, bid) -> dict:
    sc = get_scene(app)
    b = sc.buildings
    bid = int(bid)
    f = getattr(app, "flood", None) or {}
    d = {"id": bid, "address": str(b.address[bid]), "land_use": str(b.land_use[bid]), "name": str(b.name[bid]),
         "height_m": float(b.height_m[bid]), "area_m2": float(b.area_m2[bid]), "sale_price": float(b.sale_price[bid]),
         "base_z_m": float(sc.base_z[bid]), "x": float(b.cx[bid]), "y": float(b.cy[bid]), "depth_m": 0.0, "tier": 0,
         "speed_ms": 0.0, "flooded": False, "rank": None, "hour": f.get("hour")}
    bs = f.get("buildings") if f.get("ready") else None
    if bs is not None:
        d.update(depth_m=float(bs.depth_m[bid]), tier=int(bs.tier[bid]), speed_ms=float(bs.speed_ms[bid]),
                 flooded=bool(bs.flooded[bid]))
        if bs.flooded[bid]:
            w = np.flatnonzero(bs.order == bid)
            d["rank"] = int(w[0]) + 1 if w.size else None
    return d


def describe_road(app, rid) -> dict:
    sc = get_scene(app)
    r = sc.roads
    rid = int(rid)
    f = getattr(app, "flood", None) or {}
    d = {"id": rid, "class": ROAD_CLASSES[int(r.road_class[rid])], "length_m": float(r.road_length_m[rid]),
         "flooded_pct": 0.0, "hour": f.get("hour")}
    rs = f.get("roads") if f.get("ready") else None
    if rs is not None:
        d["flooded_pct"] = float(rs.per_road_pct[rid])
    return d


# ------------------------------------------------------------------------------------------------ camera flight
def _ease(s: float) -> float:
    return s * s * (3.0 - 2.0 * s)


def fly_to_building(app, bid, dist: float = 120.0, pitch: float = 50.0, yaw: float | None = None,
                    duration: float | None = None):
    """Smooth transition to a building (selects it).  Duration defaults to 0.8-2.4 s depending on the distance."""
    sc = get_scene(app)
    cam = app.camera
    bid = int(bid)
    cx, cy = float(sc.buildings.cx[bid]), float(sc.buildings.cy[bid])
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return None
    select_building(app, bid)
    start = (float(cam.target[0]), float(cam.target[1]), float(cam.dist), float(cam.yaw), float(cam.pitch))
    travel = math.hypot(cx - start[0], cy - start[1])
    if duration is None:
        duration = float(np.clip(0.8 + 0.45 * math.log10(1.0 + travel / 60.0) + 0.25 * abs(math.log2(max(start[2], 1.0) / dist)) * 0.3,
                                 0.8, 2.4))
    yaw_end = start[3] if yaw is None else float(yaw)
    dy = ((yaw_end - start[3] + 180.0) % 360.0) - 180.0
    app._flight = {"start": start, "end": (cx, cy, float(dist), start[3] + dy, float(pitch)), "dur": max(duration, 1e-3),
                   "t": 0.0, "arc": float(min(0.25 * travel, 5000.0)), "last": time.perf_counter()}
    return app._flight


def step_flight(app, dt: float) -> bool:
    """Advance the active flight by dt seconds.  Returns True while a flight is active."""
    fl = getattr(app, "_flight", None)
    if fl is None:
        return False
    cam = app.camera
    if cam._drag is not None or cam._keys:                 # the user took over
        app._flight = None
        return False
    fl["t"] += dt
    s = min(fl["t"] / fl["dur"], 1.0)
    e = _ease(s)
    a, b = fl["start"], fl["end"]
    x = a[0] + (b[0] - a[0]) * e
    y = a[1] + (b[1] - a[1]) * e
    dist = math.exp(math.log(max(a[2], 1.0)) * (1 - e) + math.log(max(b[2], 1.0)) * e) + fl["arc"] * math.sin(math.pi * e)
    yaw = a[3] + (b[3] - a[3]) * e
    pitch = a[4] + (b[4] - a[4]) * e
    cam.set_view(x, y, dist, yaw, pitch)
    if s >= 1.0:
        app._flight = None
        return False
    return True


def update_flight(app):
    """Per-frame hook (called from the stats hook): real-time dt since the last call."""
    fl = getattr(app, "_flight", None)
    if fl is None:
        return
    now = time.perf_counter()
    dt = min(now - fl.get("last", now), 0.1)
    fl["last"] = now
    step_flight(app, dt)
