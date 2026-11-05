"""Cinematic tools, pure logic (no imgui, no wgpu: unit-testable).  The runtime pieces are
`bgv/gpu/layers/cinematic.py` (controller layer), `bgv/ui/panels_cinematic.py` (panel, menu, captions) and
`bgv/export_cinematic.py` (offscreen PNG / MP4 export).

    BookmarkStore      camera bookmarks per domain (name, view, optionally hour + storm), JSON round trip
    view_lerp, ViewFlight   smooth camera transitions (ease in / out, distance in log space, rise over long hops)
    TimeLapse          continuous play of the hour with a rate in hours per second, range, loop / ping-pong / once
    build_stops, order_route   tour route from layers/infrastructure.geojson + the most flooded named roads
    TourPath           spline camera path through the stops: legs with ease in / out and a slow orbit while dwelling
    TourPlan           which storm and hour belong to which moment of the tour
    Shot classes       what the exporter renders: state_at(seconds) -> ShotState (view, hour, storm, caption)

A camera view is the 5-tuple (x, y, dist, yaw, pitch) of `OrbitCamera.set_view` (world metres from the grid centre).  Names on
captions come from the data files only; an unnamed place is skipped, never given an invented name.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------ easing / views
View = tuple


def smootherstep(k: float) -> float:
    k = min(max(float(k), 0.0), 1.0)
    return k * k * k * (k * (k * 6.0 - 15.0) + 10.0)


def smoothstep(k: float) -> float:
    k = min(max(float(k), 0.0), 1.0)
    return k * k * (3.0 - 2.0 * k)


def lerp_angle(a: float, b: float, k: float) -> float:
    d = ((b - a + 180.0) % 360.0) - 180.0
    return (a + d * k) % 360.0


def bearing_deg(dx: float, dy: float) -> float:
    """Compass heading (0 = north, 90 = east) of a world-metre direction."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def hop_metres(a: View, b: View) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def flight_seconds(a: View, b: View) -> float:
    """How long a camera hop between two views takes: short hops about a second, long ones a few."""
    hop = hop_metres(a, b)
    zoom = abs(math.log(max(b[2], 1.0) / max(a[2], 1.0)))
    return float(min(max(0.9 + 0.55 * math.log2(1.0 + hop / 400.0) + 0.5 * zoom, 0.9), 5.0))


def view_lerp(a: View, b: View, k: float, rise: float = 0.25, max_rise: float = 2500.0) -> View:
    """Pose between views a and b at eased parameter k in [0, 1] (the caller applies the easing).  Distance interpolates in
    log space and a long hop rises (sine bump, `rise` x hop up to `max_rise` m) so the camera zooms out and back in."""
    k = min(max(float(k), 0.0), 1.0)
    d = math.exp(math.log(max(a[2], 1e-3)) + (math.log(max(b[2], 1e-3)) - math.log(max(a[2], 1e-3))) * k)
    d += math.sin(math.pi * k) * min(hop_metres(a, b) * rise, max_rise)
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, d, lerp_angle(a[3], b[3], k), a[4] + (b[4] - a[4]) * k)


def hermite(p0, p1, m0, m1, s: float):
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1)


# ------------------------------------------------------------------------------------------------ bookmarks
MAX_BOOKMARKS = 99


@dataclass
class Bookmark:
    name: str
    view: list                       # x, y, dist, yaw, pitch
    t: float | None = None           # hour when saved
    storm: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "view": [float(v) for v in self.view], "t": self.t, "storm": self.storm}

    @staticmethod
    def from_dict(d) -> "Bookmark | None":
        try:
            v = [float(x) for x in d["view"]]
            if len(v) != 5 or not all(math.isfinite(x) for x in v):
                return None
            t = d.get("t")
            return Bookmark(str(d.get("name") or "View")[:60], v, None if t is None else float(t), d.get("storm") or None)
        except Exception:
            return None

    def lnglat(self, frame) -> tuple[float, float]:
        return frame.world_to_lnglat(self.view[0], self.view[1])


class BookmarkStore:
    """Bookmarks of every domain ({'city': [...], 'county': [...]}); `to_dict` / `load` go into the layer state."""

    def __init__(self):
        self.items: dict[str, list[Bookmark]] = {}

    def list(self, domain: str) -> list[Bookmark]:
        return self.items.setdefault(domain, [])

    def add(self, domain: str, name: str | None, view, t=None, storm=None, slot: int | None = None) -> int:
        """Append (or, with `slot`, overwrite that 0-based slot).  Returns the index.  A blank name becomes 'View N'."""
        lst = self.list(domain)
        name = (name or "").strip() or self.next_name(domain)
        bm = Bookmark(name[:60], [float(v) for v in view], None if t is None else float(t), storm)
        if slot is not None and 0 <= slot < len(lst):
            bm.name = (name if (name or "").strip() else lst[slot].name)[:60]
            lst[slot] = bm
            return slot
        if len(lst) >= MAX_BOOKMARKS:
            raise ValueError(f"at most {MAX_BOOKMARKS} bookmarks per area")
        lst.append(bm)
        return len(lst) - 1

    def next_name(self, domain: str) -> str:
        used = {b.name for b in self.list(domain)}
        n = len(used) + 1
        while f"View {n}" in used:
            n += 1
        return f"View {n}"

    def rename(self, domain: str, i: int, name: str):
        name = (name or "").strip()
        if name and 0 <= i < len(self.list(domain)):
            self.list(domain)[i].name = name[:60]

    def delete(self, domain: str, i: int):
        lst = self.list(domain)
        if 0 <= i < len(lst):
            del lst[i]

    def move(self, domain: str, i: int, j: int):
        lst = self.list(domain)
        if 0 <= i < len(lst) and 0 <= j < len(lst):
            lst.insert(j, lst.pop(i))

    def get(self, domain: str, i: int) -> Bookmark | None:
        lst = self.list(domain)
        return lst[i] if 0 <= i < len(lst) else None

    def to_dict(self) -> dict:
        return {k: [b.to_dict() for b in v] for k, v in self.items.items() if v}

    def load(self, d) -> "BookmarkStore":
        self.items = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, list):
                    got = [b for b in (Bookmark.from_dict(x) for x in v[:MAX_BOOKMARKS]) if b is not None]
                    if got:
                        self.items[str(k)] = got
        return self


# ------------------------------------------------------------------------------------------------ live flight
class ViewFlight:
    """Smooth camera transition driven from a layer update: `start(cam, target)`, then `update(cam, dt)` until it returns False."""

    def __init__(self):
        self.active = False
        self._a = self._b = None
        self._k = 0.0
        self._dur = 1.0
        self._last = None

    def start(self, cam, target: View, seconds: float | None = None) -> float:
        a = (float(cam.target[0]), float(cam.target[1]), float(cam.dist), float(cam.yaw), float(cam.pitch))
        b = (float(target[0]), float(target[1]), float(target[2]), float(target[3]) % 360.0, float(target[4]))
        self._a, self._b = a, b
        self._dur = float(seconds) if seconds else flight_seconds(a, b)
        self._k = 0.0
        self.active = True
        self._last = None
        return self._dur

    def cancel(self):
        self.active = False

    def update(self, cam, dt: float) -> bool:
        if not self.active:
            return False
        if self._last is not None and getattr(cam, "moving", False):      # the user took over (drag, keys, wheel)
            self.active = False
            return False
        self._k += dt / max(self._dur, 1e-3)
        k = smootherstep(self._k)
        v = view_lerp(self._a, self._b, k)
        cam.set_view(*v)
        _keep_eye_above_ground(cam)
        self._last = v
        if self._k >= 1.0:
            self.active = False
            cam.set_view(*self._b)
            _keep_eye_above_ground(cam)
        return True


def _keep_eye_above_ground(cam):
    try:
        cam._keep_eye_above_ground()
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------ time-lapse
RATE_MIN, RATE_MAX = 0.05, 12.0
RATE_PRESETS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LOOP_MODES = ("loop", "pingpong", "once")


def rate_from_slider(v: float) -> float:
    """Slider 0..1 -> hours per second, logarithmic between RATE_MIN and RATE_MAX."""
    return float(RATE_MIN * (RATE_MAX / RATE_MIN) ** min(max(v, 0.0), 1.0))


def slider_from_rate(r: float) -> float:
    r = min(max(float(r), RATE_MIN), RATE_MAX)
    return math.log(r / RATE_MIN) / math.log(RATE_MAX / RATE_MIN)


@dataclass
class TimeLapse:
    """Drives the continuous hour t.  The renderer blends the two hourly flow frames around t in the shader, so any rate
    shows continuous water; this class only moves t smoothly (hours per real second)."""
    playing: bool = False
    rate: float = 1.0 / 1.5            # hours per second (the standard 1x of the viewer)
    t_in: float = 0.0
    t_out: float = 24.0
    mode: str = "loop"
    direction: int = 1                 # +1 forward, -1 backward (ping-pong flips it)

    def step(self, t: float, dt: float) -> float:
        """New t after `dt` seconds (wraps / reflects / stops at the range ends according to `mode`)."""
        lo, hi = sorted((min(max(self.t_in, 0.0), 24.0), min(max(self.t_out, 0.0), 24.0)))
        if hi - lo < 1e-6:
            return lo
        t = min(max(t, lo), hi) + self.direction * self.rate * dt
        if self.mode == "loop":
            if t > hi or t < lo:
                t = lo + (t - lo) % (hi - lo)
        elif self.mode == "pingpong":
            for _ in range(4):
                if t > hi:
                    t, self.direction = 2 * hi - t, -1
                elif t < lo:
                    t, self.direction = 2 * lo - t, 1
                else:
                    break
            t = min(max(t, lo), hi)
        else:                                                               # once
            if t >= hi or t <= lo:
                self.playing = False
            t = min(max(t, lo), hi)
        return t

    def to_dict(self) -> dict:
        return {"rate": self.rate, "t_in": self.t_in, "t_out": self.t_out, "mode": self.mode}

    def load(self, d):
        if isinstance(d, dict):
            try:
                self.rate = min(max(float(d.get("rate", self.rate)), RATE_MIN), RATE_MAX)
                self.t_in = min(max(float(d.get("t_in", self.t_in)), 0.0), 24.0)
                self.t_out = min(max(float(d.get("t_out", self.t_out)), 0.0), 24.0)
            except (TypeError, ValueError):
                pass
            if d.get("mode") in LOOP_MODES:
                self.mode = d["mode"]


# ------------------------------------------------------------------------------------------------ tour: data
TYPE_LABEL = {"hospital": "Hospital", "school": "School", "fire_station": "Fire station", "police": "Police station",
              "assembly_point": "Assembly point (shelter)", "clinic": "Clinic", "townhall": "Town hall",
              "community_centre": "Community centre", "road": "Road"}
# round-robin order when more places qualify than the tour has room for
TYPE_ORDER = ("hospital", "school", "fire_station", "police", "assembly_point", "townhall", "clinic", "community_centre")
DEFAULT_TYPES = ("hospital", "school", "fire_station", "police", "assembly_point")
DWELL_DIST_M = {"hospital": 520.0, "school": 480.0, "fire_station": 420.0, "police": 420.0, "assembly_point": 420.0,
                "clinic": 420.0, "townhall": 420.0, "community_centre": 420.0, "road": 700.0}


@dataclass
class Stop:
    name: str                        # from the data file, never invented
    kind: str                        # key of TYPE_LABEL
    lng: float
    lat: float
    x: float = 0.0                   # world metres (filled by `place`)
    y: float = 0.0
    peak_m: float | None = None      # peak depth at the place from the depth source (None = unknown)
    detail: str = ""                 # e.g. road class

    @property
    def label(self) -> str:
        return TYPE_LABEL.get(self.kind, self.kind.replace("_", " ").title())

    def view(self, yaw: float, pitch: float, dist: float | None = None) -> View:
        return (self.x, self.y, dist or DWELL_DIST_M.get(self.kind, 450.0), yaw, pitch)


def infrastructure_path(root, domain_key: str) -> Path:
    r = Path(root) / "layers"
    return r / "infrastructure.geojson" if domain_key == "city" else r / "county" / "infrastructure.geojson"


def load_facilities(root, domain_key: str, types=DEFAULT_TYPES) -> tuple[list[Stop], int]:
    """(named facilities of the wanted types, number skipped for having no name) from the domain's infrastructure.geojson."""
    p = infrastructure_path(root, domain_key)
    if not p.is_file():
        return [], 0
    d = json.loads(p.read_text(encoding="utf-8"))
    out, unnamed = [], 0
    for f in d.get("features", []):
        pr = f.get("properties") or {}
        kind = pr.get("type")
        if kind not in types:
            continue
        name = str(pr.get("name") or "").strip()
        if not name:
            unnamed += 1
            continue
        try:
            lng, lat = float(pr["lng"]), float(pr["lat"])
        except (KeyError, TypeError, ValueError):
            c = (f.get("geometry") or {}).get("coordinates")
            if not c:
                continue
            lng, lat = float(c[0]), float(c[1])
        out.append(Stop(name, kind, lng, lat))
    return out, unnamed


def facility_peak_from_file(root, domain_key: str, storm: str) -> dict:
    """{(name, kind): max depth m} from the infrastructure file's own `d_<storm>` (cm) column: the fallback ranking when no
    depth source is available."""
    p = infrastructure_path(root, domain_key)
    out = {}
    try:
        for f in json.loads(p.read_text(encoding="utf-8")).get("features", []):
            pr = f.get("properties") or {}
            v = pr.get(f"d_{storm}")
            if v is not None:
                out[(str(pr.get("name") or "").strip(), pr.get("type"))] = float(v) / 100.0
    except Exception:
        pass
    return out


def flooded_road_rows(root, domain_key: str, storm: str) -> list:
    """Named rows [id, name, class, evac, len_m, wet_pct, flooded_pct, max_cm] of layers/roads_flooded/<domain>/<storm>.json,
    deepest first, one row per road name."""
    p = Path(root) / "layers" / "roads_flooded" / domain_key / f"{storm}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    seen, out = set(), []
    for r in sorted(d.get("rows", []), key=lambda r: -float(r[7])):
        nm = str(r[1] or "").strip()
        if nm and nm not in seen:
            seen.add(nm)
            out.append(r)
    return out


def road_geometry(root) -> dict:
    """{osm_id: [(lng, lat), ...]} of the county roads file (it holds the way geometry of both domains' road lists)."""
    p = Path(root) / "layers" / "county" / "roads.geojson.gz"
    if not p.is_file():
        return {}
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        d = json.load(fh)
    out = {}
    for f in d.get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") == "LineString":
            out[(f.get("properties") or {}).get("osm_id")] = [(c[0], c[1]) for c in g["coordinates"]]
    return out


def polyline_samples(pts, spacing_m: float = 25.0, max_n: int = 24) -> list:
    """Points along a lng/lat polyline about `spacing_m` apart (at most `max_n`, always including both ends)."""
    if len(pts) < 2:
        return list(pts)
    lat0 = math.radians(pts[0][1])
    kx, ky = 111320.0 * math.cos(lat0), 110574.0
    segs = [(math.hypot((b[0] - a[0]) * kx, (b[1] - a[1]) * ky)) for a, b in zip(pts, pts[1:])]
    total = sum(segs)
    n = int(min(max(math.ceil(total / spacing_m), 1), max_n))
    out = []
    for i in range(n + 1):
        s = total * i / n
        for (a, b), L in zip(zip(pts, pts[1:]), segs):
            if s <= L or L == segs[-1]:
                k = 0.0 if L <= 0 else min(s / L, 1.0)
                out.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
                break
            s -= L
    return out


PLACE_RADIUS_M = 25.0              # depth read-outs take the maximum within this distance of the place


def build_stops(root, domain_key: str, storm: str, frame, *, peak_many=None, types=DEFAULT_TYPES, max_facilities: int = 10,
                max_roads: int = 4, only_wet: bool = False, wet_m: float = 0.1, roads: bool = True) -> tuple[list[Stop], dict]:
    """The places of a tour for one storm.  `peak_many([(lng, lat), ...]) -> [metres | None]` ranks and annotates them (the depth
    source's peak, one batch for all places); without it the files' own `d_<storm>` column / `max_cm` is used.  Returns
    (unordered stops, info dict with counts: 'unnamed', 'facilities', 'roads')."""
    fac, unnamed = load_facilities(root, domain_key, types)
    file_peaks = facility_peak_from_file(root, domain_key, storm)
    peaks = peak_many([(s.lng, s.lat) for s in fac]) if (peak_many and fac) else [None] * len(fac)
    for s, pk in zip(fac, peaks):
        if pk is None or not math.isfinite(pk):
            pk = file_peaks.get((s.name, s.kind))
        s.peak_m = pk
    if only_wet:
        fac = [s for s in fac if (s.peak_m or 0.0) >= wet_m]
    by_kind: dict[str, list[Stop]] = {}
    for s in sorted(fac, key=lambda s: -(s.peak_m or 0.0)):
        by_kind.setdefault(s.kind, []).append(s)
    chosen: list[Stop] = []
    kinds = [k for k in TYPE_ORDER if k in by_kind]
    i = 0
    while len(chosen) < max_facilities and any(by_kind.get(k) for k in kinds):
        k = kinds[i % len(kinds)]
        if by_kind.get(k):
            chosen.append(by_kind[k].pop(0))
        i += 1
    n_roads = 0
    if roads and max_roads > 0:
        rows = flooded_road_rows(root, domain_key, storm)
        geo = road_geometry(root) if rows else {}
        classes = ("motorway", "motorway_link", "trunk", "primary", "primary_link", "secondary", "tertiary",
                   "unclassified", "residential", "other")
        cand = [(r, polyline_samples(geo[r[0]])) for r in rows[:max_roads * 3] if geo.get(r[0])][:max(max_roads * 2, max_roads)]
        flat = [q for _, smp in cand for q in smp]
        vals = peak_many(flat) if (peak_many and flat) else [None] * len(flat)
        k = 0
        scored = []
        for r, smp in cand:
            v = vals[k:k + len(smp)]
            k += len(smp)
            best, best_v = smp[len(smp) // 2], -1.0
            for q, x in zip(smp, v):
                if x is not None and math.isfinite(x) and x > best_v:
                    best, best_v = q, x
            scored.append((best_v if best_v >= 0 else float(r[7]) / 100.0, r, best))
        for pk, r, best in sorted(scored, key=lambda z: -z[0])[:max_roads]:
            hw = classes[int(r[2])] if 0 <= int(r[2]) < len(classes) else "road"
            chosen.append(Stop(str(r[1]), "road", best[0], best[1], peak_m=pk, detail=hw.replace("_", " ")))
            n_roads += 1
    for s in chosen:
        s.x, s.y = frame.lnglat_to_world(s.lng, s.lat)
    return chosen, {"unnamed": unnamed, "facilities": len(chosen) - n_roads, "roads": n_roads}


def order_route(stops: list[Stop], start_xy=(0.0, 0.0)) -> list[Stop]:
    """Short route: nearest neighbour from the stop closest to `start_xy`, then 2-opt on the open path."""
    if len(stops) < 3:
        return list(stops)
    pts = np.array([[s.x, s.y] for s in stops])
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    cur = int(np.argmin(np.hypot(pts[:, 0] - start_xy[0], pts[:, 1] - start_xy[1])))
    order, left = [cur], set(range(len(stops))) - {cur}
    while left:
        nxt = min(left, key=lambda j: d[order[-1], j])
        order.append(nxt)
        left.remove(nxt)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                a, b = order[i - 1], order[i]
                c = order[j]
                e = order[j + 1] if j + 1 < len(order) else None
                old = d[a, b] + (d[c, e] if e is not None else 0.0)
                new = d[a, c] + (d[b, e] if e is not None else 0.0)
                if new < old - 1e-6:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
    return [stops[i] for i in order]


# ------------------------------------------------------------------------------------------------ tour: path
ORBIT_DEG = 32.0                  # slow turn around the place while dwelling
INTRO_PITCH = 52.0


@dataclass
class TourState:
    view: View
    stop: int                      # index of the stop being approached / shown
    phase: str                     # 'leg' | 'dwell' | 'outro'
    k: float                       # progress inside the phase 0..1
    caption_alpha: float           # 0..1 fade of the place caption
    caption_stop: int | None       # stop whose caption shows (None between places)
    T: float


class TourPath:
    """Camera path: start view -> stop 0 -> ... -> stop n-1 -> back to the start view.  Each hop is a Hermite spline through
    the stops (Catmull-Rom tangents) travelled with smootherstep ease in / out; a dwell orbits `ORBIT_DEG` degrees around the
    place with the same easing, so speed is zero at every arrival.  `at(T)` is the pose at tour time T seconds."""

    def __init__(self, stops: list[Stop], start_view: View, dwell_s: float = 6.0, speed: float = 1.0, return_to_start: bool = True):
        if not stops:
            raise ValueError("a tour needs at least one stop")
        self.stops = list(stops)
        self.start_view = tuple(float(v) for v in start_view)
        self.dwell_s = float(dwell_s)
        self.return_to_start = return_to_start
        n = len(stops)
        pts = [np.array([s.x, s.y]) for s in stops]
        self.yaws = []
        for i, s in enumerate(stops):               # look along the direction of travel (the last stop keeps its predecessor's)
            j = min(i + 1, n - 1)
            if j == i:
                j, k = i, max(i - 1, 0)
                yaw = bearing_deg(pts[i][0] - pts[k][0], pts[i][1] - pts[k][1]) if n > 1 else self.start_view[3]
            else:
                yaw = bearing_deg(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1])
            self.yaws.append(yaw)
        self.pitches = [40.0 + 6.0 * math.sin(1.7 * i) for i in range(n)]
        self.dwell_views = [s.view(self.yaws[i], self.pitches[i]) for i, s in enumerate(stops)]
        # tangents of the position spline through [start, stops..., start]
        chain = [np.array(self.start_view[:2])] + pts + ([np.array(self.start_view[:2])] if return_to_start else [pts[-1]])
        self._chain = chain
        self._m = [(0.5 * (chain[min(i + 1, len(chain) - 1)] - chain[max(i - 1, 0)])) for i in range(len(chain))]
        # timeline: leg i (into stop i), dwell i, ..., outro leg
        self.segs: list[dict] = []
        T = 0.0
        prev = self.start_view
        for i in range(n):
            dur = self._leg_seconds(prev, self.dwell_views[i])
            self.segs.append({"kind": "leg", "i": i, "t0": T, "t1": T + dur, "a": prev, "b": self.dwell_views[i]})
            T += dur
            self.segs.append({"kind": "dwell", "i": i, "t0": T, "t1": T + self.dwell_s})
            T += self.dwell_s
            prev = self._dwell_end(i)
        if return_to_start:
            dur = self._leg_seconds(prev, self.start_view)
            self.segs.append({"kind": "outro", "i": n, "t0": T, "t1": T + dur, "a": prev, "b": self.start_view})
            T += dur
        self.duration = T

    @staticmethod
    def _leg_seconds(a: View, b: View) -> float:
        hop = hop_metres(a, b)
        return float(min(max(3.0 + 1.6 * math.log2(1.0 + hop / 700.0) + 0.6 * abs(math.log(b[2] / a[2])), 3.0), 9.0))

    def _dwell_end(self, i: int) -> View:
        v = self.dwell_views[i]
        return (v[0], v[1], v[2], (v[3] + ORBIT_DEG) % 360.0, v[4])

    def stop_times(self) -> list[tuple[float, float]]:
        """[(arrival T, departure T)] per stop."""
        out = []
        for s in self.segs:
            if s["kind"] == "leg":
                arr = s["t1"]
            elif s["kind"] == "dwell":
                out.append((arr, s["t1"]))
        return out

    def at(self, T: float) -> TourState:
        T = min(max(float(T), 0.0), self.duration)
        seg = self.segs[-1]
        for s in self.segs:
            if T <= s["t1"]:
                seg = s
                break
        span = max(seg["t1"] - seg["t0"], 1e-6)
        k = min(max((T - seg["t0"]) / span, 0.0), 1.0)
        e = smootherstep(k)
        if seg["kind"] == "dwell":
            v = self.dwell_views[seg["i"]]
            fade = min(1.0, k / 0.12, (1.0 - k) / 0.12)
            return TourState((v[0], v[1], v[2], (v[3] + ORBIT_DEG * e) % 360.0, v[4]), seg["i"], "dwell", k, max(fade, 0.0),
                             seg["i"], T)
        a, b = seg["a"], seg["b"]
        i = seg["i"]
        # position follows the spline through the stops; everything else is the generic eased view blend
        v = view_lerp(a, b, e)
        i0 = i                                      # chain index: 0 = start view, stop j = j + 1
        p0, p1 = self._chain[i0], self._chain[min(i0 + 1, len(self._chain) - 1)]
        m0, m1 = self._m[i0], self._m[min(i0 + 1, len(self._m) - 1)]
        pos = hermite(p0, p1, m0, m1, e)
        v = (float(pos[0]), float(pos[1]), v[2], v[3], v[4])
        return TourState(v, i if seg["kind"] == "leg" else n_stops_clip(i, len(self.stops)), seg["kind"], k, 0.0, None, T)

    def stop_at_or_after(self, T: float) -> int:
        """Index of the stop whose dwell contains T or comes next (for skip forward)."""
        for i, (arr, dep) in enumerate(self.stop_times()):
            if T < dep - 1e-6:
                return i
        return len(self.stops)

    def next_start(self, T: float) -> float:
        """Tour time to jump to for 'next stop': the arrival at the next stop."""
        times = self.stop_times()
        for i, (arr, dep) in enumerate(times):
            if T < arr - 1e-6:
                return arr
            if T < dep - 1e-6:
                return times[i + 1][0] if i + 1 < len(times) else self.duration
        return self.duration

    def prev_start(self, T: float) -> float:
        """Tour time for 'previous stop': the start of this dwell when it has run for > 1.5 s, else the earlier arrival."""
        times = self.stop_times()
        best = 0.0
        for arr, dep in times:
            if arr < T - 1.5:
                best = arr
        return best


def n_stops_clip(i: int, n: int) -> int:
    return min(max(i, 0), n - 1)


# ------------------------------------------------------------------------------------------------ tour: schedule
TIME_MODES = ("sweep", "hold")


@dataclass
class TourPlan:
    """Storm and hour as a function of the tour moment.  `storms` are visited in order, each for an equal share of the stops;
    with time_mode 'sweep' the hour runs 0 -> 24 once per storm group, with 'hold' it stays at `hold_t`."""
    storms: list = field(default_factory=list)
    time_mode: str = "sweep"
    hold_t: float = 12.0

    def groups(self, path: TourPath) -> list[tuple[float, float, str]]:
        n = len(path.stops)
        g = max(min(len(self.storms), n), 1)
        times = path.stop_times()
        out = []
        for gi in range(g):
            a = gi * n // g
            b = (gi + 1) * n // g
            t0 = 0.0 if gi == 0 else times[a - 1][1]          # a group starts when the camera leaves the previous group's last place
            t1 = path.duration if gi == g - 1 else times[b - 1][1]
            out.append((t0, t1, self.storms[gi] if self.storms else ""))
        return out

    def at(self, path: TourPath, T: float) -> tuple[str | None, float]:
        """(storm name or None to keep the current storm, hour)."""
        gs = self.groups(path)
        for t0, t1, name in gs:
            if T <= t1 or (t0, t1, name) == gs[-1]:
                if self.time_mode == "hold":
                    return (name or None), self.hold_t
                k = min(max((T - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
                return (name or None), 24.0 * k
        return None, self.hold_t


# ------------------------------------------------------------------------------------------------ captions
def depth_text(series, t: float, peak: float | None, label: str) -> str:
    """One caption line: depth at the place as of hour t and its peak, with the source."""
    from .data.depthtiles import mix_hours
    if series is None or not np.isfinite(np.asarray(series)).any():
        return "depth at this place: not available"
    d = mix_hours(np.asarray(series), t)
    pk = float(np.nanmax(series)) if peak is None or not math.isfinite(peak) else float(peak)
    return f"water here {d:.2f} m at hour {t:.1f}, peak {pk:.2f} m  -  {label}"


# ------------------------------------------------------------------------------------------------ shots (exporter)
@dataclass
class ShotState:
    view: View
    t: float
    storm: str | None = None
    caption: dict | None = None        # {"title", "subtitle", "alpha", "stop"}


class TimeLaw:
    """Hour as a function of clip progress u in [0, 1]: held, or a range travelled `loops` times with optional easing."""

    def __init__(self, mode: str = "range", t_in: float = 0.0, t_out: float = 24.0, loops: float = 1.0, hold: float = 12.0,
                 ease: bool = False):
        self.mode, self.t_in, self.t_out, self.loops, self.hold, self.ease = mode, t_in, t_out, loops, hold, ease

    def at(self, u: float) -> float:
        if self.mode == "hold":
            return float(self.hold)
        u = min(max(u, 0.0), 1.0)
        x = u * self.loops
        if self.loops > 1.0:
            x = 1.0 if u >= 1.0 else x % 1.0
        if self.ease:
            x = smoothstep(x)
        return float(self.t_in + (self.t_out - self.t_in) * x)


class Shot:
    duration: float = 5.0

    def state_at(self, s: float) -> ShotState:
        raise NotImplementedError


class OrbitShot(Shot):
    """The current view, optionally orbiting `degrees` around the target (0 = a static camera)."""

    def __init__(self, view: View, duration: float, time: TimeLaw, degrees: float = 0.0):
        self.view, self.duration, self.time, self.degrees = tuple(view), float(duration), time, float(degrees)

    def state_at(self, s):
        u = min(max(s / max(self.duration, 1e-6), 0.0), 1.0)
        v = self.view
        return ShotState((v[0], v[1], v[2], (v[3] + self.degrees * smoothstep(u)) % 360.0, v[4]), self.time.at(u))


class FlightShot(Shot):
    """Camera flight from view a to view b (two bookmarks) with ease in / out."""

    def __init__(self, a: View, b: View, duration: float, time: TimeLaw):
        self.a, self.b, self.duration, self.time = tuple(a), tuple(b), float(duration), time

    def state_at(self, s):
        u = min(max(s / max(self.duration, 1e-6), 0.0), 1.0)
        return ShotState(view_lerp(self.a, self.b, smootherstep(u)), self.time.at(u))


class TourShot(Shot):
    """A whole tour (TourPath + TourPlan) at its natural speed times `speed`."""

    def __init__(self, path: TourPath, plan: TourPlan, speed: float = 1.0, caption_fn=None):
        self.path, self.plan, self.speed = path, plan, float(speed)
        self.duration = path.duration / max(self.speed, 1e-3)
        self.caption_fn = caption_fn

    def state_at(self, s):
        T = min(max(s * self.speed, 0.0), self.path.duration)
        ts = self.path.at(T)
        storm, t = self.plan.at(self.path, T)
        cap = None
        if ts.caption_stop is not None:
            stop = self.path.stops[ts.caption_stop]
            cap = {"title": stop.name, "subtitle": stop.label + (f" ({stop.detail})" if stop.detail else ""),
                   "alpha": ts.caption_alpha, "stop": ts.caption_stop}
            if self.caption_fn is not None:
                cap["extra"] = self.caption_fn(stop, storm, t)
        return ShotState(ts.view, t, storm, cap)


def clip_frames(duration: float, fps: float) -> int:
    return max(int(round(duration * fps)), 1)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")[:60] or "clip"
