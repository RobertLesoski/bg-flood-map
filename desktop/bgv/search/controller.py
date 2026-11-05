"""SearchController: the state and background work behind the search / facilities / roads / report panels.  No imgui here.

One controller per UiManager (`ui._search_ctl`, created by bgv.ui.panels_search).  Data is loaded for the CURRENT domain in a worker
thread (never in the frame): address index, facilities (+ depth tiles for the county), road lists (per storm, on demand), road
geometry, footprints.  Everything that can be missing reports a user-readable message in `status` instead of raising.

    ctl.sync()                         every frame: notices a domain change, (re)starts the loading
    ctl.hits_for(text)                 typeahead rows (sync, a few ms)
    ctl.select_record(i)               fly to an address, select its building, assess it in the background (-> ctl.result)
    ctl.select_facility(i) / ctl.pick_facility(px, py)
    ctl.road_table() / ctl.select_road(row)
    ctl.request_report(kind)           the report is written after the next frame (post_frame) so the screenshot is the view
"""
from __future__ import annotations

import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import addrindex, assess as asm, facilities as fac, report as rpt, roadrisk, tiles as tl


@dataclass
class Status:
    state: str = "idle"           # idle | loading | ready | missing | error
    message: str = ""
    path: str = ""
    seconds: float = 0.0


@dataclass
class Result:
    rec: object
    assessment: object = None     # assess.Assessment when ready
    pending: bool = True
    building: int | None = None
    error: str = ""
    t0: float = field(default_factory=time.perf_counter)


def _bg_priority():
    try:
        from ..gpu import bgthread
        bgthread.lower_priority()
    except Exception:
        pass


class SearchController:
    HIT_LIMIT = 12

    def __init__(self, ui):
        self.ui = ui
        self.app = ui.app
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bgv-search")
        self._lock = threading.RLock()
        self.domain_key: str | None = None
        self.gen = 0
        self.status: dict[str, Status] = {k: Status() for k in ("index", "facilities", "tiles", "roads", "geometry", "footprints", "help")}
        self.index: addrindex.AddressIndex | None = None
        self.facilities: fac.FacilitySet | None = None
        self.tiles: tl.TileSet | None = None
        self.footprints: asm.Footprints | None = None
        self.geometry: roadrisk.RoadGeometry | None = None
        self.road_tables: dict[str, roadrisk.RoadTable | str] = {}
        self._loading_roads: set = set()
        self._geo_started = False
        self._fp_started = False
        # query / result state
        self._hits_cache: tuple | None = None
        self.result: Result | None = None
        self.hit_assess: dict[int, object] = {}
        self._hit_pending: set = set()
        self.facility_sel: int = -1
        self.road_sel: dict | None = None             # {'id', 'name', 'coords': [[lng, lat]...], 'row': i, 'storm': s}
        self.report_req: dict | None = None
        self.last_report: dict | None = None
        self._fac_z: np.ndarray | None = None
        self._bld_tree_key = None
        self.messages: list[str] = []
        # persisted panel state (settings["search_ui"])
        cfg = (self.app.settings or {}).get("search_ui") or {}
        self.markers_on: bool = bool(cfg.get("markers_on", True))
        self.types_off: set = set(cfg.get("types_off") or [])
        self.onboarded: bool = bool(cfg.get("onboarded", False))
        # panel state (not persisted)
        self.query = ""
        self.committed = False
        self.sel_hit = 0
        self.focus = False
        self.show_fac = False
        self.show_roads = False
        self.dd_hover = False
        self.road_sort = "flooded_pct"
        self.road_asc = False
        self.road_min_pct = 0.0
        self.road_evac = False
        self.road_text = ""
        self.hint_req = False

    def persist(self):
        try:
            self.app.settings["search_ui"] = {"markers_on": self.markers_on, "types_off": sorted(self.types_off),
                                              "onboarded": self.onboarded}
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------ plumbing
    @property
    def root(self) -> Path:
        return Path(self.app.root)

    @property
    def domain(self):
        return self.app.domain

    def storm_key(self) -> str:
        return str(getattr(self.app.storm, "name", ""))

    def _submit(self, fn, *a):
        gen = self.gen

        def run():
            _bg_priority()
            if gen != self.gen:
                return None
            try:
                return fn(*a)
            except Exception as e:                                   # noqa: BLE001
                self._log_exc(f"search worker {getattr(fn, '__name__', fn)}", e)
                return None
        return self._pool.submit(run)

    def _log_exc(self, where, e):
        try:
            from .. import applog
            applog.log().error("%s: %s: %s\n%s", where, type(e).__name__, e, traceback.format_exc())
        except Exception:
            pass

    def close(self):
        self.gen += 1
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self.tiles is not None:
            self.tiles.close()

    # ------------------------------------------------------------------------------------------ domain / loading
    def sync(self):
        key = self.domain.key
        if key != self.domain_key:
            self._reset(key)

    def _reset(self, key: str):
        self.gen += 1
        if self.tiles is not None:
            try:
                self.tiles.close()
            except Exception:
                pass
        self.domain_key = key
        for st in self.status.values():
            st.state, st.message, st.path = "idle", "", ""
        self.index = self.facilities = self.tiles = self.footprints = self.geometry = None
        self.road_tables.clear()
        self._loading_roads.clear()
        self._geo_started = self._fp_started = False
        self._hits_cache = None
        self.result = None
        self.hit_assess.clear()
        self._hit_pending.clear()
        self.facility_sel = -1
        self.road_sel = None
        self._fac_z = None
        self._submit(self._load_index, key)
        self._submit(self._load_facilities, key)

    def _load_index(self, key):
        st = self.status["index"]
        p = addrindex.index_path(self.root, key)
        st.state, st.path = "loading", str(p)
        t0 = time.perf_counter()
        try:
            idx = addrindex.AddressIndex.load(p)
        except addrindex.IndexError_ as e:
            st.state, st.message = ("missing" if not p.is_file() else "error"), str(e)
            return
        if key != self.domain_key:
            return
        self.index = idx
        st.state, st.seconds = "ready", time.perf_counter() - t0
        st.message = f"{idx.n:,} addresses, {len(idx.storm_keys)} storms"

    def _load_facilities(self, key):
        stf, stt = self.status["facilities"], self.status["tiles"]
        p = fac.facilities_path(self.root, key)
        stf.state, stf.path = "loading", str(p)
        try:
            fs = fac.FacilitySet.load(p)
        except fac.FacilityError as e:
            stf.state, stf.message = ("missing" if not p.is_file() else "error"), str(e)
            fs = None
        # tiles (county only)
        stt.state = "loading"
        try:
            ts = tl.TileSet.discover(self.root, key, self.app.settings, storms=None)
        except Exception as e:                                         # noqa: BLE001
            ts = None
            stt.state, stt.message = "error", f"depth tiles cannot be opened: {type(e).__name__}: {e}"
        if ts is None:
            if stt.state != "error":
                stt.state = "missing"
                stt.message = ("no depth tiles for the city yet (index / flow grid is used)" if key != "county" else
                               f"depth tiles not found in {tl.tiles_root(self.root, self.app.settings) / key} (index / flow grid is used)")
            stt.path = str(tl.tiles_root(self.root, self.app.settings) / key)
        else:
            stt.state, stt.path = "ready", str(ts.folder)
            stt.message = f"{len(ts.storms)} storms, {ts.res_m:g} m cells"
        if key != self.domain_key:
            return
        self.tiles = ts
        if fs is not None:
            t0 = time.perf_counter()
            try:
                fs.attach_tiles(ts)
            except Exception as e:                                     # noqa: BLE001
                self._log_exc("facility tiles", e)
                stt.state, stt.message = "error", f"facility depths from tiles failed: {type(e).__name__}: {e}"
            if key != self.domain_key:
                return
            self.facilities = fs
            stf.state, stf.seconds = "ready", time.perf_counter() - t0
            stf.message = f"{fs.n} facilities" + (f", depths from {'tiles' if ts else 'the index'}")

    # ------------------------------------------------------------------------------------------ lazy loads
    def ensure_geometry(self):
        if self._geo_started:
            return
        self._geo_started = True
        self._submit(self._load_geometry, self.domain_key)

    def _load_geometry(self, key):
        st = self.status["geometry"]
        st.state = "loading"
        t0 = time.perf_counter()
        fallback = None
        try:
            fallback = self.domain.scene_paths(self.app.terrain.frame)[1] if key == "county" else None
        except Exception:
            fallback = None
        try:
            g = roadrisk.RoadGeometry.load(self.root, fallback)
        except roadrisk.RoadsError as e:
            st.state, st.message = "missing", str(e)
            return
        if key != self.domain_key:
            return
        self.geometry = g
        st.state, st.seconds, st.message, st.path = "ready", time.perf_counter() - t0, f"{len(g.ways):,} ways", g.source

    def ensure_footprints(self):
        if self._fp_started or self.tiles is None:
            return
        self._fp_started = True
        self._submit(self._load_footprints, self.domain_key)

    def _load_footprints(self, key):
        st = self.status["footprints"]
        st.state = "loading"
        try:
            bpath = self.domain.scene_paths(self.app.terrain.frame)[0]
            fp = asm.Footprints.load(bpath)
        except Exception as e:                                         # noqa: BLE001
            st.state, st.message = "missing", f"building footprints for the tile lookup unavailable ({type(e).__name__}); the cell under the address point is used"
            return
        if key == self.domain_key:
            self.footprints = fp
            st.state, st.message, st.path = "ready", f"{len(fp.polys):,} footprints", str(bpath)

    def road_table(self, storm: str | None = None):
        """RoadTable of the storm, or None while loading / a message string when it cannot be shown."""
        storm = storm or self.storm_key()
        v = self.road_tables.get(storm)
        if v is None and storm not in self._loading_roads:
            self._loading_roads.add(storm)
            self.status["roads"].state = "loading"
            self._submit(self._load_roads, self.domain_key, storm)
        return v

    def _load_roads(self, key, storm):
        try:
            t = roadrisk.RoadTable.load(self.root, key, storm)
            res = t
            self.status["roads"].message = f"{t.n:,} roads with water ({storm})"
            self.status["roads"].state = "ready"
        except roadrisk.RoadsError as e:
            res = str(e)
            self.status["roads"].state, self.status["roads"].message = "missing", str(e)
        if key == self.domain_key:
            self.road_tables[storm] = res
        self._loading_roads.discard(storm)

    # ------------------------------------------------------------------------------------------ typeahead
    def hits_for(self, text: str):
        idx = self.index
        if idx is None:
            return []
        key = (text, id(idx))
        if self._hits_cache is not None and self._hits_cache[0] == key:
            return self._hits_cache[1]
        hits = idx.search(text, self.HIT_LIMIT)
        self._hits_cache = (key, hits)
        # tile figures for the rows shown (background): the dropdown shows the deepest storm from the tiles when they exist
        if self.tiles is not None:
            for h in hits:
                if h.kind == "address" and h.idx not in self.hit_assess and h.idx not in self._hit_pending:
                    self._hit_pending.add(h.idx)
                    self._submit(self._assess_hit, h.idx)
        return hits

    def _assess_hit(self, i):
        self.ensure_footprints()
        fp = self.footprints
        a = asm.assess(self.index, i, self.tiles, fp)
        if len(self.hit_assess) > 400:
            self.hit_assess.clear()
        self.hit_assess[i] = a
        self._hit_pending.discard(i)

    def hit_summary(self, i: int):
        """(depth cm, tier, source label) of the deepest storm for a dropdown row: tiles when assessed, else the index."""
        a = self.hit_assess.get(i)
        if a is not None and a.rows:
            r = max(a.rows, key=lambda r: r.shown_cm)
            return r.shown_cm, r.shown_tier, r.shown_source, True
        d, t = self.index.summary(i)
        pending = self.tiles is not None and i in self._hit_pending
        return d, t, tl.INDEX_LABEL, not pending

    # ------------------------------------------------------------------------------------------ select
    def select_record(self, i: int):
        idx = self.index
        if idx is None:
            return
        rec = idx.record(int(i))
        res = Result(rec)
        res.building = self.resolve_building(rec)           # quick guess (record point); refined in the worker with the footprint
        self.result = res
        self.facility_sel = -1
        self.fly_to(rec.lng, rec.lat, 130.0 if rec.kind == 0 else 160.0)
        self._select_building(res.building)
        self.ensure_footprints()
        self._submit(self._assess_result, res)

    def _assess_result(self, res: Result):
        try:
            if self.tiles is not None:
                for _ in range(100):                       # wait (<= 5 s) for the footprints so the result uses them
                    if self.status["footprints"].state in ("ready", "missing", "error"):
                        break
                    time.sleep(0.05)
            if self.footprints is not None and res.rec.bid >= 0:
                b2 = self.resolve_building(res.rec, self.footprints)
                if b2 is not None and b2 != res.building and self.result is res:
                    res.building = b2
                    self._select_building(b2)
            res.assessment = asm.assess(self.index, res.rec.i, self.tiles, self.footprints)
        except Exception as e:                                         # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
            self._log_exc("assess", e)
        res.pending = False

    def _select_building(self, bid):
        if bid is None:
            return
        try:
            from ..ui import bridge
            bridge.selection(self.app)["building"] = int(bid)
        except Exception:
            pass

    def resolve_building(self, rec, footprints=None) -> int | None:
        """The viewer's building id for an address record, or None (lots, no building near the point).  City: the record's `bid`
        is the feature index in buildings.geojson (checked against the point); county: nearest footprint centroid."""
        try:
            from ..ui import bridge
            sc = bridge.scene(self.app)
            if sc is None or sc.buildings is None or rec.kind != 0:
                return None
            b = sc.buildings
            lng, lat = rec.lng, rec.lat
            if footprints is not None and rec.bid >= 0:                 # county: centroid of the OSM way the record names
                poly = footprints.get(rec.bid)
                if poly:
                    ring = np.asarray(poly[0] if isinstance(poly[0][0][0], (int, float)) else poly[0][0], np.float64)
                    lng, lat = float(ring[:, 0].mean()), float(ring[:, 1].mean())
            x, y = self.app.terrain.frame.lnglat_to_world(lng, lat)
            rad = sc.pick.bld_radius if sc.pick is not None else None
            if self.domain.key == "city" and 0 <= rec.bid < b.n and math.isfinite(float(b.cx[rec.bid])):
                d = math.hypot(float(b.cx[rec.bid]) - x, float(b.cy[rec.bid]) - y)
                if d <= max(25.0, float(rad[rec.bid]) * 1.5 if rad is not None else 25.0):
                    return int(rec.bid)
            pk = sc.pick
            if pk is None or pk.bld_tree is None:
                return None
            d, k = pk.bld_tree.query([x, y])
            bid = int(pk.bld_ids[k])
            if d <= float(rad[bid]) * 1.3 + 6.0:
                return bid
        except Exception as e:                                         # noqa: BLE001
            self._log_exc("resolve_building", e)
        return None

    def record_for_building(self, bid: int) -> int:
        """Address record index of a viewer building (selection -> report), -1 when none matches."""
        idx = self.index
        if idx is None:
            return -1
        try:
            from ..ui import bridge
            b = bridge.buildings(self.app)
            if b is None or not (0 <= bid < b.n):
                return -1
            if self.domain.key == "city":
                i = idx.by_bid(bid)
                if i >= 0:
                    return i
            lng, lat = self.app.terrain.frame.world_to_lnglat(float(b.cx[bid]), float(b.cy[bid]))
            return idx.nearest(lng, lat, max_m=max(12.0, 1.0 * math.sqrt(max(float(b.area_m2[bid]), 1.0))))
        except Exception as e:                                         # noqa: BLE001
            self._log_exc("record_for_building", e)
            return -1

    # ------------------------------------------------------------------------------------------ camera
    def fly_to(self, lng: float, lat: float, dist: float, pitch: float = 52.0, seconds: float = 1.1):
        app = self.app
        x, y = app.terrain.frame.lnglat_to_world(lng, lat)
        self.ui.nav.fly(app.camera, x, y, dist, float(app.camera.yaw), pitch, seconds=seconds)

    def fly_to_bbox(self, w, s, e, n, seconds: float = 1.2):
        fr = self.app.terrain.frame
        x0, y0 = fr.lnglat_to_world(w, s)
        x1, y1 = fr.lnglat_to_world(e, n)
        ext = max(abs(x1 - x0), abs(y1 - y0), 40.0)
        self.ui.nav.fly(self.app.camera, 0.5 * (x0 + x1), 0.5 * (y0 + y1), min(max(ext * 1.5 + 120.0, 220.0), 20000.0),
                        float(self.app.camera.yaw), 50.0, seconds=seconds)

    # ------------------------------------------------------------------------------------------ facilities
    def facility_screen(self, W: float, H: float):
        """(px, py, visible) float arrays of the facility markers (window logical px), or None.  Ground height is cached."""
        fs = self.facilities
        if fs is None or fs.n == 0 or self.app.terrain is None:
            return None
        app = self.app
        if self._fac_z is None or len(self._fac_z) != fs.n:
            fr = app.terrain.frame
            xs, ys = [], []
            for lng, lat in zip(fs.lng, fs.lat):
                x, y = fr.lnglat_to_world(float(lng), float(lat))
                xs.append(x)
                ys.append(y)
            self._fac_xy = (np.array(xs), np.array(ys))
            self._fac_z = np.asarray(app.terrain.sampler.sample(*self._fac_xy), np.float64)
        x, y = self._fac_xy
        z = self._fac_z * float(app.exag) + 4.0
        vp = app.camera.view_proj()
        P = np.stack([x, y, z, np.ones_like(x)], axis=1) @ vp.T
        w = P[:, 3]
        ok = w > 1e-6
        ww = np.where(ok, w, 1.0)
        cam = app.camera
        px = (P[:, 0] / ww * 0.5 + 0.5) * cam.width
        py = (0.5 - P[:, 1] / ww * 0.5) * cam.height
        return px, py, ok

    def pick_facility(self, mx: float, my: float, max_px: float = 14.0) -> int:
        fs = self.facilities
        if fs is None or not getattr(self, "markers_on", True):
            return -1
        s = self.facility_screen(0, 0)
        if s is None:
            return -1
        px, py, ok = s
        mask = self.type_mask()
        d = np.hypot(px - mx, py - my)
        d = np.where(ok & mask, d, np.inf)
        i = int(np.argmin(d))
        return i if d[i] <= max_px else -1

    def type_mask(self):
        fs = self.facilities
        off = getattr(self, "types_off", set())
        if not off:
            return np.ones(fs.n, bool)
        return np.array([t not in off for t in fs.types])

    def select_facility(self, i: int, fly: bool = False):
        self.facility_sel = int(i)
        self.result = None
        if fly and self.facilities is not None:
            f = self.facilities.items[i]
            self.fly_to(f.lng, f.lat, 400.0)

    # ------------------------------------------------------------------------------------------ roads
    def select_road(self, table: roadrisk.RoadTable, row: int):
        wid = int(table.id[row])
        self.ensure_geometry()
        g = self.geometry
        coords = g.coords(wid) if g is not None else None
        self.road_sel = {"id": wid, "name": table.name(row), "coords": coords, "row": int(row), "storm": table.storm,
                         "pending": g is None}
        if coords:
            bb = g.bbox(wid)
            self.fly_to_bbox(*bb)
        return coords is not None

    def resolve_pending_road(self):
        """A road clicked before the geometry finished loading: fly to it as soon as it is available."""
        rs = self.road_sel
        if rs and rs.get("pending") and self.geometry is not None:
            rs["pending"] = False
            c = self.geometry.coords(rs["id"])
            rs["coords"] = c
            bb = self.geometry.bbox(rs["id"])
            if bb:
                self.fly_to_bbox(*bb)

    # ------------------------------------------------------------------------------------------ report
    def request_report(self, kind: str = "auto"):
        """Queue the report for the current result / selected building; it is written by `post_frame` (after the next frame) so
        the PNG is a screenshot of the view as it is."""
        self.report_req = {"kind": kind, "t": time.perf_counter()}

    def build_report(self):
        """ReportData for the current subject: the search result, else the selected building (via its address record, else the live
        snapshot of the current storm).  Returns (data, message)."""
        app = self.app
        label = f"{app.domain.label}" + (" (DEMO DATA)" if getattr(app.domain, "demo", False) else "")
        build = ""
        cur = self.storm_key()
        res = self.result
        if res is not None and res.assessment is not None:
            return rpt.from_assessment(res.assessment, label, current_storm=cur, build=build), ""
        if res is not None and res.pending:
            a = asm.assess(self.index, res.rec.i, self.tiles, self.footprints)
            return rpt.from_assessment(a, label, current_storm=cur, build=build), ""
        from ..ui import bridge
        sel = bridge.selection(app).get("building")
        if sel is not None and self.index is not None:
            i = self.record_for_building(int(sel))
            if i >= 0:
                a = asm.assess(self.index, i, self.tiles, self.footprints)
                return rpt.from_assessment(a, label, current_storm=cur, build=build), ""
        if sel is not None:
            fl = bridge.flood(app)
            b = bridge.buildings(app)
            if fl and b is not None and fl.get("buildings") is not None:
                bs = fl["buildings"]
                sel = int(sel)
                lng, lat = app.terrain.frame.world_to_lnglat(float(b.cx[sel]), float(b.cy[sel]))
                dm = float(bs.depth_m[sel])
                tier = int(bs.tier[sel])
                approx = fl.get("source") == "tiers"
                src = "viewer snapshot (" + ("depth-tier preview" if approx else "solver depth") + f", hour {int(fl['hour'])})"
                return rpt.from_live(str(b.address[sel]).strip(), lng, lat, label, cur, getattr(app.storm, "label", cur),
                                     int(round(dm * 100)), tier, approx, src, int(fl["hour"])), ""
        return None, "Pick a building (click it) or search an address first."

    def run_report(self, capture_png):
        """Called from post_frame.  capture_png() -> RGB array of the view (or None)."""
        req, self.report_req = self.report_req, None
        if req is None:
            return None
        data, msg = self.build_report()
        if data is None:
            self.ui.toasts.push(msg, _warn())
            return None
        png = None
        try:
            png = capture_png()
        except Exception as e:                                         # noqa: BLE001
            self._log_exc("report screenshot", e)
            data.notes.append("The screenshot of the view could not be captured: " + type(e).__name__)
        folder = Path((self.app.settings or {}).get("report_dir") or rpt.default_dir())
        try:
            hp, pp = rpt.write_report(folder, data, png)
        except Exception as e:                                         # noqa: BLE001
            self._log_exc("write report", e)
            self.ui.toasts.push(f"Report failed: {type(e).__name__}: {e}", _err(), 7.0)
            return None
        self.last_report = {"html": str(hp), "png": str(pp) if pp else None, "t": time.time()}
        self.ui.toasts.push(f"Report saved: {hp}", _ok(), 8.0)
        return hp


def _ok():
    from ..ui import theme
    return theme.OK


def _warn():
    from ..ui import theme
    return theme.WARN


def _err():
    from ..ui import theme
    return theme.ERR
