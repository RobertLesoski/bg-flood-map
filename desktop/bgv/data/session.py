""".bgview session files: camera, domain, storm, hour, layers, bookmarks and filters in one JSON file that can be reopened.

collect(app, ui) -> dict      apply(app, ui, d) -> list[str] (what could not be restored)      save(path, d) / load(path)
Unknown keys are ignored and missing ones keep the current value, so older / newer files open.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

FORMAT = "bgview"
VERSION = 1


def collect(app, ui=None) -> dict:
    cam = app.camera
    d = {"format": FORMAT, "version": VERSION, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
         "domain": app.domain.key, "storm": app.storm.name if app.storm is not None else None, "hour": float(app.t),
         "camera": [float(cam.target[0]), float(cam.target[1]), float(cam.dist), float(cam.yaw), float(cam.pitch)],
         "exag": float(getattr(app, "exag", 1.0)),
         "layers": {L.name: bool(L.enabled) for L in getattr(app, "layers", []) if getattr(L, "name", None)}}
    try:
        cin = app.get_layer("cinematic")
        if cin is not None:
            d["bookmarks"] = json.loads(json.dumps(cin.save_state().get("bookmarks"), default=str))
    except Exception:
        pass
    if ui is not None:
        st = ui.st
        d["ui"] = {"risk_tiers": list(st.risk_tiers), "risk_search": st.risk_search, "risk_min_depth": float(st.risk_min_depth),
                   "use_tiles": bool(getattr(st, "use_tiles", False))}
        d["ext"] = json.loads(json.dumps(app.settings.get("ext", {}), default=str))
    return d


def apply(app, ui, d: dict) -> list[str]:
    """Restore a session.  A domain switch must be done by the caller first (it is asynchronous in the window)."""
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise ValueError("not a .bgview session file")
    missed = []
    s = d.get("storm")
    if s:
        if s in app.storms_available:
            app.set_storm(s)
        else:
            missed.append(f"storm {s} is not available in this view")
    try:
        app.t = min(24.0, max(0.0, float(d.get("hour", app.t))))
    except (TypeError, ValueError):
        missed.append("hour")
    c = d.get("camera")
    if isinstance(c, list) and len(c) == 5:
        try:
            app.camera.set_view(*(float(v) for v in c))
        except Exception:
            missed.append("camera")
    for name, on in (d.get("layers") or {}).items():
        L = app.get_layer(name)
        if L is not None and bool(on) != bool(L.enabled):
            L.enabled = bool(on)
    bm = d.get("bookmarks")
    if bm is not None:
        try:
            cin = app.get_layer("cinematic")
            st = cin.save_state()
            st["bookmarks"] = bm
            cin.load_state(st)
        except Exception:
            missed.append("bookmarks")
    if ui is not None:
        u = d.get("ui") or {}
        try:
            if "risk_tiers" in u:
                ui.st.risk_tiers = [bool(x) for x in u["risk_tiers"]]
            ui.st.risk_search = str(u.get("risk_search", ui.st.risk_search))
            ui.st.risk_min_depth = float(u.get("risk_min_depth", ui.st.risk_min_depth))
            if "use_tiles" in u:
                ui.st.use_tiles = bool(u["use_tiles"])
            f = ui.risk_filter
            f.tiers, f.search, f.min_depth_m = tuple(ui.st.risk_tiers), ui.st.risk_search, ui.st.risk_min_depth
        except Exception:
            missed.append("filters")
        ext = d.get("ext")
        if isinstance(ext, dict):
            app.settings.setdefault("ext", {}).update(ext)
    return missed


def save(path, d: dict) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".bgview":
        p = p.with_suffix(".bgview")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return p


def load(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise ValueError("not a .bgview session file")
    return d
