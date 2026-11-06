"""'Hours flooded' heat layer: for the displayed storm, the number of hourly solver-tile snapshots with >= 10 cm of water at each
flow-grid cell, draped on the terrain with the flow-paths 'hours' ramp.  Computed once per storm in a worker thread from the
hourly COG tiles (bgv.data.duration) and cached on disk (desktop cache folder).  Off by default; switched from the Depth-duration
panel (Tools menu).  Storms without tiles show nothing (the panel says so).
"""
from __future__ import annotations

import threading

import numpy as np

from .flowpaths import FlowpathsLayer

LABEL = "Hours >= 10 cm (solver tiles, hourly snapshots, +-1 h)"


def to_rgba(hours: np.ndarray) -> np.ndarray:
    """uint8[H, W] hours (255 = no data) -> RGBA texture of the flowpaths shader (R coverage, B value, A valid)."""
    h = np.asarray(hours)
    out = np.zeros(h.shape + (4,), np.uint8)
    wet = (h >= 1) & (h <= 24)
    out[..., 0] = np.where(wet, 255, 0)
    out[..., 2] = np.where(wet, np.round(h.astype(np.float32) / 24.0 * 255.0), 0).astype(np.uint8)
    out[..., 3] = np.where(h != 255, 255, 0)
    return out


class HoursFloodedLayer(FlowpathsLayer):
    name = "hoursflooded"
    title = "Hours flooded (tiles)"
    has_ui = False
    default_enabled = False

    def __init__(self):
        super().__init__()
        self.mode = 2
        self.alpha = 0.8
        self.progress = 0.0
        self.status = ""
        self._app = None

    def init(self, app):
        super().init(app)
        self._app = app

    def update(self, app, dt):
        if not self.enabled:
            return
        self.mode = 2
        super().update(app, dt)

    def _start_load(self, app):
        name = app.storm.name
        self._loading = name
        self._arr = None
        self._rows = 0
        self.error = ""
        self.progress = 0.0
        self.status = "computing"

        def work():
            try:
                from ...data import depthapi, duration
                from ... import config
                svc = depthapi.service(app)
                g = duration.cached_hours_flooded(svc, name, app.terrain.frame, config.cache_dir() / "hoursflooded",
                                                  progress=lambda f: setattr(self, "progress", f))
                if g is None:
                    raise RuntimeError(f"no solver tiles for {name}")
                self.status = "ready"
                self._pending = (name, np.ascontiguousarray(to_rgba(g)))
            except Exception as e:
                self.status = "error"
                self._pending = (name, e)
        threading.Thread(target=work, name="bgv-hoursflooded", daemon=True).start()

    def save_state(self):
        return {"alpha": self.alpha}

    def load_state(self, d):
        try:
            self.alpha = float(d.get("alpha", self.alpha))
        except (TypeError, ValueError, AttributeError):
            pass
