"""Ground probe: depth / speed / direction under the mouse (CPU, off the render thread).

    r = sample_probe(storm, t, x, y, sampler)           synchronous (tests, one-off)
    w = ProbeWorker(); w.request(storm, t, x, y)         non-blocking: the newest request wins
    w.latest -> ProbeResult | None                       result of the newest finished request

`Storm.sample_t` may have to gunzip an hour (~0.1 s) the first time, so it never runs on the render thread.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from .units import bearing_deg


@dataclass
class ProbeResult:
    x: float
    y: float
    t: float
    storm: str
    ground_m: float
    depth_m: float
    u: float
    v: float
    speed_ms: float
    bearing: float
    valid: bool
    key: tuple = ()


def sample_probe(storm, t: float, x: float, y: float, sampler=None, key: tuple = ()) -> ProbeResult:
    s = storm.sample_t(t, x, y)
    u, v = float(s["u"]), float(s["v"])
    sp = float(s["speed"])
    g = float(sampler.sample(x, y)) if sampler is not None else float("nan")
    d = float(s["depth"])
    if d < 0.02:                          # below the data's wet threshold: report dry, no phantom velocity
        d, u, v, sp = 0.0, 0.0, 0.0, 0.0
    return ProbeResult(x, y, float(t), storm.name, g, d, u, v, sp, bearing_deg(u, v), bool(s["valid"]), key)


class ProbeWorker:
    def __init__(self):
        self._cv = threading.Condition()
        self._req = None
        self._stop = False
        self.latest: ProbeResult | None = None
        self.busy = False
        self.error: str | None = None
        self._th = threading.Thread(target=self._run, name="bgv-probe", daemon=True)
        self._th.start()

    def request(self, storm, t: float, x: float, y: float, sampler=None, key: tuple = ()):
        with self._cv:
            self._req = (storm, float(t), float(x), float(y), sampler, key)
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._req is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                req, self._req = self._req, None
                self.busy = True
            try:
                self.latest = sample_probe(*req[:5], key=req[5])
                self.error = None
            except Exception as e:                       # never kill the worker
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self.busy = False

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
