"""CPU-only start-up work that can run while the window opens and the GPU adapter is being created.

`Prefetch(root, storm_name, settings)` starts a worker thread that
    1. opens the storm (meta + frame)                              -> .storm
    2. loads / builds the terrain height + slope data (the cache build of a first run, a few hundred ms when cached)
                                                                   -> .terrain_data  (tuple for TerrainTextures(data=...))
    3. imports the modules the main thread needs next (app, layers, numba kernels) so those imports are already done.
`done` becomes True when 1. and 2. finished (3. continues; it only warms the import cache).  Any failure is stored in
`.error` and the App falls back to doing the work itself, in its own steps.
"""
from __future__ import annotations

import threading
import time


class Prefetch:
    def __init__(self, root, storm_name: str | None = None, settings: dict | None = None):
        self.root = root
        self.storm = None
        self.terrain_data = None
        self.error: BaseException | None = None
        self.done = False
        self.warm_done = False                    # numba kernels loaded (the loading screen waits for it)
        self.t_storm = self.t_terrain = self.t_imports = None
        self._name = storm_name
        self._settings = settings or {}
        self.thread = threading.Thread(target=self._run, name="bgv-prefetch", daemon=True)
        self.thread.start()

    def _run(self):
        t0 = time.perf_counter()
        try:
            from .data import flow as flowdata
            avail = flowdata.available_storms(self.root)
            first = self._name or self._settings.get("storm") or ("25yr" if "25yr" in avail else avail[0])
            if first not in avail:
                first = avail[0]
            self.storm = flowdata.Storm.open(self.root, first)
            self.t_storm = time.perf_counter() - t0
            from .data import terrain as terraindata
            self.terrain_data = terraindata.load_terrain_gpu_data(self.root, self.storm.frame)
            self.t_terrain = time.perf_counter() - t0
        except BaseException as e:                 # the App repeats the work itself
            self.error = e
            self.storm = None
            self.terrain_data = None
        finally:
            self.done = True
        try:                                       # warm the import cache for the main thread
            from . import tools  # noqa: F401
            from . import app  # noqa: F401
            from .gpu.layers import registry
            registry.load_layer_classes(lambda m: None)
            from .data import stats  # noqa: F401
            from .gpu import fastcull
            fastcull.warmup()
            self.t_imports = time.perf_counter() - t0
        except BaseException:
            pass
        finally:
            self.warm_done = True


def main(argv=None) -> int:
    """Headless start-up measurement: the windowed start-up sequence (bgv/launch.py) with an offscreen canvas.

        python -m bgv.startup [--json]

    Times are seconds from process start.  The real window adds the glfw window creation (~0.1 - 0.2 s) and shows a GDI
    splash from `window` on; `first_loading_frame` is when the imgui loading screen is first drawn by the GPU."""
    import json
    import sys
    t_proc0 = time.perf_counter()
    as_json = argv is not None and "--json" in argv or (argv is None and "--json" in sys.argv)
    res: dict = {}
    from . import config
    root = config.data_root()
    pre = Prefetch(root, None, {})
    box: dict = {}

    def make_ctx():
        from .gpu.context import GpuContext
        box["ctx"] = GpuContext.create(verbose=False)
        box["t"] = time.perf_counter() - t_proc0
    th = threading.Thread(target=make_ctx, name="gpu-init", daemon=True)
    th.start()
    from rendercanvas.offscreen import OffscreenRenderCanvas
    canvas = OffscreenRenderCanvas(size=(1920, 1080), pixel_ratio=1.0)
    res["window_object_s"] = time.perf_counter() - t_proc0
    th.join()
    res["gpu_ready_s"] = box["t"]
    from .app import App
    ctx = box["ctx"]
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
    app = App(ctx, root, quiet=True, settings={}, colour_format=fmt, defer=True, prefetch=pre)
    app.attach_canvas(canvas, ui=True)
    canvas.set_physical_size(1920, 1080)
    canvas.request_draw(app.draw_frame)
    res["app_created_s"] = time.perf_counter() - t_proc0
    frames = []
    while app.loading:
        t = time.perf_counter()
        canvas.draw()
        if "first_loading_frame_s" not in res:
            res["first_loading_frame_s"] = time.perf_counter() - t_proc0
        frames.append((time.perf_counter() - t) * 1000.0)
        if time.perf_counter() - t_proc0 > 120:
            break
    res["loaded_s"] = time.perf_counter() - t_proc0
    res["loading_frames"] = len(frames)
    res["loading_worst_frame_ms"] = max(frames) if frames else 0.0
    res["loading_mean_frame_ms"] = sum(frames) / max(len(frames), 1)
    res["prefetch_s"] = {"storm": pre.t_storm, "terrain": pre.t_terrain, "imports": pre.t_imports}
    res["errors"] = list(app.errors)
    app.shutdown()
    print(json.dumps(res, indent=1) if as_json else
          "[startup] GPU %.2f s, first loading frame %.2f s, loaded %.2f s; %d loading frames, worst %.0f ms; errors %s"
          % (res["gpu_ready_s"], res["first_loading_frame_s"], res["loaded_s"], res["loading_frames"],
             res["loading_worst_frame_ms"], res["errors"] or "none"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
