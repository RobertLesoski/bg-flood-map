"""Headless tools: --render (one offscreen frame to PNG), --benchmark (scripted camera path, frame-time stats),
--selftest (3 s window).  None of them (except --selftest) opens a window."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .app import App
from .gpu.context import GpuContext


def parse_size(s: str | None) -> tuple[int, int]:
    w, _, h = (s or "1920x1080").lower().partition("x")
    return max(int(w), 16), max(int(h), 16)


def parse_floats(s: str, n: int, what: str) -> list[float]:
    parts = [p for p in s.replace(";", ",").split(",") if p.strip() != ""]
    if len(parts) != n:
        raise ValueError(f"{what} needs {n} comma-separated numbers, got {s!r}")
    return [float(p) for p in parts]


def make_app(args, ctx: GpuContext, colour_format: str = "rgba8unorm-srgb", settings: dict | None = None) -> App:
    return App(ctx, config.data_root(), storm=args.storm, msaa=args.msaa, quiet=getattr(args, "quiet", False),
               settings=settings if settings is not None else {}, colour_format=colour_format)


def apply_view_args(app: App, args):
    cam = app.camera
    if getattr(args, "exag", None) is not None:
        app.set_exag(args.exag)
    if getattr(args, "sun", None):
        app.sun_az, app.sun_el = parse_floats(args.sun, 2, "--sun")
    if getattr(args, "hour", None) is not None:
        app.t = float(args.hour)
    if getattr(args, "camera", None):
        x, y, dist, yaw, pitch = parse_floats(args.camera, 5, "--camera")
        cam.set_view(x, y, dist, yaw, pitch)
    if getattr(args, "at", None):
        lng, lat = parse_floats(args.at, 2, "--at")
        X, Y = app.terrain.frame.lnglat_to_world(lng, lat)
        cam.set_view(X, Y, cam.dist, cam.yaw, cam.pitch)
    if getattr(args, "ripple", None) is not None:
        lw = app.get_layer("water")
        if lw:
            lw.ripple = args.ripple
    if getattr(args, "haze", None) is not None:
        app.haze_density = args.haze


def wait_for_flow(app: App, all_hours: bool = False, timeout: float = 120.0) -> float:
    t0 = time.perf_counter()
    while True:
        app.flow.pump()
        done = (app.flow.progress()[0] >= app.flow.progress()[1]) if all_hours else app.flow.ready_for(app.t)
        if done and (app.flow._cur is None):
            break
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError("flow hours did not finish loading")
        time.sleep(0.002)
    for fn in list(getattr(app, "ready_waiters", {}).values()):     # layers with async data (buildings, roads, stats)
        try:
            fn(timeout)
        except Exception as e:
            app.report_error("ready_waiter", e)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------------------------------- --render
def run_render(args) -> int:
    from PIL import Image
    from rendercanvas.offscreen import OffscreenRenderCanvas
    t_start = time.perf_counter()
    w, h = parse_size(args.size)
    ctx = GpuContext.create(verbose=not args.quiet)
    canvas = OffscreenRenderCanvas(size=(w, h), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
    app = make_app(args, ctx, fmt)
    app.attach_canvas(canvas, ui=args.ui)
    apply_view_args(app, args)
    wait_for_flow(app, all_hours=False)
    canvas.set_physical_size(w, h)
    canvas.request_draw(app.draw_frame)
    app.anim_time = float(args.anim)
    img = canvas.draw()
    # second draw: all one-time costs (shader compile, first-use texture clears) are paid; keep the animation phase
    app.anim_time = float(args.anim)
    app._t_last = None
    img = canvas.draw()
    arr = np.asarray(img)
    out = Path(args.render)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr[..., :3].copy()).save(out, optimize=True)
    if not args.quiet:
        print(f"[bgv] wrote {out} ({w}x{h}) in {time.perf_counter() - t_start:.1f} s; storm {app.storm.name} t={app.t:.2f}",
              flush=True)
    app.shutdown()
    return 0


# ------------------------------------------------------------------------------------------------- --benchmark
def _stats(a) -> dict:
    a = np.asarray(a, np.float64)
    if a.size == 0:
        return {}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "min": float(a.min()), "n": int(a.size)}


def bench_camera(app: App, f: float, base_dist: float):
    """Scripted path, f in [0,1): overview -> dive to the downtown / river area -> orbit at street scale -> pull back."""
    cam = app.camera
    fr = app.terrain.frame
    X, Y = fr.lnglat_to_world(-86.44, 36.99)
    if f < 0.25:                      # overview, slow orbit
        k = f / 0.25
        cam.set_view(0.0, 0.0, base_dist, 360.0 * 0.25 * k, 55.0)
    elif f < 0.5:                     # dive
        k = (f - 0.25) / 0.25
        s = k * k * (3 - 2 * k)
        cam.set_view(X * s, Y * s, base_dist * (1 - s) + 700.0 * s, 90.0 - 70.0 * s, 55.0 - 20.0 * s)
    elif f < 0.8:                     # street-scale orbit
        k = (f - 0.5) / 0.3
        cam.set_view(X, Y, 700.0 - 450.0 * math.sin(k * math.pi), 20.0 + 360.0 * k, 35.0 + 10.0 * math.sin(k * 2 * math.pi))
    else:                             # pull back out
        k = (f - 0.8) / 0.2
        s = k * k * (3 - 2 * k)
        cam.set_view(X * (1 - s), Y * (1 - s), 700.0 * (1 - s) + base_dist * s, 20.0 - 20.0 * s, 35.0 + 20.0 * s)


def run_benchmark(args) -> int:
    import wgpu
    t_proc0 = getattr(sys.modules.get("__main__"), "T_PROCESS_START", time.perf_counter())
    w, h = parse_size(args.size)
    ctx = GpuContext.create(verbose=not (args.quiet or args.json))
    args.quiet = args.quiet or args.json
    app = make_app(args, ctx, "rgba8unorm-srgb")
    apply_view_args(app, args)
    tex = ctx.device.create_texture(label="bench-target", size=(w, h, 1), format=app.colour_format,
                                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    view = tex.create_view()
    base_dist = 16000.0
    # first frame (includes shader compile): time to first frame from process start
    wait_for_flow(app, all_hours=False)
    app.camera.set_view(0.0, 0.0, base_dist, 0.0, 55.0)
    app.step(0.0)
    app.render_scene(view, w, h, 0.0)
    ctx.sync()
    t_first = time.perf_counter() - t_proc0
    wait_for_flow(app, all_hours=True)
    secs = float(args.seconds)
    fps_target = 60.0
    n = max(int(secs * fps_target), 30)
    dt = 1.0 / fps_target
    walls, cpus, gpus = [], [], []
    app.playing = True
    app.t = 6.0
    app.perf.gc.enter_frames()                       # as in the window: data frozen, collector off during frames, young GC every ~2 s
    for i in range(n + 30):
        f = (max(i - 30, 0) / n) % 1.0
        bench_camera(app, f, base_dist)
        app.t = 6.0 + 16.0 * f if i >= 30 else 6.0
        app.playing = False
        app.step(dt)
        t0 = time.perf_counter()
        cpu = app.render_scene(view, w, h, dt)
        ctx.sync()
        t1 = time.perf_counter()
        app.perf.gc.tick()
        g = app.timer.read_blocking()
        if i >= 30:
            walls.append((t1 - t0) * 1000.0)
            cpus.append(cpu)
            if g is not None:
                gpus.append(g)
    # free-running (pipelined, only a final sync): what a vsync-off window would reach
    m = max(int(n // 2), 60)
    t0 = time.perf_counter()
    cpus_free = []
    for i in range(m):
        bench_camera(app, (i / m) % 1.0, base_dist)
        app.t = 6.0 + 16.0 * (i / m)
        app.step(dt)
        cpus_free.append(app.render_scene(view, w, h, dt))
        app.perf.gc.tick()
    ctx.sync()
    free_fps = m / (time.perf_counter() - t0)
    # storm switch: how long until water is back, and the worst frame while the new storm decodes + uploads
    others = [s for s in app.storms_available if s != app.storm.name]
    switch = {}
    if others:
        bench_camera(app, 0.1, base_dist)
        app.t = 12.0
        app.step(dt)
        fts = []
        ts = time.perf_counter()
        app.set_storm(others[0])
        call_ms = (time.perf_counter() - ts) * 1000.0
        ready_ms = None
        while True:
            f0 = time.perf_counter()
            app.step(dt)
            app.render_scene(view, w, h, dt)
            ctx.sync()
            fts.append((time.perf_counter() - f0) * 1000.0)
            if ready_ms is None and app.flow.ready_for(app.t):
                ready_ms = (time.perf_counter() - ts) * 1000.0
            if app.flow.progress()[0] >= app.flow.progress()[1] and app.flow._cur is None:
                break
            if time.perf_counter() - ts > 60:
                break
        switch = {"call_ms": call_ms, "water_ready_ms": ready_ms, "all_hours_ms": (time.perf_counter() - ts) * 1000.0,
                  "frame_ms_during_load": _stats(fts), "max_pump_ms": app.flow.max_pump_ms}
    mean = float(np.mean(walls))
    res = {
        "adapter": ctx.adapter_name, "backend": ctx.backend, "size": [w, h], "msaa": app.sample_count,
        "frames": len(walls), "storm": args.storm or "25yr", "gpu_timestamps": bool(gpus),
        "frame_ms_wall_synced": _stats(walls), "fps_synced_mean": 1000.0 / mean,
        "cpu_record_ms": _stats(cpus), "cpu_record_ms_pipelined": _stats(cpus_free),
        "gpu_pass_ms": _stats(gpus) if gpus else None,
        "fps_free_running": free_fps,
        "startup_to_first_frame_s": t_first,
        "storm_switch": switch,
        "vram_mb_estimate": ctx.vram_mb(), "vram_breakdown_mb": ctx.vram_breakdown(),
        "chunks_last": [app.terrain_mesh.n_visible, app.terrain_mesh.nx * app.terrain_mesh.ny],
        "terrain_load_s": app.t_startup.get("terrain"), "init_s": app.t_startup.get("init"),
        "errors": list(app.errors),
        "quality": {"mode": app.perf.gov.mode, "level": app.perf.gov.level, "render_scale": app.render_scale},
    }
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        s = res["frame_ms_wall_synced"]
        print(f"[bench] {ctx.adapter_name} {w}x{h} msaa{app.sample_count}: {res['fps_synced_mean']:.1f} fps (synced) "
              f"mean {s['mean']:.2f} ms  p95 {s['p95']:.2f}  p99 {s['p99']:.2f}  max {s['max']:.2f}")
        if gpus:
            g = res["gpu_pass_ms"]
            print(f"[bench] GPU pass: mean {g['mean']:.2f} ms  p95 {g['p95']:.2f}  p99 {g['p99']:.2f}  max {g['max']:.2f}")
        print(f"[bench] CPU record mean {res['cpu_record_ms']['mean']:.2f} ms (synced), "
              f"{res['cpu_record_ms_pipelined']['mean']:.2f} ms (pipelined); free-running {free_fps:.1f} fps")
        print(f"[bench] first frame {t_first:.2f} s after process start; VRAM ~{res['vram_mb_estimate']:.0f} MB "
              f"{res['vram_breakdown_mb']}")
        if switch:
            print(f"[bench] storm switch: call {switch['call_ms']:.1f} ms, water ready {switch['water_ready_ms']:.0f} ms, "
                  f"all hours {switch['all_hours_ms']:.0f} ms, worst frame {switch['frame_ms_during_load'].get('max', 0):.1f} ms")
        if app.errors:
            print("[bench] errors:", list(app.errors))
    app.shutdown()
    return 0


# --------------------------------------------------------------------------------------------------- windows
def open_window(args, title="BG Flood Viewer", selftest=False):
    """Create the real window + app and return (app, canvas, loop)."""
    root = config.data_root()                          # first: a wrong data folder must fail before any window appears
    from rendercanvas.auto import RenderCanvas, loop
    w, h = parse_size(args.size) if args.size else (1600, 900)
    ctx = GpuContext.create()
    canvas = RenderCanvas(size=(w, h), title=title, update_mode="continuous", max_fps=1000,
                          vsync=not args.no_vsync)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
    settings = {} if selftest else config.load_settings()
    app = App(ctx, root, storm=args.storm, msaa=args.msaa, settings=settings, colour_format=fmt)
    app.attach_canvas(canvas, ui=True)
    apply_view_args(app, args)
    canvas.request_draw(app.draw_frame)
    return app, canvas, loop


def run_window(args) -> int:
    """The real window: it opens at once with a loading screen (bgv/launch.py); everything else loads behind it."""
    from . import launch
    config.data_root()                                  # a wrong data folder must fail before any window appears
    L, canvas, loop = launch.start(args)

    def on_close(ev=None):
        app = L.app
        if app is None or app.loading:                  # never save the settings of a half-built app
            return
        try:
            app.save_settings()
        except Exception:
            pass
    canvas.add_event_handler(on_close, "close")
    loop.run()
    on_close()
    if L.error is not None:
        raise L.error
    if L.app is not None:
        L.app.shutdown()
        err = getattr(L.app, "fatal_error", None)
        if err is not None:                             # the frame loop failed for good (see App.draw_frame)
            raise err
    return 0


def run_selftest(args) -> int:
    from . import launch
    t0 = time.perf_counter()
    config.data_root()
    L, canvas, loop = launch.start(args, title="BG Flood Viewer (selftest)", selftest=True)
    n0 = {"n": 0, "t": None}
    canvas.add_event_handler(lambda ev: None, "close")

    st = {}

    def send(kind, x=0.0, y=0.0, button=0, buttons=(), **kw):
        # the same call the glfw callbacks make (canvas.submit_event): proves mouse / keyboard reach the camera on screen
        canvas.submit_event({"event_type": kind, "x": float(x), "y": float(y), "button": button, "buttons": tuple(buttons),
                             "modifiers": (), **kw})

    def on_loaded(app):
        app.playing = True
        app.t = 8.0
        orig = app.draw_frame
        script = []                                     # (seconds after the first frame, action), run from the frame loop:
        # loop.call_later fired early / unreliably under the real glfw loop, the frame loop's own clock is exact

        def draw():
            t_a = time.perf_counter()
            orig()
            now = time.perf_counter()
            if "first_ms" not in st:
                st["first_ms"] = (now - t_a) * 1000.0
            st.setdefault("stamps", []).append((t_a, now))
            if n0["t"] is None:
                n0["t"] = now
            n0["n"] += 1
            while script and now - n0["t"] >= script[0][0]:
                script.pop(0)[1]()
        canvas.request_draw(draw)

        def centre():
            ov = app.get_layer("overlay")
            L_ = ov.manager.L if ov is not None and ov.manager is not None else None
            if L_ is None or L_.sx1 <= L_.sx0:
                lw, lh = canvas.get_logical_size()
                return lw / 2, lh / 2
            return (L_.sx0 + L_.sx1) / 2, (L_.sy0 + L_.sy1) / 2

        def start_drag():
            cx, cy = centre()
            st["t_start"] = app.camera.target.copy()
            st["d_start"] = float(app.camera.dist)
            send("pointer_move", cx, cy)
            send("pointer_down", cx, cy, 1, (1,))
            t_now = time.perf_counter() - n0["t"]
            for i in range(1, 7):
                script.append((t_now + 0.05 * i, lambda i=i: send("pointer_move", cx - 25 * i, cy + 6 * i, 0, (1,))))
            script.append((t_now + 0.45, lambda: send("pointer_up", cx - 150, cy + 36, 1, ())))
            script.sort(key=lambda s: s[0])

        def wheel():
            cx, cy = centre()
            st["t_pan_end"] = app.camera.target.copy()
            send("pointer_move", cx, cy)                 # imgui tracks the last pointer position, not the wheel event's;
            # its want_capture_mouse flag is one frame old, so the wheel follows a few frames after the move
            script.append((time.perf_counter() - n0["t"] + 0.06, lambda: send("wheel", cx, cy, dx=0.0, dy=-250.0)))
            script.sort(key=lambda s: s[0])

        def keys_down():
            st["d_zoom_end"] = float(app.camera.dist)
            st["t_key0"] = app.camera.target.copy()
            send("key_down", key="w")
            script.append((time.perf_counter() - n0["t"] + 0.45, lambda: send("key_up", key="w")))
            script.sort(key=lambda s: s[0])

        def close():
            st["t_end"] = app.camera.target.copy()
            canvas.close()
        MOTION0, MOTION_S = 3.2, float(getattr(args, "selftest_seconds", 8.0))

        def motion_start():
            st["t_end"] = app.camera.target.copy()
            st["motion_t0"] = time.perf_counter()
            cx, cy = centre()
            send("pointer_move", cx, cy)
            send("pointer_down", cx, cy, 1, (1,))
            st["m_c"] = (cx, cy)

        def motion_step():
            # one pointer move per frame (an orbit drag) for MOTION_S seconds, so every frame has real camera motion
            t_ = time.perf_counter() - st["motion_t0"]
            cx, cy = st["m_c"]
            send("pointer_move", cx + 200 * math.sin(t_ * 1.7), cy + 60 * math.sin(t_ * 1.1), 0, (1,))

        def motion_end():
            cx, cy = st["m_c"]
            send("pointer_up", cx, cy, 1, ())
            st["motion_t1"] = time.perf_counter()
            canvas.close()
        script.extend([(0.7, start_drag), (1.5, wheel), (2.0, keys_down), (MOTION0, motion_start)])
        n_steps = int(MOTION_S * 240)                  # more steps than frames at any rate; each is a no-op once the motion ended
        script.extend([(MOTION0 + 0.002 + i * 0.004, motion_step) for i in range(n_steps)])
        script.append((MOTION0 + MOTION_S, motion_end))
    L.on_loaded.append(on_loaded)
    loop.call_later(90.0, canvas.close)                 # safety: never stay open if loading hangs
    loop.run()
    el = time.perf_counter() - (n0["t"] or t0)
    app = L.app
    errs = list(app.errors) if app is not None else ["app never started"]
    t = L.times
    print("[selftest] start-up: window %.2f s, first loading frame %.2f s, GPU ready %.2f s, loaded %.2f s (from process start)"
          % (t.get("window", -1), t.get("first_loading_frame", -1), t.get("gpu", -1), t.get("loaded", -1)))
    print(f"[selftest] {n0['n']} frames in {el:.2f} s -> {n0['n'] / max(el, 1e-6):.1f} fps; errors: {errs or 'none'}")
    if L.error is not None:
        print(f"[selftest] start-up failed: {type(L.error).__name__}: {L.error}")
    t_exit0 = time.perf_counter()
    if app is not None:
        app.shutdown()
    t_stop = time.perf_counter() - t_exit0
    panned = float(np.hypot(*(st["t_pan_end"][:2] - st["t_start"][:2]))) if "t_pan_end" in st else 0.0
    zoom = st["d_zoom_end"] / st["d_start"] if st.get("d_zoom_end") and st.get("d_start") else 1.0
    walked = float(np.hypot(*(st["t_end"][:2] - st["t_key0"][:2]))) if "t_key0" in st and "t_end" in st else 0.0
    print(f"[selftest] input through the window canvas: drag panned {panned:.0f} m, wheel zoom x{zoom:.2f}, "
          f"W key moved {walked:.0f} m; shutdown after the close {t_stop * 1000:.0f} ms")
    ok_input = panned > 50.0 and zoom < 0.9 and walked > 20.0
    ok_fps = True
    stamps = st.get("stamps", [])
    if "motion_t0" in st and "motion_t1" in st and len(stamps) > 10:
        ms = [(a, b) for a, b in stamps if st["motion_t0"] <= a <= st["motion_t1"]]
        iv = np.diff([a for a, _ in ms]) * 1000.0
        post = [(b - a) * 1000.0 for a, b in stamps]
        if len(iv) > 5:
            fps = 1000.0 / float(iv.mean())
            p95, p99, mx = (float(np.percentile(iv, q)) for q in (95, 99, 100))
            worst_post = max(post)
            print(f"[selftest] first frame after loading {st.get('first_ms', -1):.0f} ms; steady motion {len(iv)} frames: "
                  f"{fps:.1f} fps, interval mean {iv.mean():.1f} p95 {p95:.1f} p99 {p99:.1f} max {mx:.1f} ms; "
                  f"slowest frame after loading {worst_post:.0f} ms (vsync {'off' if getattr(args, 'no_vsync', False) else 'on'})")
            min_fps = 40.0
            if fps < min_fps or worst_post > 1000.0:
                ok_fps = False
                print(f"[selftest] FRAME RATE CHECK FAILED: need >= {min_fps:.0f} fps steady and no frame > 1 s after loading")
    if not ok_input:
        print("[selftest] INPUT CHECK FAILED: the scripted mouse / key events did not move the camera")
    return 0 if (not errs and n0["n"] > 10 and L.error is None and ok_input and ok_fps) else 1
