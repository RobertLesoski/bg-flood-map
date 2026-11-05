"""Developer tool: render ONE offscreen frame of the UI in a chosen state (headless, never opens a window).

    python -m bgv.ui.devshot --out x.png --size 1920x1080 --storm 25yr --hour 12 \
        [--mouse 960,540] [--select auto|<id>] [--show about,help,source,perf] [--hide-ui] [--scale 1.0]
        [--pixel-ratio 1.0] [--camera x,y,dist,yaw,pitch] [--search TEXT] [--tiers 1,1,0,0] [--collapse stats,info]
        [--timeline both|flooded|wet_area] [--play-hours N] [--toast TEXT]

Waits for the stats hub (current snapshot; the whole timeline unless --fast) so the panels show real numbers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--storm", default="25yr")
    ap.add_argument("--hour", type=float, default=12.0)
    ap.add_argument("--camera")
    ap.add_argument("--at", help="lng,lat of the camera target")
    ap.add_argument("--mouse")
    ap.add_argument("--select")
    ap.add_argument("--show", default="")
    ap.add_argument("--hide-ui", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pixel-ratio", type=float, default=1.0)
    ap.add_argument("--search", default="")
    ap.add_argument("--tiers", default="")
    ap.add_argument("--collapse", default="")
    ap.add_argument("--expand", default="")
    ap.add_argument("--timeline", default="")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--toast", default="")
    ap.add_argument("--anim", type=float, default=4.0)
    ap.add_argument("--no-sidebar", action="store_true")
    ap.add_argument("--no-atrisk", action="store_true")
    ap.add_argument("--imperial", action="store_true")
    ap.add_argument("--domain", choices=("city", "county"))
    ap.add_argument("--tile-stats", action="store_true", help="open the tile comparison in the statistics")
    ap.add_argument("--no-tiles", action="store_true")
    ap.add_argument("--theme", choices=("dark", "light"), default="dark")
    ap.add_argument("--palette", choices=("std", "cvd"), default="std")
    ap.add_argument("--sun")
    ap.add_argument("--enable", default="", help="comma list of layer names to switch on (e.g. tiledepth)")
    ap.add_argument("--disable", default="", help="comma list of layer names to switch off (e.g. water)")
    ap.add_argument("--wait-tiles", action="store_true", help="wait for the footprint index / statistics before drawing")
    args = ap.parse_args(argv)

    import numpy as np
    from PIL import Image
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from .. import config, tools
    from ..gpu.context import GpuContext
    from . import bridge

    w, h = tools.parse_size(args.size)
    ctx = GpuContext.create(verbose=False)
    canvas = OffscreenRenderCanvas(size=(w, h), pixel_ratio=args.pixel_ratio)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:               # the subset of CLI args tools.make_app / apply_view_args read
        storm = args.storm
        msaa = 4
        quiet = True
        exag = None
        sun = args.sun
        hour = args.hour
        camera = args.camera
        at = args.at
        ripple = None
        haze = None
        domain = args.domain
    app = tools.make_app(A, ctx, fmt)
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=not args.fast)
    ov = app.get_layer("overlay")
    if ov is None:
        print("no overlay layer", file=sys.stderr)
        return 2
    m = ov.manager
    st = ov.state
    st.ui_scale = args.scale
    st.hidden = args.hide_ui
    st.seen_help = True
    st.show_sidebar = not args.no_sidebar
    st.show_atrisk = not args.no_atrisk
    st.imperial = args.imperial
    st.tile_stats = bool(args.tile_stats)
    st.use_tiles = not args.no_tiles
    st.theme_mode = args.theme
    app.palette = args.palette
    for nm in [x.strip() for x in args.enable.split(",") if x.strip()]:
        lay = app.get_layer(nm)
        if lay is not None:
            lay.enabled = True
    for nm in [x.strip() for x in args.disable.split(",") if x.strip()]:
        lay = app.get_layer(nm)
        if lay is not None:
            lay.enabled = False
    for k in (args.show or "").split(","):
        k = k.strip()
        if k == "about":
            m.show_about = True
        elif k == "help":
            m.show_help = True
        elif k == "source":
            m.show_source = True
        elif k == "perf":
            st.show_perf = True
        elif k == "atrisk":
            m.narrow_list = True
    for k in (args.collapse or "").split(","):
        if k.strip():
            st.sections[k.strip()] = False
    for k in (args.expand or "").split(","):
        if k.strip():
            st.sections[k.strip()] = True
    if args.timeline:
        st.timeline_mode = args.timeline
    if args.search:
        m.risk_filter.search = args.search
    if args.tiers:
        m.risk_filter.tiers = tuple(bool(int(x)) for x in args.tiers.split(","))
        m.sync_risk_filter()
    if args.toast:
        m.toasts.push(args.toast, None, 600.0)
    # statistics
    hub = None
    if bridge.ensure_hub(app):
        hub = app._stats_hub
        hub.wait_ready(120.0)
        if not args.fast:
            hub.wait_timeline(180.0)
    canvas.set_physical_size(w, h)
    canvas.request_draw(app.draw_frame)
    app.anim_time = args.anim
    if args.select:
        b = bridge.buildings(app)
        fl = bridge.flood(app)
        if args.select == "auto" and fl and fl.get("buildings") is not None and len(fl["buildings"].order):
            bid = int(fl["buildings"].order[0 if True else 0])
        else:
            bid = int(args.select) if args.select != "auto" else None
        if bid is not None:
            bridge.selection(app)["building"] = bid
            if b is not None and not args.camera:
                from .cameranav import building_view
                app.camera.set_view(*building_view(float(b.cx[bid]), float(b.cy[bid]), float(b.height_m[bid]), app.camera))
    if args.wait_tiles:
        from ..data import depthapi
        svc = depthapi.service(app)
        for _ in range(3):                                   # tick builds the index once the scene is ready
            canvas.draw()
        t0 = time.time()
        while svc.index_state in ("idle", "building") and time.time() - t0 < 300:
            canvas.draw()
            time.sleep(0.2)
        svc.building_depths(app.storm.name, int(round(app.t)), block=True)
        svc.building_depths(app.storm.name, "peak", block=True)
        print(f"[devshot] tile index {svc.index_state} {svc.index_msg} in {time.time() - t0:.1f} s")
    td = app.get_layer("tiledepth")
    if td is not None and td.enabled:                         # the overlay window is read in a worker and uploaded over several frames
        t0 = time.time()
        while (td._key is None or td._up is not None or td._ready is not None) and time.time() - t0 < 90:
            app._t_last = None
            canvas.draw()
            time.sleep(0.05)
        for _ in range(3):
            canvas.draw()
        print(f"[devshot] tile overlay: {td.status}, level {td.level}, {time.time() - t0:.1f} s")
    io = app.imgui_renderer.backend.io
    if args.mouse:
        mx, my = (float(v) for v in args.mouse.split(","))
        io.add_mouse_pos_event(mx, my)
    for i in range(4):
        app.anim_time = args.anim
        app._t_last = None
        img = canvas.draw()
        if args.mouse:
            io.add_mouse_pos_event(mx + (i % 2) * 0.1, my)
            time.sleep(0.15)                      # let the probe worker finish
    arr = np.asarray(img)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr[..., :3].copy()).save(out, optimize=True)
    print(f"[devshot] wrote {out} {w}x{h}  errors: {list(app.errors) or 'none'}  panel errors: {m.panel_errors or 'none'}")
    app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
