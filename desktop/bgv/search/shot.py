"""Developer tool: render the search / facilities / roads / report / help UI headless (offscreen, never opens a window).

    python -m bgv.search.shot --out x.png --domain county --storm 100yr --size 1920x1080 \
        [--query "1309 scottsville r"] [--pick 0] [--facility 5|name] [--fac] [--roads] [--road-row 0] [--help-tab Help|Shortcuts|"Data sources"|About]
        [--hint] [--report DIR] [--camera x,y,dist,yaw,pitch] [--no-markers] [--hour 12] [--scale 1.0] [--help-query TEXT]
        [--click-facility N] [--bench SECONDS]

--pick N chooses the N-th typeahead hit (as Enter would); --report writes a risk report for the result into DIR (HTML + PNG);
--bench N renders N seconds of frames with the UI on and prints the frame time (fps) after the other steps.
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
    ap.add_argument("--domain", default="city")
    ap.add_argument("--storm", default=None)
    ap.add_argument("--hour", type=float, default=12.0)
    ap.add_argument("--camera")
    ap.add_argument("--query", default="")
    ap.add_argument("--typing", action="store_true", help="leave the dropdown open (do not pick)")
    ap.add_argument("--pick", type=int, default=None)
    ap.add_argument("--facility", default=None)
    ap.add_argument("--fac", action="store_true")
    ap.add_argument("--roads", action="store_true")
    ap.add_argument("--road-row", type=int, default=None)
    ap.add_argument("--help-tab", dest="help_tab", default=None)
    ap.add_argument("--help-query", default="")
    ap.add_argument("--hint", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--bench", type=float, default=0.0)
    ap.add_argument("--tiles-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    from PIL import Image
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from .. import tools
    from ..gpu.context import GpuContext
    from ..ui import bridge, panels_search

    w, h = tools.parse_size(args.size)
    ctx = GpuContext.create(verbose=False)
    canvas = OffscreenRenderCanvas(size=(w, h), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        storm = args.storm
        msaa = 4
        quiet = True
        exag = None
        sun = None
        hour = args.hour
        camera = args.camera
        at = None
        ripple = None
        haze = None
        domain = args.domain
        county_dem = None
        county_demo = None
        county_pbf = None
    settings = {}
    if args.tiles_dir:
        settings["tiles_dir"] = args.tiles_dir
    if args.report:
        settings["report_dir"] = args.report
    app = tools.make_app(A, ctx, fmt, settings)
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=False)
    ov = app.get_layer("overlay")
    m, st = ov.manager, ov.state
    st.ui_scale = args.scale
    st.seen_help = True
    if bridge.ensure_hub(app):
        app._stats_hub.wait_ready(180.0)
    canvas.set_physical_size(w, h)
    canvas.request_draw(app.draw_frame)

    def frames(n=3):
        img = None
        for _ in range(n):
            app._t_last = None
            img = canvas.draw()
        return img

    frames(3)
    c = panels_search.controller(m)
    c.sync()
    # wait for the background loads
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 120:
        frames(1)
        sts = [c.status[k].state for k in ("index", "facilities")]
        if all(s in ("ready", "missing", "error") for s in sts) and c.status["tiles"].state != "loading":
            break
        time.sleep(0.05)
    c.markers_on = not args.no_markers
    if args.hint:
        c.onboarded, c.hint_req = False, True
    else:
        c.onboarded = True
    if args.fac:
        c.show_fac = True
    if args.roads:
        c.show_roads = True
    if args.query:
        c.query = args.query
        c.committed = False
        c.force_dd = bool(args.typing)
        frames(2)
        hits = c.hits_for(args.query)
        if args.pick is not None and hits:
            panels_search._choose(m, c, hits[min(args.pick, len(hits) - 1)])
            m.nav.cancel()
            # jump the camera to the end of the flight
            if m.nav._b is not None:
                app.camera.set_view(*m.nav._b)
            for _ in range(200):
                if c.result is None or not c.result.pending:
                    break
                frames(1)
                time.sleep(0.05)
        elif args.typing:
            time.sleep(1.0)                                   # tile figures of the shown rows
            frames(2)
    if args.facility is not None and c.facilities is not None:
        fs = c.facilities
        try:
            i = int(args.facility)
        except ValueError:
            i = next((k for k, f in enumerate(fs.items) if args.facility.lower() in f.name.lower()), -1)
        if i >= 0:
            c.select_facility(i, fly=True)
            if m.nav._b is not None:
                app.camera.set_view(*m.nav._b)
                m.nav.cancel()
    if args.road_row is not None:
        tab = None
        for _ in range(100):
            tab = c.road_table(c.storm_key())
            if tab is not None:
                break
            frames(1)
            time.sleep(0.05)
        if tab is not None and not isinstance(tab, str):
            c.show_roads = True
            c.ensure_geometry()
            for _ in range(200):
                if c.geometry is not None or c.status["geometry"].state in ("missing", "error"):
                    break
                time.sleep(0.05)
            view = tab.view(c.road_sort, c.road_asc, c.road_min_pct, c.road_evac, c.road_text)
            c.select_road(tab, int(view[args.road_row]))
            if m.nav._b is not None:
                app.camera.set_view(*m.nav._b)
                m.nav.cancel()
    if args.help_tab:
        m.show_help = True
        hs = getattr(m, "_help_state", None)
        frames(2)
        hs = getattr(m, "_help_state", None)
        if hs is not None:
            hs.want_tab = args.help_tab
            hs.tab = args.help_tab
            if args.help_query:
                hs.query = args.help_query
    if args.report:
        c.request_report()
    time.sleep(0.2)
    img = frames(5)
    if args.report:
        ov.post_frame(app)
        frames(1)
        if c.last_report:
            print("[shot] report:", c.last_report)
    arr = np.asarray(img)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr[..., :3].copy()).save(out, optimize=True)
    print(f"[shot] wrote {out} {w}x{h} errors: {list(app.errors) or 'none'} panel errors: {m.panel_errors or 'none'}")
    print("[shot] status:", {k: (v.state, v.message[:80]) for k, v in c.status.items()})
    if args.bench > 0:
        ft = []
        t_end = time.perf_counter() + args.bench
        while time.perf_counter() < t_end:
            t1 = time.perf_counter()
            app._t_last = None
            canvas.draw()
            ft.append((time.perf_counter() - t1) * 1000.0)
        a = np.array(ft)
        print(f"[bench] frames {len(a)} mean {a.mean():.2f} ms p95 {np.percentile(a, 95):.2f} p99 {np.percentile(a, 99):.2f} "
              f"max {a.max():.2f} -> {1000.0 / a.mean():.0f} fps (offscreen, UI on, panels as shown)")
    app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
