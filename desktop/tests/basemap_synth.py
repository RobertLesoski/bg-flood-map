"""Synthetic OSM-look basemap composite for verification (no network): paper, parks, roads with casings, building footprints.

Built from the project's own buildings / roads data in the flow-grid frame, so a draped result can be checked against the
extruded buildings: footprints in the image must sit exactly under the 3D buildings (proves the uv identity of the drape).

    synth_composite(root, frame, long_px=4096) -> BasemapComposite   (rgba written, mips built, filled=True, no tiles flag)
    make_app(settings=None, size=(1920, 1080)) -> (app, canvas, ctx)  offscreen app (no window) with the basemap layer on and the
                                                                      synthetic composite uploaded
    python tests/basemap_synth.py out.png [--camera x,y,dist,yaw,pitch] [--size WxH] [--hour h] [--off]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DESKTOP = Path(__file__).resolve().parents[1]
if str(DESKTOP) not in sys.path:
    sys.path.insert(0, str(DESKTOP))

from bgv.data import basemap as bm                   # noqa: E402
from bgv.geo import R_MERC                           # noqa: E402

PAPER = (242, 239, 233)
BUILD = (217, 208, 201)
BUILD_EDGE = (196, 182, 171)
ROAD_FILL = {"motorway": (233, 144, 160), "trunk": (249, 178, 156), "primary": (252, 214, 164), "secondary": (247, 250, 191),
             "tertiary": (255, 255, 255), "residential": (255, 255, 255), "service": (255, 255, 255)}
ROAD_W = {"motorway": 9, "trunk": 8, "primary": 7, "secondary": 6, "tertiary": 5, "residential": 3.5, "service": 2.2}


def _ll_to_px(comp, lng, lat):
    lng = np.asarray(lng, np.float64)
    lat = np.asarray(lat, np.float64)
    x = R_MERC * np.radians(lng)
    y = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    fr = comp.frame
    px = (x - fr.xmin) / (fr.xmax - fr.xmin) * comp.width
    py = (fr.ymax - y) / (fr.ymax - fr.ymin) * comp.height
    return px, py


def synth_composite(root, frame, long_px: int = 4096, calib: bool = False) -> "bm.BasemapComposite":
    from PIL import Image, ImageDraw
    comp = bm.BasemapComposite(frame, long_px)
    S = 2 if long_px <= 4096 else 1                                    # supersampling for clean edges (RAM)
    W, H = comp.width * S, comp.height * S
    im = Image.new("RGB", (W, H), PAPER)
    dr = ImageDraw.Draw(im)
    layers = Path(root) / "layers"
    mpp = (frame.xmax - frame.xmin) / comp.width / frame.k              # ground metres per composite pixel (approx)
    # park-ish washes: coarse value noise thresholded, purely cosmetic
    rng = np.random.default_rng(7)
    g = rng.random((H // 512 + 2, W // 512 + 2))
    from PIL import Image as _I
    gi = _I.fromarray((g * 255).astype(np.uint8)).resize((W, H), _I.BICUBIC)
    a = np.asarray(gi, np.float32) / 255.0
    park = a > 0.62
    arr = np.asarray(im).copy()
    arr[park] = (200, 225, 190)
    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)
    # roads: casing then fill, widest first
    rd = json.loads((layers / "roads.geojson").read_text(encoding="utf-8"))
    feats = rd["features"]
    order = sorted(feats, key=lambda f: ROAD_W.get(f["properties"].get("highway"), 2.2))
    for casing in (True, False):
        for f in order:
            hw = f["properties"].get("highway")
            wm = ROAD_W.get(hw, 2.2)
            wpx = max(wm / max(mpp, 1e-6) * S * (1.35 if casing else 1.0), (3.0 if calib else 1.0) * S)
            col = (160, 150, 140) if casing else ROAD_FILL.get(hw, (255, 255, 255))
            if calib:
                col = (120, 20, 120) if casing else (230, 60, 230)
            geom = f["geometry"]
            lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
            for ln in lines:
                c = np.asarray(ln, np.float64)
                px, py = _ll_to_px(comp, c[:, 0], c[:, 1])
                pts = list(zip((px * S).tolist(), (py * S).tolist()))
                dr.line(pts, fill=col, width=int(round(wpx)), joint="curve")
    # buildings
    bd = json.loads((layers / "buildings.geojson").read_text(encoding="utf-8"))
    for f in bd["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0], np.float64)
            px, py = _ll_to_px(comp, ring[:, 0], ring[:, 1])
            dr.polygon(list(zip((px * S).tolist(), (py * S).tolist())), fill=(230, 40, 40) if calib else BUILD, outline=BUILD_EDGE)
    im = im.resize((comp.width, comp.height), Image.LANCZOS)
    comp.rgba[..., :3] = np.asarray(im, np.uint8)
    comp.rgba[..., 3] = 255
    comp.rebuild_mips()
    return comp


def cached_synth(root, frame, long_px, scratch: Path, calib: bool = False):
    p = scratch / f"synth_{long_px}{'_calib' if calib else ''}.npy"
    if p.is_file():
        comp = bm.BasemapComposite(frame, long_px)
        comp.rgba[...] = np.load(p)
        comp.rebuild_mips()
        return comp
    comp = synth_composite(root, frame, long_px, calib)
    scratch.mkdir(parents=True, exist_ok=True)
    np.save(p, comp.rgba)
    return comp


def install(app, comp):
    """Put a ready-made composite into the (enabled) basemap layer: texture uploaded, no loader, no network."""
    from bgv.gpu.layers.basemap import BasemapTexture
    lay = app.get_layer("basemap")
    lay.comp = comp
    lay.tex = BasemapTexture(app.ctx, comp)
    app.ctx.track("basemap", lay.tex.bytes_total)
    lay.handle.set_texture(lay.tex.view)
    lay.tex.adopt_fallback()                                       # same order as BasemapLayer.update (GPU clear, then bands)
    lay.tex.upload_all()
    comp.tiles_done.setdefault(comp.zoom, set()).add((0, 0))      # "has real tiles" for the shader flag
    return lay


def make_app(settings=None, size=(1920, 1080), storm="25yr", msaa=4, ui=False):
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import config
    from bgv.gpu.context import GpuContext
    from bgv.app import App
    ctx = GpuContext.create(verbose=False)
    canvas = OffscreenRenderCanvas(size=size, pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
    app = App(ctx, config.data_root(), storm=storm, msaa=msaa, quiet=True, settings=settings or {}, colour_format=fmt)
    app.attach_canvas(canvas, ui=ui)
    return app, canvas, ctx


def render(app, canvas, out: Path, cam=None, hour=12.0, size=(1920, 1080)):
    from PIL import Image
    from bgv.tools import wait_for_flow
    if cam:
        app.camera.set_view(*cam)
    app.t = float(hour)
    wait_for_flow(app, all_hours=False)
    canvas.set_physical_size(*size)
    canvas.request_draw(app.draw_frame)
    app.anim_time = 4.0
    canvas.draw()
    app.anim_time = 4.0
    app._t_last = None
    arr = np.asarray(canvas.draw())
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr[..., :3].copy()).save(out, optimize=True)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--camera", default="0,0,16000,0,55")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--hour", type=float, default=12.0)
    ap.add_argument("--opacity", type=float, default=0.85)
    ap.add_argument("--px", type=int, default=4096)
    ap.add_argument("--calib", action="store_true", help="loud roads / footprints (drape alignment check)")
    ap.add_argument("--ui", action="store_true", help="include the sidebar")
    ap.add_argument("--off", action="store_true", help="basemap layer disabled (reference render)")
    ap.add_argument("--exag", type=float, default=None)
    ap.add_argument("--scratch", default=r"C:\Users\Minec\AppData\Local\Temp\claude\C--Users-Minec-Documents-BG-3D-Map\6d20f384-9e95-4606-82dd-08b76f90f515\scratchpad")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    settings = {"layers": {"basemap": {"enabled": not a.off, "state": {"opacity": a.opacity}}}}
    app, canvas, ctx = make_app(settings, (w, h), ui=a.ui)
    from bgv import config
    t0 = time.time()
    comp = cached_synth(config.data_root(), app.terrain.frame, a.px, Path(a.scratch), a.calib)
    print(f"synthetic composite {comp.width}x{comp.height} ready in {time.time() - t0:.1f} s")
    install(app, comp)
    if a.exag is not None:
        app.set_exag(a.exag)
    cam = [float(v) for v in a.camera.split(",")]
    render(app, canvas, Path(a.out), cam, a.hour, (w, h))
    print("wrote", a.out, "errors:", list(app.errors))
    app.shutdown()


if __name__ == "__main__":
    main()


def real_composite(frame, long_px: int = 4096, levels=(12,)):
    """Composite of the REAL OSM tiles already in desktop/cache/tiles at `levels` (disk cache only, never the network)."""
    comp = bm.BasemapComposite(frame, long_px)
    src = bm.TileSource(network=False)
    ld = bm.BasemapLoader(comp, src, threads=2)
    ld.start(download=False, plan=ld.plan(levels=list(levels)))
    ld.join(60)
    return comp
