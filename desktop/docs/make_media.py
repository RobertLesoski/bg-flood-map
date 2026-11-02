import sys, math, time
from pathlib import Path
sys.path.insert(0, r"C:\Users\Minec\Documents\BG_3D_Map\desktop")
sys.argv = ["bgv"]
import numpy as np
from PIL import Image
from rendercanvas.offscreen import OffscreenRenderCanvas
from bgv import tools
from bgv.__main__ import build_parser
from bgv.gpu.context import GpuContext

OUT = Path(__file__).resolve().parent / "media"  # frames per clip; encode with ffmpeg (e.g. pip install imageio-ffmpeg)
W, H = 1280, 720
args = build_parser().parse_args(["--storm", "100yr", "--quiet", "--size", f"{W}x{H}"])
ctx = GpuContext.create(verbose=False)
canvas = OffscreenRenderCanvas(size=(W, H), pixel_ratio=1.0)
fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
app = tools.make_app(args, ctx, fmt)
app.attach_canvas(canvas, ui=False)
tools.wait_for_flow(app, all_hours=True)
canvas.set_physical_size(W, H)
canvas.request_draw(app.draw_frame)
fr = app.terrain.frame
X, Y = fr.lnglat_to_world(-86.44, 36.99)


def grab(path, anim, dt):
    app.anim_time = anim
    img = canvas.draw()
    Image.fromarray(np.asarray(img)[..., :3].copy()).save(path, quality=92)


def clip(name, n, setup, fps):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        setup(i / (n - 1))
        app.step(1.0 / fps)
        grab(d / f"{i:04d}.jpg", i / fps, 1.0 / fps)
    print("clip", name, n, flush=True)


# warm up
app.t = 12.0
app.camera.set_view(X, Y, 2500, 30, 42)
for _ in range(3):
    app.step(0.016); canvas.draw()


def rise(k):
    app.t = 24.0 * k
    app.camera.set_view(X + 150, Y - 250, 2600 - 500 * k, 25 + 25 * k, 44)


clip("rise", 240, rise, 24)


def fly(k):
    app.t = 8.0 + 8.0 * k
    tools.bench_camera(app, k * 0.99, 14000.0)


clip("fly", 300, fly, 30)


def orbit(k):
    app.t = 14.0
    app.camera.set_view(X - 400, Y + 200, 1500, 360 * k, 38)


clip("orbit", 180, orbit, 30)
app.shutdown()
print("done")

