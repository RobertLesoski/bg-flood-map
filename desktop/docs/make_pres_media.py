"""Presentation media (2026-11-05): county screenshots with UI and an ev6 hour 1-24 time-lapse GIF."""
import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\Minec\Documents\BG_3D_Map\desktop")
sys.argv = ["bgv"]
import numpy as np
from PIL import Image
from rendercanvas.offscreen import OffscreenRenderCanvas
from bgv import tools
from bgv.__main__ import build_parser
from bgv.gpu.context import GpuContext

OUT = Path(__file__).resolve().parent / "media" / "presentation_2026-11-05"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1600, 900
storm = sys.argv[1] if len(sys.argv) > 1 else "ev6"
args = build_parser().parse_args(["--domain", "county", "--storm", "ev6", "--quiet", "--size", f"{W}x{H}"])
ctx = GpuContext.create(verbose=False)
canvas = OffscreenRenderCanvas(size=(W, H), pixel_ratio=1.0)
fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
app = tools.make_app(args, ctx, fmt)
app.attach_canvas(canvas, ui=True)
tools.wait_for_flow(app, all_hours=True)
canvas.set_physical_size(W, H)
canvas.request_draw(app.draw_frame)
fr = app.terrain.frame
X, Y = fr.lnglat_to_world(-86.44, 36.99)      # downtown Bowling Green


def grab(path):
    for _ in range(3):
        app.step(1 / 30)
        img = canvas.draw()
    Image.fromarray(np.asarray(img)[..., :3].copy()).save(path)


def view(name, t, x, y, dist, yaw, pitch):
    app.t = t
    app.camera.set_view(x, y, dist, yaw, pitch)
    grab(OUT / f"{name}.png")
    print("shot", name, flush=True)


view("01_ev6_h1_downtown", 1.0, X, Y, 3500, 20, 45)
view("02_ev6_h12_downtown", 12.0, X, Y, 3500, 20, 45)
view("03_ev6_h24_downtown", 24.0, X, Y, 3500, 20, 45)
view("04_ev6_h18_county", 18.0, X, Y, 22000, 0, 55)
view("05_ev6_h18_closeup", 18.0, X + 300, Y - 400, 1200, 60, 35)

frames = []
for i in range(48):                            # hour 1 -> 24 time-lapse
    app.t = 1.0 + 23.0 * i / 47
    app.camera.set_view(X, Y, 4200 - 1200 * i / 47, 20 + 30 * i / 47, 48)
    for _ in range(2):
        app.step(1 / 30)
        img = canvas.draw()
    frames.append(Image.fromarray(np.asarray(img)[..., :3].copy()).resize((960, 540)))
frames[0].save(OUT / "ev6_timelapse.gif", save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)
print("gif done", len(frames), flush=True)
app.shutdown()
