"""F12 screenshots: render the scene (+ the imgui draw data of the frame just drawn) into a private texture, read it
back, save a PNG in Documents\\BG_Flood_Screenshots.  Runs once per request, on the main thread, after the frame."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


def default_dir() -> Path:
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    docs = home / "Documents"
    return (docs if docs.is_dir() else home) / "BG_Flood_Screenshots"


def make_path(storm: str, hour: int, folder: Path | None = None) -> Path:
    folder = folder or default_dir()
    return folder / f"bgflood_{storm}_h{int(hour):02d}_{time.strftime('%Y%m%d_%H%M%S')}.png"


def capture_rgba(app, include_ui: bool = True) -> np.ndarray:
    """uint8[H, W, 4] RGBA of the current view at the canvas' physical resolution."""
    import wgpu
    from imgui_bundle import imgui
    pw, ph = app.canvas.get_physical_size()
    dev = app.ctx.device
    fmt = app.colour_format
    tex = dev.create_texture(label="screenshot", size=(pw, ph, 1), format=fmt,
                             usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    view = tex.create_view()
    try:
        app.render_scene(view, pw, ph, 0.0)
        r = getattr(app, "imgui_renderer", None)
        if include_ui and r is not None:
            imgui.set_current_context(r.imgui_context)
            dd = imgui.get_draw_data()
            enc = dev.create_command_encoder(label="screenshot-ui")
            rp = enc.begin_render_pass(color_attachments=[{"view": view, "resolve_target": None, "clear_value": (0, 0, 0, 1),
                                                           "load_op": wgpu.LoadOp.load, "store_op": wgpu.StoreOp.store}])
            r.backend.render(dd, rp)
            rp.end()
            dev.queue.submit([enc.finish()])
        data = dev.queue.read_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
                                      {"offset": 0, "bytes_per_row": pw * 4, "rows_per_image": ph}, (pw, ph, 1))
        arr = np.frombuffer(data, np.uint8).reshape(ph, pw, 4).copy()
    finally:
        try:
            tex.destroy()
        except Exception:
            pass
    if fmt.startswith("bgra"):
        arr = arr[..., [2, 1, 0, 3]]
    arr[..., 3] = 255
    return arr


def mark_demo(arr: np.ndarray, text: str = "DEMO DATA  -  synthetic, not a simulation result") -> np.ndarray:
    """Stamp a clearly visible DEMO DATA banner into an RGBA / RGB image (in place) - screenshots and renders of the synthetic
    county demo data must never pass for real results, whether or not the interface is in the picture."""
    from PIL import Image, ImageDraw, ImageFont
    h, w = arr.shape[:2]
    img = Image.fromarray(np.ascontiguousarray(arr[..., :3]))
    d = ImageDraw.Draw(img, "RGBA")
    size = max(14, int(min(w, h) * 0.022))
    try:
        font = ImageFont.truetype("arialbd.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    tw = d.textlength(text, font=font)
    pad = size * 0.5
    x0, y0 = w - tw - 3 * pad, h - size - 3.2 * pad
    d.rounded_rectangle([x0, y0, x0 + tw + 2 * pad, y0 + size + 1.6 * pad], radius=pad, fill=(40, 28, 0, 215),
                        outline=(255, 190, 60, 255), width=2)
    d.text((x0 + pad, y0 + 0.7 * pad), text, fill=(255, 205, 90, 255), font=font)
    arr[..., :3] = np.asarray(img)
    return arr


def save_png(arr: np.ndarray, path: Path) -> Path:
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr[..., :3]).save(path, optimize=False, compress_level=3)
    return path


def target_dir(app=None) -> Path:
    """Where F12 saves: settings["screenshot_dir"] (or BGV_SCREENSHOT_DIR) when set, else Documents/BG_Flood_Screenshots."""
    d = (getattr(app, "settings", None) or {}).get("screenshot_dir") or os.environ.get("BGV_SCREENSHOT_DIR")
    return Path(d) if d else default_dir()


def take(app, include_ui: bool = True) -> Path:
    arr = capture_rgba(app, include_ui)
    if getattr(app.storm, "synthetic", False):
        mark_demo(arr)
    from .units import snapshot_hour
    return save_png(arr, make_path(app.storm.name, snapshot_hour(app.t), target_dir(app)))
