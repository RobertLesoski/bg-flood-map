"""Offscreen clip export: PNG sequence (always) and an MP4 when ffmpeg exists, else a script that builds one.

    spec = ExportSpec(width=1920, height=1080, fps=30, shot=OrbitShot(...), out_dir=Path(...), name="orbit")
    job = ExportJob(app, spec)
    while not job.tick(): ...                # one frame per call (the window stays alive), or job.run_blocking()
    job.progress, job.state ('prepare'|'render'|'encode'|'done'|'cancelled'|'error'), job.message, job.cancel()

Every frame is rendered at the chosen resolution (up to 4K) into the exporter's own render targets: the window's targets,
camera size, render scale and play state are saved and put back, so a running viewer carries on unchanged.  Each frame gets
the honesty stamp (flood model unvalidated, derived velocities, data source) and a storm / hour caption burned in; synthetic
demo data is additionally marked DEMO DATA.  `clip.json` in the output folder records what was rendered.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import tools_cinematic as tc

MAX_SIDE = 4096
FFMPEG_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
HONESTY_LINE = "Flood model output, not validated.  Flow velocity is derived from the water-surface slope (Manning), not solver output."


def default_dir(app=None) -> Path:
    d = ((getattr(app, "settings", None) or {}).get("cinematic") or {}).get("export_dir") or os.environ.get("BGV_EXPORT_DIR")
    if d:
        return Path(d)
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    docs = home / "Documents"
    return (docs if docs.is_dir() else home) / "BG_Flood_Clips"


def find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return exe if exe and Path(exe).is_file() else None
    except Exception:
        return None


def ffmpeg_command(ffmpeg: str, fps: float, frames_dir: Path, mp4: Path) -> list[str]:
    return [ffmpeg, "-y", "-framerate", f"{fps:g}", "-i", str(frames_dir / "frame_%06d.png"), *FFMPEG_ARGS, str(mp4)]


def write_build_scripts(folder: Path, fps: float, mp4_name: str = "clip.mp4") -> list[Path]:
    """make_mp4.bat / make_mp4.sh next to the frames: run them where ffmpeg is installed (https://ffmpeg.org)."""
    args = " ".join(FFMPEG_ARGS)
    bat = folder / "make_mp4.bat"
    bat.write_text("@echo off\r\nrem Builds the MP4 from the PNG frames in this folder. Needs ffmpeg on PATH (https://ffmpeg.org/download.html).\r\n"
                   "cd /d \"%~dp0\"\r\nwhere ffmpeg >nul 2>nul || (echo ffmpeg was not found on PATH & pause & exit /b 1)\r\n"
                   f"ffmpeg -y -framerate {fps:g} -i frame_%%06d.png {args} {mp4_name}\r\npause\r\n", encoding="ascii")
    sh = folder / "make_mp4.sh"
    sh.write_text("#!/bin/sh\n# Builds the MP4 from the PNG frames in this folder. Needs ffmpeg on PATH.\ncd \"$(dirname \"$0\")\"\n"
                  f"ffmpeg -y -framerate {fps:g} -i frame_%06d.png {args} {mp4_name}\n", encoding="ascii", newline="\n")
    return [bat, sh]


@dataclass
class ExportSpec:
    width: int
    height: int
    fps: float
    shot: tc.Shot
    out_dir: Path
    name: str = "clip"
    make_mp4: bool = True
    keep_png: bool = True
    preroll: int = 10                       # frames rendered and thrown away first (shader compile, particles settle)
    depth_label: str = ""                   # source text of the depth read-outs in captions ('' = none shown)
    stamp: bool = True

    def n_frames(self) -> int:
        return tc.clip_frames(self.shot.duration, self.fps)


def even(v: int) -> int:
    return max(16, min(int(v), MAX_SIDE) // 2 * 2)


# ------------------------------------------------------------------------------------------------ stamping (worker thread)
_FONTS: dict = {}


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    key = (size, bold)
    f = _FONTS.get(key)
    if f is None:
        for name in (("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")) + ("DejaVuSans.ttf",):
            try:
                f = ImageFont.truetype(name, size)
                break
            except Exception:
                continue
        if f is None:
            f = ImageFont.load_default()
        _FONTS[key] = f
    return f


def burn_stamps(img, info: dict) -> None:
    """Draw the storm / hour caption, the place caption and the honesty stamp into the PIL RGB image `img` (in place).
    info: line_top (storm | hour), lines_bottom [str], caption {title, subtitle, extra, alpha} or None, demo bool."""
    from PIL import ImageDraw
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    u = max(H / 1080.0, 0.8)                                    # all sizes scale with the frame
    pad = int(14 * u)
    # ---- top-left: storm and hour
    f1 = _font(int(26 * u), True)
    top = info.get("line_top", "")
    if top:
        tw = d.textlength(top, font=f1)
        d.rounded_rectangle([pad, pad, pad + tw + 2 * pad, pad + 26 * u * 1.55], radius=int(8 * u), fill=(8, 12, 26, 175))
        d.text((2 * pad, pad + 5 * u), top, fill=(214, 226, 255, 255), font=f1)
    # ---- bottom strip: honesty stamp + source
    f2 = _font(int(15 * u))
    lines = info.get("lines_bottom", [])
    lh = 15 * u * 1.35
    if lines:
        h = pad * 1.2 + lh * len(lines)
        d.rectangle([0, H - h, W, H], fill=(6, 9, 20, 185))
        d.rectangle([0, H - h, W, H - h + max(1, int(2 * u))], fill=(255, 179, 71, 230))
        y = H - h + pad * 0.7
        for i, ln in enumerate(lines):
            d.text((pad, y), ln, fill=(255, 214, 150, 255) if i == 0 else (205, 212, 232, 255), font=f2)
            y += lh
    else:
        h = 0
    # ---- lower third: the place
    cap = info.get("caption")
    if cap and cap.get("alpha", 0) > 0.01:
        a = float(cap["alpha"])
        ft, fs, fe = _font(int(40 * u), True), _font(int(20 * u)), _font(int(17 * u))
        title, sub, extra = cap.get("title", ""), cap.get("subtitle", ""), cap.get("extra", "")
        w = max(d.textlength(title, font=ft), d.textlength(sub, font=fs), d.textlength(extra, font=fe) if extra else 0) + 3 * pad
        w = min(w, W - 4 * pad)
        h3 = 40 * u * 1.35 + 20 * u * 1.5 + (17 * u * 1.5 if extra else 0) + pad * 1.2
        x0, y1 = 2 * pad, H - h - 2 * pad
        d.rounded_rectangle([x0, y1 - h3, x0 + w, y1], radius=int(10 * u), fill=(8, 12, 26, int(190 * a)))
        d.rectangle([x0, y1 - h3, x0 + max(3, int(5 * u)), y1], fill=(77, 171, 247, int(255 * a)))
        y = y1 - h3 + pad * 0.5
        d.text((x0 + pad * 1.4, y), title, fill=(255, 255, 255, int(255 * a)), font=ft)
        y += 40 * u * 1.35
        d.text((x0 + pad * 1.4, y), sub, fill=(150, 200, 255, int(255 * a)), font=fs)
        y += 20 * u * 1.5
        if extra:
            d.text((x0 + pad * 1.4, y), extra, fill=(214, 220, 238, int(235 * a)), font=fe)


def _save_frame(arr: np.ndarray, path: Path, info: dict, demo: bool, stamp: bool):
    from PIL import Image
    if demo:
        from .ui.screenshot import mark_demo
        mark_demo(arr)
    img = Image.fromarray(np.ascontiguousarray(arr[..., :3]))
    if stamp:
        burn_stamps(img, info)
    img.save(path, optimize=False, compress_level=1)


# ------------------------------------------------------------------------------------------------ job
class ExportJob:
    def __init__(self, app, spec: ExportSpec, log=None):
        self.app, self.spec = app, spec
        self.W, self.H = even(spec.width), even(spec.height)
        self.n = spec.n_frames()
        self.i = -spec.preroll                      # negative: pre-roll frames
        self.state = "prepare"
        self.message = ""
        self.error: str | None = None
        self.t0 = time.perf_counter()
        self.frame_ms: list[float] = []
        self.mp4: Path | None = None
        self.scripts: list[Path] = []
        self.ffmpeg_used: str | None = None
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bgv-export")
        self._pending: list = []
        self._saved = None
        self._targets = None
        self._tex = None
        self._enc = None
        self._wait0 = None
        self._last_hour = None
        self._log = log or (lambda m: None)

    # ------------------------------------------------------------------ public
    @property
    def progress(self) -> float:
        if self.state in ("done",):
            return 1.0
        if self.state == "encode":
            return 0.99
        return min(max(self.i, 0) / max(self.n, 1), 0.99)

    @property
    def folder(self) -> Path:
        return self.spec.out_dir

    def cancel(self):
        if self.state == "encode":
            try:
                self._proc.terminate()
            except Exception:
                pass
            self.state, self.message = "cancelled", f"Encoding cancelled; the frames are in {self.spec.out_dir}"
        elif self.state in ("prepare", "render"):
            self._finish_render_state(cancel=True)

    def run_blocking(self, on_progress=None):
        while not self.tick():
            if on_progress:
                on_progress(self)
        return self.state

    def eta_s(self) -> float | None:
        if len(self.frame_ms) < 3 or self.i <= 0:
            return None
        return (self.n - self.i) * float(np.mean(self.frame_ms[-12:])) / 1000.0

    # ------------------------------------------------------------------ state machine
    def tick(self) -> bool:
        """Advance by one frame (or one poll of the encoder).  True when finished (done, cancelled or error)."""
        try:
            if self.state == "prepare":
                self._prepare()
            elif self.state == "render":
                self._render_next()
            elif self.state == "encode":
                self._poll_encode()
        except Exception as e:                       # never take the viewer down: report and restore
            self.error = f"{type(e).__name__}: {e}"
            self.message = "Export failed: " + self.error
            self._restore()
            self.state = "error"
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        return self.state in ("done", "cancelled", "error")

    def _prepare(self):
        sp, app = self.spec, self.app
        sp.out_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(sp.out_dir).free
        need = int(self.n * self.W * self.H * 1.2)
        if free < need * 1.2:
            raise OSError(f"not enough disk space in {sp.out_dir}: about {need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")
        cam = app.camera
        self._saved = {"view": (float(cam.target[0]), float(cam.target[1]), float(cam.dist), float(cam.yaw), float(cam.pitch)),
                       "t": app.t, "playing": app.playing, "storm": app.storm.name, "scale": app.render_scale,
                       "targets": (app._targets, app._target_size, getattr(app, "_depth_view", None), getattr(app, "_msaa_view", None)),
                       "cam_wh": (cam.width, cam.height), "anim": app.anim_time}
        app.playing = False
        app.render_scale = 1.0
        self.state = "render"
        self.message = "Rendering"
        self._log(f"export: {self.n} frames {self.W}x{self.H} @ {sp.fps:g} fps into {sp.out_dir}")

    def _wait_flow(self, st: tc.ShotState) -> bool:
        """Switch storm / wait until the hours around st.t are uploaded.  False = not yet (try again next tick)."""
        app = self.app
        if st.storm and st.storm != app.storm.name:
            app.set_storm(st.storm)
            self._wait0 = time.perf_counter()
        app.flow.pump()
        if app.flow.ready_for(st.t) and app.flow._cur is None:
            self._wait0 = None
            return True
        if self._wait0 is None:
            self._wait0 = time.perf_counter()
        if time.perf_counter() - self._wait0 > 180.0:
            raise TimeoutError("flow hours did not finish loading")
        time.sleep(0.004)
        return False

    def _render_next(self):
        sp, app = self.spec, self.app
        t_frame = time.perf_counter()
        idx = max(self.i, 0)
        st = sp.shot.state_at(idx / sp.fps)
        if not self._wait_flow(st):
            return
        cam = app.camera
        cam.stop_motion()
        app.t = float(st.t)
        cam.set_view(*st.view)
        try:
            cam._keep_eye_above_ground()
        except Exception:
            pass
        dt = 1.0 / sp.fps
        self._cam_wh = (cam.width, cam.height)
        cam.resize(self.W, self.H)                   # the view-projection of this frame uses the clip's aspect ratio
        app.step(dt)
        hr = int(min(24, max(0, math.floor(app.t + 0.5))))
        if hr != self._last_hour:                    # statistics (building colours) of the new snapshot hour
            self._last_hour = hr
            for fn in list(getattr(app, "ready_waiters", {}).values()):
                try:
                    fn(60.0)
                except Exception:
                    pass
        arr = self._render_and_read(dt)
        if self.i >= 0:
            info = self._stamp_info(st)
            path = sp.out_dir / f"frame_{self.i + 1:06d}.png"
            self._pending = [f for f in self._pending if not f.done()]
            while len(self._pending) > 6:                # bounded queue: the encoder threads must not fall far behind
                self._pending[0].result()
                self._pending = [f for f in self._pending if not f.done()]
            demo = bool(getattr(app.storm, "synthetic", False))
            self._pending.append(self._pool.submit(_save_frame, arr, path, info, demo, sp.stamp))
        self.frame_ms.append((time.perf_counter() - t_frame) * 1000.0)
        self.i += 1
        if self.i >= self.n:
            self._finish_render_state(cancel=False)

    def _render_and_read(self, dt: float) -> np.ndarray:
        import wgpu
        app = self.app
        dev = app.ctx.device
        if self._tex is None:
            self._tex = dev.create_texture(label="export-target", size=(self.W, self.H, 1), format=app.colour_format,
                                           usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        cam = app.camera
        # the window's targets as they are RIGHT NOW (its quality governor may have re-created them since the last frame)
        self._win = (app._targets, app._target_size, getattr(app, "_depth_view", None), getattr(app, "_msaa_view", None),
                     getattr(self, "_cam_wh", (cam.width, cam.height)), app.render_scale)
        # install the exporter's own scene targets (created on the first frame by render_scene itself)
        if self._targets is None:
            app._targets, app._target_size = None, (0, 0)
        else:
            app._targets, app._target_size, app._depth_view, app._msaa_view = self._targets
        try:
            app.render_scale = 1.0
            app.render_scene(self._tex.create_view(), self.W, self.H, dt)
            self._targets = (app._targets, app._target_size, app._depth_view, app._msaa_view)
            data = dev.queue.read_texture({"texture": self._tex, "mip_level": 0, "origin": (0, 0, 0)},
                                          {"offset": 0, "bytes_per_row": self.W * 4, "rows_per_image": self.H},
                                          (self.W, self.H, 1))
        finally:
            self._put_back_window_targets()
        arr = np.frombuffer(data, np.uint8).reshape(self.H, self.W, 4).copy()
        if app.colour_format.startswith("bgra"):
            arr = arr[..., [2, 1, 0, 3]]
        arr[..., 3] = 255
        return arr

    def _put_back_window_targets(self):
        app = self.app
        w = getattr(self, "_win", None)
        if w is None:
            return
        app._targets, app._target_size, app._depth_view, app._msaa_view = w[:4]
        app.camera.width, app.camera.height = w[4]
        app.render_scale = w[5]
        self._win = None                              # consumed: a later call must never restore stale targets

    # ------------------------------------------------------------------ stamps
    def _stamp_info(self, st: tc.ShotState) -> dict:
        app, sp = self.app, self.spec
        from .ui import units
        from .ui.menus import source_info
        storm_label = app.storm_label(app.storm.name)
        hr = units.snapshot_hour(app.t)
        top = f"{app.domain.label}   |   {storm_label}   |   Hour {hr} of 24   t = {units.hour_label(app.t)}"
        src, _full = source_info(app.storm)
        bottom = [HONESTY_LINE, f"Water: {src}."]
        if sp.depth_label:
            bottom[1] += f"   Depth read-outs: {sp.depth_label}."
        if getattr(app.storm, "synthetic", False):
            bottom.insert(0, "DEMO DATA - synthetic, not a simulation result")
        cap = None
        if st.caption:
            cap = dict(st.caption)
        return {"line_top": top, "lines_bottom": bottom, "caption": cap}

    # ------------------------------------------------------------------ end of render / encode
    def _restore(self):
        app, s = self.app, self._saved
        if self._targets and self._targets[0]:
            for tex in self._targets[0].values():
                try:
                    tex.destroy()
                except Exception:
                    pass
        self._targets = None
        if self._tex is not None:
            try:
                self._tex.destroy()
            except Exception:
                pass
            self._tex = None
        if s is None:
            return
        self._put_back_window_targets()
        app.render_scale = s["scale"]
        app.playing = s["playing"]
        app.anim_time = s["anim"]
        try:
            if app.storm.name != s["storm"]:
                app.set_storm(s["storm"])
            app.t = s["t"]
            app.camera.set_view(*s["view"])
        except Exception:
            pass
        try:
            from .gpu.context import tex_bytes
            pw, ph = app._target_size
            if pw:
                nb = tex_bytes(pw, ph, 1, 4) * app.sample_count * (2 if app.sample_count > 1 else 1)
                app.ctx.track("render targets", nb)
        except Exception:
            pass
        self._saved = None

    def _finish_render_state(self, cancel: bool):
        for f in self._pending:
            try:
                f.result()
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
        self._pending = []
        self._restore()
        self._pool.shutdown(wait=True)
        written = len(list(self.spec.out_dir.glob("frame_*.png")))
        if cancel:
            self.state = "cancelled"
            self.message = f"Cancelled after {written} frames (kept in {self.spec.out_dir})"
            return
        self._write_meta(written)
        self._start_encode()

    def _write_meta(self, written: int):
        app, sp = self.app, self.spec
        meta = {"name": sp.name, "frames": written, "fps": sp.fps, "size": [self.W, self.H], "domain": app.domain.key,
                "storm": app.storm.name, "duration_s": sp.shot.duration, "depth_source": sp.depth_label,
                "water_source": getattr(app.storm, "source", ""), "synthetic": bool(getattr(app.storm, "synthetic", False)),
                "honesty": HONESTY_LINE, "mean_frame_ms": float(np.mean(self.frame_ms)) if self.frame_ms else None,
                "rendered": time.strftime("%Y-%m-%d %H:%M:%S")}
        (sp.out_dir / "clip.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def _start_encode(self):
        sp = self.spec
        self._proc = None
        if not sp.make_mp4:
            self.state, self.message = "done", f"Done: {sp.out_dir}"
            return
        ff = find_ffmpeg()
        if ff is None:
            self.scripts = write_build_scripts(sp.out_dir, sp.fps, f"{sp.name}.mp4")
            self.state = "done"
            self.message = (f"Done: {sp.out_dir}. ffmpeg was not found on PATH, so no MP4 was made: run make_mp4.bat "
                            f"(or make_mp4.sh) in that folder where ffmpeg is installed")
            return
        self.mp4 = sp.out_dir / f"{sp.name}.mp4"
        self.ffmpeg_used = ff
        cmd = ffmpeg_command(ff, sp.fps, sp.out_dir, self.mp4)
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._log_fh = open(sp.out_dir / "ffmpeg.log", "wb")
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self._log_fh, creationflags=flags)
        except OSError as e:
            self.scripts = write_build_scripts(sp.out_dir, sp.fps, f"{sp.name}.mp4")
            self.state, self.message = "done", f"Done: {sp.out_dir}. ffmpeg could not start ({e}); run make_mp4.bat there"
            return
        self.state, self.message = "encode", "Encoding MP4"

    def _poll_encode(self):
        rc = self._proc.poll()
        if rc is None:
            time.sleep(0.02)
            return
        self._log_fh.close()
        try:
            err = (self.spec.out_dir / "ffmpeg.log").read_bytes().decode("utf-8", "replace")[-400:]
        except OSError:
            err = ""
        if rc == 0 and self.mp4 and self.mp4.is_file():
            if not self.spec.keep_png:
                for f in self.spec.out_dir.glob("frame_*.png"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
            self.state, self.message = "done", f"Done: {self.mp4}"
        else:
            self.scripts = write_build_scripts(self.spec.out_dir, self.spec.fps, f"{self.spec.name}.mp4")
            self.state, self.message = "done", f"Frames are saved in {self.spec.out_dir}; ffmpeg failed ({err.strip()[-120:]})"
