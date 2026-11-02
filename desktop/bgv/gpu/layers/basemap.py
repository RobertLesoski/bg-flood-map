"""Basemap layer: optional OpenStreetMap-style background for the terrain (UI + texture owner, draws nothing itself).

The layer owns ONE rgba8unorm-srgb texture (composite of XYZ tiles in the flow-grid frame, full CPU mip chain, anisotropic
sampler) and exposes it to the terrain shader as `app.basemap` (a `BasemapHandle`, dict-style access works too):

    app.basemap.texture_view   GPU texture view (a 1x1 grey placeholder until the real composite exists)
    app.basemap.sampler        linear / mip-linear / anisotropic x8, clamp-to-edge
    app.basemap.bind_group_layout, .bind_group     ready-made bind group (binding 0 uniform, 1 texture, 2 sampler)
    app.basemap.version        int, +1 whenever bind_group / texture_view were replaced (rebuild cached bind groups)
    app.basemap.enabled        True when the user ticked "Map background" AND the composite is uploaded
    app.basemap.opacity        0..1 slider
    app.basemap.has_tiles      False while only the procedural fallback is available (no real tile loaded yet)
    app.basemap.uv_scale, .uv_offset   (1,1) / (0,0): the composite covers exactly the flow grid, so basemap uv == terrain uv
    app.basemap.attribution    "(c) OpenStreetMap contributors"  (the overlay layer should print it while enabled)
    app.basemap.write_params(queue)   call ONCE PER FRAME from the terrain pass (16-32 bytes): writes opacity = 0 when the
                                      layer is off, because a disabled layer receives no update() calls.

TERRAIN INTEGRATION (DONE, in terrain.py): TerrainLayer.init puts `app.basemap.bind_group_layout` at group 1 of the terrain
pipeline and prepends `basemap_wgsl(1)` to TERRAIN_WGSL (a stub, BASEMAP_OFF_WGSL, when `app.basemap` does not exist);
TerrainLayer.update calls `app.basemap.write_params(queue)` every frame (opacity 0 while the layer is off), draw() binds
`app.basemap.bind_group` at group 1.  Fragment shader (uniform control flow): `let bmc = basemap_color(in.uv)` at the top, the
terrain-normal grain is faded by (1 - 0.85 a), and at the end of the lighting
    col = mix(col, bmc.rgb * relief(light / lref), bmc.a)
where `light` is the terrain lighting (sun * hillshade * cast shadow + ambient, times AO) and `lref` the same for flat, fully
sunlit ground: flat ground shows the tile colours unchanged, slopes / shadows / AO modulate them (relief stays readable).

Uploads: the CPU composite is dirty-tracked in 64-row bands; `update()` uploads at most `upload_budget` bytes per frame
(default 0.8 MB, incl. mips) so tile arrival never causes a hitch.  Nothing touches the network until the user presses
"Download tiles for this area".
"""
from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import numpy as np
import wgpu

from ...data import basemap as bm
from ...data.basemap import BAND_ROWS, ATTRIBUTION
from ..context import tex_bytes
from . import Layer

TEX_FORMAT = "rgba8unorm-srgb"
VRAM_BUDGET_MB = 200.0
DETAILS = [("standard", "Standard  4096 px"), ("high", "High  6144 px"), ("ultra", "Ultra  8192 px")]


def basemap_wgsl(group: int = 1) -> str:
    """WGSL for the terrain fragment shader: bindings + `basemap_color(uv)`.  `group` = bind group index you chose."""
    return f"""
// ---- basemap drape (bgv/gpu/layers/basemap.py) ------------------------------------------------------------------
struct BasemapParams {{
  p: vec4<f32>,      // x = opacity (0 = off / not ready), y = 1 while only the procedural fallback exists
  uv: vec4<f32>,     // xy = uv scale, zw = uv offset (identity: the composite covers exactly the flow grid)
}};
@group({group}) @binding(0) var<uniform> basemap: BasemapParams;
@group({group}) @binding(1) var basemap_tex: texture_2d<f32>;
@group({group}) @binding(2) var basemap_samp: sampler;

// rgb: LINEAR colour (the texture is *-srgb), a: opacity.  Uses implicit derivatives (mips + anisotropy): call it from
// UNIFORM control flow in a fragment shader, then branch on .a if you want to.
fn basemap_color(terrain_uv: vec2<f32>) -> vec4<f32> {{
  let buv = terrain_uv * basemap.uv.xy + basemap.uv.zw;
  let c = textureSample(basemap_tex, basemap_samp, buv).rgb;
  return vec4<f32>(c, basemap.p.x);
}}
// vertex-stage / compute variant (no derivatives): explicit mip level
fn basemap_color_lod(terrain_uv: vec2<f32>, lod: f32) -> vec4<f32> {{
  let buv = terrain_uv * basemap.uv.xy + basemap.uv.zw;
  return vec4<f32>(textureSampleLevel(basemap_tex, basemap_samp, buv, lod).rgb, basemap.p.x);
}}
"""


BASEMAP_WGSL = basemap_wgsl(1)


def draw_attribution(app, layout=None, scale: float = 1.0) -> bool:
    """OSM licence: the credit must be visible whenever the map is shown.  Call inside the imgui frame (the UI manager does,
    every frame): draws "(c) OpenStreetMap contributors" in the foreground list, at most once per frame, only while the
    basemap is showing.  `layout` = the UI layout (sx1 / sy1 = right / bottom edge of the 3D view): the credit sits bottom-right
    of the view, just above the scale bar; without a layout it goes to the bottom-left of the window.  Returns True when drawn."""
    bmh = getattr(app, "basemap", None)
    if bmh is None or not bmh.enabled or bmh.opacity <= 0.01:
        return False
    if getattr(app, "_bm_attr_frame", -1) == app.frame_index:
        return True
    from imgui_bundle import imgui
    app._bm_attr_frame = app.frame_index
    text = bmh.attribution
    ts = imgui.calc_text_size(text)
    pad = 6.0 * scale
    if layout is not None:
        x = layout.sx1 - 16.0 * scale - ts.x                       # right-aligned with, and just above, the scale bar box
        y = layout.sy1 - 44.0 * scale - 3.0 * ts.y - 2.0
    else:
        vp = imgui.get_main_viewport()
        x = vp.pos.x + 12.0
        y = vp.pos.y + vp.size.y - 34.0 - ts.y
    dl = imgui.get_foreground_draw_list()
    dl.add_rect_filled((x - pad, y - 2.0), (x + ts.x + pad, y + ts.y + 2.0), imgui.get_color_u32((0.04, 0.06, 0.11, 0.62)), 4.0)
    dl.add_text((x, y), imgui.get_color_u32((1.0, 1.0, 1.0, 0.92)), text)
    return True


def _dim(imgui, text: str):
    imgui.push_text_wrap_pos(0.0)
    imgui.text_disabled(text)
    imgui.pop_text_wrap_pos()


# ====================================================================================================== GPU texture
class BasemapTexture:
    """The GPU copy of a `BasemapComposite`: texture + mips, row-band streaming upload."""

    def __init__(self, ctx, comp: bm.BasemapComposite, label: str = "basemap"):
        self.ctx = ctx
        self.comp = comp
        dev = ctx.device
        self.texture = dev.create_texture(
            label=label, size=(comp.width, comp.height, 1), format=TEX_FORMAT, mip_level_count=comp.n_levels,
            usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC
                   | wgpu.TextureUsage.RENDER_ATTACHMENT))
        self.view = self.texture.create_view()
        self.cleared = False
        self.up_gen = np.full(comp.n_bands, -1, np.int64)
        self.bytes_total = tex_bytes(comp.width, comp.height, 1, 4, comp.n_levels)
        self.last_upload_ms = 0.0
        self.max_upload_ms = 0.0

    @property
    def initial_done(self) -> bool:
        """Every band was uploaded at least once (before that the texture holds undefined / zero texels)."""
        return bool((self.up_gen >= 0).all())

    def adopt_fallback(self) -> bool:
        """GPU-side clear of every mip level to the fallback colour (no 75-300 MB upload of flat paper).  Bands written by
        tiles after the fallback generation stay dirty and are uploaded as usual.  False until the composite is filled."""
        c = self.comp
        if self.cleared:
            return True
        if not c.filled or not c.fallback_gen:
            return False
        lin = [float(bm._LIN[v]) for v in bm.FALLBACK_RGB]
        enc = self.ctx.device.create_command_encoder(label="basemap-clear")
        for k in range(c.n_levels):
            v = self.texture.create_view(base_mip_level=k, mip_level_count=1)
            p = enc.begin_render_pass(color_attachments=[{"view": v, "clear_value": (lin[0], lin[1], lin[2], 1.0),
                                                          "load_op": "clear", "store_op": "store"}])
            p.end()
        self.ctx.queue.submit([enc.finish()])
        self.up_gen[:] = c.fallback_gen
        self.cleared = True
        return True

    def pending(self) -> int:
        return int((self.comp.band_gen != self.up_gen).sum())

    def _write_band(self, b: int) -> int:
        c = self.comp
        r0, r1 = b * BAND_ROWS, min((b + 1) * BAND_ROWS, c.height)
        chunks = []
        if not c.lock.acquire(blocking=False):                    # a worker thread is writing: never stall the frame
            return 0
        try:
            gen = int(c.band_gen[b])
            for k in range(c.n_levels):
                m = c.mips[k]
                mh = m.shape[0]
                a = min(r0 >> k, mh - 1)                          # deep levels of a non power-of-two composite: clamp
                e = min(max(-(-r1 >> k), a + 1), mh)              # ceil(r1 / 2^k), at least one row
                chunks.append((k, a, e - a, np.array(m[a:e])))
        finally:
            c.lock.release()
        total = 0
        q = self.ctx.queue
        for k, a, n, data in chunks:
            w = data.shape[1]
            q.write_texture({"texture": self.texture, "mip_level": k, "origin": (0, a, 0)}, data,
                            {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": n}, (w, n, 1))
            total += data.nbytes
        self.up_gen[b] = gen
        return total

    def upload(self, budget_bytes: int = 1_500_000) -> int:
        """Upload dirty bands until `budget_bytes` were written (at least one band if any is dirty).  Returns bytes."""
        diff = np.nonzero(self.comp.band_gen != self.up_gen)[0]
        if diff.size == 0:
            return 0
        t0 = time.perf_counter()
        written = 0
        for b in diff:
            written += self._write_band(int(b))
            if written >= budget_bytes:
                break
        self.last_upload_ms = (time.perf_counter() - t0) * 1000.0
        self.max_upload_ms = max(self.max_upload_ms, self.last_upload_ms)
        return written

    def upload_all(self):
        while self.upload(1 << 60):
            pass

    def read_texel(self, x: int, y: int, level: int = 0):
        raw = self.ctx.queue.read_texture({"texture": self.texture, "mip_level": level, "origin": (x, y, 0)},
                                          {"offset": 0, "bytes_per_row": 4, "rows_per_image": 1}, (1, 1, 1))
        return tuple(np.frombuffer(raw, np.uint8)[:4].tolist())

    def destroy(self):
        try:
            self.texture.destroy()
        except Exception:
            pass


# ====================================================================================================== handle
class BasemapHandle:
    """`app.basemap`: everything the terrain shader needs.  Dict-style read access (`app.basemap["texture_view"]`)."""

    KEYS = ("texture_view", "sampler", "enabled", "opacity", "uv_scale", "uv_offset", "bind_group", "bind_group_layout",
            "version", "has_tiles", "attribution", "ready", "params_buffer")

    def __init__(self, layer: "BasemapLayer", ctx):
        self.layer = layer
        self.ctx = ctx
        dev = ctx.device
        self.attribution = ATTRIBUTION
        self.uv_scale = (1.0, 1.0)
        self.uv_offset = (0.0, 0.0)
        self.version = 0
        S = wgpu.ShaderStage
        self.bind_group_layout = dev.create_bind_group_layout(label="basemap-bgl", entries=[
            {"binding": 0, "visibility": S.FRAGMENT | S.VERTEX, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": S.FRAGMENT | S.VERTEX, "texture": {"sample_type": "float", "view_dimension": "2d"}},
            {"binding": 2, "visibility": S.FRAGMENT | S.VERTEX, "sampler": {"type": "filtering"}},
        ])
        self.sampler = dev.create_sampler(label="basemap-samp", mag_filter="linear", min_filter="linear",
                                          mipmap_filter="linear", address_mode_u="clamp-to-edge",
                                          address_mode_v="clamp-to-edge", max_anisotropy=8)
        self.params_buffer = dev.create_buffer(label="basemap-params", size=32,
                                               usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._params = np.zeros(8, np.float32)
        self._last = None
        # 1x1 placeholder so a bind group exists from the very first frame
        self._dummy = dev.create_texture(label="basemap-dummy", size=(1, 1, 1), format=TEX_FORMAT,
                                         usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        ctx.queue.write_texture({"texture": self._dummy, "mip_level": 0, "origin": (0, 0, 0)},
                                np.array([[[200, 200, 196, 255]]], np.uint8),
                                {"offset": 0, "bytes_per_row": 4, "rows_per_image": 1}, (1, 1, 1))
        self.texture_view = self._dummy.create_view()
        self.bind_group = None
        self._rebuild()

    def _rebuild(self):
        self.bind_group = self.ctx.device.create_bind_group(label="basemap-bg", layout=self.bind_group_layout, entries=[
            {"binding": 0, "resource": {"buffer": self.params_buffer, "offset": 0, "size": 32}},
            {"binding": 1, "resource": self.texture_view},
            {"binding": 2, "resource": self.sampler},
        ])
        self.version += 1

    def set_texture(self, view):
        self.texture_view = view if view is not None else self._dummy.create_view()
        self._rebuild()
        self._last = None

    # live state (read from the layer so a toggled checkbox is seen immediately)
    @property
    def ready(self) -> bool:
        return self.layer.is_ready()

    @property
    def enabled(self) -> bool:
        return bool(self.layer.enabled and self.layer.is_ready())

    @property
    def opacity(self) -> float:
        return float(self.layer.opacity)

    @property
    def has_tiles(self) -> bool:
        c = self.layer.comp
        return bool(c is not None and c.has_tiles)

    def write_params(self, queue=None):
        """Per-frame: cheap, skips the GPU write when nothing changed."""
        op = self.opacity if self.enabled else 0.0
        vals = (op, 0.0 if self.has_tiles else 1.0, 0.0, 0.0, self.uv_scale[0], self.uv_scale[1],
                self.uv_offset[0], self.uv_offset[1])
        if vals == self._last:
            return
        self._last = vals
        self._params[:] = vals
        (queue or self.ctx.queue).write_buffer(self.params_buffer, 0, self._params)

    def __getitem__(self, k):
        if k not in self.KEYS:
            raise KeyError(k)
        return getattr(self, k)

    def get(self, k, default=None):
        return getattr(self, k, default) if k in self.KEYS else default

    def keys(self):
        return self.KEYS

    def destroy(self):
        for o in (self._dummy, self.params_buffer):
            try:
                o.destroy()
            except Exception:
                pass


# ====================================================================================================== layer
class BasemapLayer(Layer):
    name = "basemap"
    title = "Map background"
    enabled = False
    default_enabled = False          # off until the user ticks it: no tile traffic is ever started unasked
    has_ui = True
    draws = False                    # the terrain shader does the drawing (see module docstring)

    UPLOAD_BUDGET = 800_000

    def __init__(self):
        self.opacity = 0.85
        self.detail = 0              # index into DETAILS
        self.url = bm.DEFAULT_URL
        self.local_dir = ""
        self.handle: BasemapHandle | None = None
        self.comp: bm.BasemapComposite | None = None
        self.tex: BasemapTexture | None = None
        self.loader: bm.BasemapLoader | None = None
        self.source: bm.TileSource | None = None
        self.plan: bm.Plan | None = None
        self.error = ""
        self._plan_busy = False
        self._confirm = False
        self._last_phase = ""
        self._edit_url = ""
        self._edit_dir = ""
        self._app = None
        self.upload_budget = self.UPLOAD_BUDGET
        self.t_upload_ms = 0.0

    # ----------------------------------------------------------------------------------------------- state
    def save_state(self) -> dict:
        return {"opacity": round(float(self.opacity), 3), "detail": int(self.detail), "url": self.url,
                "local_dir": self.local_dir}

    def load_state(self, d: dict) -> None:
        self.opacity = float(min(max(d.get("opacity", self.opacity), 0.0), 1.0))
        self.detail = int(min(max(d.get("detail", self.detail), 0), len(DETAILS) - 1))
        self.url = str(d.get("url") or bm.DEFAULT_URL)
        self.local_dir = str(d.get("local_dir") or "")

    # ----------------------------------------------------------------------------------------------- lifecycle
    def init(self, app) -> None:
        self._app = app
        self.handle = BasemapHandle(self, app.ctx)
        app.basemap = self.handle
        self._edit_url, self._edit_dir = self.url, self.local_dir

    def shutdown(self, app) -> None:
        self._teardown_data()
        if self.handle is not None:
            if getattr(app, "basemap", None) is self.handle:
                app.basemap = None
            self.handle.destroy()
            self.handle = None

    def _teardown_data(self):
        if self.loader is not None:
            self.loader.cancel()
            self.loader.join(5.0)
        self.loader = None
        if self.handle is not None:
            self.handle.set_texture(None)                  # bind group back to the placeholder BEFORE the texture dies
        if self.tex is not None:
            self.tex.destroy()
            self._app.ctx.untrack("basemap")
        self.tex = None
        self.comp = None
        self.plan = None

    def make_source(self) -> "bm.TileSource":
        """Overridable (tests): the tile source built from the user settings."""
        return bm.TileSource(self.url or None, local_dir=self.local_dir or None)

    def _begin(self, app):
        """Create composite + loader and start reading the DISK CACHE (never the network)."""
        frame = app.terrain.frame
        long_px = bm.SIZE_PRESETS[DETAILS[self.detail][0]]
        self.comp = bm.BasemapComposite(frame, long_px)
        self.source = self.make_source()
        self.loader = bm.BasemapLoader(self.comp, self.source, threads=4, duty=0.5)
        self.tex = BasemapTexture(app.ctx, self.comp)
        app.ctx.track("basemap", self.tex.bytes_total)
        self.handle.set_texture(self.tex.view)
        self._confirm = False
        self.error = ""
        self._last_phase = ""
        self.loader.start(download=False)
        self._refresh_plan()

    def _refresh_plan(self):
        if self._plan_busy or self.loader is None:
            return
        self._plan_busy = True
        loader = self.loader

        def work():
            try:
                p = loader.plan()
                if loader is self.loader:
                    self.plan = p
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self._plan_busy = False
        threading.Thread(target=work, name="basemap-plan", daemon=True).start()

    def is_ready(self) -> bool:
        return bool(self.tex is not None and self.comp is not None and self.comp.filled and self.tex.initial_done)

    def update(self, app, dt: float) -> None:
        if self.comp is None:
            self._begin(app)
        if self.tex is not None:
            t0 = time.perf_counter()
            if self.tex.adopt_fallback():
                self.tex.upload(self.upload_budget)
            self.t_upload_ms = (time.perf_counter() - t0) * 1000.0
        st = self.loader.status() if self.loader else None
        if st is not None and st.phase != self._last_phase:
            if self._last_phase in ("reading", "downloading") and not st.running:
                self._refresh_plan()                       # counts changed
            self._last_phase = st.phase
        if self.handle is not None:
            self.handle.write_params(app.ctx.queue)

    # ----------------------------------------------------------------------------------------------- UI
    def _status_text(self):
        st = self.loader.status() if self.loader else None
        c = self.comp
        if st is None:
            return (0.7, 0.7, 0.7, 1.0), "starting..."
        if st.phase == "reading":
            return (0.7, 0.85, 1.0, 1.0), f"Reading cached tiles  zoom {st.level}  {st.done}/{st.total}"
        if st.phase == "downloading":
            return (0.7, 0.85, 1.0, 1.0), (f"Downloading  zoom {st.level}  {st.done}/{st.total}   "
                                           f"{st.downloaded} new, {st.from_cache} cached" + (f", {st.failed} failed" if st.failed else ""))
        if st.phase == "unavailable":
            return (1.0, 0.75, 0.35, 1.0), f"Map unavailable: {st.message}. Plain fallback shown."
        if st.phase == "cancelled":
            return (1.0, 0.85, 0.4, 1.0), f"Cancelled. {len(c.tiles_done.get(c.zoom, ()))} of {c.n_final} detail tiles loaded."
        if st.phase == "error":
            return (1.0, 0.45, 0.4, 1.0), f"Error: {st.message}"
        if st.phase == "done":
            cov = c.coverage
            nd = len(c.tiles_done.get(c.zoom, ()))
            if cov >= 0.999:
                return (0.5, 0.9, 0.55, 1.0), f"Map ready: all {c.n_final} tiles at zoom {c.zoom}"
            if nd == 0:
                return (1.0, 0.85, 0.4, 1.0), "Low-detail overview only. Download the detail tiles below for the full map."
            return (1.0, 0.85, 0.4, 1.0), f"Partial map: {nd} of {c.n_final} detail tiles (the rest shows lower detail)"
        return (0.7, 0.7, 0.7, 1.0), ""

    def ui(self, app) -> None:
        from imgui_bundle import imgui
        draw_attribution(app)                              # on-screen credit (the overlay may already have drawn it: once per frame)
        _, self.opacity = imgui.slider_float("Opacity##bm", self.opacity, 0.0, 1.0, "%.2f")
        frame = app.terrain.frame
        names = []
        for key, label in DETAILS:
            mb = bm.composite_vram_mb(*bm.composite_dims(frame, bm.SIZE_PRESETS[key]))
            names.append(f"{label}  ({mb:.0f} MB VRAM" + (", over budget" if mb > VRAM_BUDGET_MB else "") + ")")
        imgui.set_next_item_width(-1)
        ch, idx = imgui.combo("##bmdetail", self.detail, names)
        if ch and idx != self.detail:
            self.detail = idx
            self._teardown_data()                          # rebuilt (from the disk cache only) on the next update
        col, text = self._status_text()
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(col, text)
        imgui.pop_text_wrap_pos()
        st = self.loader.status() if self.loader else None
        if st is not None and st.running:
            imgui.progress_bar(st.fraction, (-1, 6))
            if st.downloading and imgui.button("Cancel download"):
                self.loader.cancel()
        p = self.plan
        c = self.comp
        if p is not None and c is not None:
            host = urlparse(self.url).netloc or "the configured tile source"
            imgui.separator()
            imgui.text_wrapped(f"Tiles come from {host}. Its tile usage policy allows LIGHT use only: tiles are fetched once "
                               f"(max 4 connections, about 6 per second, valid User-Agent), kept on disk forever and never "
                               f"requested outside this map area.")
            _dim(imgui, f"Zoom {p.zoom}: {p.final_tiles} tiles + {p.total - p.final_tiles} overview tiles. "
                        f"{p.cached} cached, {p.missing} still to download (about {p.approx_mb:.0f} MB). "
                        f"Composite {p.composite_px[0]}x{p.composite_px[1]} px, {p.vram_mb:.0f} MB VRAM.")
            busy = st is not None and st.running
            if busy or p.missing == 0:
                imgui.begin_disabled()
            if not self._confirm:
                if imgui.button("Download tiles for this area", (-1, 0)):
                    if p.needs_confirm:
                        self._confirm = True
                    else:
                        self._start_download(confirmed=False)
            else:
                imgui.text_colored((1.0, 0.7, 0.35, 1.0), f"{p.missing} tiles (more than {bm.CONFIRM_TILES}). Confirm?")
                if imgui.button("Yes, download"):
                    self._confirm = False
                    self._start_download(confirmed=True)
                imgui.same_line()
                if imgui.button("No"):
                    self._confirm = False
            if busy or p.missing == 0:
                imgui.end_disabled()
        if imgui.tree_node("Tile source##bm"):
            ch1, self._edit_url = imgui.input_text("URL", self._edit_url)
            ch2, self._edit_dir = imgui.input_text("Local folder", self._edit_dir)
            _dim(imgui, "Local folder: z/x/y.png tiles, read before anything is downloaded.")
            if imgui.button("Apply##bmsrc") and (self._edit_url != self.url or self._edit_dir != self.local_dir):
                self.url, self.local_dir = self._edit_url, self._edit_dir
                self._teardown_data()
            imgui.tree_pop()
        imgui.text_disabled(ATTRIBUTION)
        if self.tex is not None:
            _dim(imgui, f"GPU upload {self.t_upload_ms:.2f} ms/frame, {self.tex.pending()} bands pending")

    def _start_download(self, confirmed: bool):
        try:
            self.loader.start(download=True, confirmed=confirmed, plan=self.plan)
        except bm.ConfirmationRequired:
            self._confirm = True
        except Exception as e:
            self.error = str(e)
