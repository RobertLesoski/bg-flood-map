"""Shared per-frame GPU state: frame uniforms + bind group 0, and the flow texture-array manager.

Bind group 0 (identical for every layer; `COMMON_WGSL` declares it, prepend it to your shader source):
    0 uniform  Frame            camera, sun, grid frame constants, hour / mix / exaggeration, hour->layer map
    1 texture_2d_array<f32>     flow grid, rgba8unorm, ONE layer per hour (R depth code, G/B velocity codes, A valid)
    2 sampler                   linear, clamp-to-edge (for the flow array)
    3 texture_2d<f32>           ground height, r32float, metres (read with textureLoad / ground_height())
    4 texture_2d<f32>           slope map, rgba16float with mips: R dz/dE, G dz/dN, B curvature, A 1 = real DEM
    5 sampler                   trilinear + anisotropic, clamp (for the slope map)

WGSL helpers in COMMON_WGSL: world_to_uv, ground_height, flow_at (depth + velocity at the displayed time),
vel_decode, haze, sky_color, sun_dir.  `Frame.flow.xy = (t, mix)`, hours `flow.zw`.  All metres; Z is exaggerated by
`frame.grid.w` (terrain Z scale) and depth by `frame.enc.w` (water depth scale) in the shaders that draw heights.

`FlowTextures` owns the texture array: layer = hour (0..24), layer 25 is the permanent all-zero layer; the uniform
`layer_map[h]` says which layer hour h reads, so dry hours (not in meta.hours) and hours still loading read zeros.
Decoding runs in worker threads, uploads are sliced so a frame never writes more than `upload_budget` bytes.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import wgpu

from ..data import flow as flowdata
from .bgthread import lower_priority
from .context import GpuContext, tex_bytes

N_HOURS = 25                 # layers 0..24
ZERO_LAYER = 25
N_LAYERS = 26
CHUNK_TEXELS = 128           # block size of the per-hour wet table (== terrain chunk size)
FRAME_BYTES = 368

COMMON_WGSL = """
struct Frame {
  view_proj: mat4x4<f32>,
  inv_view_proj: mat4x4<f32>,
  eye: vec4<f32>,        // xyz eye (world m), w = animation time (s)
  sun: vec4<f32>,        // xyz unit vector TOWARD the sun, w = intensity
  grid: vec4<f32>,       // size_x, size_y (ground m), cell_ground_m, terrain Z exaggeration
  gridpx: vec4<f32>,     // W, H, 1/W, 1/H   (flow / height grid pixels)
  flow: vec4<f32>,       // t (hours), mix, hour0, hour1
  enc: vec4<f32>,        // depth_max_m, vmax_ms, wet_threshold_m, water depth exaggeration
  view: vec4<f32>,       // width, height (px), terrain z min, z max (m, un-exaggerated)
  opts: vec4<f32>,       // x = water ripple strength, y = sun elevation (rad), z = haze density, w = hillshade relief boost
  layer_map: array<vec4<u32>, 7>,
};
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var flow_tex: texture_2d_array<f32>;
@group(0) @binding(2) var flow_samp: sampler;
@group(0) @binding(3) var height_tex: texture_2d<f32>;
@group(0) @binding(4) var slope_tex: texture_2d<f32>;
@group(0) @binding(5) var slope_samp: sampler;

const PI: f32 = 3.14159265;

fn world_to_uv(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(p.x / frame.grid.x + 0.5, 0.5 - p.y / frame.grid.y);
}
fn uv_to_world(uv: vec2<f32>) -> vec2<f32> {
  return vec2<f32>((uv.x - 0.5) * frame.grid.x, (0.5 - uv.y) * frame.grid.y);
}
// world XY of the centre of pixel (ix, iy)
fn texel_to_world(ix: f32, iy: f32) -> vec2<f32> {
  return vec2<f32>(((ix + 0.5) * frame.gridpx.z - 0.5) * frame.grid.x, (0.5 - (iy + 0.5) * frame.gridpx.w) * frame.grid.y);
}
fn height_texel(ix: i32, iy: i32) -> f32 {
  let c = vec2<i32>(clamp(ix, 0, i32(frame.gridpx.x) - 1), clamp(iy, 0, i32(frame.gridpx.y) - 1));
  return textureLoad(height_tex, c, 0).r;
}
// bilinear ground height in metres (un-exaggerated) at world XY
fn ground_height(p: vec2<f32>) -> f32 {
  let uv = world_to_uv(p);
  let f = uv * frame.gridpx.xy - vec2<f32>(0.5, 0.5);
  let i = vec2<i32>(floor(f));
  let t = f - floor(f);
  let a = mix(height_texel(i.x, i.y), height_texel(i.x + 1, i.y), t.x);
  let b = mix(height_texel(i.x, i.y + 1), height_texel(i.x + 1, i.y + 1), t.x);
  return mix(a, b, t.y);
}
// quality-governor knobs (bgv/perf.py), carried in the three spare words of layer_map (hours only use 0..24)
fn gov_shadow_steps() -> i32 { return clamp(i32(bitcast<f32>(frame.layer_map[6].y)), 4, 26); }
fn gov_smooth_ps() -> f32 { return bitcast<f32>(frame.layer_map[6].z); }   // sub-texel shoreline only closer than this many m/px
fn flow_layer(h: u32) -> i32 {
  return i32(frame.layer_map[h >> 2u][h & 3u]);
}
fn vel_decode(g: f32, b: f32, r: f32) -> vec2<f32> {
  // codes (0..1 normalised bytes) -> m/s; velocity only means something where the depth code is > 0
  if (r <= 0.0) { return vec2<f32>(0.0); }
  return (vec2<f32>(g, b) * 255.0 - vec2<f32>(128.0)) / 127.0 * frame.enc.y;
}
struct FlowSample {
  depth: f32,           // metres at the displayed time
  vel: vec2<f32>,       // m/s east, north
  valid: f32,           // 1 inside the model domain
};
// Flow at the displayed continuous time: codes interpolated bilinearly (hardware), each hour decoded, hours blended.
fn flow_at(uv: vec2<f32>) -> FlowSample {
  var o: FlowSample;
  let ha = u32(frame.flow.z);
  let hb = u32(frame.flow.w);
  let m = frame.flow.y;
  let a = textureSampleLevel(flow_tex, flow_samp, uv, flow_layer(ha), 0.0);
  let da = a.r * a.r * frame.enc.x;
  let va = vel_decode(a.g, a.b, a.r);
  var d = da; var v = va; var valid = a.a;
  if (m > 0.0) {
    let b = textureSampleLevel(flow_tex, flow_samp, uv, flow_layer(hb), 0.0);
    d = mix(da, b.r * b.r * frame.enc.x, m);
    v = mix(va, vel_decode(b.g, b.b, b.r), m);
    valid = max(a.a, b.a);
  }
  o.depth = d; o.vel = v; o.valid = valid;
  return o;
}
fn sun_dir() -> vec3<f32> { return frame.sun.xyz; }
fn sky_base(dir: vec3<f32>) -> vec3<f32> {
  let h = clamp(dir.z, -0.2, 1.0);
  let horizon = vec3<f32>(0.80, 0.87, 0.95);
  let zenith = vec3<f32>(0.26, 0.48, 0.82);
  var c = mix(horizon, zenith, pow(clamp(h, 0.0, 1.0), 0.55));
  let sd = max(dot(dir, frame.sun.xyz), 0.0);
  c += vec3<f32>(1.0, 0.92, 0.75) * pow(sd, 10.0) * 0.10 * frame.sun.w;      // soft glow around the sun
  return c;
}
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
  let sd = max(dot(dir, frame.sun.xyz), 0.0);
  return sky_base(dir) + vec3<f32>(1.0, 0.92, 0.75) * pow(sd, 600.0) * 1.5 * frame.sun.w;
}
// aerial perspective toward the horizon colour
fn haze(col: vec3<f32>, dist: f32, dir: vec3<f32>) -> vec3<f32> {
  let f = 1.0 - exp(-dist * frame.opts.z);
  let hc = sky_base(vec3<f32>(dir.x, dir.y, max(dir.z, 0.0) * 0.25));
  return mix(col, hc, clamp(f, 0.0, 0.92));
}
"""


def _bgl_entries():
    vis_all = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
    vf = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
    return [
        {"binding": 0, "visibility": vis_all, "buffer": {"type": "uniform"}},
        {"binding": 1, "visibility": vis_all, "texture": {"sample_type": "float", "view_dimension": "2d-array"}},
        {"binding": 2, "visibility": vis_all, "sampler": {"type": "filtering"}},
        {"binding": 3, "visibility": vis_all, "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"}},
        {"binding": 4, "visibility": vf, "texture": {"sample_type": "float", "view_dimension": "2d"}},
        {"binding": 5, "visibility": vf, "sampler": {"type": "filtering"}},
    ]


class FrameState:
    """The frame uniform buffer, bind group layout and bind group 0 (rebuilt when textures are replaced)."""

    def __init__(self, ctx: GpuContext):
        self.ctx = ctx
        dev = ctx.device
        self.layout = dev.create_bind_group_layout(label="frame-bgl", entries=_bgl_entries())
        self.buffer = dev.create_buffer(label="frame-uniforms", size=FRAME_BYTES,
                                        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.flow_sampler = dev.create_sampler(label="flow-samp", mag_filter="linear", min_filter="linear",
                                               address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        self.slope_sampler = dev.create_sampler(label="slope-samp", mag_filter="linear", min_filter="linear",
                                                mipmap_filter="linear", address_mode_u="clamp-to-edge",
                                                address_mode_v="clamp-to-edge", max_anisotropy=8)
        self.bind_group = None
        self.f = np.zeros(FRAME_BYTES // 4, np.float32)
        self.u = self.f.view(np.uint32)
        self.dirty_bg = True
        self._vkey = None
        self._lmkey = None
        self._grid_vals = None
        self.q = [26.0, 8.0, 0.0]            # governor: shadow steps, smooth-shore max m/px, spare

    def rebuild(self, flow_view, height_view, slope_view):
        self.bind_group = self.ctx.device.create_bind_group(label="frame-bg", layout=self.layout, entries=[
            {"binding": 0, "resource": {"buffer": self.buffer, "offset": 0, "size": FRAME_BYTES}},
            {"binding": 1, "resource": flow_view},
            {"binding": 2, "resource": self.flow_sampler},
            {"binding": 3, "resource": height_view},
            {"binding": 4, "resource": slope_view},
            {"binding": 5, "resource": self.slope_sampler},
        ])

    def write(self, *, view_proj, eye, anim_time, sun, sun_intensity, grid, enc, t, mix, h0, h1, exag, water_exag,
              viewport, zrange, ripple, sun_elev, haze_density, layer_map, relief=1.0):
        f = self.f
        vpa = np.asarray(view_proj, np.float64)
        vkey = vpa.tobytes()
        if vkey != self._vkey:                      # view-projection (and its inverse) only change when the camera moves
            self._vkey = vkey
            f[0:16] = vpa.T.reshape(-1)
            try:
                inv = np.linalg.inv(vpa)
            except np.linalg.LinAlgError:
                inv = np.eye(4)
            f[16:32] = inv.T.reshape(-1)
        gk = self._grid_vals
        if gk is None or gk[0] is not grid:
            gk = self._grid_vals = (grid, grid.size_m[0], grid.size_m[1], grid.cell_ground_m, grid.width, grid.height,
                                    1.0 / grid.width, 1.0 / grid.height)
        f[32:64] = (eye[0], eye[1], eye[2], anim_time, sun[0], sun[1], sun[2], sun_intensity,
                    gk[1], gk[2], gk[3], exag, gk[4], gk[5], gk[6], gk[7],
                    t, mix, h0, h1, enc[0], enc[1], enc[2], water_exag,
                    viewport[0], viewport[1], zrange[0], zrange[1], ripple, sun_elev, haze_density, relief)
        lmb = layer_map.tobytes() if hasattr(layer_map, "tobytes") else bytes(layer_map)
        if lmb != self._lmkey:                      # the hour -> layer table only changes while hours stream in
            self._lmkey = lmb
            lm = np.full(28, ZERO_LAYER, np.uint32)
            lm[:len(layer_map)] = layer_map
            self.u[64:92] = lm
        f[64 + 25] = self.q[0]                      # governor knobs ride in the spare layer_map words 25 / 26 (floats)
        f[64 + 26] = self.q[1]
        self.ctx.queue.write_buffer(self.buffer, 0, f)


# --------------------------------------------------------------------------------------------- flow textures
class _Upload:
    __slots__ = ("hour", "rgba", "row", "wet")

    def __init__(self, hour, rgba, wet):
        self.hour, self.rgba, self.row, self.wet = hour, rgba, 0, wet


class FlowTextures:
    """rgba8unorm texture array, one layer per hour.  See module docstring."""

    def __init__(self, ctx: GpuContext, width: int, height: int, upload_budget: int = 6 << 20, workers: int = 1):
        self.ctx = ctx
        self.width, self.height = width, height
        self.upload_budget = upload_budget
        self.upload_chunk = 1 << 20                      # one write_texture call carries at most this many bytes
        self.upload_ms = 2.0                             # ... and a frame stops uploading after about this much time
        self.texture = ctx.device.create_texture(
            label="flow-array", size=(width, height, N_LAYERS), format="rgba8unorm", dimension="2d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC)
        self.view = self.texture.create_view(dimension="2d-array")
        # Hours are uploaded into this single-layer staging texture in small per-frame chunks and copied into the array in ONE
        # GPU copy when complete: writing chunks straight into the (sampled) array made the driver re-transition the whole
        # 40 MB layer for every chunk (10 - 30 ms GPU stalls on the GTX 1650 while a storm loads).
        self.stage = ctx.device.create_texture(
            label="flow-stage", size=(width, height, 1), format="rgba8unorm", dimension="2d",
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC)
        ctx.track("flow staging", tex_bytes(width, height, 1, 4))
        ctx.track("flow textures", tex_bytes(width, height, N_LAYERS, 4))
        self.layer_map = np.full(N_HOURS, ZERO_LAYER, np.uint32)
        self.nx = -(-width // CHUNK_TEXELS)
        self.ny = -(-height // CHUNK_TEXELS)
        self._empty_wet = np.zeros((self.ny, self.nx), bool)
        self._wet: dict[int, np.ndarray] = {}
        self.storm = None
        self._gen = 0
        self._jobs: queue.Queue = queue.Queue()
        self._closed = False
        self._threads = [threading.Thread(target=self._worker, name=f"bgv-decode-{i}", daemon=True)
                         for i in range(workers)]
        for th in self._threads:
            th.start()
        self._done: queue.Queue = queue.Queue()
        self._slots = threading.Semaphore(3)           # decoded-but-not-uploaded hours (3 x 40 MB)
        self._free: queue.SimpleQueue = queue.SimpleQueue()     # decode buffers, reused (never a fresh 40 MB allocation)
        self._pooled: set[int] = set()
        self._cur: _Upload | None = None
        self._wanted: list[int] = []
        self._loaded: set[int] = set()
        self._failed: dict[int, str] = {}
        self.t_switch = time.perf_counter()
        self.t_first_ready: float | None = None
        self.uploaded_bytes_total = 0
        self.last_pump_ms = 0.0
        self.max_pump_ms = 0.0

    # ------------------------------------------------------------------ storm switching
    def set_storm(self, storm: flowdata.Storm, t: float = 0.0):
        """Switch to another storm of the SAME grid size.  All hours read as dry until their layer is uploaded;
        the hours around `t` load first, so water is back within ~two decode times, and no frame stalls."""
        if (storm.width, storm.height) != (self.width, self.height):
            raise ValueError("grid size changed: create a new FlowTextures")
        self._gen += 1
        gen = self._gen
        self.storm = storm
        self.layer_map[:] = ZERO_LAYER
        self._wet.clear()
        self._loaded.clear()
        self._failed.clear()
        if self._cur is not None:
            cur, self._cur = self._cur, None
            self._release(cur.rgba)
        self.t_switch = time.perf_counter()
        self.t_first_ready = None
        # drain results of the previous generation
        try:
            while True:
                it = self._done.get_nowait()
                self._release(it[2])
        except queue.Empty:
            pass
        self._wanted = sorted(storm.hours, key=lambda h: (abs(h - t), h))
        while True:                                       # drop jobs of the previous storm
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        for h in self._wanted:
            self._jobs.put((gen, storm, h))

    def _worker(self):
        lower_priority()
        while not self._closed:
            try:
                gen, storm, h = self._jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            while not self._closed and gen == self._gen:
                if self._slots.acquire(timeout=0.1):
                    break
            else:
                continue
            self._decode(gen, storm, h)

    def _take_buffer(self) -> np.ndarray:
        try:
            return self._free.get_nowait()
        except queue.Empty:
            b = np.empty((self.height, self.width, 4), np.uint8)
            self._pooled.add(id(b))
            return b

    def _release(self, arr=None):
        """Give a decode buffer back to the pool and free its slot."""
        if arr is not None and id(arr) in self._pooled:
            self._free.put(arr)
        self._slots.release()

    def _decode(self, gen: int, storm: flowdata.Storm, h: int):
        if gen != self._gen:
            self._release()
            return
        buf = None
        try:
            buf = self._take_buffer()
            arr = storm.read_hour_into(h, buf)
            wet = flowdata.block_max(arr[..., 0], CHUNK_TEXELS) > 0
            self._done.put((gen, h, arr, wet, None))
        except Exception as e:                           # corrupt file: report, keep running
            if buf is not None:
                self._free.put(buf)
            self._done.put((gen, h, None, None, f"{type(e).__name__}: {e}"))

    # ------------------------------------------------------------------ per frame
    def pump(self) -> int:
        """Upload up to `upload_budget` bytes of decoded hours.  Returns bytes written this call."""
        t0 = time.perf_counter()
        budget = self.upload_budget
        rowbytes = self.width * 4
        written = 0
        chunk = max(rowbytes, self.upload_chunk)
        tlimit = t0 + self.upload_ms * 1e-3
        while budget > 0:
            if self._cur is None:
                try:
                    gen, h, arr, wet, err = self._done.get_nowait()
                except queue.Empty:
                    break
                if gen != self._gen:
                    self._release(arr)
                    continue
                if err is not None:
                    self._failed[h] = err
                    self._release()
                    continue
                self._cur = _Upload(h, arr, wet)
            u = self._cur
            rows = min(self.height - u.row, max(1, min(budget, chunk) // rowbytes))
            self.ctx.queue.write_texture(
                {"texture": self.stage, "mip_level": 0, "origin": (0, u.row, 0)},
                u.rgba[u.row:u.row + rows],
                {"offset": 0, "bytes_per_row": rowbytes, "rows_per_image": rows},
                (self.width, rows, 1))
            u.row += rows
            budget -= rows * rowbytes
            written += rows * rowbytes
            if time.perf_counter() > tlimit and u.row < self.height:
                break                                    # time budget spent: the rest of this hour goes next frame
            if u.row >= self.height:
                enc = self.ctx.device.create_command_encoder(label="flow-layer-copy")
                enc.copy_texture_to_texture({"texture": self.stage, "mip_level": 0, "origin": (0, 0, 0)},
                                            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, u.hour)},
                                            (self.width, self.height, 1))
                self.ctx.queue.submit([enc.finish()])
                self.layer_map[u.hour] = u.hour
                self._wet[u.hour] = u.wet
                self._loaded.add(u.hour)
                self._cur = None
                self._release(u.rgba)
                if self.t_first_ready is None and len(self._loaded) >= min(2, len(self._wanted)):
                    self.t_first_ready = time.perf_counter()
        self.uploaded_bytes_total += written
        self.last_pump_ms = (time.perf_counter() - t0) * 1000.0
        self.max_pump_ms = max(self.max_pump_ms, self.last_pump_ms)
        return written

    # ------------------------------------------------------------------ queries
    def hour_ready(self, h: int) -> bool:
        """True when hour h reads real data, or is dry by definition (not in meta.hours)."""
        if self.storm is None:
            return False
        if self.storm.is_zero(h):
            return True
        return h in self._loaded or h in self._failed

    def ready_for(self, t: float) -> bool:
        h0, h1, mix = flowdata.hour_pair(t)
        return self.hour_ready(h0) and (mix <= 0 or self.hour_ready(h1))

    def progress(self) -> tuple[int, int]:
        """(hours resolved, hours expected) for the current storm.  An hour whose file is missing / corrupt counts as
        resolved (it reads dry and is reported through `failed`), so progress bars and waits always finish."""
        return len(self._loaded) + len(self._failed), len(self._wanted)

    @property
    def failed(self) -> dict[int, str]:
        return dict(self._failed)

    def chunk_wet(self, h0: int, h1: int, mix: float) -> np.ndarray:
        """bool[ny, nx]: terrain chunks (128 texels) that may hold water at the displayed time."""
        a = self._wet.get(h0, self._empty_wet)
        if mix > 0:
            return a | self._wet.get(h1, self._empty_wet)
        return a

    def close(self):
        self._gen += 1
        self._closed = True
        self.ctx.untrack("flow textures")
        self.ctx.untrack("flow staging")
        for t in (self.texture, self.stage):
            try:
                t.destroy()
            except Exception:
                pass
