"""Terrain GPU resources: height texture (r32float, metres, NaN filled), slope map (rgba16float + mips), CPU sampler.

`app.terrain` is a `TerrainTextures`:
    .frame        GridFrame of the flow grid (world <-> mercator <-> uv <-> pixel)
    .heights      float32[H, W] filled heights (metres, NOT exaggerated), row 0 = north
    .valid        bool[H, W] False where the DEM had no coverage (heights there are nearest-filled)
    .sampler      camera.HeightSampler (bilinear CPU ground height + ray casting, world metres)
    .stats        dict (min / max / mean / nan_fraction)
    .height_view, .slope_view   texture views bound in bind group 0
    .ground(x, y) un-exaggerated ground height in metres at world x, y
"""
from __future__ import annotations

import time

import numpy as np
import wgpu

from ..data import terrain as terraindata
from ..geo import GridFrame
from .camera import HeightSampler
from .context import GpuContext, tex_bytes


class TerrainTextures:
    def __init__(self, ctx: GpuContext, root, frame: GridFrame, data=None, source=None):
        """`data`: the tuple of `terraindata.load_terrain_gpu_data` when a worker thread already prepared it.  `source`: a
        `terraindata.TerrainSource` (the domain's DEM; default the city's)."""
        self.ctx = ctx
        self.frame = frame
        t0 = time.perf_counter()
        filled, mips, stats, valid = data if data is not None else terraindata.load_terrain_gpu_data(root, frame, source)
        self.t_load = time.perf_counter() - t0
        self.heights = filled
        self.valid = valid
        self.stats = stats
        sx, sy = frame.size_m
        self.sampler = HeightSampler(filled, sx, sy)
        dev, q = ctx.device, ctx.queue
        W, H = frame.width, frame.height
        t0 = time.perf_counter()
        self.height_tex = dev.create_texture(label="terrain-height", size=(W, H, 1), format="r32float",
                                             usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        q.write_texture({"texture": self.height_tex, "mip_level": 0, "origin": (0, 0, 0)}, np.ascontiguousarray(filled),
                        {"offset": 0, "bytes_per_row": W * 4, "rows_per_image": H}, (W, H, 1))
        self.slope_tex = dev.create_texture(label="terrain-slope", size=(W, H, 1), format="rgba16float",
                                            mip_level_count=len(mips),
                                            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        for lvl, m in enumerate(mips):
            h, w = m.shape[:2]
            q.write_texture({"texture": self.slope_tex, "mip_level": lvl, "origin": (0, 0, 0)}, np.ascontiguousarray(m),
                            {"offset": 0, "bytes_per_row": w * 8, "rows_per_image": h}, (w, h, 1))
        self.height_view = self.height_tex.create_view()
        self.slope_view = self.slope_tex.create_view()
        self.t_upload = time.perf_counter() - t0
        ctx.track("terrain height", tex_bytes(W, H, 1, 4))
        ctx.track("terrain slope map", tex_bytes(W, H, 1, 8, len(mips)))

    @property
    def zrange(self) -> tuple[float, float]:
        return float(self.stats["min"]), float(self.stats["max"])

    def ground(self, x, y):
        return self.sampler.sample(x, y)

    def close(self):
        for t in (self.height_tex, self.slope_tex):
            try:
                t.destroy()
            except Exception:
                pass
        self.ctx.untrack("terrain height")
        self.ctx.untrack("terrain slope map")
