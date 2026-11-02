"""GPU adapter / device selection, limits, resource accounting and GPU timestamp timing.

    ctx = GpuContext.create()          # prints the adapter; Vulkan high-performance adapter (GTX 1650), never the iGPU
    ctx.device, ctx.queue, ctx.adapter_name, ctx.features, ctx.limits
    ctx.track("flow textures", nbytes) # VRAM estimate bookkeeping (ctx.vram_mb())
    timer = GpuTimer(ctx);  timer.pass_writes() -> dict for begin_render_pass(timestamp_writes=...)
"""
from __future__ import annotations

import os
import struct
import sys

import wgpu

WANTED_FEATURES = ("timestamp-query", "float32-filterable", "indirect-first-instance", "texture-format-16bit-norm")
WANTED_LIMITS = ("max-texture-dimension-2d", "max-texture-array-layers", "max-buffer-size",
                 "max-storage-buffer-binding-size", "max-compute-workgroup-size-x", "max-compute-invocations-per-workgroup",
                 "max-bind-groups", "max-sampled-textures-per-shader-stage", "max-storage-buffers-per-shader-stage")


def _score(info) -> tuple:
    """Higher is better: discrete + Vulkan first (on this hybrid laptop the OpenGL context lands on the iGPU)."""
    kind = {"DiscreteGPU": 3, "IntegratedGPU": 1, "VirtualGPU": 0, "CPU": -5}.get(info.get("adapter_type", ""), 0)
    be = {"Vulkan": 2, "D3D12": 1, "Metal": 2}.get(info.get("backend_type", ""), -3)   # OpenGL / unknown lose
    return (kind, be)


def _fast_vulkan_adapter():
    """The high-performance Vulkan adapter without enumerating every backend (D3D12, OpenGL, WARP ...), which costs about
    0.8 s of start-up on this laptop.  Returns None unless the result is a discrete GPU (then the full scoring below runs)."""
    forced = os.environ.get("WGPU_BACKEND_TYPE")
    try:
        if not forced:
            os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"
        a = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception:
        return None
    finally:
        if not forced:
            os.environ.pop("WGPU_BACKEND_TYPE", None)
    try:
        if a is not None and a.info.get("adapter_type") == "DiscreteGPU":
            return a
    except Exception:
        pass
    return None


def pick_adapter(prefer: str | None = None):
    """Best adapter: env BGV_ADAPTER (substring of the device name) wins, then discrete+Vulkan, then wgpu's own
    high-performance choice."""
    prefer = prefer or os.environ.get("BGV_ADAPTER")
    if not prefer:
        fast = _fast_vulkan_adapter()
        if fast is not None:
            return fast
    try:
        adapters = list(wgpu.gpu.enumerate_adapters_sync())
    except Exception:
        adapters = []
    if prefer:
        for a in adapters:
            if prefer.lower() in a.info.get("device", "").lower():
                return a
    if adapters:
        best = max(adapters, key=lambda a: _score(a.info))
        if _score(best.info)[0] >= 3:
            return best
    return wgpu.gpu.request_adapter_sync(power_preference="high-performance")


class GpuContext:
    def __init__(self, adapter, device):
        self.adapter = adapter
        self.device = device
        self.queue = device.queue
        info = adapter.info
        self.adapter_name = info.get("device", "?")
        self.backend = info.get("backend_type", "?")
        self.adapter_type = info.get("adapter_type", "?")
        self.driver = info.get("description", "")
        self.features = set(device.features)
        self.limits = dict(device.limits)
        self.has_timestamps = "timestamp-query" in self.features
        self.has_f32_filterable = "float32-filterable" in self.features
        self._tracked: dict[str, int] = {}
        self._sync_buf = None

    @staticmethod
    def create(prefer: str | None = None, verbose: bool = True) -> "GpuContext":
        from . import cffiwarm
        cffiwarm.speedup()                                  # cffi type parsing 50x faster (see cffiwarm.py)
        adapter = pick_adapter(prefer)
        feats = [f for f in WANTED_FEATURES if f in adapter.features]
        alim = adapter.limits
        req = {k: alim[k] for k in WANTED_LIMITS if k in alim}
        device = adapter.request_device_sync(required_features=feats, required_limits=req)
        ctx = GpuContext(adapter, device)
        if verbose:
            print(f"[bgv] GPU adapter: {ctx.adapter_name} ({ctx.adapter_type}, {ctx.backend}) {ctx.driver}".rstrip(),
                  flush=True)
            if ctx.adapter_type != "DiscreteGPU":
                print("[bgv] WARNING: not a discrete GPU - performance will be poor on the hybrid laptop. "
                      "Set BGV_ADAPTER=<name> to force one.", file=sys.stderr, flush=True)
        return ctx

    # ------------------------------------------------------------------ accounting
    def track(self, name: str, nbytes: int):
        """Record (or update) the size of a long-lived GPU resource for the VRAM estimate."""
        self._tracked[name] = int(nbytes)

    def untrack(self, name: str):
        self._tracked.pop(name, None)

    def vram_bytes(self) -> int:
        return sum(self._tracked.values())

    def vram_mb(self) -> float:
        return self.vram_bytes() / 1048576.0

    def vram_breakdown(self) -> dict[str, float]:
        return {k: round(v / 1048576.0, 1) for k, v in sorted(self._tracked.items(), key=lambda kv: -kv[1])}

    def sync(self):
        """Block until all submitted GPU work has finished (benchmarks / screenshots only)."""
        # queue.on_submitted_work_done_* is broken in wgpu-py 0.32 (cffi callback type error); reading back a tiny
        # buffer is queued behind all earlier work and blocks until it completed.
        if self._sync_buf is None:
            self._sync_buf = self.device.create_buffer(size=16, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self.queue.read_buffer(self._sync_buf, 0, 16)


def tex_bytes(width, height, layers, bytes_per_texel, mips=1) -> int:
    total = 0
    w, h = width, height
    for _ in range(mips):
        total += w * h * layers * bytes_per_texel
        w, h = max(w // 2, 1), max(h // 2, 1)
    return total


class GpuTimer:
    """GPU time of one render pass via timestamp queries (needs 'timestamp-query'); results are read back a few
    frames late without stalling.  `last_ms` is the most recent resolved pass time, None when unsupported.
    Assumes a 1 ns timestamp period (true for NVIDIA/AMD on Vulkan; wgpu-py does not expose the period)."""

    SLOTS = 3

    def __init__(self, ctx: GpuContext):
        self.ctx = ctx
        self.enabled = ctx.has_timestamps
        self.last_ms: float | None = None
        self._frame = 0
        if not self.enabled:
            return
        dev = ctx.device
        self._qs = [dev.create_query_set(type="timestamp", count=2) for _ in range(self.SLOTS)]
        self._resolve = [dev.create_buffer(size=256, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
                         for _ in range(self.SLOTS)]
        self._read = [dev.create_buffer(size=16, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
                      for _ in range(self.SLOTS)]
        self._pending: list = [None] * self.SLOTS
        self._slot = 0

    def pass_writes(self):
        """`timestamp_writes=` for the scene render pass (begin + end of pass), or None."""
        if not self.enabled:
            return None
        self._poll()
        s = self._slot
        if self._pending[s] is not None:           # slot still in flight: skip this frame's timing
            self._cur = None
            return None
        self._cur = s
        return {"query_set": self._qs[s], "beginning_of_pass_write_index": 0, "end_of_pass_write_index": 1}

    def after_pass(self, encoder):
        """Call on the command encoder after the timed pass ended."""
        if not self.enabled or getattr(self, "_cur", None) is None:
            return
        s = self._cur
        encoder.resolve_query_set(self._qs[s], 0, 2, self._resolve[s], 0)
        encoder.copy_buffer_to_buffer(self._resolve[s], 0, self._read[s], 0, 16)
        self._cur = None
        self._slot = (s + 1) % self.SLOTS
        self._issue = s

    def after_submit(self):
        s = getattr(self, "_issue", None)
        if s is None or not self.enabled:
            return
        self._issue = None
        try:
            self._pending[s] = self._read[s].map_async("READ_NOSYNC", 0, 16)
        except Exception:
            self._pending[s] = self._read[s].map_async("READ", 0, 16)

    def _poll(self):
        for s in range(self.SLOTS):
            buf = self._read[s]
            if self._pending[s] is not None and buf.map_state == "mapped":
                try:
                    data = bytes(buf.read_mapped(0, 16))
                    t0, t1 = struct.unpack("<QQ", data)
                    if t1 >= t0:
                        self.last_ms = (t1 - t0) * 1e-6
                finally:
                    buf.unmap()
                    self._pending[s] = None

    def read_blocking(self) -> float | None:
        """Wait for the in-flight timing and return it (benchmark use)."""
        if not self.enabled:
            return None
        for s in range(self.SLOTS):
            p = self._pending[s]
            if p is not None:
                try:
                    p.sync_wait()
                except Exception:
                    pass
        self._poll()
        return self.last_ms
