"""Basemap data module: tile maths vs bgv.geo, resampling correctness, mips, fake tile server (no real network)."""
import http.server
import io
import math
import threading
import time

import numpy as np
import pytest
from PIL import Image

from bgv.data import basemap as bm
from bgv.geo import GridFrame, lnglat_to_merc, merc_to_lnglat

# the real Bowling Green flow grid (25yr meta.json) - literal so the test does not need the data root
BG = GridFrame(2947, 3368, 6.4, 5.1128, -9633395.200000001, 4425075.2, -9614534.4, 4446630.4)


def small_frame(n=300, lng=-86.45, lat=36.97):
    """A ~1.9 km square grid near Bowling Green (a handful of tiles per zoom)."""
    cell = 6.4
    x, y = lnglat_to_merc(lng, lat)
    half = n * cell / 2
    return GridFrame(n, n, cell, cell * 0.7993, x - half, y - half, x + half, y + half)


# ------------------------------------------------------------------------------------------------------ tile maths
def test_world_tile_and_span():
    assert bm.tile_bounds_merc(0, 0, 0) == pytest.approx((-bm.ORIGIN, -bm.ORIGIN, bm.ORIGIN, bm.ORIGIN))
    assert bm.tile_span_m(16) == pytest.approx(611.4962, abs=1e-3)
    assert bm.meters_per_px(16) == pytest.approx(2.3887, abs=1e-3)           # the "2.4 m/px" of the brief
    assert bm.meters_per_px(16, 37.0) == pytest.approx(2.3887 * math.cos(math.radians(37.0)), abs=1e-3)


@pytest.mark.parametrize("z", [8, 12, 15, 16])
def test_grid_corners_match_slippy_formula_and_geo(z):
    x0, x1, y0, y1 = bm.tile_range(BG, z)
    # north-west and south-east corners through bgv.geo -> lng/lat -> textbook slippy-map formula
    lng0, lat0 = merc_to_lnglat(BG.xmin, BG.ymax)
    lng1, lat1 = merc_to_lnglat(BG.xmax, BG.ymin)
    ref0 = bm.lnglat_to_tile(lng0, lat0, z)
    ref1 = bm.lnglat_to_tile(lng1, lat1, z)
    assert (x0, y0) == ref0
    assert (x1, y1) == ref1
    # the covering tiles really contain the grid, and nothing beyond one tile of margin
    bx0 = bm.tile_bounds_merc(z, x0, y0)
    bx1 = bm.tile_bounds_merc(z, x1, y1)
    assert bx0[0] <= BG.xmin and bx0[3] >= BG.ymax
    assert bx1[2] >= BG.xmax and bx1[1] <= BG.ymin
    s = bm.tile_span_m(z)
    assert BG.xmin - bx0[0] < s and bx1[2] - BG.xmax < s
    assert bx0[3] - BG.ymax < s and BG.ymin - bx1[1] < s
    # grid corners through GridFrame.world_to_merc agree (world frame <-> merc)
    sx, sy = BG.size_m
    mx, my = BG.world_to_merc(-sx / 2, sy / 2)
    assert (mx, my) == pytest.approx((BG.xmin, BG.ymax), abs=1e-6)


def test_tile_counts_for_real_grid():
    n16 = len(bm.tiles_for_frame(BG, 16))
    assert 1000 <= n16 <= 1300                                              # "about 1.2k tiles at zoom 16"
    assert len(bm.tiles_for_frame(BG, 15)) == 288
    assert len(set(bm.tiles_for_frame(BG, 14))) == 81


def test_choose_zoom_and_dims():
    assert bm.choose_zoom(BG, 4096) == 15
    assert bm.choose_zoom(BG, 6144) == 16 and bm.choose_zoom(BG, 8192) == 16
    assert bm.choose_zoom(BG, 20000) == bm.MAX_ZOOM                         # capped
    w, h = bm.composite_dims(BG, 4096)
    assert (w, h) == (3584, 4096) and abs(w / h - BG.width / BG.height) < 0.002
    for p in (4096, 6144, 8192):
        w, h = bm.composite_dims(BG, p)
        assert max(w, h) == p and w % 64 == 0
        assert bm.composite_vram_mb(w, h) == pytest.approx(w * h * 4 * (4 / 3) / 1048576, rel=0.01)
    assert bm.composite_vram_mb(*bm.composite_dims(BG, 4096)) < 80
    assert bm.composite_vram_mb(*bm.composite_dims(BG, 6144)) < 200
    assert bm.composite_vram_mb(*bm.composite_dims(BG, 8192)) > 200         # the "ultra" preset exceeds the 200 MB budget
    p = bm.max_long_px_for_vram(BG, 200)
    assert bm.composite_vram_mb(*bm.composite_dims(BG, p)) <= 200 < bm.composite_vram_mb(*bm.composite_dims(BG, p + 256))


def test_levels_progressive():
    assert bm.levels_for(16) == [12, 14, 16] and bm.levels_for(15) == [11, 13, 15]
    assert bm.levels_for(5) == [4, 5]


# ------------------------------------------------------------------------------------------------------ resampling
def _mosaic_tile(z, x, y, seed=3):
    rng = np.random.default_rng((seed, z, x, y))
    # smooth-ish random content so seams are visible if the weights were wrong
    a = rng.integers(0, 256, (32, 32, 3)).astype(np.float32)
    big = np.kron(a, np.ones((8, 8, 1), np.float32))
    return big.astype(np.uint8)


def _reference_pixel(comp, z, i, j, tile_fn):
    """Exact area average of the footprint of dest pixel (i, j) over the mosaic (independent, loop based, s >= 1)."""
    fr = comp.frame
    fx0, fy0 = bm.merc_to_tilef(fr.xmin, fr.ymax, z)
    fx1, fy1 = bm.merc_to_tilef(fr.xmax, fr.ymin, z)
    sx = (fx1 - fx0) * 256 / comp.width
    sy = (fy1 - fy0) * 256 / comp.height
    cx = (fx0 + (i + 0.5) / comp.width * (fx1 - fx0)) * 256
    cy = (fy0 + (j + 0.5) / comp.height * (fy1 - fy0)) * 256
    acc, wsum = np.zeros(3), 0.0
    for py in range(int(math.floor(cy - sy / 2)), int(math.ceil(cy + sy / 2))):
        oy = min(cy + sy / 2, py + 1) - max(cy - sy / 2, py)
        for px in range(int(math.floor(cx - sx / 2)), int(math.ceil(cx + sx / 2))):
            ox = min(cx + sx / 2, px + 1) - max(cx - sx / 2, px)
            if ox <= 0 or oy <= 0:
                continue
            t = tile_fn(z, px // 256, py // 256)
            acc += oy * ox * t[py % 256, px % 256, :3]
            wsum += oy * ox
    return acc / wsum


def test_put_tile_matches_exact_box_filter_including_seams():
    fr = small_frame()
    comp = bm.BasemapComposite(fr, long_px=512, zoom=16)                     # 1.57 source px per dest px
    z = comp.zoom
    s = (fr.xmax - fr.xmin) / bm.tile_span_m(z) * 256 / comp.width
    assert 1.0 <= s < 2.0                                                   # the regime choose_zoom guarantees
    x0, x1, y0, y1 = bm.tile_range(fr, z)
    cache = {}

    def getter(zz, xx, yy):
        if (xx, yy) not in cache:
            cache[(xx, yy)] = _mosaic_tile(zz, xx, yy)
        return cache[(xx, yy)]
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            assert comp.put_tile(z, x, y, getter(z, x, y), getter)
    rng = np.random.default_rng(0)
    pts = [(int(i), int(j)) for i, j in rng.integers(2, comp.width - 2, (120, 2))]
    # plus pixels sitting on tile seams
    fx0, fy0 = bm.merc_to_tilef(fr.xmin, fr.ymax, z)
    fx1, fy1 = bm.merc_to_tilef(fr.xmax, fr.ymin, z)
    for t in range(x0 + 1, x1 + 1):
        i = int((t - fx0) / (fx1 - fx0) * comp.width)
        pts += [(i - 1, 60), (i, 60), (i, 20), (i + 1, 300)]
    for t in range(y0 + 1, y1 + 1):
        j = int((t - fy0) / (fy1 - fy0) * comp.height)
        pts += [(60, j - 1), (60, j), (20, j), (300, j + 1)]
    worst = 0.0
    for (i, j) in pts:
        if not (0 <= i < comp.width and 0 <= j < comp.height):
            continue
        ref = _reference_pixel(comp, z, i, j, getter)
        got = comp.rgba[j, i, :3].astype(np.float64)
        worst = max(worst, float(np.abs(got - ref).max()))
    assert worst <= 1.0 + 1e-6, worst


def test_upsampling_is_bilinear_and_seamless():
    """Low zoom tiles are up-sampled: a horizontal ramp stays a ramp (no seams, no blockiness)."""
    fr = small_frame()
    z = 12
    comp = bm.BasemapComposite(fr, long_px=512, zoom=16)

    def tile_fn(zz, xx, yy):
        gx = (xx * 256 + np.arange(256))[None, :].repeat(256, 0)
        r = (gx % 200).astype(np.uint8)                                      # slope 1 / px, wraps (rare, avoided below)
        return np.stack([r, np.full_like(r, 90), np.full_like(r, 30)], -1)
    x0, x1, y0, y1 = bm.tile_range(fr, z)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            comp.put_tile(z, x, y, tile_fn(z, x, y), tile_fn)
    row = comp.rgba[200, :, 0].astype(int)
    d = np.diff(row)
    ok = d[(np.abs(d) < 50)]                                                 # ignore the wrap discontinuity
    assert ok.max() <= 2 and ok.min() >= 0                                   # smooth monotone steps, no jumps
    assert comp.has_tiles and comp.tiles_done[z]


def test_box_mip_linear_light_and_dims():
    src = np.zeros((4, 6, 4), np.uint8)
    src[..., 3] = 255
    src[0::2, 0::2, :3] = 255                                                # checker: half white, half black
    src[1::2, 1::2, :3] = 255
    dst = np.zeros((2, 3, 4), np.uint8)
    bm.box_mip(src, dst, 0, 0, 3, 2)
    # linear 0.5 -> sRGB 0.7354 -> 187.5
    assert abs(int(dst[0, 0, 0]) - 188) <= 1 and dst[0, 0, 3] == 255
    comp = bm.BasemapComposite(small_frame(), long_px=256)
    dims = [m.shape[:2] for m in comp.mips]
    assert dims[0] == (256, 256) and dims[1] == (128, 128) and dims[-1] == (1, 1) and len(dims) == 9
    comp.rgba[:] = 255
    comp.rebuild_mips()
    assert (comp.mips[-1] == 255).all() and (comp.mips[3] == 255).all()


def test_fallback_and_dirty_bands():
    comp = bm.BasemapComposite(small_frame(), long_px=256)
    assert not comp.filled and not comp.has_tiles
    comp.fill_fallback()
    assert comp.filled and not comp.has_tiles and comp.coverage == 0.0
    assert comp.rgba[..., 3].min() == 255 and tuple(comp.rgba[5, 5, :3]) == bm.FALLBACK_RGB
    assert all(tuple(m[0, 0, :3]) == bm.FALLBACK_RGB for m in comp.mips)      # every level, no mip maths
    assert (comp.band_gen == comp.fallback_gen).all()                        # one generation = 'fallback everywhere'
    comp.band_gen[:] = 0
    tile = np.full((256, 256, 3), 10, np.uint8)
    x0, _, y0, _ = bm.tile_range(comp.frame, comp.zoom)
    comp.put_tile(comp.zoom, x0, y0, tile)
    assert (comp.band_gen > 0).any() and (comp.band_gen == 0).any()          # only the touched bands changed


# ------------------------------------------------------------------------------------------------------ fake server
def png_bytes(z, x, y):
    rng = np.random.default_rng((z, x, y))
    a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    im = Image.fromarray(np.kron(a, np.ones((16, 16, 1), np.uint8)))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


class FakeServer:
    """Local tile server thread.  `rules(z, x, y, n) -> (status, body, headers)` per request (n = hit count of that tile)."""

    def __init__(self, rules=None):
        self.hits = {}
        self.agents = set()
        self.total = 0
        self.lock = threading.Lock()
        self.rules = rules
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                parts = self.path.strip("/").split("/")
                z, x, y = int(parts[0]), int(parts[1]), int(parts[2].split(".")[0])
                with outer.lock:
                    outer.total += 1
                    n = outer.hits[(z, x, y)] = outer.hits.get((z, x, y), 0) + 1
                    outer.agents.add(self.headers.get("User-Agent"))
                status, body, headers = (200, png_bytes(z, x, y), {}) if outer.rules is None else outer.rules(z, x, y, n)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}" + "/{z}/{x}/{y}.png"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def make_loader(tmp_path, url, frame=None, long_px=256, **src_kw):
    frame = frame or small_frame()
    kw = dict(delay=0.0, min_interval=0.0, backoff=0.01, retries=2, timeout=3.0, cache_root=tmp_path / "tiles")
    kw.update(src_kw)
    src = bm.TileSource(url, **kw)
    comp = bm.BasemapComposite(frame, long_px)
    return bm.BasemapLoader(comp, src, threads=4), src, comp


def wait(loader, timeout=20):
    loader.join(timeout)
    assert not loader.running
    return loader.status()


@pytest.fixture()
def server():
    s = FakeServer()
    yield s
    s.close()


def test_plan_counts_and_estimate(tmp_path, server):
    loader, src, comp = make_loader(tmp_path, server.url)
    p = loader.plan()
    assert p.total == sum(len(t) for _, t in p.levels) and p.cached == 0 and p.missing == p.total
    assert [z for z, _ in p.levels] == bm.levels_for(comp.zoom)
    assert p.approx_mb == pytest.approx(p.missing * bm.APPROX_KB_PER_TILE / 1024)
    assert not p.needs_confirm
    assert server.total == 0                                                 # planning never touches the network
    big = bm.BasemapLoader(bm.BasemapComposite(BG, 6144), bm.TileSource(None, cache_root=tmp_path / "x"))
    pb = big.plan()
    assert pb.zoom == 16 and 1100 <= pb.final_tiles <= 1300 and not pb.needs_confirm
    assert pb.total == 9 + 81 + 1152


def test_download_caches_polite_and_composites(tmp_path, server):
    loader, src, comp = make_loader(tmp_path, server.url)
    plan = loader.plan()
    seen = []
    loader.progress = lambda st: seen.append(st.done)
    loader.start(download=True)
    st = wait(loader)
    assert st.phase == "done", st
    assert st.downloaded == plan.total and st.failed == 0 and st.done == plan.total
    assert server.total == plan.total                                        # every tile exactly once
    assert all(n == 1 for n in server.hits.values())
    assert server.agents == {bm.USER_AGENT}
    for z, tiles in plan.levels:                                             # disk cache layout z/x/y.png
        for x, y in tiles:
            assert (tmp_path / "tiles" / str(z) / str(x) / f"{y}.png").is_file()
    assert comp.coverage == 1.0 and comp.has_tiles
    assert seen and max(seen) == plan.total
    # footprint only: every request lies in some level's plan
    allowed = {(z, x, y) for z, t in plan.levels for x, y in t}
    assert set(server.hits) == allowed
    # second run: all from the cache, zero network
    before = server.total
    loader2 = bm.BasemapLoader(bm.BasemapComposite(small_frame(), 256), src, threads=4)
    loader2.start(download=True)
    st2 = wait(loader2)
    assert server.total == before and st2.from_cache == plan.total and st2.downloaded == 0
    assert loader2.plan().missing == 0


def test_cache_only_read_makes_no_requests(tmp_path, server):
    loader, src, comp = make_loader(tmp_path, server.url)
    loader.start(download=True)
    wait(loader)
    n = server.total
    loader3, _, comp3 = make_loader(tmp_path, server.url)
    loader3.start(download=False)
    st = wait(loader3)
    assert server.total == n and st.phase == "done" and comp3.coverage == 1.0
    # composites of both runs are identical
    assert np.array_equal(comp.rgba, comp3.rgba)


def test_offline_unavailable_keeps_procedural_fallback(tmp_path):
    loader, src, comp = make_loader(tmp_path, bm.DEFAULT_URL, network=False)
    loader.start(download=True)                                              # network disabled -> nothing fetched
    st = wait(loader)
    assert st.phase == "unavailable" and "no" in st.message
    assert comp.filled and not comp.has_tiles and tuple(comp.rgba[7, 7, :3]) == bm.FALLBACK_RGB
    assert src.requests == 0


def test_unreachable_server_aborts_after_failures(tmp_path):
    s = FakeServer()
    url = s.url
    s.close()                                                                # nothing listens any more
    loader, src, comp = make_loader(tmp_path, url, frame=BG, long_px=512, retries=0, timeout=1.0)
    loader.start(download=True)
    st = wait(loader)
    assert st.phase == "unavailable" and "network" in st.message
    assert st.failed >= 8 and src.requests < loader.plan().total             # gave up early instead of hammering
    assert comp.filled and not comp.has_tiles


def test_404_is_final_and_429_retry_after_is_honoured(tmp_path):
    frame = small_frame()
    z = bm.choose_zoom(frame, 256)
    missing = bm.tiles_for_frame(frame, z)[0]
    slow = bm.tiles_for_frame(frame, z)[-1]
    stamps = {}

    def rules(zz, x, y, n):
        if (zz, x, y) == (z, *missing):
            return 404, b"nope", {}
        if (zz, x, y) == (z, *slow):
            stamps.setdefault("t", []).append(time.time())
            if n == 1:
                return 429, b"slow down", {"Retry-After": "1"}
        return 200, png_bytes(zz, x, y), {}
    s = FakeServer(rules)
    try:
        loader, src, comp = make_loader(tmp_path, s.url)
        loader.start(download=True)
        st = wait(loader)
    finally:
        s.close()
    assert st.phase == "done" and st.absent == 1 and st.failed == 0
    assert s.hits[(z, *missing)] == 1                                        # 404: no retries
    assert s.hits[(z, *slow)] == 2                                           # 429 then OK
    assert stamps["t"][1] - stamps["t"][0] >= 0.95                           # waited the Retry-After second
    assert not src.cache_path(z, *missing).exists() and src.cache_path(z, *slow).is_file()


def test_403_stops_everything(tmp_path):
    s = FakeServer(lambda z, x, y, n: (403, b"blocked", {}))
    try:
        loader, src, comp = make_loader(tmp_path, s.url)
        loader.start(download=True)
        st = wait(loader)
    finally:
        s.close()
    assert st.phase == "unavailable" and "403" in st.message
    assert s.total <= 8                                                      # a handful of in-flight requests at most
    assert not comp.has_tiles


def test_garbage_response_is_not_cached(tmp_path):
    s = FakeServer(lambda z, x, y, n: (200, b"<html>captive portal</html>", {}))
    try:
        loader, src, comp = make_loader(tmp_path, s.url, retries=0)
        loader.start(download=True)
        wait(loader)
    finally:
        s.close()
    assert not any((tmp_path / "tiles").rglob("*.png")) if (tmp_path / "tiles").exists() else True
    assert not comp.has_tiles


def test_cancel(tmp_path):
    def rules(z, x, y, n):
        time.sleep(0.05)
        return 200, png_bytes(z, x, y), {}
    s = FakeServer(rules)
    try:
        loader, src, comp = make_loader(tmp_path, s.url, delay=0.05)
        loader.start(download=True)
        time.sleep(0.25)
        loader.cancel()
        st = wait(loader)
    finally:
        s.close()
    assert st.phase == "cancelled" and 0 < st.downloaded < loader.plan().total + st.downloaded


def test_confirmation_for_large_downloads(tmp_path, server, monkeypatch):
    loader, src, comp = make_loader(tmp_path, server.url)
    monkeypatch.setattr(bm, "CONFIRM_TILES", 5)
    p = loader.plan()
    assert p.needs_confirm
    with pytest.raises(bm.ConfirmationRequired) as e:
        loader.start(download=True)
    assert e.value.n == p.missing and server.total == 0
    monkeypatch.setattr(bm, "MAX_TILES_HARD", 5)
    with pytest.raises(ValueError):
        loader.start(download=True, confirmed=True)
    assert server.total == 0


def test_local_folder_is_read_first_and_never_written(tmp_path):
    frame = small_frame()
    z = bm.choose_zoom(frame, 256)
    local = tmp_path / "local"
    plan_tiles = [(zz, x, y) for zz in bm.levels_for(z) for x, y in bm.tiles_for_frame(frame, zz)]
    for zz, x, y in plan_tiles:
        p = local / str(zz) / str(x) / f"{y}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(png_bytes(zz, x, y))
    src = bm.TileSource(None, cache_root=tmp_path / "cache", local_dir=local)    # no URL: purely local
    comp = bm.BasemapComposite(frame, 256)
    loader = bm.BasemapLoader(comp, src)
    assert loader.plan().missing == 0
    loader.start(download=False)
    st = wait(loader)
    assert st.phase == "done" and comp.coverage == 1.0
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").rglob("*.png"))


def test_damaged_cache_file_is_dropped_and_refetched(tmp_path, server):
    loader, src, comp = make_loader(tmp_path, server.url)
    z = comp.zoom
    x, y = bm.tiles_for_frame(comp.frame, z)[0]
    p = src.cache_path(z, x, y)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a png")
    loader.start(download=False)
    st = wait(loader)
    assert st.failed >= 1 and not p.exists()                                 # forgotten, so a download can replace it
    loader2, _, comp2 = make_loader(tmp_path, server.url)
    loader2.start(download=True)
    assert wait(loader2).phase == "done" and comp2.coverage == 1.0
