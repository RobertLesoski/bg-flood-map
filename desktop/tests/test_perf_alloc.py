"""Load-robust perf regressions: count bytes / allocations, never time."""
import tracemalloc

import numpy as np

from bgv.data.cog import TileCache


def test_tile_cache_stays_within_its_byte_budget_after_many_storm_switches():
    c = TileCache(max_bytes=8 * 1024 * 1024)
    for storm in range(100):                          # 100 storm switches x 24 hours of 256x256 float32 tiles
        for h in range(24):
            c.put((storm, h), np.zeros((256, 256), np.float32))
            assert c.bytes <= c.max_bytes
    assert c.stats()["tiles"] == 8 * 1024 * 1024 // (256 * 256 * 4)


def test_compare_grid_conversion_allocates_one_float_grid():
    from bgv.analysis.compare import DepthGridProvider

    class Src:
        def source_key(self, s):
            return s

    p = DepthGridProvider(Src(), frame=None)
    cm = np.full((512, 512), 150, np.uint16)
    p._lru[p.key("a", 3)] = cm
    p.grid("a", 3)                                    # warm
    tracemalloc.start()
    out = p.grid("a", 3)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert out.dtype == np.float32 and abs(float(out[0, 0]) - 1.5) < 1e-6
    assert peak < 1.5 * out.nbytes                    # was 2x (astype + multiply)
