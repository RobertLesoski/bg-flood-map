"""Tiny helpers shared by the data modules for the on-disk cache (`desktop/cache/*.npz`, never inside the data root).

A cache file is named  `<kind>_<frame hash>_<input hash>.npz`.  The frame hash covers the flow-grid geometry, the input
hash covers the size + mtime of every source file and a code `VERSION` string, so editing a source file or changing the
builder invalidates exactly the right files.  Writing is atomic (temp file + os.replace); a damaged file is treated as a
cache miss.  Files are stored UNCOMPRESSED so a load is a few memcpy's (np.load of an uncompressed npz).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import cache_dir


def file_sig(path) -> str:
    st = os.stat(path)
    return f"{st.st_size}-{st.st_mtime_ns}"


def frame_key(frame) -> str:
    return (f"{frame.width}x{frame.height}c{frame.cell_m!r}g{frame.cell_ground_m!r}"
            f"b{frame.xmin!r},{frame.ymin!r},{frame.xmax!r},{frame.ymax!r}")


def _h(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def cache_path(kind: str, frame, inputs: list, version: str, extra: str = "") -> Path:
    """`inputs` is a list of source file paths (their size+mtime go into the key)."""
    sig = "|".join(file_sig(p) for p in inputs) + "|" + version + "|" + extra
    return cache_dir() / f"{kind}_{_h(frame_key(frame))}_{_h(sig)}.npz"


def save_npz(path: Path, arrays: dict, meta: dict) -> None:
    """Atomic write; removes older files of the same kind+frame (different input hash)."""
    path = Path(path)
    payload = dict(arrays)
    payload["_meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **payload)           # uncompressed
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    else:
        try:
            tmp.unlink()
        except OSError:
            pass
        return
    prefix = path.name.rsplit("_", 1)[0] + "_"
    for other in path.parent.glob(prefix + "*.npz"):
        if other != path:
            try:
                other.unlink()
            except OSError:
                pass


def load_npz(path: Path):
    """Returns (arrays dict, meta dict) or None when missing / damaged."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "_meta_json"}
            meta = json.loads(bytes(z["_meta_json"]).decode("utf-8"))
        return arrays, meta
    except Exception:
        return None
