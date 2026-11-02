"""Paths and persisted settings. Import `data_root()` / `cache_dir()`; never hard-code paths elsewhere."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

DEFAULT_DATA_ROOT = r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map"
_override: str | None = None


def set_data_root(path: str | None) -> None:
    global _override
    _override = path


def _has_flow(p) -> bool:
    try:
        return bool(p) and (Path(p) / "layers" / "flow").is_dir()
    except OSError:
        return False


def data_root() -> Path:
    """Resolution order: --data (an explicit choice: wrong -> error, never a silent fallback), BGV_DATA, the default
    folder, the deployed copy (`<data root>/desktop`)."""
    if _override:
        if _has_flow(_override):
            return Path(_override)
        raise FileNotFoundError(f"--data {_override!r}: no layers/flow folder in there")
    for cand in (os.environ.get("BGV_DATA"), DEFAULT_DATA_ROOT):
        if _has_flow(cand):
            return Path(cand)
    # deployed copy lives in <data root>\desktop
    here = Path(__file__).resolve().parents[2]
    if (here.parent / "layers" / "flow").is_dir():
        return here.parent
    raise FileNotFoundError("flow data not found: pass --data <folder containing layers/flow> or set BGV_DATA")


def source_root() -> Path:
    """The folder that contains the `bgv` package (desktop/)."""
    return Path(__file__).resolve().parents[1]


_cache_dir: Path | None = None


def _writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / f".w{os.getpid()}"
        probe.write_bytes(b"x")
        probe.unlink()
        return True
    except OSError:
        return False


def cache_dir() -> Path:
    """<source>/cache, or BGV_CACHE; when that folder cannot be written (read-only install, locked folder) a folder in the
    system temp directory is used instead.  Decided once per process."""
    global _cache_dir
    if _cache_dir is not None and _cache_dir.is_dir():
        return _cache_dir
    first = Path(os.environ["BGV_CACHE"]) if os.environ.get("BGV_CACHE") else source_root() / "cache"
    for p in (first, Path(tempfile.gettempdir()) / "bgv_cache"):
        if _writable(p):
            _cache_dir = p
            if p != first:
                print(f"[bgv] cache folder {first} is not writable; using {p}", flush=True)
            return p
    raise OSError(f"no writable cache folder (tried {first} and the temp directory)")


def reset_cache_dir():
    global _cache_dir
    _cache_dir = None


def settings_path() -> Path:
    return source_root() / "user_settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(d: dict) -> None:
    try:
        tmp = settings_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        os.replace(tmp, settings_path())
    except Exception:
        pass