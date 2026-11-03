"""County buildings and roads from an OpenStreetMap PBF extract, cached as two compact GeoJSON files.

    sc = ensure_osm_scene(pbf_path, frame, progress)      # first call ~1 min (reads the whole PBF once), later calls instant
    sc.buildings_path, sc.roads_path                      # GeoJSON files in <cache>/osm/<key>/ - the same files the
                                                          # city loaders read (bgv.data.buildings / roads), so everything
                                                          # downstream (pixel index, statistics, meshes) is shared
    sc.info                                               # counts, bbox, seconds, osm file signature

The box is the grid's lng/lat extent (plus 0.01 degree, rounded outward to 0.01), so the demo grid, the real county grid and
a slightly different crop share one extraction.  The key covers the PBF's size + mtime, the box and the reader version: a new
Geofabrik download or a reader fix rebuilds, nothing else does.  Nothing is ever written next to the PBF.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..geo import GridFrame, merc_to_lnglat
from . import osmpbf


@dataclass
class OsmScene:
    buildings_path: Path
    roads_path: Path
    info: dict
    cached: bool = False


def frame_bbox(frame: GridFrame, pad_deg: float = 0.01) -> tuple[float, float, float, float]:
    """(west, south, east, north) of the grid in degrees, padded and rounded outward to 0.01 degree."""
    xs = (frame.xmin, frame.xmax)
    ys = (frame.ymin, frame.ymax)
    pts = [merc_to_lnglat(x, y) for x in xs for y in ys]
    w = min(p[0] for p in pts) - pad_deg
    e = max(p[0] for p in pts) + pad_deg
    s = min(p[1] for p in pts) - pad_deg
    n = max(p[1] for p in pts) + pad_deg
    r = lambda v, f: round(f(v * 100.0) / 100.0, 2)         # noqa: E731
    return r(w, math.floor), r(s, math.floor), r(e, math.ceil), r(n, math.ceil)


def _key(pbf: Path, bbox) -> str:
    st = os.stat(pbf)
    s = f"{osmpbf.VERSION}|{st.st_size}|{st.st_mtime_ns}|{bbox!r}"
    return hashlib.sha1(s.encode()).hexdigest()[:14]


def cache_folder(pbf, bbox) -> Path:
    return config.cache_dir() / "osm" / _key(Path(pbf), bbox)


def _write_geojson(path: Path, feats: list):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)


def ensure_osm_scene(pbf, frame: GridFrame, progress=None, *, bbox=None, rebuild: bool = False) -> OsmScene:
    """The cached county scene for `frame`, built from `pbf` when missing.  Raises FileNotFoundError for a missing PBF."""
    pbf = Path(pbf)
    if not pbf.is_file():
        raise FileNotFoundError(f"OpenStreetMap extract not found: {pbf}")
    bbox = tuple(bbox) if bbox is not None else frame_bbox(frame)
    folder = cache_folder(pbf, bbox)
    bp, rp, ip = folder / "buildings.geojson", folder / "roads.geojson", folder / "info.json"
    if not rebuild and bp.is_file() and rp.is_file() and ip.is_file():
        try:
            info = json.loads(ip.read_text(encoding="utf-8"))
            return OsmScene(bp, rp, info, cached=True)
        except Exception:
            pass
    t0 = time.perf_counter()
    res = osmpbf.read_pbf(pbf, bbox, progress=progress)
    n_lab = osmpbf.label_near_roads(res)
    folder.mkdir(parents=True, exist_ok=True)
    _write_geojson(bp, res.buildings)
    _write_geojson(rp, res.roads)
    info = dict(res.stats, bbox=list(bbox), labelled_near_road=n_lab, pbf=str(pbf), pbf_size=os.path.getsize(pbf),
                built_s=time.perf_counter() - t0, reader=osmpbf.VERSION)
    ip.write_text(json.dumps(info, indent=1), encoding="utf-8")
    # one extraction per PBF + box: drop the folders of older keys of the same pbf (a stale PBF version)
    try:
        for other in folder.parent.iterdir():
            if other.is_dir() and other != folder:
                try:
                    oi = json.loads((other / "info.json").read_text(encoding="utf-8"))
                except Exception:
                    oi = {}
                if oi.get("pbf") == str(pbf) and oi.get("bbox") == list(bbox):
                    shutil.rmtree(other, ignore_errors=True)
    except OSError:
        pass
    return OsmScene(bp, rp, info, cached=False)
