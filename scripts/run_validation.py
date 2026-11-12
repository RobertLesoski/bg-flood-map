"""One reproducible entry point for the ev4 validation steps.

    python scripts/run_validation.py [--model PATH_TO_ev4_max_depth.tif] [--model-scale 0.01]

What it does (every step writes to verification/derived/, never to raw inputs):
  1. Hashes every input file it uses (SHA-256) and records library versions and the seed.
  2. Runs the unit tests for the validation code (scripts/tests/test_validate_opera_ev4.py).
  3. Runs the model-vs-OPERA comparison if --model is given; otherwise records it as BLOCKED
     and names the input needed.
  4. Writes a manifest: verification/derived/run_manifest_<stamp>.json with every status and output.

Fixed seed: SEED is set for any randomized step (none currently; kept so later steps stay reproducible).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import unittest

import numpy as np

SEED = 20261112
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DERIVED = os.path.join(REPO, "verification", "derived")
OPERA_DIR = os.path.join(REPO, "verification", "opera")
BOUNDARY = os.path.join(REPO, ".claude", "worktrees", "agent-ab76444f0e4e07897", "layers",
                        "county_boundary_21227.geojson")
INPUTS = [
    os.path.join(REPO, "verification", "observed_depths.csv"),
    os.path.join(REPO, "layers", "storm_info.json"),
    os.path.join(REPO, "scripts", "storms_out", "bwg_hourly.csv"),
]


def sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
    try:
        import PIL
        out["pillow"] = PIL.__version__
    except ImportError:
        out["pillow"] = None
    try:
        import rasterio
        out["rasterio"] = rasterio.__version__
    except ImportError:
        out["rasterio"] = None
    return out


def run_unit_tests() -> dict:
    import importlib.util
    path = os.path.join(REPO, "scripts", "tests", "test_validate_opera_ev4.py")
    spec = importlib.util.spec_from_file_location("test_validate_opera_ev4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    suite = unittest.TestLoader().loadTestsFromModule(mod)
    result = unittest.TestResult()
    suite.run(result)
    return {"status": "passed" if result.wasSuccessful() else "failed",
            "tests_run": result.testsRun,
            "failures": [str(t) for t, _ in result.failures + result.errors]}


def run_opera_comparison(model: str, scale: float, nodata: float) -> dict:
    cmd = [sys.executable, os.path.join(HERE, "validate_opera_ev4.py"),
           "--opera-dir", OPERA_DIR, "--boundary", BOUNDARY, "--model", model,
           "--model-scale", str(scale), "--model-nodata", str(nodata), "--out", DERIVED]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"status": "computed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="ev4 max-depth GeoTIFF (blocked if omitted)")
    ap.add_argument("--model-scale", type=float, default=0.01)
    ap.add_argument("--model-nodata", type=float, default=65535)
    args = ap.parse_args(argv)

    np.random.seed(SEED)
    os.makedirs(DERIVED, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "versions": versions(),
        "inputs": {p: sha256(p) for p in INPUTS + [BOUNDARY]},
        "opera_tiles": {os.path.basename(p): sha256(p) for p in
                        sorted(os.path.join(OPERA_DIR, f) for f in os.listdir(OPERA_DIR)
                               if f.lower().endswith(".tif"))} if os.path.isdir(OPERA_DIR) else {},
        "steps": {},
    }
    manifest["steps"]["unit_tests"] = run_unit_tests()
    if args.model and os.path.exists(args.model):
        manifest["steps"]["opera_vs_model"] = run_opera_comparison(args.model, args.model_scale, args.model_nodata)
    else:
        manifest["steps"]["opera_vs_model"] = {
            "status": "blocked",
            "needed": "ev4 max-depth raster: layers/cog/county/ev4/max_depth.tif or runs/county_v2/ev4/max_depth.tif "
                      "(on Vista, /scratch/11931/rlesoski/bgflood/runs/county_v2/ev4/max_depth.tif)",
        }
    manifest["steps"]["road_validation_bootstrap"] = {
        "status": "blocked",
        "needed": "per-road observed/control table from docs/VALIDATION_EV4.md (layers/validation/ev4_observations.json "
                  "and ev4 depth raster) - not present in this repository",
    }
    manifest["steps"]["sensitivity_sweep"] = {
        "status": "blocked",
        "needed": "ev4 model inputs (DEM, roughness, rain files) and a runnable model; design in verification/SENSITIVITY_DESIGN.md",
    }
    path = os.path.join(DERIVED, f"run_manifest_{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("manifest:", path)
    for k, v_ in manifest["steps"].items():
        print(f"  {k}: {v_['status']}")
    return 0 if manifest["steps"]["unit_tests"]["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
