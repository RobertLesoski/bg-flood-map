#!/usr/bin/env python3
"""
DSOC Flood Map - installation self-check (standard library only, Python 3.8+).

Checks the folder this file lives in (or --root DIR): Python version, key files and their sizes, map layers,
the hourly flood GeoJSON layers (first and last bytes of every file, so a cut-off download shows), and the fluid flow grids in layers/flow/<storm>/ against FLOW_FORMAT.md
(meta.json fields, every hNN.rgba.gz and flowpaths.rgba.gz present, non-empty, valid gzip header, and the gzip
trailer size equal to width*height*4).

  python selfcheck.py              readable table (OK / WARN / FAIL), exit code 1 if anything FAILs
  python selfcheck.py --brief      only the problems and the summary
  python selfcheck.py --summary    just the summary line when nothing FAILs, the --brief table when something does (the launcher)
  python selfcheck.py --deep       also decompress every flow grid and verify its CRC (slower, catches corrupt downloads)
  python selfcheck.py --json       machine-readable result on stdout
  python selfcheck.py --strict     exit code 1 on WARN as well
"""

import argparse
import gzip
import json
import os
import platform
import re
import shutil
import struct
import sys
import textwrap
import time
import zlib

MIN_PYTHON = (3, 8)
STORMS = ["5yr", "10yr", "25yr", "50yr", "100yr"]      # expected flow storms ("2yr" is optional)
MAX_GRID = 6000
MIN_GZIP = 18                                         # smallest possible gzip file (10 header + 8 trailer)

# (path relative to the folder, minimum plausible size in bytes, what happens without it)
CORE_FILES = [
    ("serve.py", 8 * 1024, "FAIL", "the local web server"),
    ("viewer.html", 20 * 1024, "FAIL", "the 3D map itself"),
    ("flood_report.html", 10 * 1024, "WARN", "the flood probability report"),
    ("flood_methodology.txt", 1024, "WARN", "the report's methodology download"),
    ("dsoc_logo.png", 1024, "WARN", "the logo in the page header"),
]
# GeoJSON layers the viewer / report load at start-up. Without them features silently go missing.
MAP_LAYERS = [
    ("fema", 100 * 1024, "FEMA flood zones"),
    ("sinkholes", 200 * 1024, "karst sinkholes"),
    ("riverine", 4 * 1024, "riverine flood zones"),
    ("roads", 50 * 1024, "road network, used by viewer and report"),
    ("buildings", 1024 * 1024, "building footprints, used by viewer and report"),
    ("all_parcels", 2 * 1024 * 1024, "all parcels"),
]
DEFAULT_HPC = {"2yr": 11, "v2_5yr": 24, "v2_10yr": 24, "v2_25yr": 24, "v2_50yr": 24, "v2_100yr": 24}

REQUIRED_META = ["format", "version", "storm", "label", "source", "source_note", "velocity_method", "crs", "width",
                 "height", "cell_m", "cell_ground_m", "bounds_merc", "bounds_lnglat", "depth_encoding",
                 "vel_encoding", "wet_threshold_m", "qmax_m2s", "hours", "built"]
OPTIONAL_META = ["storm_depth_in", "hourly", "run_summary"]

FLOW_HINT = "flow data not built yet: see RUN_LOCALLY.md"


class Row:
    def __init__(self, name, status, detail, items=None):
        self.name, self.status, self.detail, self.items = name, status, detail, items or []

    def as_dict(self):
        return {"name": self.name, "status": self.status, "detail": self.detail, "items": self.items}


class Report:
    def __init__(self):
        self.rows = []
        self.next_steps = []

    def add(self, name, status, detail, items=None):
        self.rows.append(Row(name, status, detail, items))

    def step(self, text):
        if text not in self.next_steps:
            self.next_steps.append(text)

    def count(self, status):
        return sum(1 for r in self.rows if r.status == status)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0


def worst(statuses):
    return "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "OK")


def join_problems(problems, limit=3):
    msgs = [m for _, m in problems]
    if len(msgs) <= limit:
        return "; ".join(msgs)
    return "; ".join(msgs[:limit]) + "; +%d more (use --json or -v for all)" % (len(msgs) - limit)


# --------------------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------------------

def check_python(rep):
    v = sys.version_info
    ver = "%d.%d.%d" % (v[0], v[1], v[2])
    if v[:2] < MIN_PYTHON:
        rep.add("Python", "FAIL", "%s is too old, need %d.%d or newer (%s)" % (ver, MIN_PYTHON[0], MIN_PYTHON[1], sys.executable))
        rep.step("Install Python 3.8 or newer from https://www.python.org/downloads/ (tick 'Add python.exe to PATH').")
    else:
        rep.add("Python", "OK", "%s (%s)" % (ver, sys.executable))


def check_folder(rep, root):
    if not os.path.isdir(root):
        rep.add("Folder", "FAIL", "not a folder: %s" % root)
        return False
    probe = os.path.join(root, ".selfcheck_write_test")
    try:
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        rep.add("Folder", "OK", root)
    except OSError as e:
        rep.add("Folder", "WARN", "%s is read-only (%s); the server will work but cannot record its port" % (root, e))
    return True


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def check_core_files(rep, root):
    for rel, minsize, missing_status, what in CORE_FILES:
        size = file_size(os.path.join(root, rel))
        if size is None:
            rep.add(rel, missing_status, "missing (%s)" % what)
            if missing_status == "FAIL":
                rep.step("%s is missing: the folder looks incomplete. Re-copy the whole map folder." % rel)
        elif size < minsize:
            rep.add(rel, missing_status, "only %s, expected at least %s (damaged or cut off?)" % (human(size), human(minsize)))
        else:
            rep.add(rel, "OK", human(size))


def geojson_shape_problem(path, features_last=False):
    """Cheap look at the first and last bytes: catches empty files and downloads that were cut off.
    features_last: the file was written by our own tools, whose "features" array is the last member, so it must close
    with "]}" (white space ignored). That also catches a cut that happens to fall right after a brace."""
    try:
        with open(path, "rb") as f:
            head = f.read(2048)
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64))
            tail = f.read().rstrip()
    except OSError as e:
        return "cannot read (%s)" % e
    if not head.lstrip().startswith(b"{"):
        return "does not start like JSON"
    if b"FeatureCollection" not in head and b"features" not in head:
        return "does not look like a GeoJSON FeatureCollection"
    if not tail.endswith(b"}"):
        return "ends abruptly (download cut off?)"
    if features_last and not b"".join(tail.split()).endswith(b"]}"):
        return "ends abruptly (download cut off?)"
    return None


def check_layers(rep, root):
    bad = []
    parts = []
    for name, minsize, what in MAP_LAYERS:
        path = os.path.join(root, "layers", name + ".geojson")
        size = file_size(path)
        if size is None:
            bad.append(("FAIL", "layers/%s.geojson missing (%s)" % (name, what)))
            continue
        if size < minsize:
            bad.append(("FAIL", "layers/%s.geojson is only %s, expected at least %s" % (name, human(size), human(minsize))))
            continue
        problem = geojson_shape_problem(path)
        if problem:
            bad.append(("FAIL", "layers/%s.geojson %s" % (name, problem)))
            continue
        parts.append("%s %s" % (name, human(size)))
    if bad:
        rep.add("Map layers", "FAIL", join_problems(bad), [m for _, m in bad])
        rep.step("Rebuild the missing map layers with the scripts listed in README.md (section Data Setup), or copy layers/ from a working install.")
    else:
        rep.add("Map layers", "OK", "%d of %d present: %s" % (len(parts), len(MAP_LAYERS), ", ".join(parts)))
    size = file_size(os.path.join(root, "dem_bg.tif"))
    if size is None:
        rep.add("dem_bg.tif", "WARN", "missing (only needed to rebuild flow data, not to view the map)")
    elif size < 10 * 1024 * 1024:
        rep.add("dem_bg.tif", "WARN", "only %s, expected tens of MB (damaged?)" % human(size))
    else:
        with open(os.path.join(root, "dem_bg.tif"), "rb") as f:
            magic = f.read(4)
        if magic[:2] in (b"II", b"MM") and magic[2:4] in (b"*\x00", b"\x00*", b"+\x00", b"\x00+"):
            rep.add("dem_bg.tif", "OK", "%s, valid TIFF header" % human(size))
        else:
            rep.add("dem_bg.tif", "FAIL", "%s but not a TIFF file" % human(size))


def hpc_scenarios(root):
    """Scenario names and hour counts as the viewer declares them; falls back to the defaults."""
    try:
        with open(os.path.join(root, "viewer.html"), "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        found = re.findall(r"'([A-Za-z0-9_]+)'\s*:\s*\{\s*label:\s*'[^']*',\s*badge:\s*'[^']*',\s*depth:\s*[\d.]+,\s*hours:\s*(\d+)", text)
        if found:
            return {n: int(h) for n, h in found}
    except OSError:
        pass
    return dict(DEFAULT_HPC)


def check_hpc(rep, root):
    base = os.path.join(root, "layers", "flood_hpc")
    scen = hpc_scenarios(root)
    problems, parts = [], []
    for name, hours in scen.items():
        d = os.path.join(base, name)
        have, empty, broken = 0, [], []
        for h in range(1, hours + 1):
            p = os.path.join(d, "flood_h%02d.geojson" % h)
            size = file_size(p)
            if size is None:
                continue
            if size == 0:
                empty.append(h)
                continue
            # these 7-115 MB files are written in one go: a half-written one or a saved error page is not a usable hour
            why = geojson_shape_problem(p, features_last=True)
            if why:
                broken.append("flood_h%02d.geojson %s" % (h, why))
            else:
                have += 1
        if empty:
            problems.append(("FAIL", "%s: zero-length hour file(s) %s" % (name, ",".join("h%02d" % h for h in empty))))
        if broken:
            shown = "; ".join(broken[:3]) + ("; +%d more" % (len(broken) - 3) if len(broken) > 3 else "")
            problems.append(("FAIL", "%s: %s" % (name, shown)))
        if have == 0 and not (empty or broken):
            problems.append(("WARN", "%s: not built" % name))
        elif have < hours and not (empty or broken):
            problems.append(("WARN", "%s: %d of %d hours" % (name, have, hours)))
        elif have:
            parts.append("%s %dh" % (name, have))
    if not problems:
        rep.add("Hourly flood layers", "OK", ", ".join(parts))
        return
    status = worst([s for s, _ in problems])
    detail = join_problems(problems, limit=6)
    if parts:
        detail += " | ok: " + ", ".join(parts)
    rep.add("Hourly flood layers", status, detail, [m for _, m in problems])
    rep.step("Hourly GeoJSON flood layers (the flat, non-fluid view): run  python scripts/reproject_hpc.py v2_all  from the repo folder.")


# ---- flow grids ----

def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def meta_problems(meta, storm):
    p = []
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        p.append(("FAIL", "meta.json missing field(s): " + ", ".join(missing)))
    for k in OPTIONAL_META:
        if k not in meta and k != "run_summary":
            p.append(("WARN", "meta.json has no '%s'" % k))
    if missing:
        return p, None, None
    if meta["format"] != "bgflow" or meta["version"] != 1:
        p.append(("FAIL", "meta.json format/version is %r/%r, expected 'bgflow'/1" % (meta["format"], meta["version"])))
    if meta["storm"] != storm:
        p.append(("FAIL", "meta.json storm is %r but the folder is %r" % (meta["storm"], storm)))
    if meta["source"] not in ("rasters", "tiers"):
        p.append(("FAIL", "meta.json source %r is not 'rasters' or 'tiers'" % (meta["source"],)))
    if meta["crs"] != "EPSG:3857":
        p.append(("FAIL", "meta.json crs is %r, expected EPSG:3857" % (meta["crs"],)))
    w, h = meta["width"], meta["height"]
    if not (isinstance(w, int) and isinstance(h, int) and 0 < w <= MAX_GRID and 0 < h <= MAX_GRID):
        p.append(("FAIL", "grid size %r x %r is not a whole number up to %d" % (w, h, MAX_GRID)))
        w = h = None
    if not (num(meta["cell_m"]) and meta["cell_m"] > 0):
        p.append(("FAIL", "cell_m must be a positive number"))
    bm = meta["bounds_merc"]
    if not (isinstance(bm, list) and len(bm) == 4 and all(num(x) for x in bm) and bm[2] > bm[0] and bm[3] > bm[1]):
        p.append(("FAIL", "bounds_merc must be [xmin, ymin, xmax, ymax] with max > min"))
    elif w and num(meta["cell_m"]) and meta["cell_m"] > 0:
        if abs((bm[2] - bm[0]) / meta["cell_m"] - w) > 1 or abs((bm[3] - bm[1]) / meta["cell_m"] - h) > 1:
            p.append(("FAIL", "bounds_merc / cell_m does not match width x height"))
    bl = meta["bounds_lnglat"]
    if not (isinstance(bl, list) and len(bl) == 4 and all(num(x) for x in bl) and -180 <= bl[0] < bl[2] <= 180 and -90 <= bl[1] < bl[3] <= 90):
        p.append(("FAIL", "bounds_lnglat must be [west, south, east, north] in degrees"))
    elif not (-88.5 < bl[0] and bl[2] < -84.5 and 36.0 < bl[1] and bl[3] < 38.0):
        p.append(("WARN", "bounds_lnglat is far from Bowling Green, KY"))
    de, ve = meta["depth_encoding"], meta["vel_encoding"]
    if not (isinstance(de, dict) and de.get("type") == "sqrt" and de.get("max_m") == 8.0):
        p.append(("FAIL", "depth_encoding must be sqrt with max_m 8.0 (the viewer decodes with those constants)"))
    if not (isinstance(ve, dict) and ve.get("vmax_ms") == 4.0 and ve.get("zero_code") == 128):
        p.append(("FAIL", "vel_encoding must have vmax_ms 4.0 and zero_code 128"))
    if meta["wet_threshold_m"] != 0.02 or meta["qmax_m2s"] != 2.0:
        p.append(("FAIL", "wet_threshold_m must be 0.02 and qmax_m2s 2.0"))
    hours = meta["hours"]
    if not (isinstance(hours, list) and all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 24 for x in hours)
            and hours == sorted(set(hours))):
        p.append(("FAIL", "hours must be an ascending list of whole hours between 1 and 24"))
        hours = None
    elif not hours:
        p.append(("WARN", "meta.hours is empty (nothing wet in this storm)"))
    if hours and isinstance(meta.get("hourly"), list) and len(meta["hourly"]) != len(hours):
        p.append(("WARN", "meta.hourly has %d entries for %d hours" % (len(meta["hourly"]), len(hours))))
    return p, (w * h * 4 if w and h else None), hours


def gzip_problem(path, expect_isize, deep):
    try:
        size = os.path.getsize(path)
    except OSError:
        return "missing"
    if size == 0:
        return "zero-length file"
    if size < MIN_GZIP:
        return "only %d bytes, too small to be a gzip file" % size
    try:
        with open(path, "rb") as f:
            head = f.read(3)
            f.seek(-4, 2)
            isize = struct.unpack("<I", f.read(4))[0]
    except OSError as e:
        return "cannot read (%s)" % e
    if head != b"\x1f\x8b\x08":
        return "not a gzip file (bad header)"
    if expect_isize is not None and isize != expect_isize % (1 << 32):
        return "gzip trailer says %d bytes, expected %d (width*height*4): cut off or wrong grid size" % (isize, expect_isize)
    if deep:
        total = 0
        try:
            with gzip.open(path, "rb") as g:
                while True:
                    chunk = g.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
        except (OSError, EOFError, zlib.error) as e:
            return "gzip data is corrupt (%s)" % e
        if expect_isize is not None and total != expect_isize:
            return "decompressed to %d bytes, expected %d" % (total, expect_isize)
    return None


def check_storm(rep, sdir, storm, deep):
    name = "flow/" + storm
    problems = []
    metap = os.path.join(sdir, "meta.json")
    msize = file_size(metap)
    if msize is None:
        rep.add(name, "FAIL", "meta.json missing (build did not finish?)")
        rep.step("Rebuild layers/flow/%s: see RUN_LOCALLY.md." % storm)
        return
    if msize == 0:
        rep.add(name, "FAIL", "meta.json is zero-length")
        rep.step("Rebuild layers/flow/%s: see RUN_LOCALLY.md." % storm)
        return
    try:
        with open(metap, "r", encoding="utf-8-sig") as f:       # tolerate a BOM; the browser's json() does too
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError("top level is not an object")
    except (OSError, ValueError) as e:
        rep.add(name, "FAIL", "meta.json does not parse: %s" % e)
        rep.step("Rebuild layers/flow/%s: see RUN_LOCALLY.md." % storm)
        return
    problems, isize, hours = meta_problems(meta, storm)
    present = {}
    try:
        names = os.listdir(sdir)
    except OSError as e:
        rep.add(name, "FAIL", "cannot read folder: %s" % e)
        return
    total = msize
    for n in names:
        p = os.path.join(sdir, n)
        if not os.path.isfile(p):
            continue
        size = os.path.getsize(p)
        total += size if n != "meta.json" else 0
        m = re.match(r"^h(\d\d)\.rgba\.gz$", n)
        if m:
            present[int(m.group(1))] = p
        if size == 0 and n != "meta.json":
            problems.append(("FAIL", "%s is zero-length" % n))
    if hours is not None:
        for hr in hours:
            p = present.get(hr)
            if p is None:
                problems.append(("FAIL", "h%02d.rgba.gz is listed in meta.hours but missing" % hr))
                continue
            if os.path.getsize(p) == 0:
                continue                                       # already reported as zero-length
            g = gzip_problem(p, isize, deep)
            if g:
                problems.append(("FAIL", "h%02d.rgba.gz: %s" % (hr, g)))
        extra = sorted(set(present) - set(hours))
        if extra:
            problems.append(("WARN", "unlisted hour file(s) ignored by the viewer: " + ", ".join("h%02d" % x for x in extra)))
    fp = os.path.join(sdir, "flowpaths.rgba.gz")
    if not os.path.isfile(fp):
        problems.append(("FAIL", "flowpaths.rgba.gz missing"))
    elif os.path.getsize(fp) > 0:
        g = gzip_problem(fp, isize, deep)
        if g:
            problems.append(("FAIL", "flowpaths.rgba.gz: %s" % g))
    status = worst([s for s, _ in problems])
    if status == "FAIL":
        rep.step("Rebuild or re-fetch layers/flow/%s: see RUN_LOCALLY.md." % storm)
    if status == "OK" or (problems and status == "WARN"):
        w, h = meta.get("width"), meta.get("height")
        hs = hours or []
        span = ("h%02d-h%02d" % (hs[0], hs[-1])) if hs else "none"
        info = "%s, %d hours (%s), %sx%s @ %s m, %s%s" % (
            meta.get("source"), len(hs), span, w, h, meta.get("cell_m"), human(total), ", CRC verified" if deep else "")
        if problems:
            info += " | " + join_problems(problems)
        rep.add(name, status, info, [m for _, m in problems])
    else:
        rep.add(name, status, join_problems(problems), [m for _, m in problems])


def check_flow(rep, root, deep):
    base = os.path.join(root, "layers", "flow")
    try:
        dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
                      key=lambda s: (int(re.match(r"\d+", s).group()) if re.match(r"\d+", s) else 10 ** 6, s))
    except OSError:
        dirs = []
    storms = [d for d in dirs if re.match(r"^\d+yr$", d)]
    for d in dirs:
        if d not in storms:
            rep.add("flow/" + d, "WARN", "not a storm folder (names look like 5yr, 100yr); the viewer ignores it, probably a leftover of an unfinished build")
    if not storms:
        rep.add("Flow data", "WARN", FLOW_HINT + " (the map works without it; the fluid view and flow vectors need it)")
        rep.step(FLOW_HINT[0].upper() + FLOW_HINT[1:] + " (RUN_LOCALLY.md is in the code folder, next to scripts/).")
        return
    for d in storms:
        check_storm(rep, os.path.join(base, d), d, deep)
    missing = [s for s in STORMS if s not in storms]
    for s in missing:
        rep.add("flow/" + s, "WARN", "not built yet")
    if missing:
        rep.step("Flow data for %s is not built yet: see RUN_LOCALLY.md." % ", ".join(missing))


# ---- references from the HTML pages ----

# Files the page cannot work without (scripts, styles, images, modules) ...
HARD_REFS = [
    re.compile(r"""<script[^>]+\bsrc\s*=\s*["']([^"']+)["']""", re.I),
    re.compile(r"""<link[^>]+\bhref\s*=\s*["']([^"']+)["']""", re.I),
    re.compile(r"""<img[^>]+\bsrc\s*=\s*["']([^"']+)["']""", re.I),
    re.compile(r"""\bfrom\s+["'](\.{0,2}/[^"']+)["']"""),
    re.compile(r"""\bimport\s+["'](\.{0,2}/[^"']+)["']"""),            # side-effect import: import './x.js'
    re.compile(r"""\bimport\(\s*["'](\.{0,2}/[^"']+)["']\s*\)"""),
]
# ... and data the page fetches with a fixed name and copes with losing (it shows a message or skips the feature)
SOFT_REFS = [re.compile(r"""\bfetch\(\s*["'](\./[^"']+)["']""")]


def local_refs(text, patterns):
    refs = set()
    for pat in patterns:
        for m in pat.findall(text):
            ref = m.split("#", 1)[0].split("?", 1)[0]
            if not ref or re.match(r"^([a-z][a-z0-9+.-]*:|//|#)", ref, re.I) or "{" in ref:
                continue
            refs.add(ref)
    return refs


def check_references(rep, root):
    for page in ("viewer.html", "flood_report.html"):
        path = os.path.join(root, page)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        gone = lambda r: not os.path.exists(os.path.normpath(os.path.join(root, *r.lstrip("./").split("/"))))
        hard = local_refs(text, HARD_REFS)
        soft = local_refs(text, SOFT_REFS) - hard
        hard_missing = sorted(r for r in hard if gone(r))
        soft_missing = sorted(r for r in soft if gone(r))
        label = page + " links"
        problems = []
        if hard_missing:
            problems.append(("FAIL", "needed file(s) missing: " + ", ".join(hard_missing[:6])))
            rep.step("%s needs file(s) that are not in this folder (%s). Re-copy the whole map folder." % (page, ", ".join(hard_missing[:4])))
        if soft_missing:
            problems.append(("WARN", "optional data it fetches is missing: " + ", ".join(soft_missing[:6])))
        if problems:
            rep.add(label, worst([s for s, _ in problems]), "; ".join(m for _, m in problems), hard_missing + soft_missing)
        else:
            n = len(hard) + len(soft)
            rep.add(label, "OK", "all %d local file%s it loads exist" % (n, "" if n == 1 else "s"))


# --------------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------------

def summary_line(rep):
    return "Summary: %d OK, %d WARN, %d FAIL" % (rep.count("OK"), rep.count("WARN"), rep.count("FAIL"))


def print_table(rep, brief, root, verbose):
    out = sys.stdout
    cols = shutil.get_terminal_size((110, 24)).columns
    namew = min(30, max(12, max(len(r.name) for r in rep.rows)))
    detw = max(40, cols - namew - 9)
    out.write("DSOC Flood Map - installation self-check\n")
    out.write("Folder: %s\n\n" % rep.root)
    out.write(" %-6s %-*s %s\n" % ("STATUS", namew, "CHECK", "DETAIL"))
    out.write(" %s %s %s\n" % ("-" * 6, "-" * namew, "-" * min(detw, 60)))
    for r in rep.rows:
        if brief and r.status == "OK":
            continue
        lines = textwrap.wrap(r.detail, detw) or [""]
        out.write(" %-6s %-*s %s\n" % (r.status, namew, r.name[:namew], lines[0]))
        for ln in lines[1:]:
            out.write(" %-6s %-*s %s\n" % ("", namew, "", ln))
        if verbose and r.items:
            for it in r.items:
                for k, ln in enumerate(textwrap.wrap(it, detw - 4) or [""]):
                    out.write(" %-6s %-*s   %s%s\n" % ("", namew, "", "- " if k == 0 else "  ", ln))
    out.write("\n")
    out.write(summary_line(rep) + "\n")
    if rep.next_steps:
        out.write("\nWhat to do next:\n")
        for s in rep.next_steps:
            for k, ln in enumerate(textwrap.wrap(s, max(40, cols - 6)) or [""]):
                out.write("  %s %s\n" % ("-" if k == 0 else " ", ln))
    out.flush()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Check a local DSOC Flood Map install.")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)), help="folder to check (default: the folder of this file)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--brief", action="store_true", help="print only WARN/FAIL rows and the summary")
    ap.add_argument("--summary", action="store_true", help="print only the summary line when nothing FAILs, otherwise as --brief (what the launcher shows)")
    ap.add_argument("--deep", action="store_true", help="decompress every flow grid and verify its CRC (slower)")
    ap.add_argument("--strict", action="store_true", help="exit with 1 on WARN as well as FAIL")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every individual problem under its row")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    rep = Report()
    rep.root = root
    t0 = time.time()
    check_python(rep)
    if check_folder(rep, root):
        check_core_files(rep, root)
        check_layers(rep, root)
        check_hpc(rep, root)
        check_flow(rep, root, args.deep)
        check_references(rep, root)
    fail, warn = rep.count("FAIL"), rep.count("WARN")
    if args.json:
        json.dump({
            "ok": fail == 0 and not (args.strict and warn),
            "root": root,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "seconds": round(time.time() - t0, 2),
            "counts": {"OK": rep.count("OK"), "WARN": warn, "FAIL": fail},
            "checks": [r.as_dict() for r in rep.rows],
            "next_steps": rep.next_steps,
        }, sys.stdout, indent=1)
        sys.stdout.write("\n")
    elif args.summary and not fail:
        sys.stdout.write(summary_line(rep) + "\n")
    else:
        print_table(rep, args.brief or args.summary, root, args.verbose)
    return 1 if (fail or (args.strict and warn)) else 0


if __name__ == "__main__":
    sys.exit(main())
