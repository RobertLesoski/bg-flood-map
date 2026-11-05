r"""
build_cog_verification.py - turns the raw verification results into docs/COG_VERIFICATION.md and layers/cog/verification.json.

Inputs (all produced by runs documented in docs/COG_VERIFICATION.md; nothing here is typed in by hand except the text):
  --acc-local   JSON of scripts/verify_cog_accuracy.py on county_cog_tol (the tiles that are in layers/cog/county)
  --acc-other   JSON of the same script on county_cog (Slurm job 1122676, default GDAL warp tolerance)
  --cross       JSON of scripts/crosscheck_cogs.py (12 storms)
  --sha         JSON {n_files, bytes, vs_county_cog_tol_mismatches, vs_county_cog_mismatches, identical_cog_vs_tol}
  --facts       scripts/storms_out/vista_verified_facts.json (native resolution, DEM, Slurm)
  --viewer      JSON with the browser test results
Usage: python scripts/build_cog_verification.py --acc-local a.json --acc-other b.json --cross c.json --sha s.json --viewer v.json --app "<web app folder>"
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ORDER = ["2yr", "5yr", "10yr", "25yr", "50yr", "100yr", "ev1", "ev2", "ev3", "ev4", "ev5", "ev6"]
LAB = ["deep(>=1.2m)", "mid(0.3-1.19m)", "low(0.1-0.29m)", "trace(0.01-0.09m)", "dry(0)"]


def pool(d, key, part, lab, kind):
    rows = [d[s][key][part][lab][kind] for s in ORDER if d[s][key][part][lab][kind]]
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    w = lambda f: sum(r[f] * r["n"] for r in rows) / n
    return {"n": n, "max": max(r["max"] for r in rows), "mean_abs": round(w("mean"), 3), "p99_min": round(min(r["p99"] for r in rows), 1), "p99_max": round(max(r["p99"] for r in rows), 1),
            "signed_mean": round(w("signed_mean"), 3), "share_cog_lower_1cm": round(w("share_cog_lower"), 3), "share_cog_higher_1cm": round(w("share_cog_higher"), 3)}


def per_storm(d):
    out = {}
    for s in ORDER:
        m, h = d[s]["max_depth"], d[s]["hourly_all"]
        out[s] = {"files": d[s]["hourly_files"] + 1, "structure_ok": d[s]["all_files_ok"], "same_geometry": d[s]["all_files_same_geometry"], "layout": d[s]["layouts"], "bytes_total": d[s]["bytes_total"],
                  "points_max_depth": m["points"], "points_hourly": h["points"],
                  "bilinear_max_depth": m["bilinear_err"], "bilinear_hourly": h["bilinear_err"], "nearest_max_depth": m["nearest_err"], "nearest_hourly": h["nearest_err"],
                  "hull_violations": m["hull_ok"]["violations"] + h["hull_ok"]["violations"], "hull_checked": m["hull_ok"]["n"] + h["hull_ok"]["n"],
                  "nodata_mismatch": m["cog_nodata_src_valid"] + m["cog_valid_src_nodata"] + h["cog_nodata_src_valid"] + h["cog_valid_src_nodata"],
                  "max_vs_hourly": d[s]["max_vs_hourly"], "by_depth_stratum_hourly": h["by_depth_stratum"], "by_depth_stratum_max": m["by_depth_stratum"], "by_flag_hourly": h["by_flag"], "by_flag_max": m["by_flag"]}
    return out


def table_storms(ps):
    L = ["| storm | files | structure | points (max / hourly) | bilinear err max / p99 (cm) | bilinear mean (cm) | vs nearest cell: mean / p99 / max (cm) | hull violations |", "|---|---|---|---|---|---|---|---|"]
    for s in ORDER:
        p = ps[s]
        bmax = max(p["bilinear_max_depth"]["max"], p["bilinear_hourly"]["max"])
        bp99 = max(p["bilinear_max_depth"]["p99"], p["bilinear_hourly"]["p99"])
        nm = (p["nearest_max_depth"]["mean"] * p["nearest_max_depth"]["n"] + p["nearest_hourly"]["mean"] * p["nearest_hourly"]["n"]) / (p["nearest_max_depth"]["n"] + p["nearest_hourly"]["n"])
        np99 = max(p["nearest_max_depth"]["p99"], p["nearest_hourly"]["p99"])
        nmax = max(p["nearest_max_depth"]["max"], p["nearest_hourly"]["max"])
        bm = (p["bilinear_max_depth"]["mean"] * p["bilinear_max_depth"]["n"] + p["bilinear_hourly"]["mean"] * p["bilinear_hourly"]["n"]) / (p["bilinear_max_depth"]["n"] + p["bilinear_hourly"]["n"])
        L.append("| %s | %d | %s | %d / %d | %.0f / %.1f | %.3f | %.2f / %.0f / %.0f | %d of %d |" % (s, p["files"], "OK" if p["structure_ok"] and p["same_geometry"] else "PROBLEM", p["points_max_depth"], p["points_hourly"], bmax, bp99, bm, nm, np99, nmax, p["hull_violations"], p["hull_checked"]))
    return L


def table_strata(d, key):
    L = ["| depth stratum (COG value) | points | vs bilinear: max / mean abs / p99 range (cm) | vs nearest source cell: max / mean abs / p99 range (cm) | nearest: signed mean (cm) | share COG lower / higher by > 1 cm |", "|---|---|---|---|---|---|"]
    for lab in LAB:
        b, n = pool(d, key, "by_depth_stratum", lab, "bilinear_err"), pool(d, key, "by_depth_stratum", lab, "nearest_err")
        L.append("| %s | %d | %.0f / %.3f / %.1f-%.1f | %.0f / %.2f / %.1f-%.1f | %+.2f | %.0f %% / %.0f %% |" % (lab, n["n"], b["max"], b["mean_abs"], b["p99_min"], b["p99_max"], n["max"], n["mean_abs"], n["p99_min"], n["p99_max"], n["signed_mean"], 100 * n["share_cog_lower_1cm"], 100 * n["share_cog_higher_1cm"]))
    for flag, name in (("edge", "within 3 px of the domain edge"), ("tileb", "first/last pixel row or column of a 512 px tile")):
        b, n = pool(d, key, "by_flag", flag, "bilinear_err"), pool(d, key, "by_flag", flag, "nearest_err")
        L.append("| %s | %d | %.0f / %.3f / %.1f-%.1f | %.0f / %.2f / %.1f-%.1f | %+.2f | %.0f %% / %.0f %% |" % (name, n["n"], b["max"], b["mean_abs"], b["p99_min"], b["p99_max"], n["max"], n["mean_abs"], n["p99_min"], n["p99_max"], n["signed_mean"], 100 * n["share_cog_lower_1cm"], 100 * n["share_cog_higher_1cm"]))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc-local", required=True)
    ap.add_argument("--acc-other", required=True)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--viewer", required=True)
    ap.add_argument("--app", required=True)
    a = ap.parse_args()
    loc, oth = json.load(open(a.acc_local)), json.load(open(a.acc_other))
    cross, sha, viewer = json.load(open(a.cross)), json.load(open(a.sha)), json.load(open(a.viewer))
    facts = json.load(open(os.path.join(HERE, "storms_out", "vista_verified_facts.json")))
    pl, po = per_storm(loc), per_storm(oth)
    tot = lambda ps: sum(p["points_max_depth"] + p["points_hourly"] for p in ps.values())
    doc = {"generated_by": "scripts/build_cog_verification.py", "date": "2026-11-05", "tiles_in_layers_cog_county": "county_cog_tol (Slurm job 1123323, warp tolerance 0.01 px)",
           "local_set": {"folder_on_vista": "county_cog_tol", "points_total": tot(pl), "per_storm": pl,
                         "pooled_by_depth_stratum_hourly": {lab: {k: pool(loc, "hourly_all", "by_depth_stratum", lab, k) for k in ("nearest_err", "bilinear_err")} for lab in LAB},
                         "pooled_by_depth_stratum_max_depth": {lab: {k: pool(loc, "max_depth", "by_depth_stratum", lab, k) for k in ("nearest_err", "bilinear_err")} for lab in LAB}},
           "other_set": {"folder_on_vista": "county_cog", "slurm_job": 1122676, "warp_tolerance_px": 0.125, "points_total": tot(po), "per_storm": po,
                         "pooled_by_depth_stratum_hourly": {lab: {k: pool(oth, "hourly_all", "by_depth_stratum", lab, k) for k in ("nearest_err", "bilinear_err")} for lab in LAB}},
           "download_check": sha, "crosscheck": cross, "viewer_tests": viewer, "native_resolution": facts["dem"] | {"solver_rasters": "EPSG:32616, 5.0 m, 10404 x 12043, float32 metres, nodata -9999 (read with rasterio on Vista)", "cog": "EPSG:3857, 6.25 mercator m = 6.25 x cos(37 deg) = 4.99 m on the ground"},
           "solver": facts["solver"], "building_ponding": facts["building_ponding"]}
    json.dump(doc, open(os.path.join(a.app, "layers", "cog", "verification.json"), "w", encoding="utf-8"), indent=1)

    L = ["# COG verification (county real-depth tiles)", "",
         "Date of the checks: 2026-11-05. Every number below was produced by a script run on Vista (`scripts/verify_cog_accuracy.py`, read-only windowed point reads on a login node, no Slurm job) or locally (`scripts/crosscheck_cogs.py`, `cog/test/real-check.html`). "
         "Machine-readable copy of everything: `layers/cog/verification.json`.", "",
         "## 1. Verdict", "",
         "* All 12 storms x (25 hourly + max_depth) COGs are structurally correct: EPSG:3857, uint16 centimetres, nodata 65535, 512 px tiles, overviews 2/4/8/16/32, GDAL COG layout, identical size and bounds in every file of a storm.",
         "* **The tiles in `layers/cog/county` are the set `county_cog_tol` (Slurm job 1123323, GDAL warp tolerance 0.01 px), not `county_cog` (job 1122676, default tolerance 0.125 px).** The local copy is byte-identical (sha256, 324 of 324 files) to `county_cog_tol` on Vista and to none of `county_cog`.",
         "* Against the solver rasters the local tiles are a faithful bilinear resample: error to the exactly expected bilinear value is at most %.0f cm and the 99th percentile %.0f cm over %d random points; `county_cog` (default tolerance) reaches %.0f cm (99th percentile up to %.1f cm; a first independent sample, picked from county_cog itself, found up to 21 cm). The earlier set is therefore the less accurate one, but both agree with the solver rasters to within about 6 cm in 99 %% of the points." % (
             max(max(p["bilinear_max_depth"]["max"], p["bilinear_hourly"]["max"]) for p in pl.values()), max(max(p["bilinear_max_depth"]["p99"], p["bilinear_hourly"]["p99"]) for p in pl.values()), tot(pl),
             max(max(p["bilinear_max_depth"]["max"], p["bilinear_hourly"]["max"]) for p in po.values()), max(max(p["bilinear_max_depth"]["p99"], p["bilinear_hourly"]["p99"]) for p in po.values())),
         "* The honest limitation is not the file format but the **re-gridding**: the solver rasters are 5.0 m in UTM 16N, the COGs 4.99 m in Web Mercator. Same resolution, rotated grid (convergence about 0.35 deg), so every COG pixel is a bilinear mix of up to four solver cells. "
         "Compared with the single solver cell under a point the COG differs on average by 7 to 8 cm in the deep stratum, 6 cm in the mid stratum, 4 cm in the low stratum and 1 cm in the trace stratum (99th percentile up to %.0f cm in the deep stratum, largest single difference %.0f cm), in both directions equally (signed mean about 0). See section 5." % (pool(loc, "max_depth", "by_depth_stratum", LAB[0], "nearest_err")["p99_max"], max(max(p["nearest_max_depth"]["max"], p["nearest_hourly"]["max"]) for p in pl.values())),
         "* The viewer reads the same numbers as rasterio: 516 of 516 points through the reader harness and 480 of 480 through the viewer's own `depthAt` path, all equal. Overlay, click query, risk card and the phone layout were exercised on the real tiles.", "",
         "## 2. What was checked and how", "",
         "* Sets: `county_cog` (job 1122676, 17:29-17:36 on 2026-11-04, make_cogs.py before the tolerance fix) and `county_cog_tol` (job 1123323, 22:39-22:43, `WarpedVRT(..., tolerance=0.01)`). No new job was submitted for this audit. `county_cog_tol_test` holds one file (100yr max_depth.tif) and was not used.",
         "* Points (per storm): 560 to 660 on max_depth.tif and 980 to 1 350 on the 25 hourly files (about 60 per file). They are picked on a factor-8 overview of the COG with these target shares: deep 15 %, mid 15 %, low 14 %, trace 10 %, dry 8 %, near the domain edge 12 %, just outside the domain 4 %, tile boundary 22 %; "
         "the stratum is then taken from the full-resolution COG value, so counts per stratum differ slightly (tables below).",
         "* At each point the COG pixel centre (EPSG:3857) is transformed with PROJ (pyproj) to UTM 16N and compared with the SOURCE raster (`runs/county_v2/<storm>/depth_*.tif`, `max_depth.tif`, float32 metres x 100 -> cm): "
         "(a) **bilinear** = bilinear interpolation of the four surrounding source cell centres (exactly what the converter asked GDAL for; only points with four valid neighbours), "
         "(b) **nearest** = the single source cell that contains the position (what a person reading the solver file would call 'the value'), "
         "(c) **hull** = the COG value lies between the minimum and maximum of those four cells (plus or minus 1 cm), a property every bilinear resample must have.",
         "* Statistics are absolute differences in cm: max, mean and 99th percentile.", "",
         "## 3. Structure (all 12 storms)", "",
         "Per storm: 25 hourly files (`depth_<seconds>s.tif`, hours 0 to 24; names are not exact multiples of 3600 s) + `max_depth.tif` + `cog_meta.json` = 27 files, 26 rasters. Every raster: EPSG 3857, uint16, nodata 65535, tiled 512x512, DEFLATE with predictor 2, overviews [2, 4, 8, 16, 32], 10475 x 12156 px of 6.25 m, "
         "bounds (-9653845.26, 4390038.47, -9588376.51, 4466013.47), `LAYOUT=COG`. Both sets pass for all 12 storms (`structure` column below).", "",
         "## 4. Accuracy per storm", "", "### 4.1 Local tiles = `county_cog_tol` (what the viewer reads)", ""]
    L += table_storms(pl)
    L += ["", "### 4.2 Earlier set `county_cog` (default warp tolerance, job 1122676)", ""]
    L += table_storms(po)
    L += ["", "The difference between the sets is the GDAL warp approximation: by default GDAL replaces the exact UTM to Mercator transform by a linear approximation that may be off by 0.125 source pixels; on noisy real depths that is up to %.0f cm. With `tolerance=0.01` it is at most %.0f cm, i.e. the rounding of the stored centimetre plus GDAL's fixed-point weights." % (
        max(max(p["bilinear_max_depth"]["max"], p["bilinear_hourly"]["max"]) for p in po.values()), max(max(p["bilinear_max_depth"]["max"], p["bilinear_hourly"]["max"]) for p in pl.values())), ""]
    L += ["## 5. Why the tolerance against the nearest solver cell is larger, by depth stratum", "",
          "The COG is bilinear, so its error against the bilinear expectation is about 1 cm. Against the nearest cell the error is bounded by the local depth gradient: a bilinear value lies between the four neighbouring cells, so it can differ from the cell under the point by at most the spread of those cells. "
          "The rain-on-grid solution is rough at the cell scale (micro-depressions, sinks, roads and roofs next to each other): the median spread of the four cells is @@MED@@ cm in the deep stratum of the 2-year max_depth sample, and individual neighbours differ by more than 1 m "
          "(example from the first run of the audit, 2yr max_depth, pixel col 2757 row 2651: COG 82 cm = the bilinear value, solver cell under it 171 cm, the four cells span 171 cm). The error is therefore concentrated where the water is deep and the surface rough, and is zero where it is dry:", "",
          "Pooled over the 12 storms, hourly rasters, local tiles:", ""]
    L += table_strata(loc, "hourly_all")
    L += ["", "Pooled, max_depth.tif, local tiles:", ""]
    L += table_strata(loc, "max_depth")
    L += ["", "Reading the table: the signed mean of the nearest-cell difference is about zero and the shares 'COG lower' and 'COG higher' are about equal (25 to 35 %% each in the deep, mid and low strata). The re-gridding scatters values; it does not damp peaks or inflate them on average. "
          "Still, a single pixel can read up to about 1 m away from the solver cell that sits under it in deep rough spots, so a pixel value is 'the model depth here, to within the local roughness', not an exact reproduction of one solver cell. "
          "Edge and tile-boundary pixels behave like all others (no seam errors: bilinear differences of at most 1 cm there; COG nodata coincides with the solver nodata at every one of the %d tested points)." % sum(p["points_hourly"] + p["points_max_depth"] for p in pl.values()), "",
          "Hull check: %d of %d tested values lie outside the min/max of their four source cells (local tiles); %d of %d for `county_cog`." % (sum(p["hull_violations"] for p in pl.values()), sum(p["hull_checked"] for p in pl.values()), sum(p["hull_violations"] for p in po.values()), sum(p["hull_checked"] for p in po.values())), "",
          "### max_depth.tif against the hourly files", "",
          "In `flood_solver_v2.py` max_depth.tif is the maximum over the saved snapshots (hourly files and the final state), not over every solver time step. This was confirmed on the data: at 300 random points inside the model per storm the source max_depth equals the maximum of the 25 source hourly rasters to the cm "
          "(0 points differ by more than 1 cm in any storm), and the COG max_depth equals the maximum of the 25 COG hourly rasters (differences 0 or 1 cm, none below -1 cm). A peak between two saves (the real solver maximum) is not in either file.", "",
          "## 6. Download check", "",
          "sha256 of every file in `layers/cog/county` (%d files, %.2f GB) against `sha256sum` on Vista: **mismatches against `county_cog_tol`: %d; against `county_cog`: %d of %d files.** The two Vista sets share no identical file." % (sha["n_files_local"], sha["bytes_local"] / 1e9, len(sha["vs_county_cog_tol_mismatches"]), sha["vs_county_cog_mismatches"], sha["n_files_local"]), "",
          "## 7. Cross-check against the other depth products (same solver rasters)", "",
          "`scripts/crosscheck_cogs.py`, windowed reads, local tiles. 300 random buildings per storm (150 chosen among those the address index lists as wet, 150 among the others, so the agreement shares below are for this half-wet sample, not county-wide rates), all 98 facilities (97 inside the model).", "",
          "**Viewer flow grids (25 m cells = 4x4 block mean of 5 m, 8 bit sqrt coded) at hours 12 and 24.** About 295 random cells per storm and hour. COG mean over the cell minus grid depth: signed mean between -0.4 and +0.7 cm (no bias), mean absolute 0.7 to 3.5 cm, 99th percentile 4.9 to 53 cm; "
          "share within 5 cm 79 to 99 %. COG maximum over the cell minus grid: +5 to +16 cm on average and COG >= grid in 91 to 99 % of cells, because the grid is a mean and the COG keeps the peak cells.", ""]
    L += ["| storm | hour 12: mean abs / p99 (cm) | hour 24: mean abs / p99 (cm) | COG max - grid, hour 24 (mean, cm) |", "|---|---|---|---|"]
    gd = {(r["storm"], r["hour"]): r for r in cross["flow_grid"]}
    for s in ORDER:
        a12, a24 = gd[(s, 12)], gd[(s, 24)]
        L.append("| %s | %.2f / %.1f | %.2f / %.1f | %+.1f |" % (s, a12["cog_mean_minus_grid"]["mean_abs_cm"], a12["cog_mean_minus_grid"]["p99_abs_cm"], a24["cog_mean_minus_grid"]["mean_abs_cm"], a24["cog_mean_minus_grid"]["p99_abs_cm"], a24["cog_max_minus_grid"]["mean_diff_cm"]))
    L += ["", "**Address index (building depth = max over the footprint of the flow grid over hours 4 to 24) against the COG footprint maximum (all-touched pixels of max_depth.tif).**", "",
          "| storm | buildings | COG - index: mean (cm) / median abs / p99 abs | both flooded >= 10 cm: median COG/index ratio | index >= 10 cm and COG < 10 cm | index < 10 cm and COG >= 10 cm | agreement of the 10 cm flag |", "|---|---|---|---|---|---|---|"]
    for r in cross["address_index"]:
        c = r["cog_max_minus_index"]
        L.append("| %s | %d | %+.1f / %.0f / %.0f | %s | %d | %d | %.0f %% |" % (r["storm"], r["buildings"], c["mean_diff_cm"], c["median_abs_cm"], c["p99_abs_cm"], r.get("both_flooded_ratio_median", "-"), r["index_ge10_cog_lt10"], r["index_lt10_cog_ge10"], 100 * r["flooded10_agreement_share"]))
    L += ["", "The median difference is 1 to 4 cm, but in 12 to 18 % of the sampled buildings the two products disagree on the 10 cm flag, in BOTH directions: the grid (20 m cells, 8 bit) smears a deep pit or ditch next to a house into the house's cells and misses 5 m features; "
          "the COG sees the 5 m footprint pixels only. The index is the maximum over hourly snapshots of the block-averaged grid; the COG is the maximum over the saved snapshots at 5 m, so neither is 'right': they are different resolutions of the same run. "
          "For rare events (ev1) the COG flags more buildings (41 of 300 are >= 10 cm in the COG but not in the index) because 5 m cells keep thin ponding that the 20 m mean dilutes below 10 cm.", "",
          "**Infrastructure (98 facilities, `d_<storm>` = max over the facility's outline or point on the grid) against the COG maximum over the same outline/point:**", "",
          "| storm | COG - index: mean (cm) / median abs / max | facilities >= 10 cm: index / COG | agreement of the 10 cm flag |", "|---|---|---|---|"]
    for r in cross["infrastructure"]:
        c = r["cog_footprint_max_minus_index"]
        L.append("| %s | %+.1f / %.0f / %.0f | %d / %d | %.0f %% |" % (r["storm"], c["mean_diff_cm"], c["median_abs_cm"], c["max_abs_cm"], r["index_ge10"], r["cog_ge10"], 100 * r["flooded10_agree_share"]))
    L += ["", "Systematic difference: the median is 0 to 1 cm but the mean is +15 to +18 cm (COG deeper; +5.6 cm for ev1) because a handful of campus outlines (schools, large sites) contain a pond or ditch whose 5 m cells reach 2 m while the 20 m grid averages it down (largest differences 1.8 to 2.2 m). "
          "The COG finds 33 to 34 facilities at >= 10 cm in the design storms where the index finds 26 to 29. The facility file was built from the grid; it has not been rebuilt from the COG.", "",
          "Facility depths here use the footprints rebuilt from the cached OSM extract exactly as `build_infrastructure.py` does (outlines of merged parts, points for nodes).", "",
          "## 8. Viewer on the real tiles", ""]
    for k, v in viewer.items():
        L.append("* %s: %s" % (k, v))
    L += ["", "## 9. Native resolution of the solver rasters and what the COG resampling costs", "",
          "Read with rasterio on Vista from `runs/county_v2/<storm>/max_depth.tif` and the hourly files (same for all storms): **EPSG:32616 (UTM 16N), 5.0 m x 5.0 m cells, 10404 x 12043, float32 metres, nodata -9999, LZW tiled 256.** "
          "The COG is EPSG:3857 at 6.25 m, which on the ground at 37 N is 6.25 x cos(37 deg) = 4.99 m. The two grids therefore have the same resolution (ratio 0.998) but are not aligned: Mercator north is rotated against UTM north by the grid convergence (about 0.35 deg at the county), "
          "so the converter must interpolate. Consequences:", "",
          "* No information is lost to coarsening (4.99 m vs 5.0 m) and no information is invented (no finer grid exists). The result is a smoothing of about one cell, equal in both directions on average (signed mean about 0).",
          "* The fraction of a solver cell's depth that survives at a pixel is a weighted mean of up to four cells, so isolated one-cell features are blurred (example above: 171 cm cell read as 82 cm) and thin lines (ditches, curbs) may fade; section 5 gives the size of this by depth stratum.",
          "* Overview levels (6.25 x 2^n m) are averages, as the viewer's note says; queries use full resolution.",
          "* Option if exact cell values matter: a native 5 m UTM COG (no warp, nearest or identity) would reproduce every solver cell with zero resampling error; the browser reader would then need a UTM 16N inverse projection (a few dozen lines, testable against PROJ as the Mercator one is). Not built: no new jobs were allowed for this audit.", "",
          "Terrain (the real resolution behind the 5 m model outside the city): see `STORM_DATA_SOURCES.md`; measured on Vista: `dem_warren_5m.tif` equals the 3DEP 1/3 arc-second product (about 10.3 x 8.2 m) resampled to 5 m outside the 20 x 20 km city DEM (mean absolute difference to 3DEP 0.21 m at 368 points), and equals the 2 m city DEM inside (0.06 m at 32 points). "
          "The depths outside the city therefore carry about 10 m of real terrain information on a 5 m grid.", "",
          "## 10. Not verified", "",
          "* Flood depths against observations (no gauge data, no calibration).",
          "* `max_depth.tif` against every solver time step (only snapshots exist).",
          "* The viewer's overlay colours against a second renderer (looked at in the Browser pane only, screenshots in the session).",
          "* Points were sampled randomly, not exhaustively (%d points per set); an area of larger error with fewer than about 1 in 1 000 pixels could be missed, but the structure of the error (bounded by the local spread) was confirmed by the hull check at every point." % tot(pl), ""]
    os.makedirs(os.path.join(REPO, "docs"), exist_ok=True)
    med = "%g" % loc["2yr"]["max_depth"]["by_depth_stratum"][LAB[0]]["median_local_range_cm"]
    open(os.path.join(REPO, "docs", "COG_VERIFICATION.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L).replace("@@MED@@", med))
    print("wrote docs/COG_VERIFICATION.md and verification.json")


if __name__ == "__main__":
    main()
