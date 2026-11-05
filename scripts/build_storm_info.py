r"""
build_storm_info.py - writes STORM_DATA_SOURCES.md (repo root) and layers/storm_info.json (the viewer's "About this storm" panel) from FACTS ONLY:

  scripts/storms_out/storms.json          historical events: UTC window, hourly gauge depths (KBWG ASOS via IEM), gaps       (made by build_historical_storms.py)
  scripts/storms_out/vista_run_facts.json run_summary.json of every county / city run + stats of the rain files + Slurm elapsed (copied from Vista, 2026-11-04)
  <app>/layers/flow_county/<storm>/meta.json, <app>/layers/flow/<storm>/meta.json   converted grids (hours, cell size, source, hourly max depth)
  scripts/storms_out/vista_verified_facts.json  Slurm job/task times, solver md5, DEM checks, COG accuracy (all read from Vista on 2026-11-05)
  <app>/layers/cog/county/<storm>/cog_meta.json                                       real-depth COGs (when downloaded)
  design-storm constants copied from scripts/run_5_storms.sh / run_county_all.sh (NOAA Atlas 14 24 h depths, hyetograph curve)

Anything the repo or the Vista files cannot show is written as "not verified". Usage: python scripts/build_storm_info.py --app "<web app folder>"
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DESIGN = {"2yr": 3.5, "5yr": 4.2, "10yr": 4.9, "25yr": 5.9, "50yr": 6.7, "100yr": 7.6}      # NOAA Atlas 14, 24 h, inches (run_county_all.sh)
TF = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
PF = [0, .04, .10, .19, .31, .50, .70, .79, .86, .92, 1.0]
ORDER = ["2yr", "5yr", "10yr", "25yr", "50yr", "100yr", "ev1", "ev2", "ev3", "ev4", "ev5", "ev6"]
CITY_STORMS = ["5yr", "10yr", "25yr", "50yr", "100yr"]
IN = 25.4


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def fmt_hours(h):
    return "none" if not h else ("%d to %d" % (h[0], h[-1]))


def build(app):
    ev = {e["id"]: e for e in load(os.path.join(HERE, "storms_out", "storms.json"))}
    facts = load(os.path.join(HERE, "storms_out", "vista_run_facts.json"))
    vf = load(os.path.join(HERE, "storms_out", "vista_verified_facts.json")) or {}
    out = {"generated_from": "scripts/build_storm_info.py (facts from the repo, Vista run summaries and converted-grid metadata only)", "storms": {}}
    common_solver = ("flood_solver_v2.py: 2D local-inertial (LISFLOOD-FP 'acceleration') shallow-water solver on one GPU, SCS curve-number infiltration (CN 75 open land / 98 roads and buildings, "
                     "Ia = 0.2 S, assumptions not calibrated), free outflow at the domain edge (slope 0.005), CFL 0.3 (script default; the Slurm script sets only CN, outflow and slope), mass balance logged, "
                     "rain applied uniformly over the whole domain. The SCS runoff of the cumulative rain is added to each cell as a source; water that is already on the surface never infiltrates later. "
                     "Impervious cells (CN 98) are the roads and buildings found in the Manning raster, which has them only inside the city (250,641 of 125 M county cells = 6.3 km2, the same area as the 1,566,224 cells of the 2 m city run); everywhere else CN 75. "
                     "Water from outside the model (e.g. the Barren River arriving from upstream counties) is not an inflow.")
    domain = ("County run: 5 m cells (10404 x 12043, 125,295,372 valid cells, UTM 16N), about 52 x 60 km: Warren County plus a margin. Terrain dem_warren_5m.tif built by build_county_grid.sh: USGS 3DEP 1/3 arc-second tiles (pixel 9.26e-5 deg = about 10.3 m N-S and 8.2 m E-W) resampled bilinear to 5 m, with the 2 m city DEM "
              "(dem_bg_2m.tif, 20 x 20 km, about 13 % of the county grid; README: 1 m lidar merged and resampled to 2 m) on top where it exists. Checked on Vista with 400 random points: outside the city the 5 m grid differs from the 3DEP tile by 0.21 m on average (p95 0.77 m), "
              "so its real information is the 10 m product; inside the city it differs from the 2 m DEM by 0.06 m on average. Manning n 0.035 outside the city, detailed city n on top (COUNTY_RUN.md). Outside the city the terrain is coarser than the 5 m grid, so small channels and building-scale flow are not resolved there.")
    for k in ORDER:
        cf = (facts["county"] or {}).get(k, {})
        fm = load(os.path.join(app, "layers", "flow_county", k, "meta.json")) or {}
        cm = load(os.path.join(app, "layers", "cog", "county", k, "cog_meta.json"))
        s = {"id": k, "not_verified": []}
        rf = cf.get("rain_file", {})
        ct = ((vf.get("county_slurm") or {}).get("tasks") or {}).get(k, {})
        if k in DESIGN:
            din = DESIGN[k]
            s.update(title="%s design storm (24 hours)" % k.replace("yr", "-year"), type="Design storm: a synthetic 24-hour rainfall with the nominal %s return period" % k.replace("yr", "-year"),
                     period="No calendar date (synthetic).",
                     rain_source="24-hour depth %.1f in (%.1f mm) taken from NOAA Atlas 14 for Warren County, KY, as written in scripts/run_county_all.sh and run_5_storms.sh. The Atlas 14 station/point, confidence interval and download date are not recorded in the repo." % (din, din * IN),
                     rain_total="%.1f in = %.1f mm (rain file integrates to %.2f mm, the run reports %.1f mm)" % (din, din * IN, rf.get("total_mm", 0), cf.get("total_rain_mm", 0)),
                     rain_peak="%.2f mm/hr (%.2f in/hr) as the 5-minute rate in the busiest part of the curve (hours 12 to 14.4, the steepest segment of the curve); not a short-burst peak: the curve is smooth." % (rf.get("peak_mm_per_hr", 0), rf.get("peak_mm_per_hr", 0) / IN),
                     duration="24 h of rain (simulation 24 h, rasters at hours 0 to 24; the solver runs until t = 86400 s).",
                     hyetograph="One shared 11-point cumulative curve, linear in time between points (fraction of 24 h -> fraction of depth): " + ", ".join("%.0f%%->%.0f%%" % (a * 100, b * 100) for a, b in zip(TF, PF)) +
                     ". It is NOT the official NRCS Type II table: it is the curve of the first 2-year run, kept for all design storms (run_5_storms.sh: the 2yr rasters' domain-mean depth was 30.3 mm at hour 10 and 37.2 mm at hour 11; this curve predicts 30.4 and 37.4 mm, the official Type II table would give 16.1 and 20.9 mm). Spread into 5-minute steps (577 points).",
                     return_period="Nominal %s, from the Atlas 14 24-hour depth. Return period applies to the rainfall depth only, not to the flood (infiltration, antecedent wetness and the missing upstream inflow are not part of it)." % k.replace("yr", "-year"))
            s["not_verified"] += ["The Atlas 14 depth itself (not re-downloaded from NOAA here)."]
        else:
            e = ev[k]
            tot_mm = e["total_in"] * IN
            s.update(title="Historical event %s: %s" % (k[2:], e["label"]), type="Historical rain block replayed through the model (not a reconstruction of the real flood)",
                     period="24-hour window starting %s UTC (hour 0 of the model = that hour). %s" % (e["start_utc"], e["note"]),
                     rain_source="%s, station BWG (Bowling Green-Warren County Regional Airport), hourly precipitation field p01i, downloaded by scripts/build_historical_storms.py (copy: scripts/storms_out/bwg_hourly.csv). Hours with no record: %d interpolated (gaps up to 3 h, flanked by data), %d set to 0. "
                                 "Totals were recomputed from the cached CSV and agree with the rain file. Almost all gauge reports are time-stamped at :53 (253,422 of 257,702 rows); each is assigned to the clock hour of its stamp (several reports in one hour: the largest is kept), "
                                 "so the rain may be shifted by up to about an hour against real time (the IEM definition of p01i was not re-checked here)." % (e["source"], e["gaps_interpolated_h"], e["gaps_unfilled_h"]),
                     rain_total="%.2f in = %.1f mm over 24 h (rain file integrates to %.2f mm)" % (e["total_in"], tot_mm, rf.get("total_mm", 0)),
                     rain_peak="%.2f in/hr = %.1f mm/hr in the wettest hour (hour index %d of the window, 0 = first hour)" % (e["peak_in_per_hr"], e["peak_in_per_hr"] * IN, e["hourly_in"].index(max(e["hourly_in"]))),
                     duration="24 h window; rain in %d of the 24 hours; simulation 24 h." % sum(1 for x in e["hourly_in"] if x > 0),
                     hyetograph="The gauge's own hourly depths, held constant within each hour (49 time/rate points: each hour as t and t+3599 s). Hourly depths (in): " + ", ".join("%.2f" % x for x in e["hourly_in"]) + ".",
                     return_period="No return period assigned: not computed in this project (not verified).")
            s["notes"] = []
            if k == "ev6":
                s["notes"].append("The gauge record has gaps in this window; the repo note says NWS puts Warren County near 10 in. over two days while this 24 h window holds %.2f in." % e["total_in"])
            if k == "ev4":
                s["notes"].append("The window covers day 1 to 2 of the April 2 to 6, 2025 flood; the full multi-day rainfall is not in the run.")
            s["not_verified"] += ["The event description (NWS statements) is quoted from the repo script; not checked against NWS records here.", "One gauge only: uniform rain over the county, so local rain differences are missing."]
        s["rain_units"] = ("The rain file holds seconds and rate in m/s (line 1 = number of points). The solver reads the values as already m/s (a file in mm/hr or in/hr would flood the model); "
                           "it aborts if the peak exceeds 600 mm/hr, which catches the common unit mistake. Hourly gauge inches x 0.0254 / 3600 = m/s. Written by make_rain.py / make_rain_county.py on Vista (checked: totals match the target depths within 0.01 mm); "
                           "the older gen_rain_files.py (SCS Type II, mm/s) was not used for these runs.")
        if cf:
            s["solver_run"] = ("%s Domain: %s Run: Slurm array job 1114427 task %s (partition gh, one GPU, node %s), started %s, ended %s (CST), wall time %s (%.0f min); runoff depth %.1f mm of %.1f mm rain; mass-balance error %.1e %% (a check of the numerical scheme against the SCS runoff volume, not of the physics). "
                               "Rasters: depth_<seconds>s.tif each hour (hours 0 to 24; save times are not exact multiples of 3600 s) plus max_depth.tif = the maximum over those saved snapshots, so a peak between two saves is not captured." % (common_solver, domain, ct.get("task", "?"), ct.get("node", "?"), ct.get("start", "?").replace("T", " "), ct.get("end", "?").replace("T", " "), cf.get("slurm_elapsed"), cf.get("wall_minutes", 0), cf.get("runoff_depth_mm", 0), cf.get("total_rain_mm", 0), cf.get("mass_error_pct", 0)))
        conv = []
        if fm:
            conv.append("Flow grids for the viewer: layers/flow_county/%s (source '%s', %d x %d cells of %s m, hours %s, built by build_flow_from_rasters.py; depth block-averaged 4x from 5 m to 20 m; velocity DERIVED from the water-surface slope with Manning, not solver output)." % (k, fm.get("source"), fm.get("width", 0), fm.get("height", 0), fm.get("cell_m"), fmt_hours(fm.get("hours"))))
            mx = max([h.get("max_depth_m", 0) for h in fm.get("hourly", [])] or [0])
            s["grid_max_depth_m"] = round(mx, 2)
        if cm:
            acc = (vf.get("cog_accuracy_local_set") or {}).get(k) or {}
            conv.append("Real-depth COGs: layers/cog/county/%s (%d files, EPSG:3857, uint16 cm, 6.25 mercator m cells = about 5 m on the ground, made by make_cogs.py on Vista, Slurm job 1123323 with warp tolerance 0.01 px). Checked against the solver rasters at %d random points: the COG equals a bilinear resample of the solver cells to within %.0f cm (99th percentile %.0f cm); against the single solver cell under a point the difference is larger (mean %.1f cm, 99th percentile up to %.0f cm) because depth changes between neighbouring 5 m cells. Details: docs/COG_VERIFICATION.md." % (k, len(cm.get("files", {})), acc.get("max_depth_points", 0) + acc.get("hourly_points", 0), acc.get("bilinear_max_cm", 0), acc.get("bilinear_p99_cm", 0), acc.get("nearest_mean_cm", 0), acc.get("nearest_p99_cm", 0)))
        else:
            conv.append("Real-depth COGs: not present on this computer.")
        if k in CITY_STORMS:
            cc = (facts.get("city") or {}).get(k)
            cs = ((vf.get("city_slurm") or {}).get("tasks") or {}).get(k, {})
            if cc:
                s["city_solver_run"] = ("City run of the same storm and rain file: flood_solver_v2.py on the 2 m city terrain (%s valid cells), wall time %.0f min on one GPU, runoff depth %.1f mm of %.1f mm rain, mass-balance error %.1e %%. Slurm array job 1102881 (bgflood5v2, partition gh) task %s on node %s, started %s, ended %s (CST), elapsed %s; the task-to-storm mapping follows from the elapsed time matching the run's wall time." % (format(cc.get("valid_cells", 0), ","), cc.get("wall_minutes", 0), cc.get("runoff_depth_mm", 0), cc.get("total_rain_mm", 0), cc.get("mass_error_pct", 0), cs.get("task", "?"), cs.get("node", "?"), cs.get("start", "?").replace("T", " "), cs.get("end", "?").replace("T", " "), cs.get("elapsed", "?")))
            conv.append("City domain: the city run of this storm (2 m terrain, 80,030,000 valid cells) exists on Vista; the City view's flow grids (layers/flow, 6.4 m cells) are built from the depth-TIER polygons, not from the rasters (approximate: 'Preview from depth tiers').")
        s["conversion"] = conv
        s["limits"] = ["Model output, not validated against observed flooding or gauges (no calibration of CN 75, roughness or terrain).",
                       "Uniform rain from one gauge or one design curve; no upstream inflow.",
                       "Rain falls on every cell at once (rain-on-grid): shallow sheet flow of a few cm over large areas is a model feature, not a measurement; the tiers and flooded-building counts start at 0.1 m for that reason.",
                       "Outside the city the terrain is about 10 m, so building-scale values there are indicative.",
                       "Sheet ponding of 2 to 3 cm is everywhere: in a sample of 144 buildings inside the model the median footprint maximum was 3 cm in the 2-year storm and also in the smallest event (ev1), so only values of 10 cm or more are treated as flooding.",
                       "Velocities and flow paths shown in the viewer are derived from the water-surface slope, not output by the solver."]
        if k in DESIGN:
            s["limits"].append("The temporal curve is the first run's curve, not the NRCS Type II table.")
        if cf.get("wall_minutes") is None:
            s["not_verified"].append("Run summary not found for this storm.")
        s["not_verified"].append("Observed flooding for comparison: none is in the repo (no validation).")
        out["storms"][k] = s
    return out


def md(info):
    L = ["# Storm data sources (all 12 storms)", "",
         "Generated by `scripts/build_storm_info.py` from facts in the repo, the Vista run summaries (copied into `scripts/storms_out/vista_run_facts.json`) and the converted-grid metadata. "
         "The same text is shown in the viewer under **About this storm** (`layers/storm_info.json`). Anything that could not be checked is marked *not verified*. "
         "Nothing here is a validation of the model against observed floods.", ""]
    for k in ORDER:
        s = info["storms"][k]
        L += ["## %s" % s["title"], "", "- **Type:** %s" % s["type"], "- **Date / period:** %s" % s["period"], "- **Rainfall source:** %s" % s["rain_source"],
              "- **Total depth:** %s" % s["rain_total"], "- **Peak intensity:** %s" % s["rain_peak"], "- **Duration:** %s" % s["duration"], "- **Return-period basis:** %s" % s["return_period"],
              "- **Temporal distribution (hyetograph):** %s" % s["hyetograph"], "- **Rain file units:** %s" % s["rain_units"], "- **Solver run:** %s" % s.get("solver_run", "not found")]
        for n in s.get("notes", []):
            L.append("- **Note:** %s" % n)
        if s.get("city_solver_run"):
            L.append("- **City solver run:** %s" % s["city_solver_run"])
        L.append("- **Outputs and conversion:**")
        L += ["  - %s" % c for c in s["conversion"]]
        L.append("- **Known limits:**")
        L += ["  - %s" % c for c in s["limits"]]
        L.append("- **Not verified:**")
        L += ["  - %s" % c for c in s["not_verified"]]
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    a = ap.parse_args()
    info = build(a.app)
    with open(os.path.join(REPO, "STORM_DATA_SOURCES.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md(info))
    with open(os.path.join(a.app, "layers", "storm_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print("wrote STORM_DATA_SOURCES.md and layers/storm_info.json")
