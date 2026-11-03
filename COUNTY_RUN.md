# Whole of Warren County on Vista, all 12 storms

One script does it: `scripts/run_county_all.sh`.

| What | Value |
|---|---|
| Domain | Warren County plus a margin, about 52 x 60 km, 5 m cells (about 125 M cells, one H100 each) |
| Storms | 2yr 5yr 10yr 25yr 50yr 100yr (24 h, NOAA Atlas 14) and ev1..ev6 (KBWG gauge blocks) |
| Solver | `flood_solver_v2.py` (SCS infiltration, edge outflow, mass balance), same settings as the city runs |
| Terrain | USGS 3DEP 1/3 arc-second (about 10 m, downloaded by the script, no login), with the 2 m city DEM on top where it exists |
| Roughness | n = 0.035 everywhere, the detailed city Manning's n on top over the city |
| Output | `$SCRATCH/bgflood/runs/county_v2/<storm>/` (hourly depth rasters, `max_depth.tif`, `run_summary.json`) |
| Cost estimate | roughly 8 to 10 node-hours per storm, 100 to 120 in total (8,299 SU were available on 2026-10-26) |

## Run it

From your PC, copy the script (and the six historical rain files, if they are not already on Vista):

```bash
scp scripts/run_county_all.sh rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/
scp scripts/storms_out/storm_ev*.rain rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/runs/
```

On a Vista login node:

```bash
cd $SCRATCH/bgflood
bash run_county_all.sh --dry-run      # builds nothing, checks everything, writes the job file
bash run_county_all.sh                # downloads the DEM, checks rain files, preflight, submits all 12
```

Options: `--only "25yr ev4"` (just those), `--skip-dem`, `--throttle 4` (at most 4 at once).
Cancel everything with `scancel -n warrenv2`.

The script stops with a FATAL line, and submits nothing, if: a rain file is missing or has wrong units, the solver is
an old version, the city DEM and 3DEP disagree by more than 3 m (units or datum), the grid is not UTM 16N at 5 m, or
the grid is over 150 M cells.

## What it does not model (say so when you present it)

- Water entering the county from upstream (the Barren River arrives from Allen and Simpson counties) is not an inflow;
  the edges only let water out.
- Rain is uniform over the county (single gauge) and the curve number is 75 for open land everywhere (not calibrated).
- Outside the city the terrain is about 10 m resolution, so small channels and building-scale flow are not resolved there.
- Roughness outside the city is one flat value; swap in NLCD land cover for better results.

## Afterwards

Download `runs/county_v2/<storm>/` and convert it with the same flow-grid pipeline as the city
(`scripts/flowgrid.py`, see `FLOW_FORMAT.md`). The viewer reads grids of any size, but the county grids have not been
built or tested in it yet.

## Converting county results

`scripts/build_flow_from_rasters.py` turns one storm folder (`depth_*s.tif`, `max_depth.tif`, `run_summary.json`) into the
viewer flow grids (`meta.json`, `h01..h24.rgba.gz`, `flowpaths.rgba.gz`, `"source": "rasters"`, FLOW_FORMAT.md). It streams the
rasters in row strips (about 0.5 GB RAM per worker), takes the terrain from `dem_warren_5m.tif`, Manning n from
`mannings_n_county.tif`, and writes a 2.6k x 3k grid (25 m mercator cell, about 20 m on the ground, about 31 MB raw and a few
MB gzipped per hour). Velocity is derived from the simulated water-surface slope, not solver output (the meta says so).
Convert ON VISTA: a storm is then a few tens of MB to download instead of about 5 GB of rasters.

**1. On Vista** (once: copy the scripts; then one command per storm or all)

```bash
# from the PC:
scp scripts/build_flow_from_rasters.py scripts/flowgrid.py scripts/convert_county_all.sh \
    rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/
# on Vista:
cd $SCRATCH/bgflood
bash convert_county_all.sh 100yr                 # one storm on the login node (WORKERS=4 to limit, FORCE=1 to rebuild)
bash convert_county_all.sh                       # all 12, resumable: finished storms are skipped
sbatch convert_county_all.sh                     # all 12 as a CPU array job, one storm per task (check `sinfo` for the partition)
```

Output: `$SCRATCH/bgflood/county_flow/<storm>/` for `2yr 5yr 10yr 25yr 50yr 100yr ev1..ev6`. A killed run resumes hour by hour
(work folder `county_flow/.<storm>.building`); a storm is only moved into place after `flowgrid.py validate` found no problem.
Check one: `python3 flowgrid.py validate $SCRATCH/bgflood/county_flow/100yr`.

**2. Download only the small folders** (Ubuntu / WSL terminal on the PC; `/mnt/c` is the Windows C: drive)

```bash
mkdir -p "/mnt/c/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/layers/flow_county"
rsync -avh --progress rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/county_flow/ \
      "/mnt/c/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/layers/flow_county/"
```

Re-running the same rsync only copies what changed. (The viewer reads `layers/flow/<storm>`; see the notes below on what else the
county needs before it can show them.)

**3. Alternative: convert at home** (needs the rasters on this PC)

```powershell
C:\Users\Minec\anaconda3\python.exe scripts\build_flow_from_rasters.py --run D:\county_v2\100yr `
  --out "C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map\layers\flow_county\100yr" --storm 100yr `
  --label "100-year storm (county)" --dem D:\dem_warren_5m.tif --dem-units metres --mann D:\mannings_n_county.tif --workers 3
```

Disk warning: one storm is 25 rasters of about 190 MB to 1 GB each (about 5 GB; 12 storms are about 60 GB), this PC has 56 GB
free: download, convert and delete one storm at a time. `--workers 3` keeps RAM under 3 GB on an 8 GB machine. Without `--dem`
the surface is the depth alone and the velocity only indicative (the meta says so); the city's `dem_bg.tif` is in FEET
(`--dem-units feet`), the Vista DEMs are metres.

## Real-depth click-to-query: Cloud-Optimized GeoTIFFs (Vista, CPU)

`scripts/make_cogs.py` (helper) + `scripts/vista_make_cogs.sh` turn `runs/county_v2/<storm>/max_depth.tif` and the hourly `depth_*s.tif` (city: `runs/output_v2/<storm>`)
into COGs: EPSG:3857, uint16 depth in CENTIMETRES (65535 = outside the model), 512 px tiles, DEFLATE + predictor 2, average overviews, about 5 m on the ground
(6.25 mercator m; `--res-m 12.5` makes them 4x smaller). Hour N = `depth_<N*3600>s.tif`; `cog_meta.json` lists bounds/units/files. `serve.py` already supports HTTP Range,
so geotiff.js can `fromUrl` them directly (read image 0 for queries). Size ESTIMATE (not measured, rasters are on Vista): the grid is about 10.5k x 12.2k px, mostly dry, so
roughly 5 to 15 MB per raster, 150 to 400 MB per storm (26 rasters), 2 to 5 GB for all 12; `max_depth.tif` alone about 10 MB per storm.

```bash
# PC -> Vista (once)
scp scripts/make_cogs.py scripts/vista_make_cogs.sh rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/
# on Vista
cd $SCRATCH/bgflood
bash vista_make_cogs.sh --dry-run                 # checks the storms, writes cog_job.sh
bash vista_make_cogs.sh --domain both             # one CPU task per storm (--only "25yr ev4", --here for an idev session, --partition gg)
du -sh county_cog/* city_cog/*
python3 make_cogs.py --verify county_cog/100yr/max_depth.tif
# Vista -> PC (WSL/Ubuntu terminal)
mkdir -p "/mnt/c/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/layers/cog/county" "/mnt/c/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/layers/cog/city"
rsync -avh --progress rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/county_cog/ "/mnt/c/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/layers/cog/county/"
rsync -avh --progress rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/city_cog/ "/mnt/c/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/layers/cog/city/"
```
Resumable: re-running skips finished files. Download only `max_depth.tif` per storm first if space is short.

## Rain-scaling scenarios (+10 % / +20 %), script only

`scripts/vista_climate_scenarios.sh` runs the 100yr and 25yr rain multiplied by 1.10 and 1.20 (`100yr_p10 100yr_p20 25yr_p10 25yr_p20`) into
`runs/county_climate/<name>/`. These are SCENARIO MULTIPLIERS on the same hyetograph and nothing else (same terrain, infiltration, solver): not a climate projection.
```bash
scp scripts/vista_climate_scenarios.sh rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/
cd $SCRATCH/bgflood && bash vista_climate_scenarios.sh --dry-run && bash vista_climate_scenarios.sh     # needs run_county_all.sh done once
```

## Afterwards: viewing the county
Flow grids: `layers/flow_county/` (rsync above). Terrain: `scp -o MACs=hmac-sha2-256 rlesoski@vista.tacc.utexas.edu:/scratch/11931/rlesoski/bgflood/dem/dem_warren_5m.tif "<data root>\layers\"`.
Start `python -m bgv --domain county` (or choose County on the loading screen). Buildings/roads are read from `kentucky-latest.osm.pbf` once (about a minute) and cached.
No data yet? `python -m bgv --make-county-demo <folder>` for synthetic DEMO DATA.
