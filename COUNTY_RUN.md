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
