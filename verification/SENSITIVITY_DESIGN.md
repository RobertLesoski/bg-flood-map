# Sensitivity sweep design (Phase 4, blocked)

Status: BLOCKED. The ev4 model cannot be run locally: the DEM, roughness, rain file and solver outputs are on Vista (/scratch/11931/rlesoski/bgflood). No model run has been done here.

Inputs needed to run: the bgflood solver and its ev4 configuration (flood_solver_v2.py, county_flow/, dem/), and access to Vista or a copy of dem/ (11 GB) and the ev4 rain file (runs/storm_ev4.rain).

Design, to run once inputs are available:
1. Most influential inputs (ranked by expected effect on extent and depth):
   a. Rainfall total (airport 4.97 in rolling 24 h vs. the 10.1 in Warren County total from the Kentucky Mesonet over four days). Two runs: scale 1.0x and 2.0x of the ev4 rain.
   b. DEM vertical error. Scale the DEM by a uniform +/-0.5 m offset (typical LiDAR/USGS 3DEP vertical RMSE is of this order; confirm the DEM's own metadata).
   c. Manning roughness n: scale by 0.75x and 1.25x.
   d. Rolling 24 h vs. calendar-day window (already flagged in storm_info.json).
2. Outputs compared per run: wet area (km2), max-depth at the observed points in verification/observed_depths.csv, and OPERA CSI (scripts/validate_opera_ev4.py).
3. Conclusions to test: whether the extent-level OPERA score is stable across the sweep (if it changes sign or rank, the comparison is not robust to input error); whether depths at the anecdotal points change by more than their reporting precision (inches).
4. Report which conclusions hold (ranked order of wet area by storm) and which change.

Run script: not written, because the solver invocation depends on the bgflood configuration, which is not in this session. Write it once the ev4 run is reproduced on Vista.
