"""
Flood Damage Assessment — Bowling Green, KY
Computes expected dollar damage per parcel under each rainfall scenario.

Method:
  1. Load DEM → recompute elevation thresholds per rain level (same as rainfall_sim.py)
  2. Sample ground elevation at each parcel centroid from DEM
  3. Flood depth = max(0, water_surface_elevation - ground_elevation)  [feet]
  4. Apply NACCS residential depth-damage curve (damage fraction of structure value)
  5. Multiply by property value proxy (sale_price or zoning-based fallback)

Outputs:
  layers/flood_damage.geojson  — per-parcel features with damage columns per scenario
  layers/damage_summary.csv    — aggregate totals per rain level + return period

Rainfall ↔ return period mapping (Warren County, KY 24-hr events, approximate):
  0.25" → 1-yr    0.50" → 2-yr    0.75" → 3-yr    1.00" → 5-yr
  1.25" → 7-yr    1.50" → 10-yr   2.00" → 25-yr   2.50" → 50-yr
  3.00" → 100-yr
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import geopandas as gpd
import rasterio
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ── Paths ──────────────────────────────────────────────────────────────────────
DEM_PATH     = Path(r"C:\Users\Minec\Documents\BG_3D_Map\dem_bg.tif")
PARCELS_JSON = Path(r"C:\Users\Minec\Documents\BG_3D_Map\layers\all_parcels.geojson")
OUT_JSON     = Path(r"C:\Users\Minec\Documents\BG_3D_Map\layers\flood_damage.geojson")
OUT_CSV      = Path(r"C:\Users\Minec\Documents\BG_3D_Map\layers\damage_summary.csv")

UTM16N = "EPSG:32616"
WGS84  = "EPSG:4326"

# ── Rainfall levels → elevation percentile (must match rainfall_sim.py) ────────
RAIN_PCT = [
    (0.25,  3),
    (0.50,  7),
    (0.75, 12),
    (1.00, 18),
    (1.25, 25),
    (1.50, 33),
    (2.00, 43),
    (2.50, 55),
    (3.00, 67),
]

RETURN_PERIOD = {
    0.25: 1,
    0.50: 2,
    0.75: 3,
    1.00: 5,
    1.25: 7,
    1.50: 10,
    2.00: 25,
    2.50: 50,
    3.00: 100,
}

# ── Depth-damage curve (NACCS, 1-story residential, no basement) ───────────────
# Depth above first floor (ft) → fraction of structure value lost
# Source: USACE North Atlantic Coast Comprehensive Study (2015), Table B-3
_DD_DEPTH_FT   = [-2, -1,  0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   14,   16]
_DD_FRACTION   = [0,   0,  0.072, 0.173, 0.261, 0.340, 0.407, 0.463, 0.508, 0.543, 0.568, 0.584, 0.591, 0.591, 0.591, 0.591, 0.591]

# Fallback property value by zoning class ($/parcel) when sale_price is missing
ZONING_FALLBACK = {
    "R":  180_000,   # residential
    "C":  450_000,   # commercial
    "I":  600_000,   # industrial
    "A":   80_000,   # agricultural
    "MH":  90_000,   # mobile home
}
DEFAULT_VALUE = 150_000  # if zoning unknown


# ── Step 1: Load DEM, recompute elevation thresholds ──────────────────────────
print("=" * 60)
print("Flood Damage Assessment — Bowling Green, KY")
print("=" * 60)

print("\n[1] Loading DEM and computing flood elevation thresholds...")
with rasterio.open(DEM_PATH) as src:
    elev      = src.read(1).astype(np.float32)
    nodata    = src.nodata
    transform = src.transform
    h, w      = elev.shape
    dem_crs   = src.crs.to_epsg()

if nodata is not None:
    elev[np.isclose(elev, nodata)] = np.nan
elev[elev < -500] = np.nan

valid = np.isfinite(elev)
print(f"  DEM shape: {h}×{w}  Elevation range: {np.nanmin(elev):.0f}–{np.nanmax(elev):.0f} ft")

# Karst depression mask (replicate rainfall_sim.py logic)
fill_val     = float(np.nanmean(elev))
smooth_coarse = gaussian_filter(np.where(valid, elev, fill_val), sigma=20)
depression    = np.maximum(smooth_coarse - elev, 0)
depr_thresh   = np.percentile(depression[valid], 80)
in_sinkhole   = valid & (depression > depr_thresh)

# Compute min_rain raster (same as rainfall_sim.py) to get elevation thresholds
rain_levels = [r for r, _ in RAIN_PCT]
pct_levels  = [p for _, p in RAIN_PCT]
elev_thresh  = [float(np.nanpercentile(elev[valid], p)) for p in pct_levels]  # ft

print(f"\n  Rain level → Elevation threshold (ft above which cells are dry):")
for rain, thresh in zip(rain_levels, elev_thresh):
    print(f"    {rain:.2f}\"  →  ≤ {thresh:.0f} ft floods  (return period ≈ {RETURN_PERIOD[rain]}-yr)")


# ── Step 2: Load parcels, sample ground elevation from DEM ────────────────────
print("\n[2] Loading parcels and sampling ground elevation from DEM...")
gdf = gpd.read_file(str(PARCELS_JSON))
print(f"  Parcels loaded: {len(gdf):,}")

# Reproject centroids to DEM CRS for sampling
cents = gdf.geometry.centroid
cents_reproj = gpd.GeoDataFrame({"geometry": cents}, crs=gdf.crs)
if gdf.crs.to_epsg() != dem_crs:
    cents_reproj = cents_reproj.to_crs(epsg=dem_crs)

xs = cents_reproj.geometry.x.values
ys = cents_reproj.geometry.y.values

# Affine → pixel coordinates
cols_px = np.floor((xs - transform.c) / transform.a).astype(np.int64)
rows_px = np.floor((ys - transform.f) / transform.e).astype(np.int64)

in_bounds = (rows_px >= 0) & (rows_px < h) & (cols_px >= 0) & (cols_px < w)
ground_elev = np.full(len(gdf), np.nan, dtype=np.float32)

idx = np.where(in_bounds)[0]
sampled = elev[rows_px[idx], cols_px[idx]]
sampled[~np.isfinite(sampled)] = np.nan
ground_elev[idx] = sampled

gdf["ground_elev_ft"] = ground_elev
n_valid = np.isfinite(ground_elev).sum()
print(f"  Ground elevation sampled: {n_valid:,} of {len(gdf):,} parcels")


# ── Step 3: Determine property value per parcel ───────────────────────────────
print("\n[3] Estimating property values...")

sale_price = pd.to_numeric(gdf.get("sale_price", pd.Series(dtype=float)), errors="coerce")

# Zoning-based fallback
zoning = gdf.get("ZONING", pd.Series("", index=gdf.index)).fillna("").str.upper().str[:1]
fallback = zoning.map(ZONING_FALLBACK).fillna(DEFAULT_VALUE)

# Use sale_price where > $5,000; otherwise use fallback
prop_value = sale_price.where((sale_price > 5_000) & sale_price.notna(), fallback)
prop_value = prop_value.fillna(DEFAULT_VALUE)
gdf["prop_value_usd"] = prop_value.values

print(f"  Using sale_price:     {(sale_price > 5000).sum():,} parcels")
print(f"  Using zoning fallback: {(~(sale_price > 5000)).sum():,} parcels")
print(f"  Total portfolio value: ${prop_value.sum():,.0f}")


# ── Step 4: Compute flood depth and damage fraction per scenario ───────────────
print("\n[4] Computing flood depth and damage per scenario...")

summary_rows = []

for rain, wse_ft in zip(rain_levels, elev_thresh):
    col_depth  = f"depth_{rain:.2f}in_ft"
    col_dmg_pct = f"dmg_pct_{rain:.2f}in"
    col_dmg_usd = f"dmg_usd_{rain:.2f}in"

    # Flood depth = water surface elevation − ground elevation (ft), floored at 0
    # Parcels with missing DEM sample get depth 0 (no damage assigned)
    safe_elev = np.where(np.isfinite(ground_elev), ground_elev, wse_ft + 1)
    depth = np.maximum(0.0, wse_ft - safe_elev)
    # No flooding if parcel doesn't flood at this rain level (parcel_min_rain_in > rain)
    if "parcel_min_rain_in" in gdf.columns:
        min_rain = pd.to_numeric(gdf["parcel_min_rain_in"], errors="coerce").values
        not_flooded = (min_rain > rain) | ~np.isfinite(min_rain)
        depth[not_flooded] = 0.0

    # Apply depth-damage curve (linear interpolation between NACCS breakpoints)
    dmg_frac = np.interp(depth, _DD_DEPTH_FT, _DD_FRACTION)
    dmg_frac[depth == 0] = 0.0  # unflooded parcels → no damage

    dmg_usd = dmg_frac * prop_value.values

    gdf[col_depth]   = np.round(depth,    2)
    gdf[col_dmg_pct] = np.round(dmg_frac * 100, 1)
    gdf[col_dmg_usd] = np.round(dmg_usd, 0)

    n_flooded   = (depth > 0).sum()
    total_dmg   = dmg_usd.sum()
    mean_depth  = depth[depth > 0].mean() if n_flooded else 0
    rp          = RETURN_PERIOD[rain]

    summary_rows.append({
        "rain_in":        rain,
        "return_period_yr": rp,
        "parcels_flooded":  int(n_flooded),
        "mean_depth_ft":    round(float(mean_depth), 2),
        "total_damage_usd": int(total_dmg),
        "avg_damage_per_parcel_usd": int(total_dmg / n_flooded) if n_flooded else 0,
    })

    print(f"  {rain:.2f}\" ({rp:>3}-yr): {n_flooded:>5,} flooded  "
          f"avg depth {mean_depth:.1f} ft  damage ${total_dmg:>12,.0f}")


# ── Step 5: Expected Annual Damage (EAD) via trapezoidal integration ──────────
print("\n[5] Computing Expected Annual Damage (EAD)...")

df_sum = pd.DataFrame(summary_rows)

# Exceedance probability = 1 / return_period
df_sum["exceedance_prob"] = 1.0 / df_sum["return_period_yr"]

# Sort by exceedance probability descending (most frequent first)
df_sum = df_sum.sort_values("exceedance_prob", ascending=False).reset_index(drop=True)

# EAD = area under damage-vs-exceedance-probability curve (trapezoid rule)
# Add a (prob=1, damage=0) anchor and (prob=0, damage=max) anchor
probs  = np.concatenate([[1.0], df_sum["exceedance_prob"].values, [0.0]])
damages = np.concatenate([[0.0], df_sum["total_damage_usd"].values,
                           [df_sum["total_damage_usd"].iloc[-1]]])

ead = float(np.trapz(damages, -probs))  # negative because we integrate left→right over prob
print(f"  EAD (expected annual damage): ${ead:,.0f}")
df_sum["ead_contribution_usd"] = ""  # placeholder


# ── Step 6: Parcel-level EAD column ───────────────────────────────────────────
ead_cols = [f"dmg_usd_{r:.2f}in" for r in rain_levels]
dmg_matrix = gdf[ead_cols].values.astype(float)

probs_sorted = df_sum["exceedance_prob"].values  # already sorted descending
probs_full   = np.concatenate([[1.0], probs_sorted, [0.0]])
dmg_extended = np.column_stack([
    np.zeros(len(gdf)),
    dmg_matrix,
    dmg_matrix[:, -1],
])

parcel_ead = np.trapz(dmg_extended, -probs_full, axis=1)
gdf["ead_usd"] = np.round(parcel_ead, 0)

total_portfolio_ead = float(parcel_ead.sum())
print(f"  Portfolio EAD: ${total_portfolio_ead:,.0f}  (should match ≈ ${ead:,.0f})")


# ── Step 7: Risk tier per parcel ──────────────────────────────────────────────
def assign_risk(ead):
    if ead >= 5_000:   return "critical"
    if ead >= 1_000:   return "high"
    if ead >= 200:     return "moderate"
    if ead > 0:        return "low"
    return "none"

gdf["damage_risk_tier"] = gdf["ead_usd"].apply(assign_risk)
print("\n  Risk tier distribution (by EAD):")
print(gdf["damage_risk_tier"].value_counts().to_string())


# ── Step 8: Save outputs ──────────────────────────────────────────────────────
print("\n[8] Saving outputs...")

# Drop columns that might cause serialization issues; keep geometry + damage columns
keep_cols = (
    ["geometry", "ground_elev_ft", "prop_value_usd", "ead_usd", "damage_risk_tier"]
    + [c for c in gdf.columns if c.startswith("depth_") or
                                   c.startswith("dmg_pct_") or
                                   c.startswith("dmg_usd_")]
)
# Add key identifier columns if present
for col in ["PVA_PARCEL", "ADDRESS", "SiteAddress", "ZONING", "LAND_USE",
            "sale_price", "parcel_min_rain_in"]:
    if col in gdf.columns and col not in keep_cols:
        keep_cols.insert(1, col)

gdf_out = gdf[[c for c in keep_cols if c in gdf.columns]].copy()
gdf_out.to_file(str(OUT_JSON), driver="GeoJSON")
sz = OUT_JSON.stat().st_size / 1e6
print(f"  ✓ flood_damage.geojson: {len(gdf_out):,} parcels  {sz:.1f} MB → {OUT_JSON}")

df_sum["portfolio_ead_usd"] = ""
df_sum.loc[df_sum.index[-1], "portfolio_ead_usd"] = int(total_portfolio_ead)
df_sum.to_csv(str(OUT_CSV), index=False)
print(f"  ✓ damage_summary.csv → {OUT_CSV}")

print("\n" + "=" * 60)
print("DAMAGE SUMMARY")
print("=" * 60)
print(df_sum[["rain_in", "return_period_yr", "parcels_flooded",
              "mean_depth_ft", "total_damage_usd"]].to_string(index=False))
print(f"\nExpected Annual Damage (EAD): ${ead:,.0f}")
print(f"Highest-risk parcels (critical EAD ≥ $5k): "
      f"{(gdf['damage_risk_tier'] == 'critical').sum():,}")
print("=" * 60)
