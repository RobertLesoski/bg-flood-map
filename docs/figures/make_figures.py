"""Figure sources for DOCUMENTATION.md. Run: python docs/figures/make_figures.py (writes SVG next to this file)."""
import json, os, numpy as np
import matplotlib; matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams["svg.fonttype"] = "none"; plt.rcParams["font.family"] = "DejaVu Sans"; plt.rcParams["font.size"] = 9
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
C = ["#00204D", "#31446B", "#666970", "#958F78", "#CBBA69", "#FFEA46"]   # cividis samples
def save(n): plt.savefig(os.path.join(H, n), bbox_inches="tight"); plt.close()
facts = json.load(open(os.path.join(R, "scripts/storms_out/vista_run_facts.json")))["county"]
order = ["2yr","5yr","10yr","25yr","50yr","100yr","ev1","ev2","ev3","ev4","ev5","ev6"]

# 1 data flow
fig, ax = plt.subplots(figsize=(9.2, 5.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 62)
def box(x, y, w, h, t, c="#E8EEF7", ec=C[0]):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=c, ec=ec, lw=1)); ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=7.5)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=C[1], lw=1))
ax.text(10, 60, "RAW INPUTS", ha="center", weight="bold"); ax.text(41, 60, "SOLVER (TACC Vista)", ha="center", weight="bold"); ax.text(70, 60, "CONVERSION", ha="center", weight="bold"); ax.text(92, 60, "VIEWERS", ha="center", weight="bold")
box(0, 48, 20, 8, "NOAA Atlas 14\n24 h depths\n(6 design storms)"); box(0, 37, 20, 8, "KBWG hourly gauge\n(IEM ASOS p01i)\n6 events")
box(0, 26, 20, 8, "USGS 3DEP 1/3 arc-s\n(county, ~10 m)"); box(0, 15, 20, 8, "1 m lidar -> 2 m city DEM\n(exact tiles not\nrecorded in repo)")
box(0, 4, 20, 8, "OpenStreetMap (Geofabrik)\nbuildings, roads, POIs\n+ city PVA parcels")
box(28, 41, 26, 11, "make_rain.py / make_rain_county\n.rain files (m/s)\n24 h, 5-min or hourly", "#FFF6C9")
box(28, 22, 26, 13, "dem_warren_5m.tif (5 m UTM 16N)\nmannings_n_county.tif\n(0.035 + city n on top)", "#FFF6C9")
box(28, 4, 26, 12, "flood_solver_v2.py\nlocal-inertial + SCS CN\n1 H100 per storm\n12 county, 5 city runs", "#CBBA69")
box(60, 40, 22, 10, "depth_<s>.tif hourly\nmax_depth.tif\nrun_summary.json", "#E8EEF7")
box(60, 25, 22, 11, "make_cogs.py -> COG tiles\n6.25 m EPSG:3857, cm\nbuild_flow_from_rasters.py\n-> 25 m flow grids", "#E8EEF7")
box(60, 6, 22, 14, "build_address_index.py\nbuild_infrastructure.py\nbuild_county_layers.py\n(buildings, roads,\nfacilities)", "#E8EEF7")
box(86, 36, 14, 14, "Web viewer\nviewer.html\n+ COG reader", "#E8EEF7"); box(86, 16, 14, 14, "Desktop viewer\n(wgpu, Python)", "#E8EEF7")
arr(20, 52, 28, 47); arr(20, 41, 28, 44); arr(20, 30, 28, 29); arr(20, 19, 28, 26); arr(41, 41, 41, 35); arr(41, 22, 41, 16)
arr(54, 10, 60, 44); arr(54, 10, 60, 30); arr(54, 8, 60, 12); arr(82, 44, 86, 44); arr(82, 31, 86, 24); arr(82, 12, 86, 20)
save("fig1_dataflow.svg")

# 2 stencil
fig, ax = plt.subplots(figsize=(6.4, 3.6)); ax.axis("off"); ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.3, 3.6)
for i, x in enumerate((0.5, 3.5)):
    ax.add_patch(Rectangle((x, 1), 2, 2, fc="#E8EEF7", ec=C[0], lw=1.2))
    ax.text(x+1, 2.3, "cell %s\nZ, H, n" % "ij"[i], ha="center", fontsize=9)
ax.annotate("", (3.5, 2.0), (2.5, 2.0), arrowprops=dict(arrowstyle="<->", color=C[4], lw=2))
ax.text(3.0, 1.55, "face i+1/2\nQx (m2/s)", ha="center", fontsize=8)
ax.text(3.0, 0.3, "Qx(new) = [Qx - g*hf*dt*(eta_j-eta_i)/dx] / [1 + g*dt*n^2*|Qx|/hf^(7/3)]\nhf = max(eta_i, eta_j) - max(Z_i, Z_j);  eta = Z + H", ha="center", fontsize=8)
ax.text(3.0, 3.3, "Fluxes live on cell faces; depths live in cell centres", ha="center", fontsize=9, weight="bold")
save("fig2_stencil.svg")

# 3 hyetographs
tf = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])*24; pf = np.array([0,.04,.10,.19,.31,.50,.70,.79,.86,.92,1.0])
hrs = np.arange(25); cum = np.interp(hrs, tf, pf); d100 = 7.6*25.4*np.diff(cum)
def hourly(ev):
    t, r = [], []
    L = open(os.path.join(R, "scripts/storms_out/storm_%s.rain" % ev)).read().split("\n"); n = int(L[0])
    for l in L[1:n+1]: a, b = map(float, l.split()); t.append(a); r.append(b)
    t = np.array(t); r = np.array(r); out = []
    for h in range(24):
        m = (t >= h*3600) & (t <= h*3600+3599); out.append(r[m].mean()*3600*1000 if m.any() else 0)
    return np.array(out)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
axs[0].bar(np.arange(24)+.5, d100, width=.9, color=C[1]); axs[0].set_title("100-year design storm: 193.0 mm in 24 h\n(hourly sums of the 11-point curve)")
e4 = hourly("ev4"); axs[1].bar(np.arange(24)+.5, e4, width=.9, color=C[4]); axs[1].set_title("ev4, 2025-04-03 13:00 UTC: 126.2 mm in 24 h\n(KBWG gauge, mm/h from storm_ev4.rain)")
for a in axs: a.set_xlabel("hour of the 24 h window"); a.spines[["top","right"]].set_visible(False)
axs[0].set_ylabel("rain (mm per hour)"); save("fig3_hyetographs.svg")
print("ev4 total mm", e4.sum(), "d100 total", d100.sum(), "d100 max/h", d100.max())

# 4 tier legend
fig, ax = plt.subplots(figsize=(8, 1.9)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 20)
tiers = [("below 0.1 m\nnot flooded", "#F2F2F2", "#000"), ("shallow\n0.1 to 0.3 m", "#FFEA46", "#000"), ("moderate\n0.3 to 0.6 m", "#CBBA69", "#000"), ("deep\n0.6 to 1.2 m", "#666970", "#fff"), ("severe\n1.2 m and deeper", "#00204D", "#fff")]
for i, (t, c, tc) in enumerate(tiers):
    ax.add_patch(Rectangle((i*20, 5), 20, 10, fc=c, ec="#444")); ax.text(i*20+10, 10, t, ha="center", va="center", color=tc, fontsize=8.5)
ax.text(50, 18, "Depth tiers (flowquery.py TIER_EDGES_M = 0.1, 0.3, 0.6, 1.2 m); cividis-like palette", ha="center", fontsize=8.5)
save("fig4_tiers.svg")

# 5 run times
fig, ax = plt.subplots(figsize=(9, 3.4)); wm = [facts[s]["wall_minutes"]/60 for s in order]
ax.bar(order, wm, color=[C[1]]*6+[C[4]]*6); ax.set_ylabel("wall time (hours, one H100)"); ax.spines[["top","right"]].set_visible(False)
for i, v in enumerate(wm): ax.text(i, v+.15, "%.1f" % v, ha="center", fontsize=8)
ax.set_title("County solver run times, Slurm array 1114427 (sum %.1f node-hours)" % sum(wm)); save("fig5_runtimes.svg")

# 6 rain vs runoff
fig, ax = plt.subplots(figsize=(9, 3.4)); x = np.arange(12)
ax.bar(x-.2, [facts[s]["total_rain_mm"] for s in order], .4, color=C[2], label="rain (mm)"); ax.bar(x+.2, [facts[s]["runoff_depth_mm"] for s in order], .4, color=C[0], label="SCS runoff (mm)")
ax.set_xticks(x); ax.set_xticklabels(order); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); ax.set_title("Rain and modelled runoff depth per storm (county runs)"); save("fig6_rain_runoff.svg")

# 7 comparison matrix
rows = ["This project", "FEMA NFHL/FIRM", "NWS National Water Model", "Fathom US", "HEC-RAS 2D study", "LISFLOOD-FP (software)"]
cols = ["Pluvial (rain-on-grid)", "River flooding", "Storm drains", "Calibrated/validated", "Open code/data", "Regulatory use"]
M = [["yes","no","no","no (weak check)","yes","no"],["not verified","yes","n/a","not verified","data free","yes"],["partly (1 km/250 m)","yes","no","not verified","data free","no"],["yes (global map)","yes (inland)","no","publisher-reported","no (licensed)","not verified"],["yes (option)","yes","yes (pipe option)","per study","software free","per study"],["yes (v8)","yes","no","user-dependent","yes (GPL, non-comm.)","no"]]
fig, ax = plt.subplots(figsize=(9.4, 3.8)); ax.axis("off"); ax.set_xlim(0, len(cols)+1.6); ax.set_ylim(0, len(rows)+1)
for j, c in enumerate(cols): ax.text(1.6+j+.5, len(rows)+.5, c, ha="center", va="center", fontsize=7.5, weight="bold")
for i, r in enumerate(rows):
    y = len(rows)-1-i; ax.text(1.5, y+.5, r, ha="right", va="center", fontsize=8, weight="bold")
    for j, v in enumerate(M[i]):
        col = C[4] if v.startswith("yes") else ("#F2F2F2" if v in ("no", "n/a") else C[3])
        ax.add_patch(Rectangle((1.6+j, y), 1, 1, fc=col, ec="white", lw=2)); ax.text(2.1+j, y+.5, v, ha="center", va="center", fontsize=6.5)
ax.set_title("Capability matrix (yellow = yes, tan = qualified / not verified, grey = no); sources in chapter 6", fontsize=9); save("fig7_matrix.svg")
print("ok")
