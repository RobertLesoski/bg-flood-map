"""One-click risk report: a self-contained HTML file (inline CSS + inline SVG, optional embedded PNG of the view).

    data = from_assessment(a, domain_label, ...)           # an address search result (assess.Assessment)
    data = from_live(address, lng, lat, rows=[...], ...)   # a picked building without an address record (current storm only)
    html = build_html(data)
    paths = write_report(folder, data, png_rgb=None)       # -> (html_path, png_path | None)

Contents: address / location, date generated, depth + tier per storm with the SOURCE of every number, the two sources side by
side (tiles vs index) with their difference where both exist, a bar chart, the tier legend, data notes per storm source,
the uncertainty paragraph and the 'not validated against observed floods' notice.  Tier palette = ColorBrewer YlGnBu (4 classes),
colour-blind safe and monotonic in lightness; every tier is also written out (name, range), so colour is never the only cue.
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

TIER_NAMES = ("Dry / below 10 cm", "Shallow", "Moderate", "Deep", "Severe")
TIER_RANGES = ("< 10 cm", "10 - 30 cm", "30 - 60 cm", "60 - 120 cm", ">= 120 cm")
# YlGnBu (ColorBrewer, 4 classes) lightness-ordered, colour-blind safe; index 0 = below the flood threshold
TIER_COLORS = ("#f2f2f2", "#c7e9b4", "#41b6c4", "#2c7fb8", "#253494")
TIER_TEXT_ON = ("#222222", "#222222", "#0b1020", "#ffffff", "#ffffff")

NOT_VALIDATED = ("These depths are model results. They have NOT been validated against observed floods "
                 "(no gauge, high-water-mark or photo comparison has been done).")
UNCERTAINTY = ("Model uncertainty is not quantified here. Depths depend on the rainfall assumption of the storm, the terrain "
               "model, the grid resolution and the simplified runoff and drainage treatment of the solver. A footprint depth is the "
               "MAXIMUM over the model cells the building touches, so it can overstate the depth at a doorway; a dry result means "
               "'below 10 cm in the model', not 'safe'. Use as a screening aid, not for insurance, engineering or evacuation decisions.")


@dataclass
class ReportRow:
    key: str
    label: str
    depth_cm: int
    tier: int
    source: str                        # label of the source of depth_cm
    approx: bool = False               # depth-tier (preview) source: severe is only '>= 120 cm'
    index_cm: int | None = None
    tile_cm: int | None = None
    first_hour: int | None = None
    hours: int | None = None
    disagree: bool = False
    grid_source: str = ""              # storm data source kind ('rasters' | 'tiers')


@dataclass
class ReportData:
    title: str
    address: str
    lng: float
    lat: float
    domain_label: str
    rows: list = field(default_factory=list)
    detail: list = field(default_factory=list)          # [(label, value)] parcel, land use, assessed value ...
    notes: list = field(default_factory=list)
    sources: list = field(default_factory=list)         # [(heading, text)] data source lines
    generated: dt.datetime | None = None
    current_storm: str | None = None
    screenshot_name: str | None = None
    viewer: str = "DSOC Flood Map (desktop viewer)"
    build: str = ""


def depth_text(cm: int, tier: int, approx: bool) -> str:
    if cm is None:
        return "-"
    if approx and tier == 4:
        return ">= 120 cm (>= 1.2 m, >= 3.9 ft)"
    if cm < 1:
        return "dry"
    pre = "~" if approx else ""
    return f"{pre}{cm} cm ({pre}{cm / 100.0:.2f} m, {pre}{cm / 30.48:.1f} ft)"


def from_assessment(a, domain_label: str, *, current_storm: str | None = None, generated=None, screenshot_name=None,
                    extra_notes=(), storm_notes=None, build: str = "") -> ReportData:
    rec = a.rec
    rows = []
    for r in a.rows:
        approx = r.source_kind == "tiers" and r.shown_source == "index (grid)"
        rows.append(ReportRow(r.key, r.short, int(r.shown_cm), int(r.shown_tier), r.shown_source, approx, r.index.depth_cm, r.tile_cm,
                              r.index.first_hour or None, r.index.hours or None, bool(r.comparison and r.comparison.disagree),
                              r.source_kind))
    detail = []
    if rec.parcel:
        detail.append(("Parcel (PVA id)", rec.parcel))
    if rec.use:
        detail.append(("Land-use code", rec.use))
    if rec.value:
        detail.append(("Assessed value", f"${rec.value:,}"))
    detail.append(("Record type", "lot (no building footprint; depth at the lot point)" if rec.kind == 1 else "building / address"))
    notes = list(a.notes) + list(extra_notes)
    sources = []
    if a.tile_label:
        sources.append((a.tile_label, "Depth tiles written from the solver depth rasters (maximum over all hours of each storm). "
                        "Used as the headline depth wherever a tile exists for the storm."))
    sources.append(("index (grid)", "Address index built from the storm flow grids (maximum over all hours); also supplies the "
                    "first hour flooded and the number of hours flooded."))
    for k, v in (storm_notes or {}).items():
        sources.append((k, v))
    return ReportData("Flood risk report", rec.address or "(no street address)", rec.lng, rec.lat, domain_label, rows, detail, notes,
                      sources, generated or dt.datetime.now(), current_storm, screenshot_name, build=build)


def from_live(address: str, lng: float, lat: float, domain_label: str, storm_key: str, storm_label: str, depth_cm: int, tier: int,
              approx: bool, source: str, hour: int, *, detail=(), generated=None, screenshot_name=None) -> ReportData:
    """A picked building with no address record: only the storm on screen is known (the viewer holds one storm at a time)."""
    row = ReportRow(storm_key, storm_label, int(depth_cm), int(tier), source, approx)
    notes = [f"No address record matches this building, so only the storm shown on screen ({storm_label}, snapshot hour {hour}) is "
             "reported; the other storms are not available for it.",
             "The value is the building's depth at that hour, not the maximum over the storm."]
    return ReportData("Flood risk report", address or "(building without a street address)", lng, lat, domain_label, [row], list(detail),
                      notes, [(source, "Depth taken from the viewer's current snapshot.")], generated or dt.datetime.now(), storm_key,
                      screenshot_name)


# ------------------------------------------------------------------------------------------------ html
def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _swatch(tier: int) -> str:
    return (f'<span class="sw" style="background:{TIER_COLORS[tier]}"></span>')


def _chart(rows) -> str:
    """Inline SVG bar chart: depth (cm) per storm, bars in tier colours with the number printed on / next to the bar."""
    n = len(rows)
    if n == 0:
        return ""
    W, left, right, top, bot = 760, 80, 20, 14, 46
    bw = max(14.0, min(46.0, (W - left - right) / n * 0.62))
    mx = max([r.depth_cm for r in rows] + [130])
    H = 230
    ph = H - top - bot
    sc = ph / mx

    def y(cm):
        return top + ph - cm * sc
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Depth per storm in centimetres" class="chart">']
    for edge, name in ((10, "10"), (30, "30"), (60, "60"), (120, "120")):
        if edge <= mx:
            s.append(f'<line x1="{left}" x2="{W - right}" y1="{y(edge):.1f}" y2="{y(edge):.1f}" class="grid"/>'
                     f'<text x="{left - 6}" y="{y(edge) + 4:.1f}" text-anchor="end" class="axis">{name}</text>')
    s.append(f'<text x="{left - 6}" y="{y(0) + 4:.1f}" text-anchor="end" class="axis">0</text>')
    s.append(f'<text x="14" y="{top + ph / 2:.0f}" transform="rotate(-90 14 {top + ph / 2:.0f})" text-anchor="middle" class="axis">depth, cm</text>')
    step = (W - left - right) / n
    for k, r in enumerate(rows):
        cx = left + step * (k + 0.5)
        h = max(r.depth_cm * sc, 0.0)
        s.append(f'<rect x="{cx - bw / 2:.1f}" y="{y(r.depth_cm):.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{TIER_COLORS[r.tier]}" '
                 f'stroke="#34406a" stroke-width="1"><title>{_e(r.label)}: {_e(depth_text(r.depth_cm, r.tier, r.approx))}</title></rect>')
        lab = (">=" if (r.approx and r.tier == 4) else "") + (str(r.depth_cm) if r.depth_cm > 0 else "dry")
        s.append(f'<text x="{cx:.1f}" y="{y(r.depth_cm) - 4:.1f}" text-anchor="middle" class="val">{_e(lab)}</text>')
        s.append(f'<text x="{cx:.1f}" y="{H - bot + 16}" text-anchor="middle" class="axis">{_e(r.key)}</text>')
        if r.disagree:
            s.append(f'<text x="{cx:.1f}" y="{H - bot + 32}" text-anchor="middle" class="warn">tiles &#8800; index</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root{--bg:#ffffff;--fg:#14182b;--dim:#52597a;--line:#d5d9ea;--card:#f6f8fd;--accent:#1f5fae;--warn:#8a4b00;--warnbg:#fff4e0}
@media (prefers-color-scheme: dark){:root{--bg:#0d1020;--fg:#e4e7f5;--dim:#9aa1c2;--line:#2a3158;--card:#141935;--accent:#74c0fc;--warn:#ffcf8a;--warnbg:#2b2110}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Segoe UI",system-ui,Arial,sans-serif}
main{max-width:900px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 2px}h2{font-size:17px;margin:30px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--dim);margin:0 0 18px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.sw{display:inline-block;width:14px;height:14px;border:1px solid #34406a;border-radius:3px;vertical-align:-2px;margin-right:6px}
.notice{background:var(--warnbg);color:var(--warn);border:1px solid var(--warn);border-radius:10px;padding:12px 16px;margin:16px 0;font-weight:600}
.dim{color:var(--dim)}.chart{width:100%;height:auto;max-width:760px}.chart .grid{stroke:var(--line);stroke-dasharray:3 3}
.chart text{fill:var(--fg);font:12px "Segoe UI",Arial,sans-serif}.chart .axis{fill:var(--dim)}.chart .val{font-weight:700}.chart .warn{fill:var(--warn);font-weight:700;font-size:11px}
img.shot{max-width:100%;border:1px solid var(--line);border-radius:8px}.legend span.t{display:inline-block;margin:2px 14px 2px 0}
.flag{color:var(--warn);font-weight:700}footer{margin-top:36px;color:var(--dim);font-size:12.5px}
@media print{.notice{break-inside:avoid}body{font-size:12.5px}main{padding:0}}
"""


def build_html(d: ReportData, png_data_uri: str | None = None) -> str:
    gen = d.generated or dt.datetime.now()
    rows = d.rows
    flooded = [r for r in rows if r.depth_cm >= 10]
    worst = max(rows, key=lambda r: r.depth_cm) if rows else None
    first = flooded[0] if flooded else None
    out = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
           f"<title>{_e(d.title)} - {_e(d.address)}</title><style>{CSS}</style></head><body><main>",
           f"<h1>{_e(d.title)}</h1><p class=\"sub\"><b>{_e(d.address)}</b><br>{_e(d.domain_label)} &middot; "
           f"{d.lat:.5f} N, {abs(d.lng):.5f} W &middot; generated {_e(gen.strftime('%Y-%m-%d %H:%M'))}</p>",
           f"<div class=\"notice\">{_e(NOT_VALIDATED)}</div>"]
    # summary
    if rows:
        if worst is not None and worst.depth_cm >= 10:
            summ = (f"Flooded (>= 10 cm) in <b>{len(flooded)} of {len(rows)}</b> modelled storms; first in <b>{_e(first.label)}</b>; deepest in "
                    f"<b>{_e(worst.label)}</b>: {_e(depth_text(worst.depth_cm, worst.tier, worst.approx))} "
                    f"<span class=\"dim\">({_e(worst.source)})</span>.")
        else:
            summ = (f"Below the 10 cm flood threshold in all {len(rows)} modelled storms "
                    "<span class=\"dim\">(this is a model result, not a guarantee of safety)</span>.")
        out.append(f"<div class=\"card\">{summ}</div>")
    # table
    out.append("<h2>Depth by storm</h2>")
    both = any(r.tile_cm is not None for r in rows)
    out.append("<table><thead><tr><th>Storm</th><th>Tier</th><th class=\"n\">Depth</th><th>Source of this depth</th>"
               + ("<th class=\"n\">Tiles</th><th class=\"n\">Index</th><th class=\"n\">Difference</th>" if both else "")
               + "<th class=\"n\">First hour</th><th class=\"n\">Hours flooded</th></tr></thead><tbody>")
    for r in rows:
        cur = " (on screen)" if d.current_storm == r.key else ""
        delta = ""
        if r.tile_cm is not None and r.index_cm is not None:
            dd = r.tile_cm - r.index_cm
            delta = f"{dd:+d} cm" + (' <span class="flag">tiers differ</span>' if r.disagree else "")
        out.append(f"<tr><td>{_e(r.label)}{_e(cur)}</td><td>{_swatch(r.tier)}{_e(TIER_NAMES[r.tier])}</td>"
                   f"<td class=\"n\">{_e(depth_text(r.depth_cm, r.tier, r.approx))}</td><td>{_e(r.source)}</td>"
                   + (f"<td class=\"n\">{'-' if r.tile_cm is None else str(r.tile_cm) + ' cm'}</td>"
                      f"<td class=\"n\">{'-' if r.index_cm is None else str(r.index_cm) + ' cm'}</td><td class=\"n\">{delta or '-'}</td>" if both else "")
                   + f"<td class=\"n\">{'-' if not r.first_hour else r.first_hour}</td><td class=\"n\">{'-' if not r.hours else r.hours}</td></tr>")
    out.append("</tbody></table>")
    if both:
        out.append("<p class=\"dim\">Where tiles and index disagree on the tier, both numbers are shown; the tiles are finer (about 5 m on the "
                   "ground) than the flow grid behind the index, and a campus / large footprint can differ strongly between them.</p>")
    out.append("<h2>Chart</h2><div class=\"card\">" + _chart(rows) + "</div>")
    out.append("<div class=\"legend\"><b>Tiers</b> &nbsp;" + "".join(
        f"<span class=\"t\">{_swatch(t)}{_e(TIER_NAMES[t])} <span class=\"dim\">{_e(TIER_RANGES[t])}</span></span>" for t in range(5))
        + "<br><span class=\"dim\">Palette: ColorBrewer YlGnBu, colour-blind safe; every value is also written as text.</span></div>")
    if png_data_uri or d.screenshot_name:
        src = png_data_uri or _e(d.screenshot_name)
        out.append(f"<h2>View in the viewer</h2><p><img class=\"shot\" alt=\"Screenshot of the viewer at this location\" src=\"{src}\"></p>")
    if d.detail:
        out.append("<h2>Property</h2><table><tbody>" + "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in d.detail) + "</tbody></table>")
    out.append("<h2>Where the numbers come from</h2><ul>" + "".join(f"<li><b>{_e(h)}</b>: {_e(t)}</li>" for h, t in d.sources) + "</ul>")
    if d.notes:
        out.append("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in d.notes) + "</ul>")
    out.append(f"<h2>Uncertainty</h2><p>{_e(UNCERTAINTY)}</p>")
    out.append(f"<div class=\"notice\">{_e(NOT_VALIDATED)}</div>")
    out.append(f"<footer>Generated {_e(gen.strftime('%Y-%m-%d %H:%M:%S'))} by {_e(d.viewer)}{(' ' + _e(d.build)) if d.build else ''}. "
               "Flood model by the Disaster Science Operations Center (DSOC) | AI Research Lab, Western Kentucky University. "
               "Indicative only; not an official flood map.</footer></main></body></html>")
    return "".join(out)


def png_bytes(rgb) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb[..., :3]).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")
    return (s[:n] or "location").strip("_")


def default_dir() -> Path:
    import os
    env = os.environ.get("BGV_REPORT_DIR")
    if env:
        return Path(env)
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    docs = home / "Documents"
    return (docs if docs.is_dir() else home) / "BG_Flood_Reports"


def write_report(folder, d: ReportData, png_rgb=None, now: dt.datetime | None = None) -> tuple[Path, Path | None]:
    """Write <folder>/risk_<address>_<time>.html (+ the PNG next to it, also embedded so the HTML stands alone)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = f"risk_{slug(d.address)}_{stamp}"
    png_path = None
    uri = None
    if png_rgb is not None:
        data = png_bytes(png_rgb)
        png_path = folder / (base + ".png")
        png_path.write_bytes(data)
        uri = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        d.screenshot_name = png_path.name
    html_path = folder / (base + ".html")
    tmp = html_path.with_suffix(".tmp")
    tmp.write_text(build_html(d, uri), encoding="utf-8")
    tmp.replace(html_path)
    return html_path, png_path
