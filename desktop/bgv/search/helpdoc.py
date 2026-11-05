"""Content of the Help / About window (pure python, unit-tested): topics, shortcut list, a tiny markdown parser for
STORM_DATA_SOURCES.md, the data-file status table and the text search.

    topics() -> [Topic]                     built-in help topics (title + blocks)
    SHORTCUTS -> [(group, [(keys, what)])]
    storm_sources(repo_root) -> (blocks | None, path, message)    renders STORM_DATA_SOURCES.md when present, else a placeholder
    parse_markdown(text) -> [Block]         kinds: h1 h2 h3 p bullet code table hr
    search(topics, query) -> [(Topic, [Block])]
    data_files(root, domain_key, settings) -> [(label, path, found, note)]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

STORM_SOURCES_NAME = "STORM_DATA_SOURCES.md"
PLACEHOLDER = ("Per-storm data sources are not documented yet. When the file STORM_DATA_SOURCES.md exists in the repository root "
               "(next to desktop/), this page shows it, one section per storm. Until then, open the storm's own note under "
               "Help > Data source details (the badge in the top bar) for the source of the storm on screen.")


@dataclass
class Block:
    kind: str
    text: str = ""
    rows: list = field(default_factory=list)       # table rows (list of list of str)


@dataclass
class Topic:
    key: str
    title: str
    blocks: list


def parse_markdown(text: str) -> list[Block]:
    """Headings, bullets, tables, fenced code, rules and paragraphs.  Inline markup (**bold**, `code`, links) is flattened to text."""
    out: list[Block] = []
    para: list[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0

    def flush():
        if para:
            out.append(Block("p", _inline(" ".join(para))))
            para.clear()
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            out.append(Block("code", "\n".join(code)))
            i = j + 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            flush()
            out.append(Block("h%d" % len(m.group(1)), _inline(m.group(2))))
        elif re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", s):
            flush()
            out.append(Block("hr"))
        elif s.startswith("|") and s.endswith("|") and s.count("|") >= 2:
            flush()
            rows = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [_inline(c.strip()) for c in lines[j].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip()) or not any(c.strip() for c in cells):
                    rows.append(cells)
                j += 1
            out.append(Block("table", rows=rows))
            i = j
            continue
        elif re.match(r"^[-*+]\s+", s):
            flush()
            out.append(Block("bullet", _inline(re.sub(r"^[-*+]\s+", "", s))))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return out


def _inline(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s)
    s = re.sub(r"(?<!\w)([*_])(.+?)\1(?!\w)", r"\2", s)
    return s.replace("`", "")


def repo_root(source_root=None) -> Path:
    """The repository root = the folder that contains `desktop/` (parent of the folder that holds the bgv package)."""
    from .. import config
    return Path(source_root or config.source_root()).parent


def storm_sources(root=None):
    """(blocks or None, path, message).  `root` = repository root (default: derived from the package location)."""
    p = (Path(root) if root else repo_root()) / STORM_SOURCES_NAME
    if not p.is_file():
        return None, p, PLACEHOLDER
    try:
        return parse_markdown(p.read_text(encoding="utf-8", errors="replace")), p, ""
    except OSError as e:
        return None, p, f"{STORM_SOURCES_NAME} could not be read ({e}). " + PLACEHOLDER


SHORTCUTS = [
    ("Search and reports", [
        ("Ctrl+F  or  /", "Search an address (typeahead; Enter jumps to the first result, Up / Down choose)"),
        ("M", "Show / hide the critical-facility markers"),
        ("L", "Open / close the roads-at-risk list"),
        ("Ctrl+P", "Save a risk report (HTML + PNG) for the searched address or the selected building"),
        ("Esc", "Clear the search box / close the result card (then deselect, as before)"),
    ]),
    ("Mouse", [
        ("Left drag", "Pan the map"),
        ("Right / middle drag", "Orbit (rotate and tilt)"),
        ("Mouse wheel", "Zoom toward the cursor"),
        ("Click a building", "Select it (card + highlight)"),
        ("Click a facility marker", "Show its flood depths and the first storm that floods it"),
        ("Click the compass", "Face north"),
        ("Click / drag in the timeline chart", "Seek in time"),
    ]),
    ("Keyboard", [
        ("W A S D  /  arrows", "Move the camera (Shift = faster)"),
        ("Q  E", "Rotate left / right"),
        ("R  F", "Tilt (F flies to the selection when one exists)"),
        ("+  -", "Zoom in / out"),
        ("Space", "Play / pause"),
        ("[  ]", "Step one hour back / forward"),
        ("1 .. 9", "Pick the n-th storm"),
        ("Home", "Overview (reset view)"),
        ("H", "Hide / show the interface"),
        ("F1", "This window"),
        ("F3", "Performance overlay"),
        ("F11", "Fullscreen"),
        ("F12", "Screenshot (Shift+F12: without the interface)"),
    ]),
]


def _p(t):
    return Block("p", t)


def _b(t):
    return Block("bullet", t)


def topics() -> list[Topic]:
    return [
        Topic("start", "Quick start", [
            Block("h2", "Quick start"),
            _p("Pick a storm with the number keys or in the sidebar, drag the timeline to watch the water rise, and click a building to see "
               "its depth. Everything on screen is a model result for one storm at one hour."),
            _b("Search an address: press Ctrl+F, type a number and a street, press Enter."),
            _b("Critical facilities (hospitals, schools, fire, police, shelters ...): press M or use Find > Critical facilities. A ring "
               "around a marker means water of 10 cm or more reaches it in the storm on screen."),
            _b("Roads at risk: press L for the list of roads with water, sortable by flooded share, depth, length and evacuation flag."),
            _b("Report: Ctrl+P writes a self-contained HTML report and a PNG of the view for the searched address or selected building."),
            _b("Switch between the city and the whole county in the Domain menu."),
        ]),
        Topic("search", "Searching for an address", [
            Block("h2", "Searching for an address"),
            _p("Type a house number and the start of a street ('1309 scott'); street suffixes and directions are abbreviated for you "
               "('Road' = 'Rd', 'West' = 'W'). A street name alone lists the matching streets, a number alone lists addresses with that "
               "number. The index holds the addresses with a building footprint and the parcels without one ('lot')."),
            _p("The result card shows, for every storm, the depth and tier at that address and which source the number comes from. For the "
               "county two sources exist: the solver depth tiles (finer cells) and the index built from the flow grid; both are shown with "
               "their difference. Where they disagree on the tier the row is flagged."),
            _p("A building's depth is the MAXIMUM over the model cells its footprint touches, over all hours of the storm. That is a screening "
               "value, not the depth at the front door."),
        ]),
        Topic("numbers", "Reading depths, tiers and sources", [
            Block("h2", "Reading depths, tiers and sources"),
            _b("Tiers: shallow 10-30 cm, moderate 30-60 cm, deep 60-120 cm, severe 120 cm and more. Below 10 cm counts as not flooded."),
            _b("'solver depth (tiles, 6.25 m; verification pending)': read from the depth tiles of the solver output. The verification of these "
               "tiles against the raw solver rasters is not finished, so treat the numbers as provisional."),
            _b("'index (grid)': from the address index, built from the storm flow grids (county: 20 m cells; city: about 5 m cells). The "
               "city has no depth tiles yet."),
            _b("'Preview from depth tiers' (city storms): the source only knows the tier, so 'severe' means 120 cm or more, not a depth."),
            _b("The viewer shows the state at the selected hour; the search card, the facility list and the road list use the MAXIMUM over "
               "the whole storm."),
        ]),
        Topic("facilities", "Critical facilities", [
            Block("h2", "Critical facilities"),
            _p("Hospitals, clinics, fire stations, police, schools, shelters and assembly points, community centres and town halls from "
               "OpenStreetMap. Marker colour and shape show the type (shape is the second cue for colour-blind readers). A ring in the tier "
               "colour marks a facility flooded (10 cm or more) in the storm on screen."),
            _p("Click a marker or a row of the facilities list for the first storm that floods it and the depth in every storm. In the county "
               "the depth is the tile cell under the facility point; the index value (max over the campus outline on the flow grid) and the "
               "deepest water within 30 m of the point are shown next to it, because a large campus can be wet at one corner and dry at its "
               "centre point."),
        ]),
        Topic("roads", "Roads at risk", [
            Block("h2", "Roads at risk"),
            _p("The list shows the OpenStreetMap road ways that have water in the storm on screen. Flooded % is the share of the way's length "
               "with 10 cm or more (sampled every 10 m); max depth is the deepest sample. Values are maxima over the storm, not the hour "
               "on screen. Click a row to fly to the road; it is outlined on the map."),
            _p("Evacuation flag: 'OSM tag' = the way carries an evacuation-route tag in OpenStreetMap; 'major' = motorway, trunk or primary "
               "road, treated as an evacuation corridor. Neither is an official evacuation plan."),
        ]),
        Topic("report", "Risk report", [
            Block("h2", "Risk report"),
            _p("Ctrl+P (or Find > Save risk report) writes risk_<address>_<time>.html and a PNG of the view to Documents\\BG_Flood_Reports (set "
               "BGV_REPORT_DIR to change it). The HTML is self-contained (no internet, nothing to install): depth per storm, tiers with a "
               "colour-blind-safe palette, the source of every number, the uncertainty note and the notice that the results have not been "
               "validated against observed floods."),
        ]),
        Topic("honesty", "Limits and honesty notes", [
            Block("h2", "Limits and honesty notes"),
            _b("Model results, not observations: not validated against observed floods."),
            _b("Velocity is derived from the water-surface slope (Manning), not solver output."),
            _b("Indicative only; not an official flood map, not for insurance or engineering decisions."),
            _b("Dry in the model means below 10 cm, not safe."),
        ]),
        Topic("missing", "When something is missing", [
            Block("h2", "When something is missing"),
            _p("Every feature says what is missing and where it looked instead of failing silently: the address index "
               "(layers\\address_index.json.gz, layers\\county\\...), the facilities file (infrastructure.geojson), the road lists "
               "(layers\\roads_flooded\\<domain>\\<storm>.json), the depth tiles (layers\\cog\\county\\<storm>\\max_depth.tif, folder "
               "configurable with BGV_TILES_DIR or the tiles_dir setting) and STORM_DATA_SOURCES.md. The About tab lists each file with its "
               "status and path, and the log file location."),
        ]),
    ]


def search(tops: list[Topic], query: str):
    """[(topic, matching blocks)] for a text query: every word must occur in the block (case-insensitive)."""
    words = query.lower().split()
    if not words:
        return []
    res = []
    for t in tops:
        hit = []
        for b in t.blocks:
            txt = (b.text + " " + " ".join(" ".join(r) for r in b.rows)).lower()
            if all(w in txt for w in words):
                hit.append(b)
        if hit:
            res.append((t, hit))
    return res


def search_shortcuts(query: str):
    words = query.lower().split()
    out = []
    for g, items in SHORTCUTS:
        for k, d in items:
            if all(w in (k + " " + d + " " + g).lower() for w in words):
                out.append((g, k, d))
    return out


def data_files(root, domain_key: str, settings: dict | None = None):
    """[(label, path, found, note)] of the files the search features read for a domain."""
    from . import addrindex, facilities, tiles
    root = Path(root)
    sub = "county" if domain_key == "county" else ""
    rows = []

    def add(label, p, note=""):
        p = Path(p)
        rows.append((label, str(p), p.exists(), note))
    add("Address index", addrindex.index_path(root, domain_key))
    add("Critical facilities", facilities.facilities_path(root, domain_key))
    add("Flooded roads (folder)", root / "layers" / "roads_flooded" / domain_key)
    add("Road geometry", root / "layers" / "county" / "roads.geojson.gz", "way geometry for zoom-to-road (covers the city too)")
    td = tiles.tiles_root(root, settings) / domain_key
    add("Depth tiles (folder)", td, "county only; the city has none yet" if domain_key != "county" else "max_depth.tif per storm")
    _ = sub
    add("Storm data sources", repo_root() / STORM_SOURCES_NAME, "written by the data team; placeholder shown until present")
    return rows
