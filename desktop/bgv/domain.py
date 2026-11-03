"""Domains: the area the viewer shows.  Two exist:

    city    Bowling Green: `<root>/layers/flow/<storm>` flow grids (about 5 m cells), `dem_bg.tif` terrain (FEET),
            `layers/buildings.geojson` + `layers/roads.geojson`.  The default, and unchanged by everything below.
    county  Warren County (about 52 x 60 km): `<root>/layers/flow_county/<storm>` flow grids built by
            scripts/build_flow_from_rasters.py (2631 x 3052 cells of about 20 m, source "rasters"), terrain from a GeoTIFF
            (`<root>/layers/dem_warren_5m.tif` by default, UTM 16N, METRES, 5 m cells; `--county-dem <path>` overrides), buildings
            and roads cut from the OpenStreetMap extract `<root>/kentucky-latest.osm.pbf` (cached, bgv.data.countyscene).

A `Domain` is a small immutable description (where the files are, camera defaults, labels).  `resolve(key, root, settings)`
checks that everything the domain needs exists and returns it, or raises `DomainUnavailable(reason)` - the reason is shown as
the tooltip of the disabled Domain menu entry.  `domain_infos` lists both domains for the menus.

County data order: REAL files in the data root first; only when the data root has no county flow grids at all the DEMO folder
(`settings["county_demo"]`, env BGV_COUNTY_DEMO, `--county-demo`, else `<cache>/county_demo`) is used - synthetic data written by
`python -m bgv --make-county-demo <folder>` (bgv.data.demo), always labelled DEMO DATA in the UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .data import flow as flowdata
from .data.terrain import TerrainSource, city_source
from .geo import GridFrame

CITY = "city"
COUNTY = "county"
KEYS = (CITY, COUNTY)
COUNTY_FLOW_SUBDIR = "flow_county"
COUNTY_DEM_NAME = "dem_warren_5m.tif"
DEMO_DEM_NAME = "dem_warren_demo.tif"
PBF_NAME = "kentucky-latest.osm.pbf"

CITY_PRESETS = {
    # name: (lng, lat, distance m, yaw deg, pitch deg)   (None lng = grid centre)
    "Overview": (None, None, 14000.0, 0.0, 55.0),
    "Downtown": (-86.4436, 36.9903, 1700.0, 20.0, 50.0),
    "Barren River": (-86.4305, 37.0040, 1800.0, 330.0, 45.0),
}
COUNTY_PRESETS = {
    "Overview": (None, None, None, 0.0, 55.0),                     # distance fitted to the grid (Domain.overview_dist)
    "Bowling Green": (-86.4436, 36.9903, 9000.0, 20.0, 50.0),
    "Downtown": (-86.4436, 36.9903, 1700.0, 20.0, 50.0),
    "Smiths Grove": (-86.2063, 37.0506, 4500.0, 20.0, 50.0),
}


class DomainUnavailable(Exception):
    """The requested domain cannot be shown; `str(e)` is the explanation for the user."""


@dataclass(frozen=True)
class Domain:
    key: str
    label: str                           # "Bowling Green" / "Warren County"
    flow_root: Path                      # folder holding layers/<flow_subdir>/<storm>
    flow_subdir: str
    storms: tuple
    terrain: TerrainSource
    root: Path                           # the data root (OSM extract, city geojson)
    buildings_geojson: Path | None = None    # city: layers/buildings.geojson
    roads_geojson: Path | None = None
    pbf: Path | None = None                  # county: the OpenStreetMap extract
    demo: bool = False                   # flow grids / terrain come from the demo folder (synthetic)
    presets: dict = field(default_factory=dict)
    default_exag: float = 2.0
    default_haze: float = 1.0e-5
    notice: str = ""

    # ---------------------------------------------------------------------------------- camera
    def overview_dist(self, frame: GridFrame | None) -> float:
        if self.key == CITY or frame is None:
            return 14000.0
        return float(max(frame.size_m) * 1.1)

    def overview(self, frame: GridFrame | None) -> dict:
        return dict(x=0.0, y=0.0, dist=self.overview_dist(frame), yaw=0.0, pitch=55.0)

    def preset_view(self, frame: GridFrame, name: str):
        lng, lat, dist, yaw, pitch = self.presets[name]
        if dist is None:
            dist = self.overview_dist(frame)
        if lng is None:
            return 0.0, 0.0, dist, yaw, pitch
        x, y = frame.lnglat_to_world(lng, lat)
        return x, y, dist, yaw, pitch

    # ---------------------------------------------------------------------------------- data
    def open_storm(self, name: str):
        return flowdata.Storm.open(self.flow_root, name, self.flow_subdir)

    def storm_label(self, name: str) -> str:
        if self.key == CITY:
            return {"5yr": "5-year storm", "10yr": "10-year storm", "25yr": "25-year storm", "50yr": "50-year storm",
                    "100yr": "100-year storm", "2yr": "2-year storm"}.get(name, name)
        return flowdata.storm_label(self.flow_root, name, self.flow_subdir)

    def scene_paths(self, frame: GridFrame, progress=None):
        """(buildings geojson, roads geojson, info dict): the city's files, or the county's, cut from the OSM extract on
        first use (about a minute, cached in <cache>/osm)."""
        if self.key == CITY:
            return self.buildings_geojson, self.roads_geojson, {}
        if self.pbf is None:
            raise FileNotFoundError(f"no OpenStreetMap extract for the county: put {PBF_NAME} in {self.root}")
        from .data import countyscene
        sc = countyscene.ensure_osm_scene(self.pbf, frame, progress)
        return sc.buildings_path, sc.roads_path, sc.info

    @property
    def title(self) -> str:
        return self.label + (" (DEMO DATA)" if self.demo else "")


@dataclass
class DomainInfo:
    key: str
    label: str
    available: bool
    reason: str = ""                     # why it is not available (tooltip), or a hint when it is
    demo: bool = False
    storms: int = 0


# -------------------------------------------------------------------------------------------------- locations
def county_demo_root(settings: dict | None = None) -> Path:
    s = (settings or {}).get("county_demo")
    if s:
        return Path(s)
    if os.environ.get("BGV_COUNTY_DEMO"):
        return Path(os.environ["BGV_COUNTY_DEMO"])
    return config.cache_dir() / "county_demo"


def county_dem_path(root, settings: dict | None = None) -> Path:
    p = (settings or {}).get("county_dem")
    return Path(p) if p else Path(root) / "layers" / COUNTY_DEM_NAME


def find_pbf(root, settings: dict | None = None) -> Path | None:
    p = (settings or {}).get("county_pbf")
    if p and Path(p).is_file():
        return Path(p)
    root = Path(root)
    if (root / PBF_NAME).is_file():
        return root / PBF_NAME
    try:
        for f in sorted(root.glob("*.osm.pbf")):
            return f
    except OSError:
        pass
    return None


# -------------------------------------------------------------------------------------------------- resolve
def _city(root) -> Domain:
    root = Path(root)
    storms = tuple(flowdata.available_storms(root, "flow"))
    if not storms:
        raise DomainUnavailable(f"no storms found in {root / 'layers' / 'flow'}")
    return Domain(CITY, "Bowling Green", root, "flow", storms, city_source(root), root,
                  buildings_geojson=root / "layers" / "buildings.geojson", roads_geojson=root / "layers" / "roads.geojson",
                  presets=CITY_PRESETS, default_exag=2.0, default_haze=1.0e-5)


def _county(root, settings) -> Domain:
    root = Path(root)
    real = tuple(flowdata.available_storms(root, COUNTY_FLOW_SUBDIR))
    pbf = find_pbf(root, settings)
    if real:
        dem = county_dem_path(root, settings)
        if not dem.is_file():
            raise DomainUnavailable(
                f"county flow grids found in {root / 'layers' / COUNTY_FLOW_SUBDIR}, but the county terrain is missing: {dem}. "
                f"Download dem_warren_5m.tif from Vista into {root / 'layers'} (see COUNTY_RUN.md) or pass --county-dem <file>.")
        return Domain(COUNTY, "Warren County", root, COUNTY_FLOW_SUBDIR, real,
                      TerrainSource(dem, "metres", True, f"county DEM {dem.name}"), root, pbf=pbf,
                      presets=COUNTY_PRESETS, default_exag=4.0, default_haze=0.45e-5)
    demo = county_demo_root(settings)
    dstorms = tuple(flowdata.available_storms(demo, COUNTY_FLOW_SUBDIR))
    ddem = demo / "layers" / DEMO_DEM_NAME
    if dstorms and ddem.is_file():
        return Domain(COUNTY, "Warren County", demo, COUNTY_FLOW_SUBDIR, dstorms,
                      TerrainSource(ddem, "metres", True, "synthetic demo DEM"), root, pbf=pbf, demo=True,
                      presets=COUNTY_PRESETS, default_exag=4.0, default_haze=0.45e-5,
                      notice="Showing SYNTHETIC DEMO DATA for the county (no real county flow grids found).")
    raise DomainUnavailable(
        f"No county data yet: expected flow grids in {root / 'layers' / COUNTY_FLOW_SUBDIR}\\<storm> and the terrain "
        f"{county_dem_path(root, settings)} (see COUNTY_RUN.md, 'Afterwards'). "
        f"To try the county view with synthetic data run: python -m bgv --make-county-demo <folder>")


def resolve(key: str, root, settings: dict | None = None) -> Domain:
    """The domain `key` ('city' | 'county') for the data root, or DomainUnavailable(reason)."""
    key = (key or CITY).lower()
    if key == CITY:
        return _city(root)
    if key == COUNTY:
        return _county(root, settings or {})
    raise DomainUnavailable(f"unknown domain {key!r} (use city or county)")


def resolve_or_city(key: str, root, settings: dict | None = None):
    """(Domain, notice): the requested domain, or the city plus an explanation when it is unavailable.  The notice is only
    ever a FALLBACK explanation ('' when the requested domain is shown; demo data speaks through its DEMO DATA badges)."""
    try:
        d = resolve(key, root, settings)
        return d, ""
    except DomainUnavailable as e:
        if (key or CITY).lower() == CITY:
            raise
        return _city(root), f"{key} view unavailable, showing the city: {e}"


def domain_infos(root, settings: dict | None = None) -> list[DomainInfo]:
    out = []
    for key, label in ((CITY, "Bowling Green"), (COUNTY, "Warren County")):
        try:
            d = resolve(key, root, settings)
            out.append(DomainInfo(key, d.label + (" - DEMO DATA" if d.demo else ""), True, d.notice, d.demo, len(d.storms)))
        except DomainUnavailable as e:
            out.append(DomainInfo(key, label, False, str(e)))
    return out


def pick_storm(settings: dict | None, key: str, avail, explicit: str | None = None) -> str:
    """The storm to open first in a domain: the explicit one (CLI), the one remembered for that domain, the city's legacy
    `storm` setting, 25yr, else the first available."""
    avail = list(avail)
    if explicit and explicit in avail:
        return explicit
    s = settings or {}
    cands = [((s.get("domain_views") or {}).get(key) or {}).get("storm")]
    if key == CITY:
        cands.append(s.get("storm"))
    cands.append("25yr" if key == CITY else "100yr")
    for c in cands:
        if c and c in avail:
            return c
    return avail[0]


def apply_cli(settings: dict, args) -> dict:
    """CLI flags -> settings (so they are remembered): --domain, --county-dem, --county-demo, --county-pbf."""
    for flag, key in (("domain", "domain"), ("county_dem", "county_dem"), ("county_demo", "county_demo"),
                      ("county_pbf", "county_pbf")):
        v = getattr(args, flag, None)
        if v:
            settings[key] = str(v).lower() if key == "domain" else str(v)
    return settings
