r"""
geodata.py - where the data lives and how footprints, parcels and domains are loaded (shared by the data builders).

  APP_DEFAULT                 the folder the web server serves (layers/, kentucky-latest.osm.pbf, cache/)
  get_domain(app, key)        Domain for 'city' (layers/flow, Bowling Green) or 'county' (layers/flow_county, Warren County)
  load_parcels(app)           all_parcels.geojson + the assessed value of flood_damage.geojson (same feature order)
  join_points_to_parcels()    point-in-polygon (shapely STRtree) -> parcel index per point, -1 none
  load_city_buildings(app)    layers/buildings.geojson -> BuildingSet (id = 0-based feature index)
  write_json_gz / write_json  atomic, reproducible writers

No network access anywhere in here.
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import shape

APP_DEFAULT = "C:/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map"
PBF_NAME = "kentucky-latest.osm.pbf"
COUNTY_UTM_BOX = (524851.0, 4056081.0, 576871.0, 4116296.0)      # EPSG:32616 xmin, ymin, xmax, ymax of the county model grid


@dataclass
class Domain:
    key: str                      # 'city' | 'county'
    label: str
    flow_dir: str                 # <app>/layers/flow | layers/flow_county
    out_dir: str                  # <app>/layers      | layers/county
    bbox: tuple                   # lng/lat (west, south, east, north) of the model domain
    app: str = APP_DEFAULT

    def layer(self, name):
        return os.path.join(self.out_dir, name)


def _utm_box_to_lnglat(box, epsg=32616):
    from rasterio.warp import transform
    xs = [box[0], box[2], box[0], box[2]]
    ys = [box[1], box[1], box[3], box[3]]
    lng, lat = transform("EPSG:%d" % epsg, "EPSG:4326", xs, ys)
    return (min(lng), min(lat), max(lng), max(lat))


def city_bbox(app):
    """Bowling Green model domain = bounds of dem_bg.tif (UTM 16N) in lng/lat; falls back to the union of the flow grids."""
    p = os.path.join(app, "dem_bg.tif")
    if os.path.isfile(p):
        try:
            import rasterio
            with rasterio.open(p) as d:
                b = d.bounds
                return _utm_box_to_lnglat((b.left, b.bottom, b.right, b.top), d.crs.to_epsg() or 32616)
        except Exception:                                      # noqa: BLE001
            pass
    from flowquery import load_storms
    bs = [s.grid.lnglat_bounds() for s in load_storms(os.path.join(app, "layers", "flow"), log=None)]
    if not bs:
        raise FileNotFoundError("no dem_bg.tif and no city flow grids under %s" % app)
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def get_domain(app, key):
    app = str(app)
    if key == "city":
        return Domain("city", "Bowling Green", os.path.join(app, "layers", "flow"), os.path.join(app, "layers"),
                      city_bbox(app), app)
    if key == "county":
        return Domain("county", "Warren County", os.path.join(app, "layers", "flow_county"),
                      os.path.join(app, "layers", "county"), _utm_box_to_lnglat(COUNTY_UTM_BOX), app)
    raise ValueError("domain must be 'city' or 'county', got %r" % (key,))


def cache_dir(app):
    d = os.path.join(app, "cache", "bgscripts")
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------------ writers
def _atomic(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path, obj):
    _atomic(path, dumps(obj))
    return os.path.getsize(path)


def write_json_gz(path, obj, level=9):
    raw = dumps(obj)
    import io
    bio = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=level, fileobj=bio, mtime=0) as g:
        g.write(raw)
    _atomic(path, bio.getvalue())
    return os.path.getsize(path), len(raw)


def read_json_any(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ parcels / buildings
@dataclass
class ParcelSet:
    geoms: np.ndarray             # shapely polygons (lng/lat)
    pva: list
    address: list                 # SiteAddress / ADDRESS ('' when none)
    land_use: list
    value: np.ndarray             # assessed value USD (prop_value_usd), 0 unknown
    sale_price: np.ndarray
    n: int = 0
    tree: object = field(default=None, repr=False)

    def __post_init__(self):
        self.n = len(self.geoms)


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _valid(g):
    return g if g.is_valid else shapely.make_valid(g)


def load_parcels(app, log=print):
    base = os.path.join(app, "layers")
    pa = os.path.join(base, "all_parcels.geojson")
    if not os.path.isfile(pa):
        pa = os.path.join(base, "parcels.geojson")
    with open(pa, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    vals = np.zeros(len(feats))
    fd = os.path.join(base, "flood_damage.geojson")
    if os.path.isfile(fd):
        try:
            with open(fd, encoding="utf-8") as f:
                dfe = json.load(f)["features"]
            if len(dfe) == len(feats) and all((a["properties"].get("PVA_PARCEL") == b["properties"].get("PVA_PARCEL"))
                                              for a, b in zip(feats, dfe)):
                vals = np.array([_num(x["properties"].get("prop_value_usd")) for x in dfe])
            else:                                               # different order: join by parcel id
                by = {}
                for x in dfe:
                    p = x["properties"]
                    by.setdefault(p.get("PVA_PARCEL"), _num(p.get("prop_value_usd")))
                vals = np.array([by.get(f["properties"].get("PVA_PARCEL"), 0.0) for f in feats])
        except Exception as e:                                  # noqa: BLE001
            if log:
                log("  (flood_damage.geojson not usable for assessed values: %s)" % e)
    geoms, pva, addr, lu, sale = [], [], [], [], []
    for f in feats:
        p = f.get("properties") or {}
        try:
            g = _valid(shape(f["geometry"]))
        except Exception:                                       # noqa: BLE001
            g = shapely.Polygon()
        geoms.append(g)
        pva.append(str(p.get("PVA_PARCEL") or ""))
        addr.append(str(p.get("SiteAddress") or p.get("ADDRESS") or "").strip())
        lu.append(str(p.get("LAND_USE") or ""))
        sale.append(_num(p.get("sale_price")))
    ps = ParcelSet(np.array(geoms, dtype=object), pva, addr, lu, vals, np.array(sale))
    ps.tree = shapely.STRtree(ps.geoms)
    return ps


def join_points_to_parcels(points, parcels):
    """Index of the parcel containing each shapely Point (the smallest one when several overlap), -1 when none."""
    out = np.full(len(points), -1, dtype=np.int64)
    if len(points) == 0 or parcels.n == 0:
        return out
    qi, ti = parcels.tree.query(np.asarray(points, dtype=object), predicate="within")
    if qi.size == 0:
        return out
    area = shapely.area(parcels.geoms[ti])
    order = np.lexsort((area, qi))                      # per point: smallest parcel first
    qi, ti = qi[order], ti[order]
    first = np.concatenate([[True], qi[1:] != qi[:-1]])
    out[qi[first]] = ti[first]
    return out


@dataclass
class BuildingSet:
    ids: list                     # city: feature index; county: OSM way id
    geoms: np.ndarray             # shapely polygons (lng/lat)
    address: list                 # the source's own address ('' if none)
    land_use: list
    props: list                   # original property dicts
    domain: str = "city"


def load_city_buildings(app, log=print):
    p = os.path.join(app, "layers", "buildings.geojson")
    with open(p, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    geoms, addr, lu, props = [], [], [], []
    for f in feats:
        pr = f.get("properties") or {}
        try:
            g = _valid(shape(f["geometry"]))
        except Exception:                                       # noqa: BLE001
            g = shapely.Polygon()
        geoms.append(g)
        addr.append(str(pr.get("ADDRESS") or "").strip())
        lu.append(str(pr.get("LAND_USE") or ""))
        props.append(pr)
    return BuildingSet(list(range(len(feats))), np.array(geoms, dtype=object), addr, lu, props, "city")


def in_box(lng, lat, box):
    return (lng >= box[0]) & (lng <= box[2]) & (lat >= box[1]) & (lat <= box[3])
