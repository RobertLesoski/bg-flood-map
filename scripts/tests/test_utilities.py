"""build_utilities on a synthetic PBF + synthetic depth raster.   python -m pytest scripts/tests/test_utilities.py -q"""
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "desktop" / "tests"))
import build_utilities as bu  # noqa: E402
import pbf_writer as pw       # noqa: E402

LON, LAT = -86.45, 36.98
BOX = (-86.5, 36.95, -86.4, 37.02)


def make_pbf(path):
    b1 = pw.Block()
    nodes = [(1, LON, LAT, {"man_made": "water_tower", "name": "Test Tower", "operator": "BGMU"}),
             (2, LON + 0.01, LAT, {"power": "generator", "generator:source": "hydro", "name": "Hydro One"}),
             (3, LON + 0.02, LAT, {"man_made": "storage_tank", "content": "oil"}),
             (4, LON + 0.03, LAT, {"waterway": "weir", "name": "Old Weir"}),
             (5, LON + 0.05, LAT),
             (6, LON + 0.051, LAT),
             (7, LON + 0.0, LAT + 0.01), (8, LON + 0.001, LAT + 0.01), (9, LON + 0.001, LAT + 0.011), (10, LON, LAT + 0.011)]
    b1.add_nodes(nodes)
    b2 = pw.Block()
    b2.add_ways([(100, {"power": "line", "voltage": "69000"}, [5, 6]),
                 (101, {"power": "substation", "name": "Sub A"}, [7, 8, 9, 10, 7]),
                 (102, {"highway": "residential"}, [5, 6])])
    return pw.write_pbf(path, [b1, b2])


def test_classify():
    assert bu.classify({"man_made": "storage_tank", "content": "water"}) == "water_tank"
    assert bu.classify({"man_made": "storage_tank", "content": "oil"}) is None
    assert bu.classify({"power": "plant", "plant:source": "hydro"}) == "hydro_plant"
    assert bu.classify({"power": "plant", "plant:source": "gas"}) == "power_plant"
    assert bu.classify({"power": "tower"}) is None and bu.classify({"waterway": "dam"}) == "dam"


def test_read_utilities_from_a_synthetic_pbf(tmp_path):
    p = make_pbf(tmp_path / "t.pbf")
    feats = bu.read_utilities(str(p), BOX, log=None)
    by = {f["osm_id"]: f for f in feats}
    assert set(by) == {"n1", "n2", "n4", "w100", "w101"}, set(by)
    assert by["n1"]["props"]["type"] == "water_tower" and by["n1"]["props"]["operator"] == "BGMU"
    assert by["n2"]["props"]["type"] == "hydro_plant" and by["n4"]["props"]["type"] == "weir"
    assert by["w100"]["kind"] == "line" and by["w101"]["kind"] == "area" and by["w101"]["props"]["type"] == "substation"


def _raster_and_depth():
    x0, y0 = bu.to_merc(LON - 0.01, LAT + 0.02)
    rs = bu.Raster(float(x0), float(y0), 6.25, 1200, 800)
    arr = np.zeros(rs.width * rs.height, np.uint16)
    return rs, arr


def test_exposure_points_areas_and_line_share(tmp_path):
    feats = bu.read_utilities(str(make_pbf(tmp_path / "t.pbf")), BOX, log=None)
    rs, arr = _raster_and_depth()
    by = {f["osm_id"]: i for i, f in enumerate(feats)}
    img = arr.reshape(rs.height, rs.width)
    x, y = bu.to_merc(LON, LAT)
    r, c = int((rs.y0 - y) / 6.25), int((x - rs.x0) / 6.25)
    img[r, c] = 45                                             # the tower's pixel: 45 cm
    xs, ys = bu.to_merc(LON + 0.0005, LAT + 0.0105)
    img[int((rs.y0 - ys) / 6.25), int((xs - rs.x0) / 6.25)] = 12   # one pixel inside the substation
    x5, y5 = bu.to_merc(LON + 0.05, LAT)
    x6, _ = bu.to_merc(LON + 0.051, LAT)
    c5, c6, r5 = int((x5 - rs.x0) / 6.25), int((x6 - rs.x0) / 6.25), int((rs.y0 - y5) / 6.25)
    half = (c5 + c6) // 2
    img[r5, c5:half] = 35                                      # first half of the line is wet (35 cm)
    img[r5, 0] = 65535
    d, f10, f30 = bu.exposure(feats, rs, {"ev6": arr, "100yr": arr.copy()}, ["ev6", "100yr"])
    assert d[by["n1"], 0] == 45 and d[by["n1"], 1] == 45
    assert d[by["w101"], 0] == 12
    assert d[by["n2"], 0] == 0
    assert d[by["w100"], 0] == 35 and 35 <= f10[by["w100"], 0] <= 65 and f30[by["w100"], 0] == f10[by["w100"], 0]
    assert bu.first_flooded(d[by["n1"]], ["ev6", "100yr"]) == "100yr"      # design storms only
    assert bu.first_flooded(d[by["n2"]], ["ev6", "100yr"]) is None
