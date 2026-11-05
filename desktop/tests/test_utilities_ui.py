"""Utilities layer data + search (no GPU)."""
import numpy as np

from bgv.data import utilities as ud


def doc():
    def pt(t, name, lng, lat, d, first=None, city=True, op=""):
        return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lng, lat]},
                "properties": {"type": t, "name": name, "operator": op, "osm_id": "n1", "lng": lng, "lat": lat, "in_city": city, "d": d, "first": first}}
    line = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[-86.5, 37.0], [-86.4, 37.0]]},
            "properties": {"type": "power_line", "voltage": "69000", "osm_id": "w2", "in_city": False, "d": [5, 40], "first": "100yr", "f10": [0, 30], "f30": [0, 10]}}
    return {"metadata": {"storms": ["ev6", "100yr"], "counts": {"water_tower": 2, "substation": 1, "power_line": 1}},
            "features": [pt("water_tower", "Reservoir Hill Water Tower", -86.45, 36.98, [0, 20], "100yr", op="BGMU"),
                         pt("water_tower", "", -86.3, 36.9, [0, 0], None, city=False),
                         pt("substation", "East Sub", -86.4, 36.99, [12, 12]), line]}


def test_load_search_and_depth():
    us = ud.UtilitySet(doc())
    assert len(us.features) == 4 and us.features[3].is_line
    assert us.depth_cm(us.features[0], "100yr") == 20 and us.depth_cm(us.features[0], "50yr") is None
    assert [f.i for f in us.search("water tower")][:1] == [0]          # the named one first
    assert us.search("1309 scottsville rd") == [] and us.search("123") == []   # an address is not a utility
    assert [f.name for f in us.search("east sub")] == ["East Sub"]
    assert len(us.search("substation")) == 1 and us.search("tbl 123 main st") == []


def test_domain_view_city_keeps_city_features_only():
    class Fr:
        def merc_to_world(self, x, y):
            return x * 0.001, y * 0.001

    us = ud.UtilitySet(doc())
    city = ud.build_view(us, "city", Fr())
    county = ud.build_view(us, "county", Fr())
    assert len(city.ids) == 2 and len(county.ids) == 3 and len(county.line_ids) == 1 and len(city.line_ids) == 0
    assert county.line_off.tolist() == [0, 2] and county.line_xyz.shape == (2, 3) and np.isfinite(county.xy).all()


def test_all_groups_have_distinct_shape_colour_pairs():
    pairs = {(v[2], v[3]) for v in ud.GROUPS.values()}
    assert len(pairs) == len(ud.GROUPS)
    assert set(ud.GROUP_OF_TYPE) == {t for v in ud.GROUPS.values() for t in v[1]}
