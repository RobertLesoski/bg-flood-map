"""Tests for build_storm_blurbs.py (layers/storm_blurbs.json, the storm picker texts).  python -m pytest scripts/tests/test_storm_blurbs.py -q"""
import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import build_storm_blurbs as bb  # noqa: E402
import build_storm_info as bsi  # noqa: E402


APP = Path("C:/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map")


class BlurbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = bb.build(str(APP))["storms"]

    def test_twelve_with_fields(self):
        self.assertEqual(list(self.d), bsi.ORDER)
        for k, s in self.d.items():
            for f in ("short", "long", "total", "peak", "duration", "honesty", "model_run", "name"):
                self.assertTrue(s[f], (k, f))
            self.assertLess(len(s["short"]), 80, k)

    def test_design_facts(self):
        for k, depth in bsi.DESIGN.items():
            self.assertIn("%.1f in" % depth, self.d[k]["total"])
            self.assertIn("%d%%" % bb.CHANCE[k], self.d[k]["long"])
        self.assertIn("1% chance", self.d["100yr"]["long"])

    def test_event_numbers_match_data(self):
        for e in json.load(open(SCRIPTS / "storms_out" / "storms.json", encoding="utf-8")):
            s = self.d[e["id"]]
            self.assertIn("%.2f in" % e["total_in"], s["total"])
            self.assertIn("%.2f in/hr" % e["peak_in_per_hr"], s["peak"])
            self.assertIn(e["start_utc"][:4], s["long"])

    def test_context_and_honesty(self):
        for k in ("ev1", "ev2", "ev3", "ev4", "ev5", "ev6"):
            c = self.d[k]["context"]
            self.assertIn("not verified by this project", c["flag"])
            self.assertIn("Not validated for depth", self.d[k]["honesty"])
            self.assertIn("AUC 0.84", self.d[k]["honesty"])
            for l in c["links"]:
                self.assertTrue(l["url"].startswith("https://"))
        for k in ("ev1", "ev3"):
            self.assertIn("No flood reports were found", self.d[k]["context"]["text"])
            self.assertEqual(self.d[k]["context"]["links"], [])
        self.assertEqual(len(self.d["ev4"]["context"]["links"]), 2)
        self.assertIn("196", self.d["ev4"]["context"]["text"])
        self.assertIn("4.93", self.d["ev5"]["context"]["text"])
        self.assertIsNone(self.d["25yr"]["context"])

    def test_ev4_validation(self):
        v = self.d["ev4"]["validation"]
        t = " ".join(v["lines"])
        for x in ("16 of 21", "27 of 100", "42 of 100", "0.62 m", "0.08 m"):
            self.assertIn(x, t)
        self.assertEqual(len(v["limits"]), 4)
        self.assertNotIn("validation", self.d["ev3"])

    def test_model_run(self):
        self.assertIn("5 m", self.d["ev4"]["model_run"]["county"])
        self.assertIn("2026-11-0", self.d["ev4"]["model_run"]["county"])
        self.assertIn("2 m", self.d["25yr"]["model_run"]["city"])
        self.assertIn("No city run", self.d["ev4"]["model_run"]["city"])


if __name__ == "__main__":
    unittest.main()
