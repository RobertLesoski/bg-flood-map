"""
Tests for build_storm_info.py (STORM_DATA_SOURCES.md + layers/storm_info.json, the viewer's "About this storm" panel).

  python -m pytest scripts/tests/test_storm_info.py -q
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import build_storm_info as bsi  # noqa: E402

APP = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")
FIELDS = ("title", "type", "period", "rain_source", "rain_total", "rain_peak", "duration", "return_period", "hyetograph", "rain_units", "conversion", "limits", "not_verified")


class StormInfoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.info = bsi.build(cls.tmp)                      # no grids / COGs in the folder: the builder must still work from the facts files

    def test_twelve_storms_with_every_field(self):
        self.assertEqual(list(self.info["storms"]), bsi.ORDER)
        for k, s in self.info["storms"].items():
            for f in FIELDS:
                self.assertTrue(s.get(f), "%s: %s empty" % (k, f))
            self.assertIn("solver_run", s)

    def test_design_depths_are_the_atlas14_numbers_and_match_the_rain_files(self):
        facts = json.load(open(SCRIPTS / "storms_out" / "vista_run_facts.json"))
        for k, inches in bsi.DESIGN.items():
            rf = facts["county"][k]["rain_file"]
            self.assertAlmostEqual(rf["total_mm"], inches * 25.4, delta=0.003 * inches * 25.4, msg=k)       # the rain file integrates to the stated depth
            self.assertIn("%.1f in" % inches, self.info["storms"][k]["rain_total"])

    def test_peak_intensity_of_the_shared_curve(self):
        # steepest segment: 50 % -> 70 % of the depth between hours 12 and 14.4
        slope = max((b - a) / ((t2 - t1) * 24) for (t1, a), (t2, b) in zip(zip(bsi.TF, bsi.PF), zip(bsi.TF[1:], bsi.PF[1:])))
        facts = json.load(open(SCRIPTS / "storms_out" / "vista_run_facts.json"))
        for k, inches in bsi.DESIGN.items():
            self.assertAlmostEqual(slope * inches * 25.4, facts["county"][k]["rain_file"]["peak_mm_per_hr"], delta=0.01)

    def test_historical_blocks_agree_with_storms_json(self):
        ev = {e["id"]: e for e in json.load(open(SCRIPTS / "storms_out" / "storms.json"))}
        for k in ("ev1", "ev2", "ev3", "ev4", "ev5", "ev6"):
            e = ev[k]
            self.assertAlmostEqual(sum(e["hourly_in"]), e["total_in"], delta=0.02)
            self.assertIn(e["start_utc"], self.info["storms"][k]["period"])
            self.assertIn("station BWG", self.info["storms"][k]["rain_source"])

    def test_no_invented_return_period_for_historical_events(self):
        for k in ("ev1", "ev2", "ev3", "ev4", "ev5", "ev6"):
            self.assertIn("No return period", self.info["storms"][k]["return_period"])

    def test_run_facts_come_from_the_summaries(self):
        facts = json.load(open(SCRIPTS / "storms_out" / "vista_run_facts.json"))
        for k in bsi.ORDER:
            cf = facts["county"][k]
            self.assertIn("%.1f mm of %.1f mm" % (cf["runoff_depth_mm"], cf["total_rain_mm"]), self.info["storms"][k]["solver_run"])
            self.assertIn("1114427", self.info["storms"][k]["solver_run"])

    def test_city_runs_only_for_the_five_city_storms(self):
        for k in bsi.ORDER:
            has = bool(self.info["storms"][k].get("city_solver_run"))
            self.assertEqual(has, k in bsi.CITY_STORMS, k)
        self.assertIn("1102881", self.info["storms"]["100yr"]["city_solver_run"])

    def test_markdown_lists_every_storm_and_marks_unverified(self):
        text = bsi.md(self.info)
        for s in self.info["storms"].values():
            self.assertIn("## " + s["title"], text)
        self.assertEqual(text.count("**Not verified:**"), 12)

    @unittest.skipUnless((APP / "layers" / "storm_info.json").is_file(), "web app folder not on this machine")
    def test_shipped_json_and_markdown_are_up_to_date(self):
        info = bsi.build(str(APP))
        self.assertEqual(json.load(open(APP / "layers" / "storm_info.json", encoding="utf-8")), json.loads(json.dumps(info)))
        self.assertEqual(open(bsi.REPO + "/STORM_DATA_SOURCES.md", encoding="utf-8").read(), bsi.md(info))


if __name__ == "__main__":
    unittest.main()
