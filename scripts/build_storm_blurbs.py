r"""
build_storm_blurbs.py - writes layers/storm_blurbs.json: the plain-language text the viewer's storm picker shows for each of the 12 storms.

Numbers (rain total, peak rate, dates, run dates) are read from the same fact files as build_storm_info.py (scripts/storms_out/*.json and the
design-depth constants of build_storm_info.py), never typed here. The only hand-written parts are the wording and the 'Real-world context'
paraphrases (public reports, links, flagged as not verified by this project).

Usage: python scripts/build_storm_blurbs.py --app "<web app folder>"
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_storm_info as bsi  # noqa: E402  (DESIGN depths, ORDER, IN)

CHANCE = {"2yr": 50, "5yr": 20, "10yr": 10, "25yr": 4, "50yr": 2, "100yr": 1}      # = 100 / years, percent chance in any one year
REPLAY = "The model replays the rain of that day; it is not a reconstruction of the real flood. "
DESIGN_REPLAY = "A design storm is a statistical yardstick, not a forecast or a real day. "
VAL_PAGE = "layers/validation/VALIDATION_EV4.html"


def honesty_tail(sc):
    r = sc["road_summary"]
    return ("Not validated for depth. One comparison to the April 2025 flood (%d reported flooded roads) shows the model puts more water on flooded roads than on typical roads "
            "(AUC %.2f), which is a weak sign it finds where water collects, not proof of accurate depths." % (r["t0.1_any"]["observed_n"], r["auc_max_depth"]))


def ev4_validation(sc):
    r = sc["road_summary"]
    a, b = r["t0.3_any"], r["t0.1_any"]
    return {"title": "How well does the model match the real flood?",
            "lines": ["%d of %d reported flooded roads reach 30 cm in the model, versus %d of %d random roads." % (a["observed_hits"], a["observed_n"], a["control_flagged"], a["control_n"]),
                      "At 10 cm, all %d reported roads are flagged, but so are %d of %d random roads, so 10 cm does not tell them apart." % (b["observed_n"], b["control_flagged"], b["control_n"]),
                      "Median road maximum depth: %.2f m on reported flooded roads vs %.2f m on random roads." % (r["median_max_observed"], r["median_max_control"])],
            "limits": ["The model replays %.2f in of rain in 24 h; the real event brought about 10.5 in over four days at the airport." % sc["model_rain_total_in"],
                       "There is no river inflow, so Barren River flooding is not captured.",
                       "The observations say a road was closed or flooded, not how deep the water was.",
                       "Only %d roads could be scored." % r["t0.1_any"]["observed_n"]],
            "link": {"label": "Full validation write-up (works offline once opened)", "url": "./" + VAL_PAGE}}


def render_validation(app, repo):
    """layers/validation/VALIDATION_EV4.html (+ the figure) from docs/VALIDATION_EV4.md, so the text is reachable offline."""
    import markdown
    import shutil
    md = open(os.path.join(repo, "docs", "VALIDATION_EV4.md"), encoding="utf-8").read()
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    body = body.replace("<h1>", '<p><img src="VALIDATION_EV4_map.png" alt="Map of model maximum depth, observed flooded roads and NWS reports" style="max-width:100%"></p><h1>', 1)
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Validation of ev4 (April 2025 flood)</title>'
            '<style>body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:12px 16px;background:#fff;color:#111}table{border-collapse:collapse;display:block;overflow-x:auto}'
            'td,th{border:1px solid #888;padding:3px 6px}code,pre{overflow-wrap:anywhere;white-space:pre-wrap}a{color:#0a4a9a}@media(prefers-color-scheme:dark){body{background:#12151f;color:#e8eaf4}a{color:#74c0fc}td,th{border-color:#667}}</style></head><body>'
            '<p><a href="../../viewer.html">Back to the map</a></p>' + body + "</body></html>")
    d = os.path.join(app, "layers", "validation")
    with open(os.path.join(d, "VALIDATION_EV4.html"), "w", encoding="utf-8") as f:
        f.write(page)
    shutil.copyfile(os.path.join(repo, "docs", "VALIDATION_EV4_map.png"), os.path.join(d, "VALIDATION_EV4_map.png"))
FLAG = "From public reports, not verified by this project."
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

NWS = "https://www.weather.gov/lmk/"
NO_FLOOD = "No flood reports were found in a quick search for this day. Nothing here says it flooded."
CONTEXT = {
    "ev1": {"text": NO_FLOOD, "links": []},
    "ev2": {"text": "NWS Louisville notes heavy rain and flooding on March 4-5, 2015, followed by snow.", "links": [["NWS Louisville 2015 review", NWS + "2015_review"]]},
    "ev3": {"text": NO_FLOOD, "links": []},
    "ev4": {"text": "NWS Louisville recorded a Bowling Green four-day rainfall record of about 10 in during April 1-6, 2025. The city declared a state of emergency, flooding was reported along Nashville Road, Louisville Road and Russellville Road, and the city counted 196 damaged structures by April 10.",
            "links": [["NWS Louisville: April 2025 rain and flooding", NWS + "HistoricRainfallFloodingApril2-62025"], ["Bowling Green Daily News", "https://bgdailynews.com/?p=685107"]]},
    "ev5": {"text": "NWS Louisville recorded a Bowling Green daily rainfall record of 4.93 in on September 1, 2017.", "links": [["NWS Louisville September 2017 summary", NWS + "september_2017_summary"]]},
    "ev6": {"text": "NWS Louisville describes the May 1-2, 2010 flood, with major flooding on the Green River.", "links": [["NWS Louisville: how the May 2010 flood happened", NWS + "flood_050210_how_it_happened"]]},
}
EV_TAG = {"ev1": "Light all-day rain", "ev2": "Steady moderate rain", "ev3": "Heavy storm", "ev4": "Historic flood", "ev5": "Remnants of Hurricane Harvey", "ev6": "May 2010 flood"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def date_en(s):                       # '2013-12-06 00:00' -> 'December 6, 2013'
    y, m, d = s.split(" ")[0].split("-")
    return "%s %d, %s" % (MONTHS[int(m) - 1], int(d), y)


def hourly(e):
    c = e["cum_in"]
    return [c[0]] + [c[i] - c[i - 1] for i in range(1, len(c))]


def build(app=None):
    ev = {e["id"]: e for e in load(os.path.join(HERE, "storms_out", "storms.json"))}
    facts = load(os.path.join(HERE, "storms_out", "vista_run_facts.json"))
    vf = load(os.path.join(HERE, "storms_out", "vista_verified_facts.json"))
    sc = load(os.path.join(app, "layers", "validation", "ev4_scores.json")) if app else None
    tail = honesty_tail(sc) if sc else "Not validated for depth (the April 2025 comparison file was not found)."
    ct, cit = vf["county_slurm"]["tasks"], vf["city_slurm"]["tasks"]
    out = {"generated_from": "scripts/build_storm_blurbs.py (facts: storms.json, vista_run_facts.json, vista_verified_facts.json, build_storm_info.DESIGN)", "storms": {}}
    for k in bsi.ORDER:
        rf = ((facts["county"] or {}).get(k) or {}).get("rain_file", {})
        run = {"county": ("County: 5 m grid (about 52 x 60 km), run finished %s on TACC Vista." % ct[k]["end"].split("T")[0]) if k in ct else "County run: date not recorded."}
        if k in cit:
            run["city"] = "City: 2 m grid (about 20 x 20 km), run finished %s on TACC Vista." % cit[k]["end"].split("T")[0]
        elif k == "2yr":
            run["city"] = "City: first 2-year run (classic polygons only); its run date is not recorded in the repo."
        else:
            run["city"] = "No city run of this storm."
        if k in bsi.DESIGN:
            n = int(k.replace("yr", ""))
            din = bsi.DESIGN[k]
            peak_mm = rf["peak_mm_per_hr"]
            total = "%.1f in (%.0f mm)" % (din, din * bsi.IN)
            peak = "%.2f in/hr (%.1f mm/hr)" % (peak_mm / bsi.IN, peak_mm)
            chance = CHANCE[k]
            s = {"kind": "design", "name": "%d-year design storm" % n, "total": total, "peak": peak, "duration": "24 hours", "date": "No calendar date (synthetic)",
                 "short": "Synthetic 24 h rain, %.1f in, %d%% chance in any year" % (din, chance)}
            s["long"] = ("A made-up, smooth 24-hour rain, not a real day. Total %s, peak %s. Depth: NOAA Atlas 14 for Warren County, KY. "
                         "Its return period is %d years: rain this heavy has about a %d%% chance of happening in any given year. " % (total, peak, n, chance))
            if n == 100:
                s["long"] += "This is the 1% chance storm: a 1 in 100 chance every year, which does not mean it can only happen once a century. "
            else:
                s["long"] += "That is an average, not a schedule: a %d-year storm can happen two years in a row. " % n
            s["long"] += "The return period describes the rainfall only, not how bad the flood is."
            s["context"] = None
            s["honesty"] = DESIGN_REPLAY + tail
        else:
            e = ev[k]
            peak = "%.2f in/hr (%.1f mm/hr)" % (e["peak_in_per_hr"], e["peak_in_per_hr"] * bsi.IN)
            total = "%.2f in (%.0f mm)" % (e["total_in"], e["total_in"] * bsi.IN)
            d = date_en(e["start_utc"])
            wet = sum(1 for x in hourly(e) if x > 0.004)
            s = {"kind": "event", "name": "Event %s: %s" % (k[2:], e["label"]), "total": total, "peak": peak, "duration": "24 hours", "date": d,
                 "short": "%s, %s, %.2f in in 24 h" % (EV_TAG[k], e["start_utc"][:10], e["total_in"])}
            s["long"] = ("A real 24-hour block of rain measured at the Bowling Green airport gauge (KBWG), starting %s UTC. Total %s, wettest hour %s, rain in %d of the 24 hours. "
                         "The same rain is applied evenly over the whole map." % (e["start_utc"], total, peak, wet))
            s["context"] = {"label": "Real-world context", "text": CONTEXT[k]["text"], "links": [{"label": a, "url": b} for a, b in CONTEXT[k]["links"]], "flag": FLAG}
            s["honesty"] = REPLAY + tail
            if k == "ev4" and sc:
                s["validation"] = ev4_validation(sc)
        s["model_run"] = run
        out["storms"][k] = s
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    a = ap.parse_args()
    with open(os.path.join(a.app, "layers", "storm_blurbs.json"), "w", encoding="utf-8") as f:
        json.dump(build(a.app), f, indent=1, ensure_ascii=False)
    render_validation(a.app, os.path.dirname(HERE))
    print("wrote layers/storm_blurbs.json")
