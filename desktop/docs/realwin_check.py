"""Real-window check without synthetic input: open the viewer (county), switch domain county -> city -> county, change the
UI scale, capture ONLY the app window (PrintWindow) after each step, then close.

    python docs/realwin_check.py OUTDIR
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    import realwin_search as R
    from bgv.__main__ import build_parser
    from bgv import launch
    args = build_parser().parse_args(["--domain", "county", "--size", "1400x850"])
    L, canvas, loop = launch.start(args, title="BG Flood Viewer (window check)")
    res = {"steps": []}

    def wait_ready(app, t=180):
        t0 = time.perf_counter()
        time.sleep(1.0)
        while app.loading and time.perf_counter() - t0 < t:
            time.sleep(0.5)
        time.sleep(2.0)
        return time.perf_counter() - t0

    def step(app, h, name):
        p = out / f"{name}.png"
        R.capture(h, str(p))
        res["steps"].append({"step": name, "domain": app.domain.key, "ui_scale": app.get_layer("overlay").state.ui_scale,
                             "errors": list(app.errors)[-3:], "png": str(p)})

    def driver(app):
        try:
            time.sleep(2.5)
            h = R.find_window("")
            step(app, h, "1_county")
            loop.call_soon(lambda: app.set_domain("city"))
            res["city_s"] = wait_ready(app)
            step(app, h, "2_city")
            app.get_layer("overlay").state.ui_scale = 1.5
            time.sleep(1.5)
            step(app, h, "3_city_scale150")
            app.get_layer("overlay").state.ui_scale = 1.0
            loop.call_soon(lambda: app.set_domain("county"))
            res["county_s"] = wait_ready(app)
            step(app, h, "4_county")
        except Exception as e:
            res["error"] = repr(e)
        finally:
            loop.call_soon(canvas.close)

    L.on_loaded.append(lambda app: threading.Thread(target=driver, args=(app,), daemon=True).start())
    loop.call_later(600.0, canvas.close)
    loop.run()
    if L.app is not None:
        L.app.shutdown()
    print(json.dumps(res, indent=1))
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
