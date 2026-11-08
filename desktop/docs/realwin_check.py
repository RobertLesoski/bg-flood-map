"""Real-window check WITHOUT synthetic input: open the viewer, switch city <-> county and the UI scale from code, capture
ONLY the app window (PrintWindow) after each step, then close.   python docs/realwin_check.py OUTDIR"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(out_dir):
    import realwin_search as R
    from bgv.__main__ import build_parser
    from bgv import launch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args(["--domain", "city", "--size", "1400x850"])
    L, canvas, loop = launch.start(args, title="BG Flood Viewer (window check)")
    res = {"steps": []}

    def call(fn):                                     # run on the frame loop thread, wait for it
        ev = threading.Event()

        def w():
            try:
                fn()
            finally:
                ev.set()
        loop.call_soon(w)
        ev.wait(60)

    def shot(app, name):
        time.sleep(2.0)
        h = R.find_window("")
        p = out / f"{name}.png"
        R.capture(h, str(p))
        m = app.get_layer("overlay").manager
        res["steps"].append({"step": name, "domain": app.domain.key, "ui_scale": m.st.ui_scale, "errors": list(app.errors)[-3:]})

    def driver(app):
        try:
            time.sleep(2.0)
            shot(app, "1_city")
            call(lambda: app.set_domain("county"))
            t0 = time.perf_counter()
            while app.loading and time.perf_counter() - t0 < 180:
                time.sleep(0.5)
            res["county_switch_s"] = round(time.perf_counter() - t0, 1)
            shot(app, "2_county")
            m = app.get_layer("overlay").manager
            call(lambda: setattr(m.st, "ui_scale", 1.5))
            shot(app, "3_county_scale150")
            call(lambda: setattr(m.st, "ui_scale", 1.0))
            call(lambda: app.set_domain("city"))
            t0 = time.perf_counter()
            while app.loading and time.perf_counter() - t0 < 180:
                time.sleep(0.5)
            shot(app, "4_city_again")
        except Exception as e:
            res["error"] = repr(e)
        finally:
            loop.call_soon(canvas.close)

    L.on_loaded.append(lambda app: threading.Thread(target=driver, args=(app,), daemon=True).start())
    loop.call_later(400.0, canvas.close)
    loop.run()
    if L.app is not None:
        L.app.shutdown()
    print(json.dumps(res))
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
