"""python -m bgv  -  BG Flood desktop viewer.

    python -m bgv                                  open the viewer window
    python -m bgv --render out.png --storm 25yr --hour 12 --camera "x,y,dist,yaw,pitch" --size 1920x1080
    python -m bgv --benchmark [--seconds 10] [--json]
    python -m bgv --selftest                       3 s on-screen window, then exit
"""
import time

T_PROCESS_START = time.perf_counter()

import argparse
import os
import sys


def build_parser():
    p = argparse.ArgumentParser(prog="bgv", description="BG Flood desktop viewer (GPU, wgpu/Vulkan)")
    p.add_argument("--data", help="folder that contains layers\\flow (default: BGV_DATA or the DSOC AI folder)")
    p.add_argument("--storm", help="5yr, 10yr, 25yr, 50yr, 100yr (city); 2yr .. 100yr, ev1 .. ev6 (county)")
    p.add_argument("--domain", choices=("city", "county"), type=str.lower,
                   help="which area to show: city (Bowling Green, default) or county (Warren County); remembered in the settings")
    p.add_argument("--county-dem", metavar="TIF", help="county terrain GeoTIFF, UTM 16N, metres "
                                                       "(default <data>/layers/dem_warren_5m.tif)")
    p.add_argument("--county-demo", metavar="FOLDER", help="folder written by --make-county-demo: used when the data root has "
                                                           "no real county flow grids")
    p.add_argument("--county-pbf", metavar="FILE", help="OpenStreetMap extract for the county buildings / roads "
                                                        "(default <data>/kentucky-latest.osm.pbf)")
    p.add_argument("--make-county-demo", metavar="FOLDER", help="write a synthetic county terrain + 2 synthetic storms (clearly "
                                                                "marked DEMO DATA) into FOLDER and exit")
    p.add_argument("--hour", type=float, help="time in hours (0..24, fractional ok)")
    p.add_argument("--camera", help='"x,y,dist,yaw,pitch": target world metres (origin = grid centre), distance m, '
                                    'yaw = compass heading looked toward (deg), pitch = tilt down from horizon (deg)')
    p.add_argument("--at", help='"lng,lat": put the camera target there (keeps dist/yaw/pitch)')
    p.add_argument("--size", help="WxH (window or render size)")
    p.add_argument("--enable-layers", metavar="NAMES", help="comma-separated layer names to switch on (e.g. tiledepth)")
    p.add_argument("--msaa", type=int, default=4, choices=(1, 2, 4))
    p.add_argument("--exag", type=float, help="vertical exaggeration")
    p.add_argument("--sun", help='"azimuth,elevation" degrees (compass azimuth of the sun)')
    p.add_argument("--ripple", type=float, help="ripple strength (default 1)")
    p.add_argument("--haze", type=float, help="haze density per metre (default 2.2e-5)")
    p.add_argument("--render", metavar="OUT.png", help="render one offscreen frame to a PNG and exit")
    p.add_argument("--ui", action="store_true", help="include the sidebar in --render")
    p.add_argument("--anim", type=float, default=4.0, help="water animation time (s) for --render")
    p.add_argument("--benchmark", action="store_true", help="scripted offscreen camera path, frame-time statistics")
    p.add_argument("--seconds", type=float, default=10.0, help="benchmark length at 60 fps steps")
    p.add_argument("--json", action="store_true", help="benchmark: JSON on stdout")
    p.add_argument("--selftest", action="store_true", help="3 s window then exit")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--debug", action="store_true", help="show full tracebacks")
    return p


def _fail(e: BaseException, args, code: int) -> int:
    """Top-level failure: traceback -> desktop/logs/bgv.log, one short native message box in window mode (never a bare
    traceback window), the same plain text on stderr when there is a console."""
    from . import applog
    window_mode = not (args.render or args.benchmark or args.make_county_demo)
    applog.fatal(e, show=window_mode)
    if args.debug:
        raise e
    return code


def main(argv=None) -> int:
    for name in ("stdout", "stderr"):                 # pythonw.exe has no console: sys.stdout / sys.stderr are None
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    args = build_parser().parse_args(argv)
    from . import applog, config
    applog.install_excepthooks()
    try:
        if args.data:
            config.set_data_root(args.data)
        from . import tools
        if args.make_county_demo:
            return tools.run_make_county_demo(args)
        if args.render:
            return tools.run_render(args)
        if args.benchmark:
            return tools.run_benchmark(args)
        if args.selftest:
            return tools.run_selftest(args)
        return tools.run_window(args)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError as e:
        return _fail(e, args, 2)
    except Exception as e:
        return _fail(e, args, 1)


if __name__ == "__main__":
    sys.exit(main())
