"""Frozen-app entry point for 'BG Flood Viewer.exe'. Resolves the data folder, then runs bgv.__main__.main().

Data folder order (first one containing layers\\flow wins):
  1. --data <folder> on the command line (passed through unchanged; bgv errors if wrong)
  2. bgv_data.txt next to the exe (one line: the folder path)
  3. env BGV_DATA_ROOT, then BGV_DATA
  4. a folder named 'data' next to the exe
  5. the default path from bgv.config.DEFAULT_DATA_ROOT
Missing -> Windows message box explaining where to put it. No change to bgv code is needed: we just set BGV_DATA.
"""
import os
import sys
from pathlib import Path


def _app_dir() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def _has_flow(p) -> bool:
    return bool(p) and (Path(p) / "layers" / "flow").is_dir()


def _msgbox(text: str, title: str = "BG Flood Viewer") -> None:
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
    except Exception:
        print(text, file=sys.stderr)


def resolve_data_root():
    app = _app_dir()
    cands = []
    txt = app / "bgv_data.txt"
    if txt.is_file():
        cands.append(txt.read_text(encoding="utf-8-sig").strip().strip('"'))
    cands += [os.environ.get("BGV_DATA_ROOT"), os.environ.get("BGV_DATA"), str(app / "data")]
    try:
        from bgv.config import DEFAULT_DATA_ROOT
        cands.append(DEFAULT_DATA_ROOT)
    except Exception:
        pass
    for c in cands:
        if _has_flow(c):
            return c
    return None


def main() -> int:
    app = _app_dir()
    if sys.stdout is None or sys.stderr is None:  # windowed exe: no console, keep prints from crashing
        try:
            f = open(app / "bgv.log", "w", encoding="utf-8", buffering=1)
        except OSError:
            f = open(os.devnull, "w")
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f
    os.environ.setdefault("BGV_CACHE", str(app / "cache"))  # bgv falls back to %TEMP% if not writable
    if "--data" not in sys.argv:
        root = resolve_data_root()
        if root is None:
            _msgbox("The flood data folder was not found.\n\n"
                    "Point the viewer at the folder that contains 'layers\\flow' by ONE of:\n"
                    f"  - writing its path into  {app / 'bgv_data.txt'}\n"
                    "  - setting the environment variable BGV_DATA_ROOT\n"
                    f"  - copying/linking it as  {app / 'data'}\n"
                    "  - starting with  --data \"<folder>\"")
            return 2
        os.environ["BGV_DATA"] = root
    from bgv.__main__ import main as bgv_main
    try:
        return bgv_main() or 0
    except Exception as e:  # windowed exe has no console: surface the error
        import traceback
        log = app / "bgv_error.log"
        try:
            log.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        _msgbox(f"BG Flood Viewer stopped with an error:\n\n{e}\n\nDetails: {log}")
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
