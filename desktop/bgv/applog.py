"""Application log + last-resort error dialog.

    applog.log()                       the 'bgv' logger (rotating file desktop/logs/bgv.log, 200 kB x 3, never raises)
    applog.install_excepthooks()       uncaught exceptions (main thread and worker threads) go to the log
    applog.fatal(title, text)          log + native message box (user32.MessageBoxW) with a short human message; no
                                       box when BGV_NO_DIALOG=1 or when not on Windows (the text goes to stderr then)
    applog.human_message(exc)          short, non-technical explanation of an exception

`BGV_LOG_DIR` overrides the log folder (tests); an unwritable folder falls back to the system temp directory.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
import threading
from pathlib import Path

MAX_BYTES = 200_000
BACKUPS = 2
_logger: logging.Logger | None = None
_lock = threading.Lock()


def log_dir() -> Path:
    env = os.environ.get("BGV_LOG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "logs"


def _open_handler():
    for folder in (log_dir(), Path(tempfile.gettempdir()) / "bgv_logs"):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            h = logging.handlers.RotatingFileHandler(folder / "bgv.log", maxBytes=MAX_BYTES, backupCount=BACKUPS,
                                                     encoding="utf-8", delay=False)
            return h, folder
        except Exception:
            continue
    return None, None


def log() -> logging.Logger:
    """The shared logger; creating it never raises (no usable folder -> records are dropped)."""
    global _logger
    with _lock:
        if _logger is not None:
            return _logger
        lg = logging.getLogger("bgv")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)
        h, folder = _open_handler()
        if h is None:
            lg.addHandler(logging.NullHandler())
        else:
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
            lg.addHandler(h)
            lg.log_folder = folder
        _logger = lg
        return lg


def reset():
    """Close the file handler and forget the logger (tests, or when BGV_LOG_DIR changes)."""
    global _logger
    with _lock:
        lg = _logger
        _logger = None
    if lg is not None:
        for h in list(lg.handlers):
            try:
                h.close()
            except Exception:
                pass
            lg.removeHandler(h)


def log_path() -> Path | None:
    folder = getattr(log(), "log_folder", None)
    return None if folder is None else Path(folder) / "bgv.log"


def install_excepthooks():
    prev = sys.excepthook

    def hook(tp, val, tb):
        try:
            log().error("uncaught exception", exc_info=(tp, val, tb))
        except Exception:
            pass
        prev(tp, val, tb)

    def thread_hook(args):
        try:
            log().error("uncaught exception in thread %s", getattr(args.thread, "name", "?"),
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        except Exception:
            pass

    sys.excepthook = hook
    threading.excepthook = thread_hook


def human_message(exc: BaseException) -> str:
    """Short, plain-language explanation (the technical text is in the log)."""
    name = type(exc).__name__
    text = str(exc)
    low = (name + " " + text).lower()
    if isinstance(exc, ModuleNotFoundError) or isinstance(exc, ImportError):
        mod = getattr(exc, "name", None) or text
        return (f"A required Python package is missing ({mod}).\n\nInstall the packages with:\n"
                "  python -m pip install wgpu rendercanvas glfw imgui-bundle numpy numba scipy rasterio shapely "
                "mapbox_earcut pillow requests")
    if isinstance(exc, FileNotFoundError) or "flow data not found" in low or "no storms found" in low:
        return ("The flood data folder was not found.\n\nStart the viewer with  --data <folder that contains layers\\flow>\n"
                "or set the BGV_DATA environment variable.\n\n" + text[:300])
    if "adapter" in low or "vulkan" in low or "wgpu" in low or "device" in low and "lost" in low:
        return ("The graphics card could not be started.\n\nThe viewer needs a Vulkan-capable GPU and an up-to-date "
                "graphics driver.\n\n" + text[:300])
    if isinstance(exc, MemoryError):
        return "The computer ran out of memory. Close other programs and start the viewer again."
    return f"The viewer could not start.\n\n{name}: {text[:300]}"


def message_box(title: str, text: str, error: bool = True) -> bool:
    """Native modal message box (Windows).  Returns False when no box could be shown."""
    if os.environ.get("BGV_NO_DIALOG") == "1" or os.name != "nt":
        return False
    try:
        import ctypes
        flags = 0x10 if error else 0x40                      # MB_ICONERROR / MB_ICONINFORMATION  (MB_OK = 0)
        flags |= 0x10000 | 0x40000                           # MB_SETFOREGROUND | MB_TOPMOST
        ctypes.windll.user32.MessageBoxW(None, str(text), str(title), flags)
        return True
    except Exception:
        return False


def fatal(exc: BaseException, title: str = "BG Flood Viewer", show: bool = True) -> str:
    """Log the exception with its traceback, show a short message box and return the human message."""
    try:
        log().critical("fatal: %s: %s", type(exc).__name__, exc, exc_info=(type(exc), exc, exc.__traceback__))
    except Exception:
        pass
    msg = human_message(exc)
    p = log_path()
    if p is not None:
        msg += f"\n\nDetails: {p}"
    if show:
        message_box(title, msg)
    try:
        if sys.stderr is not None:                           # pythonw has no console: nothing to print to
            print(f"\n{title}: {msg}\n", file=sys.stderr)
    except Exception:
        pass
    return msg
