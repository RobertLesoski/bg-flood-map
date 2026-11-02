"""Helpers for background threads: they must never compete with the render thread for a core.

`lower_priority()`  Windows: THREAD_PRIORITY_BELOW_NORMAL for the calling thread (no-op elsewhere / on failure).
`spawn(target, name, *args)`  daemon thread that lowers its own priority first.
"""
from __future__ import annotations

import threading


def lower_priority(level: int = -1) -> bool:
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        return bool(k32.SetThreadPriority(k32.GetCurrentThread(), level))
    except Exception:
        return False


def spawn(target, name: str, *args) -> threading.Thread:
    def run():
        lower_priority()
        target(*args)
    th = threading.Thread(target=run, name=name, daemon=True)
    th.start()
    return th
