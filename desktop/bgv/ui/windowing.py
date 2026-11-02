"""Window helpers (glfw backend of rendercanvas): minimum size, fullscreen toggle.  Every call is best effort and does
nothing on the offscreen canvas."""
from __future__ import annotations

MIN_W, MIN_H = 1024, 600


def _window(canvas):
    return getattr(canvas, "_window", None) if canvas is not None else None


def set_min_size(canvas, w: int = MIN_W, h: int = MIN_H) -> bool:
    win = _window(canvas)
    if win is None:
        return False
    try:
        import glfw
        glfw.set_window_size_limits(win, int(w), int(h), glfw.DONT_CARE, glfw.DONT_CARE)
        return True
    except Exception:
        return False


class Fullscreen:
    def __init__(self):
        self.on = False
        self._saved = None

    def toggle(self, canvas) -> bool:
        win = _window(canvas)
        if win is None:
            return False
        try:
            import glfw
            if not self.on:
                mon = glfw.get_primary_monitor()
                mode = glfw.get_video_mode(mon)
                x, y = glfw.get_window_pos(win)
                w, h = glfw.get_window_size(win)
                self._saved = (x, y, w, h)
                glfw.set_window_monitor(win, mon, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
                self.on = True
            else:
                x, y, w, h = self._saved or (80, 80, 1600, 900)
                glfw.set_window_monitor(win, None, x, y, w, h, 0)
                self.on = False
            return True
        except Exception:
            return False
