"""Small imgui drawing helpers shared by the panels (all take logical pixels)."""
from __future__ import annotations

import math
import time

from imgui_bundle import imgui

from . import theme
from .theme import col32

V2 = imgui.ImVec2


def text_c(color, s: str):
    imgui.text_colored(imgui.ImVec4(*color), s)


def bold(s: str, color=None):
    """Bold text line (falls back to a coloured line when the bold face is not loaded)."""
    if theme.Fonts.bold is not None:
        imgui.push_font(theme.Fonts.bold, 0.0)
        if color is None:
            imgui.text(s)
        else:
            imgui.text_colored(imgui.ImVec4(*color), s)
        imgui.pop_font()
    else:
        imgui.text_colored(imgui.ImVec4(*(color or theme.ACCENT2)), s)


def font_text(font, size, s: str, color=None):
    """Text with a given face (theme.Fonts.regular / .bold) and base size (0 = current)."""
    if font is not None:
        imgui.push_font(font, size)
    if color is None:
        imgui.text(s)
    else:
        imgui.text_colored(imgui.ImVec4(*color), s)
    if font is not None:
        imgui.pop_font()


_DIM = None


def dim(s: str, wrap: bool = False):
    global _DIM
    if _DIM is None:
        _DIM = imgui.ImVec4(*theme.TEXT_DIM)
    if wrap:
        imgui.push_style_color(imgui.Col_.text, _DIM)
        imgui.text_wrapped(s)
        imgui.pop_style_color()
    else:
        imgui.text_colored(_DIM, s)


def pill(dl, x: float, y: float, label: str, fg, bg, pad=(8.0, 3.0), rounding: float = 9.0, border=None) -> float:
    """Rounded badge drawn with the draw list at (x, y); returns its width."""
    ts = imgui.calc_text_size(label)
    w = ts.x + pad[0] * 2
    h = ts.y + pad[1] * 2
    dl.add_rect_filled(V2(x, y), V2(x + w, y + h), col32(bg), rounding)
    if border is not None:
        dl.add_rect(V2(x, y), V2(x + w, y + h), col32(border), rounding, 1.0)
    dl.add_text(V2(x + pad[0], y + pad[1]), col32(fg), label)
    return w


def pill_size(label: str, pad=(8.0, 3.0)):
    ts = imgui.calc_text_size(label)
    return ts.x + pad[0] * 2, ts.y + pad[1] * 2


def shadow_text(dl, pos, color, s: str, shadow=(0.0, 0.0, 0.0, 0.75)):
    dl.add_text(V2(pos[0] + 1.0, pos[1] + 1.0), col32(shadow), s)
    dl.add_text(V2(pos[0], pos[1]), col32(color), s)


def hbar(frac: float, color, height: float = 8.0, width: float = -1.0, bg=None):
    """Thin horizontal bar (current line, full width by default)."""
    p = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail().x
    w = avail if width < 0 else width
    dl = imgui.get_window_draw_list()
    dl.add_rect_filled(p, V2(p.x + w, p.y + height), col32(bg or theme.with_alpha(theme.BG3, 0.9)), height * 0.5)
    f = min(max(frac, 0.0), 1.0)
    if f > 0:
        dl.add_rect_filled(p, V2(p.x + max(w * f, height), p.y + height), col32(color), height * 0.5)
    imgui.dummy(V2(w, height))


def icon_button(sid: str, kind: str, size: float, active: bool = False, tip: str = "") -> bool:
    """Transport icon buttons drawn with polygons: play, pause, prev, next, first."""
    p = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button(sid, V2(size, size))
    hov = imgui.is_item_hovered()
    held = imgui.is_item_active()
    dl = imgui.get_window_draw_list()
    base = theme.mix(theme.BG3, theme.ACCENT, 0.55 if active else 0.25)
    if hov:
        base = theme.mix(base, theme.ACCENT, 0.35)
    if held:
        base = theme.mix(base, theme.ACCENT2, 0.4)
    r = size * 0.5
    dl.add_rect_filled(p, V2(p.x + size, p.y + size), col32(base), 6.0)
    c = V2(p.x + r, p.y + r)
    ic = col32((0.96, 0.98, 1.0, 1.0))
    u = size * 0.26
    if kind == "play":
        dl.add_triangle_filled(V2(c.x - u * 0.7, c.y - u * 1.1), V2(c.x - u * 0.7, c.y + u * 1.1), V2(c.x + u * 1.2, c.y), ic)
    elif kind == "pause":
        dl.add_rect_filled(V2(c.x - u * 0.95, c.y - u), V2(c.x - u * 0.25, c.y + u), ic, 1.0)
        dl.add_rect_filled(V2(c.x + u * 0.25, c.y - u), V2(c.x + u * 0.95, c.y + u), ic, 1.0)
    elif kind in ("prev", "next"):
        s = -1.0 if kind == "prev" else 1.0
        dl.add_triangle_filled(V2(c.x - s * u * 0.2, c.y - u), V2(c.x - s * u * 0.2, c.y + u), V2(c.x + s * u * 1.1, c.y), ic)
        dl.add_rect_filled(V2(c.x - s * u * 1.1 - 1.2, c.y - u), V2(c.x - s * u * 1.1 + 1.2, c.y + u), ic, 1.0)
    elif kind == "first":
        dl.add_triangle_filled(V2(c.x + u * 0.9, c.y - u), V2(c.x + u * 0.9, c.y + u), V2(c.x - u * 0.3, c.y), ic)
        dl.add_rect_filled(V2(c.x - u * 1.0 - 1.2, c.y - u), V2(c.x - u * 1.0 + 1.2, c.y + u), ic, 1.0)
    if tip and hov:
        imgui.set_tooltip(tip)
    return clicked


def chip_toggle(sid: str, label: str, on: bool, color, height: float = 0.0, tip: str = "") -> bool:
    """Toggle chip with a coloured dot.  Returns True when clicked."""
    ts = imgui.calc_text_size(label)
    h = height or (ts.y + 8.0)
    w = ts.x + 30.0
    p = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button(sid, V2(w, h))
    hov = imgui.is_item_hovered()
    dl = imgui.get_window_draw_list()
    bg = theme.mix(theme.BG3, color, 0.30 if on else 0.0)
    if hov:
        bg = theme.mix(bg, theme.ACCENT, 0.2)
    dl.add_rect_filled(p, V2(p.x + w, p.y + h), col32(bg), h * 0.5)
    dl.add_rect(p, V2(p.x + w, p.y + h), col32(theme.with_alpha(color, 0.9 if on else 0.35)), h * 0.5, 1.0)
    cy = p.y + h * 0.5
    dl.add_circle_filled(V2(p.x + 13.0, cy), 4.5, col32(theme.with_alpha(color, 1.0 if on else 0.35)))
    dl.add_text(V2(p.x + 23.0, cy - ts.y * 0.5), col32(theme.TEXT if on else theme.TEXT_DIM), label)
    if tip and hov:
        imgui.set_tooltip(tip)
    return clicked


def section(label: str, open_: bool, right: str = "") -> bool:
    """Collapsing section header whose open state is owned by the caller (persisted).  Returns the new open state."""
    imgui.set_next_item_open(bool(open_), imgui.Cond_.always)
    imgui.push_style_color(imgui.Col_.header, imgui.ImVec4(*theme.with_alpha(theme.mix(theme.BG2, theme.ACCENT, 0.14), 1.0)))
    opened = imgui.collapsing_header(label, imgui.TreeNodeFlags_.framed)
    imgui.pop_style_color()
    return bool(opened)


def sparkline_depth(dl, p0, size, values, t, color, ymax=None):
    """Tiny filled depth-by-hour plot drawn by hand (selected building card).  `values` has 25 entries (hour 0..24)."""
    w, h = size
    dl.add_rect_filled(V2(*p0), V2(p0[0] + w, p0[1] + h), col32(theme.with_alpha(theme.BG0, 0.55)), 4.0)
    vals = [0.0 if (v != v) else float(v) for v in values]
    top = ymax or max(max(vals), 0.3) * 1.15
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = p0[0] + 4 + (w - 8) * i / (n - 1)
        y = p0[1] + h - 4 - (h - 8) * min(v / top, 1.0)
        pts.append(V2(x, y))
    fill = theme.with_alpha(color, 0.25)
    for a, b in zip(pts, pts[1:]):
        dl.add_quad_filled(a, b, V2(b.x, p0[1] + h - 4), V2(a.x, p0[1] + h - 4), col32(fill))
    dl.add_polyline(pts, col32(color), 1.8, 0)
    tx = p0[0] + 4 + (w - 8) * min(max(t, 0.0), 24.0) / 24.0
    dl.add_line(V2(tx, p0[1] + 2), V2(tx, p0[1] + h - 2), col32(theme.ACCENT2), 1.5)
    return top


class Toasts:
    def __init__(self):
        self.items: list[list] = []        # [text, color, t_end]

    def push(self, text: str, color=None, seconds: float = 4.0):
        self.items.append([text, color or theme.TEXT, time.perf_counter() + seconds])
        self.items = self.items[-4:]

    def draw(self, x_center: float, y_bottom: float):
        now = time.perf_counter()
        self.items = [it for it in self.items if it[2] > now]
        if not self.items:
            return
        dl = imgui.get_foreground_draw_list()
        y = y_bottom
        for text, color, t_end in reversed(self.items):
            a = min(1.0, (t_end - now) / 0.6)
            ts = imgui.calc_text_size(text)
            w, h = ts.x + 28.0, ts.y + 16.0
            x0 = x_center - w * 0.5
            y0 = y - h
            dl.add_rect_filled(V2(x0, y0), V2(x0 + w, y0 + h), col32((0.05, 0.07, 0.13, 0.94 * a)), 8.0)
            dl.add_rect(V2(x0, y0), V2(x0 + w, y0 + h), col32(theme.with_alpha(theme.ACCENT, 0.7 * a)), 8.0, 1.2)
            dl.add_text(V2(x0 + 14.0, y0 + 8.0), col32(theme.with_alpha(color, a)), text)
            y = y0 - 8.0


def wrapped_lines(text: str, width: float):
    """Split `text` into lines no wider than `width` logical px (used by the draw-list based boxes)."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if imgui.calc_text_size(t).x <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
