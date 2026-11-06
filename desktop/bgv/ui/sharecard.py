"""Annotated screenshot / share card: the 3D view (captured without the interface) with a header (title, storm, hour), a depth
legend, a scale bar, a north arrow, an honesty stamp and the data credits, composed with PIL into one PNG.

compose(rgb, info) -> PIL.Image     pure (tests feed synthetic arrays); info keys: title, storm, hour, legend [(label, rgb)],
                                    metres_per_px (at the image centre, or None), north_deg (screen angle of north, 0 = up),
                                    stamp, credits [str], source.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

STAMP = "MODEL RESULT - not validated against observed floods; not for regulatory use"
CREDITS = ("Depth: 2D solver on TACC Vista (native tiles, 6.25 m)", "Terrain: USGS / KyFromAbove LiDAR",
           "Buildings / roads: (c) OpenStreetMap contributors", "Rain: NOAA Atlas 14 / historical events")


def _font(size):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def nice_length(max_m: float) -> float:
    """Largest 1/2/5 x 10^k metres <= max_m."""
    if max_m <= 0:
        return 0.0
    e = 10 ** math.floor(math.log10(max_m))
    for m in (5, 2, 1):
        if m * e <= max_m:
            return m * e
    return e


def compose(rgb: np.ndarray, info: dict):
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8), "RGB")
    W, H = img.size
    k = max(W / 1600.0, 0.6)
    head_h, foot_h = int(70 * k), int(64 * k)
    card = Image.new("RGB", (W, H + head_h + foot_h), (12, 14, 26))
    card.paste(img, (0, head_h))
    d = ImageDraw.Draw(card)
    f_big, f_mid, f_small = _font(int(30 * k)), _font(int(19 * k)), _font(int(15 * k))
    d.text((int(18 * k), int(8 * k)), info.get("title", "Bowling Green flood model"), fill=(240, 242, 250), font=f_big)
    sub = f"Storm: {info.get('storm', '?')}    Hour {info.get('hour', '?')} of 24    {info.get('source', '')}"
    d.text((int(18 * k), int(44 * k)), sub, fill=(170, 178, 205), font=f_small)
    # legend (top-left of the map)
    leg = info.get("legend") or []
    if leg:
        x0, y0 = int(14 * k), head_h + int(14 * k)
        sw = int(18 * k)
        bw = int(210 * k)
        d.rectangle((x0, y0, x0 + bw, y0 + int(14 * k) + len(leg) * (sw + int(6 * k)) + int(22 * k)), fill=(12, 14, 26))
        d.text((x0 + int(8 * k), y0 + int(4 * k)), info.get("legend_title", "Water depth"), fill=(230, 232, 240), font=f_small)
        for i, (lab, col) in enumerate(leg):
            yy = y0 + int(26 * k) + i * (sw + int(6 * k))
            d.rectangle((x0 + int(8 * k), yy, x0 + int(8 * k) + sw, yy + sw), fill=tuple(int(c) for c in col[:3]))
            d.text((x0 + int(16 * k) + sw, yy), str(lab), fill=(220, 224, 236), font=f_small)
    # north arrow (top-right of the map)
    ang = math.radians(float(info.get("north_deg", 0.0)))
    cx, cy, r = W - int(48 * k), head_h + int(52 * k), int(28 * k)
    d.ellipse((cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6), fill=(12, 14, 26))
    tip = (cx + r * math.sin(ang), cy - r * math.cos(ang))
    left = (cx + 0.45 * r * math.sin(ang - 2.4), cy - 0.45 * r * math.cos(ang - 2.4))
    right = (cx + 0.45 * r * math.sin(ang + 2.4), cy - 0.45 * r * math.cos(ang + 2.4))
    d.polygon([tip, left, (cx, cy), right], fill=(235, 70, 60))
    d.text((tip[0] - 5 * k, tip[1] - 26 * k), "N", fill=(240, 240, 240), font=f_mid)
    # scale bar (bottom-left of the map)
    mpp = info.get("metres_per_px")
    if mpp and mpp > 0:
        L = nice_length(mpp * W * 0.22)
        px = L / mpp
        x0, y0 = int(18 * k), head_h + H - int(30 * k)
        d.rectangle((x0 - 6, y0 - int(24 * k), x0 + px + 8, y0 + int(14 * k)), fill=(12, 14, 26))
        d.rectangle((x0, y0, x0 + px, y0 + int(6 * k)), fill=(240, 240, 240))
        txt = f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m"
        d.text((x0, y0 - int(22 * k)), txt + " (at view centre)", fill=(240, 240, 240), font=f_small)
    # footer: honesty stamp + credits
    fy = head_h + H + int(8 * k)
    d.text((int(18 * k), fy), info.get("stamp", STAMP), fill=(255, 179, 71), font=f_mid)
    d.text((int(18 * k), fy + int(28 * k)), "   |   ".join(info.get("credits", CREDITS)), fill=(150, 156, 180), font=f_small)
    return card


def save(card, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    card.save(p)
    return p
