"""One-line and full descriptions of each storm for the storm picker and the Help 'Storm data sources' tab.

Source: `layers/storm_blurbs.json` (built by scripts/build_storm_blurbs.py) in the data folder or the repository, else
`layers/storm_info.json` (title / type / period), else nothing.  Historic events (ev1..ev6) always carry the honesty line.
"""
from __future__ import annotations

import json
from pathlib import Path

HONESTY_EVENT = "Replays the rain of that day; not a reconstruction of the real flood; not validated."
HONESTY_DESIGN = "A design storm is a statistical yardstick, not a forecast; the model is not validated."
_CACHE: dict = {}


def _candidates(root):
    here = Path(__file__).resolve().parents[3] / "layers"
    out = []
    if root is not None:
        out.append(Path(root) / "layers")
    out.append(here)
    return out


def _load(root) -> dict:
    key = str(root)
    if key in _CACHE:
        return _CACHE[key]
    res: dict = {}
    for d in _candidates(root):
        try:
            data = json.loads((d / "storm_blurbs.json").read_text(encoding="utf-8")).get("storms") or {}
            for k, v in data.items():
                res[k] = {"short": str(v.get("short") or ""), "long": str(v.get("long") or ""), "honesty": str(v.get("honesty") or "")}
            break
        except Exception:                                                   # noqa: BLE001  (missing / damaged file: try the next source)
            continue
    if not res:
        for d in _candidates(root):
            try:
                data = json.loads((d / "storm_info.json").read_text(encoding="utf-8")).get("storms") or {}
                for k, v in data.items():
                    title = str(v.get("title") or k)
                    typ = str(v.get("type") or "")
                    per = str(v.get("period") or "")
                    res[k] = {"short": f"{typ}. {per}".strip(". ") if typ else title, "long": " ".join(x for x in (title + ".", typ + ".", per) if x.strip(". ")),
                              "honesty": ""}
                break
            except Exception:                                               # noqa: BLE001
                continue
    _CACHE[key] = res
    return res


def blurb(root, storm: str) -> dict | None:
    """{"short", "long", "honesty"} for a storm name, or None when no description exists."""
    b = _load(root).get(storm)
    if not b:
        return None
    event = storm.startswith("ev")
    h = b.get("honesty") or (HONESTY_EVENT if event else HONESTY_DESIGN)
    if event and "reconstruction" not in h:
        h = HONESTY_EVENT + " " + h
    return {"short": b["short"], "long": b["long"] or b["short"], "honesty": h}


def clear_cache():
    _CACHE.clear()
