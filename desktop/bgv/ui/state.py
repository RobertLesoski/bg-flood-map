"""Persistent UI state (panels, sections, options).  Pure python; round-trips through `to_dict` / `from_dict`
(stored under the overlay layer's state in user_settings.json).  Unknown / invalid values fall back to defaults."""
from __future__ import annotations

from dataclasses import dataclass, field

SECTIONS = ("time", "layers", "water", "quality", "stats", "info")
TIMELINE_MODES = ("flooded", "wet_area", "both")
PRESETS = ("Low", "Medium", "High", "Ultra")

# defaults: which sidebar sections start open
_SECTION_DEFAULT_OPEN = {"time": True, "layers": True, "water": False, "quality": False, "stats": True, "info": True}


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if v != v:
        return default
    return min(max(v, lo), hi)


@dataclass
class UiState:
    # ---- panels
    show_sidebar: bool = True
    show_timeline: bool = True
    show_atrisk: bool = True
    show_selected: bool = True
    show_legends: bool = True
    show_compass: bool = True
    show_scalebar: bool = True
    show_probe: bool = True
    show_perf: bool = False                # F3
    show_status: bool = True
    # ---- sections of the sidebar
    sections: dict = field(default_factory=lambda: dict(_SECTION_DEFAULT_OPEN))
    # ---- options
    ui_scale: float = 1.0
    imperial: bool = False                 # scale bar / legend units (tooltips always show both)
    timeline_mode: str = "both"
    info_details_open: bool = False
    legend_open: bool = True
    quality_preset: str = "High"
    sidebar_w: float = 340.0
    right_w: float = 420.0
    timeline_h: float = 176.0
    # ---- at-risk list
    risk_tiers: list = field(default_factory=lambda: [True, True, True, True])
    risk_search: str = ""
    risk_min_depth: float = 0.0
    # ---- first-run
    seen_help: bool = False
    # ---- not persisted
    hidden: bool = False                   # H: hide the whole UI
    fullscreen: bool = False

    PERSISTED = ("show_sidebar", "show_timeline", "show_atrisk", "show_selected", "show_legends", "show_compass",
                 "show_scalebar", "show_probe", "show_perf", "show_status", "sections", "ui_scale", "imperial",
                 "timeline_mode", "info_details_open", "legend_open", "quality_preset", "sidebar_w", "right_w",
                 "timeline_h", "risk_tiers", "risk_search", "risk_min_depth", "seen_help")

    # ------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = {}
        for k in self.PERSISTED:
            v = getattr(self, k)
            d[k] = dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v
        return d

    def load(self, d: dict | None) -> "UiState":
        """Apply a saved dict (any garbage tolerated)."""
        if not isinstance(d, dict):
            return self
        for k in ("show_sidebar", "show_timeline", "show_atrisk", "show_selected", "show_legends", "show_compass",
                  "show_scalebar", "show_probe", "show_perf", "show_status", "imperial", "info_details_open",
                  "legend_open", "seen_help"):
            if k in d:
                setattr(self, k, bool(d[k]))
        if isinstance(d.get("sections"), dict):
            for k in SECTIONS:
                if k in d["sections"]:
                    self.sections[k] = bool(d["sections"][k])
        self.ui_scale = _clamp(d.get("ui_scale", self.ui_scale), 0.75, 2.0, self.ui_scale)
        self.sidebar_w = _clamp(d.get("sidebar_w", self.sidebar_w), 260.0, 520.0, self.sidebar_w)
        self.right_w = _clamp(d.get("right_w", self.right_w), 300.0, 560.0, self.right_w)
        self.timeline_h = _clamp(d.get("timeline_h", self.timeline_h), 120.0, 320.0, self.timeline_h)
        if d.get("timeline_mode") in TIMELINE_MODES:
            self.timeline_mode = d["timeline_mode"]
        if d.get("quality_preset") in PRESETS or d.get("quality_preset") == "Custom":
            self.quality_preset = d["quality_preset"]
        rt = d.get("risk_tiers")
        if isinstance(rt, (list, tuple)) and len(rt) == 4:
            self.risk_tiers = [bool(x) for x in rt]
        if isinstance(d.get("risk_search"), str):
            self.risk_search = d["risk_search"][:80]
        self.risk_min_depth = _clamp(d.get("risk_min_depth", self.risk_min_depth), 0.0, 8.0, self.risk_min_depth)
        return self

    @classmethod
    def from_dict(cls, d: dict | None) -> "UiState":
        return cls().load(d)
