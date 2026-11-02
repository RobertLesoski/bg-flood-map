"""OverlayLayer: the whole user interface (imgui) - registered LAST in registry.py.

It draws nothing into the scene pass (`draws = False`); instead App._gui() calls `gui(app)` once per frame inside the
imgui frame, App._on_key() offers every key to `on_key(app, ev)` and App.draw_frame() calls `post_frame(app)` after the
imgui pass (screenshots).  All logic lives in `bgv/ui/*` (panels, theme, state); settings persist through the layer
state (user_settings.json -> layers.overlay.state).

If this layer is disabled or fails, App falls back to its minimal built-in sidebar.
"""
from __future__ import annotations

from . import Layer


class OverlayLayer(Layer):
    name = "overlay"
    title = "Interface"
    draws = False
    has_ui = False

    def __init__(self):
        from ...ui.state import UiState
        self.state = UiState()
        self.manager = None

    # ------------------------------------------------------------------------------------------ lifecycle
    def init(self, app):
        from ...ui.manager import UiManager
        self.manager = UiManager(app, self.state)

    def on_storm(self, app):
        m = self.manager
        if m is not None:
            m._risk_key = None
            m.risk_ids = None
            m._probe_key = None
            m.probe_worker.latest = None

    def update(self, app, dt):
        if self.manager is not None:
            self.manager.update(dt)

    def shutdown(self, app):
        if self.manager is not None:
            self.manager.probe_worker.close()
            self.manager = None

    # ------------------------------------------------------------------------------------------ hooks used by App
    def lite_frame(self) -> bool:
        """True when the imgui renderer should replay the previous draw data instead of rebuilding the UI this frame."""
        m = self.manager
        return bool(m is not None and m.lite_skip())

    def gui(self, app) -> bool:
        """Draw all windows (inside imgui new_frame / render).  Returns False when the UI cannot run."""
        if self.manager is None:
            return False
        self.manager.frame()
        return True

    def on_key(self, app, ev) -> bool:
        if self.manager is None:
            return False
        try:
            return bool(self.manager.on_key(ev))
        except Exception as e:
            app.report_error("ui.key", e)
            return False

    def post_frame(self, app):
        m = self.manager
        if m is None or not m.shot_request:
            return
        req, m.shot_request = m.shot_request, None
        try:
            from ...ui import screenshot
            from ...ui.theme import OK, ERR
            path = screenshot.take(app, include_ui=(req == "ui"))
            m.toasts.push(f"Screenshot saved: {path}", OK, 6.0)
        except Exception as e:
            app.report_error("screenshot", e)
            from ...ui.theme import ERR
            m.toasts.push(f"Screenshot failed: {type(e).__name__}: {e}", ERR, 6.0)

    # ------------------------------------------------------------------------------------------ persistence
    def save_state(self):
        return self.state.to_dict()

    def load_state(self, d):
        self.state.load(d)
