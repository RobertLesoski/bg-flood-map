"""Layer API - every visual feature of the viewer is a `Layer` subclass in its own file (see DESIGN.md "Layer API").

The app owns the window, the camera, the shared frame uniforms (bind group 0), the flow texture array, the terrain
textures and ONE render pass (shared colour + depth, optional MSAA).  A layer only adds its own pipelines/buffers.

Lifecycle, all optional except `draw`/`name`:
    init(app)             once, after the GPU, terrain and flow textures exist.  Create pipelines here.
    on_storm(app)         the storm (flow data + frame) changed; `app.storm`, `app.flow`, `app.terrain` are new.
    resize(app, w, h)     the render target size changed (physical pixels).
    update(app, dt)       once per frame before drawing; `app.t` is the current hour, `dt` real seconds.
    compute(app, enc)     record compute passes on the frame's command encoder BEFORE the render pass.
    draw(app, pass_)      record draw calls into the shared scene pass.  Bind group 0 is ALREADY set.
    ui(app)               imgui widgets; drawn inside this layer's collapsing section of the sidebar.
    save_state() / load_state(d)   JSON-able dict persisted in user_settings.json under the layer's name.
    shutdown(app)         release resources.
`enabled` is toggled by the sidebar checkbox; a disabled layer gets neither update nor draw.
Draw order == order in registry.py (terrain, roads, buildings, water, particles, arrows, overlay).
"""
from __future__ import annotations


class Layer:
    name: str = "layer"            # unique key (settings, toggles)
    title: str = ""                # sidebar label (defaults to name)
    default_enabled: bool = True
    has_ui: bool = False           # set True when ui() draws something (adds a collapsing section)
    enabled: bool = True
    # True -> draw() is called in the scene pass; False for pure-UI layers
    draws: bool = True

    def init(self, app) -> None: ...
    def on_storm(self, app) -> None: ...
    def resize(self, app, w: int, h: int) -> None: ...
    def update(self, app, dt: float) -> None: ...
    def compute(self, app, encoder) -> None: ...
    def draw(self, app, pass_) -> None: ...
    def ui(self, app) -> None: ...
    def save_state(self) -> dict: return {}
    def load_state(self, d: dict) -> None: ...
    def shutdown(self, app) -> None: ...

    def label(self) -> str:
        return self.title or self.name.replace("_", " ").title()
