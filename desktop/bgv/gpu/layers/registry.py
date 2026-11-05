"""Layer registry: ONE line per layer, in DRAW ORDER.  Each line is "module:Class".

Later features add their file `bgv/gpu/layers/<x>.py` with the class named here; a line whose module does not exist
yet is skipped silently, so agents never have to edit each other's files.  A layer that raises during init is
disabled with a message in the sidebar log - it can never take the app down.
"""
LAYERS = [
    "bgv.gpu.layers.basemap:BasemapLayer",        # optional OSM map background (owns app.basemap; draws nothing itself)
    "bgv.gpu.layers.terrain:TerrainLayer",       # hillshaded terrain (+ sky)
    "bgv.gpu.layers.roads:RoadsLayer",            # draped roads coloured by flooding           (later agent)
    "bgv.gpu.layers.buildings:BuildingsLayer",    # extruded buildings, tier colours             (later agent)
    "bgv.gpu.layers.terrain:WaterLayer",          # water surface, depth ramp, ripples, foam
    # NOT REGISTERED (experimental, unfinished): "bgv.gpu.layers.tiledepth:TileDepthLayer" - solver depth draped on the terrain. Registering it
    # currently trips an imgui ID-stack assert in the sidebar (root cause not found) and it hitches while the window is re-read; see docs/TILES.md.
    "bgv.gpu.layers.compare:CompareLayer",        # ANALYSIS: storm swipe (storm B water) / difference layer, right after storm A's water
    "bgv.gpu.layers.buildings:SelectionOverlay",  # selected-building highlight, drawn after the water (visible when submerged)
    "bgv.gpu.layers.flowpaths:FlowpathsLayer",    # flow-paths colour map on the terrain (off by default)
    "bgv.gpu.layers.particles:ParticlesLayer",    # GPU flow streaks                             (later agent)
    "bgv.gpu.layers.arrows:ArrowsLayer",          # instanced flow arrows                        (later agent)
    "bgv.gpu.layers.cinematic:CinematicLayer",    # bookmarks, tour, time-lapse, clip export (controller; UI via ui/hooks.py)
    "bgv.gpu.layers.panels:PanelsLayer",          # statistics / probe / ranking panels (UI)     (later agent)
    "bgv.analysis.layer:AnalysisLayer",           # ANALYSIS: state + background services of the analysis tools (draws nothing)
    "bgv.gpu.layers.overlay:OverlayLayer",        # labels, scale bar, picking marker            (later agent)
]


def load_layer_classes(log=print):
    """Import every registered class.  Returns [(spec, cls)]; missing optional modules are skipped."""
    import importlib
    out = []
    for spec in LAYERS:
        mod_name, _, cls_name = spec.partition(":")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            if e.name == mod_name:          # that exact module is not written yet
                continue
            log(f"[bgv] layer {spec}: import failed: {e}")
            continue
        except Exception as e:              # a broken layer must not break the app
            log(f"[bgv] layer {spec}: import failed: {type(e).__name__}: {e}")
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            log(f"[bgv] layer {spec}: class {cls_name} not found")
            continue
        out.append((spec, cls))
    return out
