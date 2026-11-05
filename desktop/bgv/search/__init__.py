"""Search, reporting and help features of the desktop viewer (see desktop/docs/SEARCH_HOOKS.md).

    addrindex   address index (layers/address_index.json.gz) loader + typeahead + per-storm flood arrays
    tiles       depth tiles (county COG max_depth.tif): point / footprint sampling, source labels, comparison with the index
    facilities  critical facilities (layers/infrastructure.geojson) with depths from the index and from the tiles
    roadrisk    flooded-road lists (layers/roads_flooded/<domain>/<storm>.json) + way geometry for zoom-to-road
    report      one-click risk report (self-contained HTML + PNG)
    assess      per-address assessment: index + tile values and their comparison
    controller  state and background loading behind the panels (no imgui)
    helpdoc     Help topics, shortcuts, STORM_DATA_SOURCES.md rendering
The imgui side is bgv/ui/panels_search.py and bgv/ui/help.py (entry points called from the shared UI files).
Every module in this package is pure CPU (no imgui, no wgpu) and unit-tested.
"""
