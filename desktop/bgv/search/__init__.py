"""Search, reporting and help features of the desktop viewer (see desktop/docs/SEARCH_HOOKS.md).

    addrindex   address index (layers/address_index.json.gz) loader + typeahead + per-storm flood arrays
    tiles       depth tiles (county COG max_depth.tif): point / footprint sampling, source labels, comparison with the index
    facilities  critical facilities (layers/infrastructure.geojson) with depths from the index and from the tiles
    roadrisk    flooded-road lists (layers/roads_flooded/<domain>/<storm>.json) + way geometry for zoom-to-road
    report      one-click risk report (self-contained HTML + PNG)
    hooks       the few entry points the shared UI files call
Every module except `hooks` is pure CPU (no imgui, no wgpu) and unit-tested.
"""
