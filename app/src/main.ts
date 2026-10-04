import maplibregl from 'maplibre-gl'
import type { ExpressionSpecification, Map as MapLibreMap } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import './style.css'

// ─── Config ─────────────────────────────────────────────────────────────────

const CENTER: [number, number] = [-86.419, 36.972]
const DEFAULT_EXAGGERATION = 1.75

// Approximate elevation thresholds (ft MSL) per rainfall level from rainfall_sim.py.
// Update these from the console output when you run: python rainfall_sim.py
// They reflect np.nanpercentile(elev[valid], p) for each percentile p in RAIN_PCT.
const ELEV_THRESHOLDS: [number, number][] = [
  [403, 0.25],
  [419, 0.50],
  [440, 0.75],
  [464, 1.00],
  [493, 1.25],
  [525, 1.50],
  [565, 2.00],
  [614, 2.50],
  [663, 3.00],
]
const ELEV_MIN = 391
const ELEV_MAX = 700  // slider max; full DEM top is 797 but ~663 covers 100-yr scenario

// RGBA(0, 105, 148, 0.65) expressed for MapLibre
const WATER_COLOR = 'rgba(0,105,148,0.65)'
const WATER_OUTLINE = 'rgba(0,80,120,0.9)'

// Building vulnerability palette: grey (safe) → amber (500-yr) → crimson (100-yr)
const BLDG_GREY    = '#94a3b8'
const BLDG_AMBER   = '#d4870a'
const BLDG_CRIMSON = '#c0392b'
const BLDG_FLOOD   = 'rgba(0,105,148,0.9)'

const LAYER_FILES: Record<string, string> = {
  flood:    '/layers/flood_sim.geojson',
  fema:     '/layers/fema.geojson',
  sinkholes:'/layers/sinkholes.geojson',
  riverine: '/layers/riverine.geojson',
  roads:    '/layers/roads.geojson',
  buildings:'/layers/buildings.geojson',
  parcels:  '/layers/all_parcels.geojson',
}

// ─── State ───────────────────────────────────────────────────────────────────

let currentRain = 0
let currentElevFt = ELEV_MIN
let buildingsData: GeoJSON.FeatureCollection | null = null
let allParcelsData: GeoJSON.FeatureCollection | null = null
const loaded: Record<string, boolean> = {}

// ─── Map Expressions ─────────────────────────────────────────────────────────

/** Building vulnerability color: crimson → amber → grey, or flood-blue when inundated. */
function buildingVulnColor(rainLevel: number): ExpressionSpecification {
  if (rainLevel === 0) {
    return ['case',
      ['<=', ['coalesce', ['get', 'flood_at_rain_in'], 9], 1.0], BLDG_CRIMSON,
      ['<=', ['coalesce', ['get', 'flood_at_rain_in'], 9], 3.0], BLDG_AMBER,
      BLDG_GREY
    ]
  }
  return ['case',
    // Currently inundated → water blue
    ['<=', ['coalesce', ['get', 'flood_at_rain_in'], 9], rainLevel], BLDG_FLOOD,
    // Dry — apply vulnerability tier
    ['<=', ['coalesce', ['get', 'flood_at_rain_in'], 9], 1.0], BLDG_CRIMSON,
    ['<=', ['coalesce', ['get', 'flood_at_rain_in'], 9], 3.0], BLDG_AMBER,
    BLDG_GREY
  ]
}

const ROAD_RISK_COLOR: ExpressionSpecification = ['match',
  ['coalesce', ['get', 'risk_tier'], ['get', 'flood_risk_tier'], 'low'],
  'critical', '#e74c3c',
  'high',     '#e67e22',
  'moderate', '#f1c40f',
  'low',      '#2ecc71',
  '#aaa'
]

const FEMA_100_FILTER: ExpressionSpecification = ['in', ['get', 'FLD_ZONE'], ['literal', ['AE', 'A', 'AO', 'AH']]]
const FEMA_500_FILTER: ExpressionSpecification = ['in', ['get', 'FLD_ZONE'], ['literal', ['X', 'X500']]]

/** Sinkhole fill by risk tier depth */
const SINK_COLOR: ExpressionSpecification = ['match',
  ['coalesce', ['get', 'risk_tier'], ['get', 'flood_risk_tier'], 'low'],
  'critical', '#7d3c98',
  'high',     '#9b59b6',
  'moderate', '#c39bd3',
  'low',      '#e8daef',
  '#e8daef'
]

// ─── Map Initialization ──────────────────────────────────────────────────────

const map: MapLibreMap = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      osm: {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a>',
        maxzoom: 19
      },
      terrain: {
        type: 'raster-dem',
        tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
        encoding: 'terrarium',
        tileSize: 256,
        maxzoom: 15
      }
    },
    layers: [{
      id: 'osm',
      type: 'raster',
      source: 'osm',
      paint: { 'raster-opacity': 0.6, 'raster-saturation': -0.5, 'raster-brightness-min': 0.02 }
    }],
    terrain: { source: 'terrain', exaggeration: DEFAULT_EXAGGERATION }
  },
  center: CENTER,
  zoom: 12,
  pitch: 52,
  bearing: -14,
  maxPitch: 80,
  hash: true
})

map.addControl(new maplibregl.NavigationControl(), 'top-right')
map.addControl(new maplibregl.ScaleControl({ unit: 'imperial' }), 'bottom-right')
map.addControl(new maplibregl.GeolocateControl({ positionOptions: { enableHighAccuracy: true } }), 'top-right')

// ─── Helpers ─────────────────────────────────────────────────────────────────

function setMsg(msg: string, pct?: number) {
  const el = document.getElementById('load-msg')
  const bar = document.getElementById('load-bar')
  if (el) el.textContent = msg
  if (bar && pct != null) bar.style.width = pct + '%'
}

function setCount(id: string, val: number | string | null) {
  const el = document.getElementById(id)
  if (el && val != null) el.textContent = typeof val === 'number' ? val.toLocaleString() : val
}

async function fetchGeoJSON(path: string): Promise<GeoJSON.FeatureCollection | null> {
  try {
    const r = await fetch(path)
    if (!r.ok) return null
    return await r.json()
  } catch {
    return null
  }
}

/** Map elevation in feet to the nearest rainfall level using ELEV_THRESHOLDS. */
function elevToRain(elevFt: number): number {
  for (const [thresh, rain] of ELEV_THRESHOLDS) {
    if (elevFt <= thresh) return rain
  }
  return 3.0
}

// ─── Layer Management ────────────────────────────────────────────────────────

function addFloodLayer(data: GeoJSON.FeatureCollection) {
  map.addSource('flood-src', { type: 'geojson', data })
  map.addLayer({
    id: 'flood-fill',
    type: 'fill',
    source: 'flood-src',
    filter: ['<=', ['get', 'min_rain_in'], 0],
    layout: { visibility: 'none' },
    paint: {
      'fill-color': WATER_COLOR,
      'fill-outline-color': WATER_OUTLINE,
      'fill-opacity': 1  // color already has alpha
    }
  })
  loaded.flood = true
  setCount('cnt-flood', (data.features?.length ?? 0) + ' bands')
}

function addFemaLayers(data: GeoJSON.FeatureCollection) {
  map.addSource('fema-src', { type: 'geojson', data })

  // 100-year: Zone AE, A (crimson)
  map.addLayer({
    id: 'fema-ae-fill',
    type: 'fill',
    source: 'fema-src',
    filter: FEMA_100_FILTER,
    paint: { 'fill-color': '#c0392b', 'fill-opacity': 0.35 }
  })
  map.addLayer({
    id: 'fema-ae-line',
    type: 'line',
    source: 'fema-src',
    filter: FEMA_100_FILTER,
    paint: { 'line-color': '#e74c3c', 'line-width': 1.5, 'line-opacity': 0.8 }
  })

  // 500-year: Zone X (amber)
  map.addLayer({
    id: 'fema-x-fill',
    type: 'fill',
    source: 'fema-src',
    filter: FEMA_500_FILTER,
    paint: { 'fill-color': '#d4870a', 'fill-opacity': 0.25 }
  })
  map.addLayer({
    id: 'fema-x-line',
    type: 'line',
    source: 'fema-src',
    filter: FEMA_500_FILTER,
    paint: { 'line-color': '#e67e22', 'line-width': 1, 'line-opacity': 0.6 }
  })

  loaded['fema-ae'] = true
  loaded['fema-x'] = true

  const ae = data.features?.filter(f => ['AE','A','AO','AH'].includes(String(f.properties?.['FLD_ZONE']))).length ?? 0
  const x  = data.features?.filter(f => ['X','X500'].includes(String(f.properties?.['FLD_ZONE']))).length ?? 0
  setCount('cnt-fema-ae', ae)
  setCount('cnt-fema-x', x)
}

function addSinkholeLayers(data: GeoJSON.FeatureCollection) {
  map.addSource('sinkholes-src', { type: 'geojson', data })
  map.addLayer({
    id: 'sinkholes-fill',
    type: 'fill',
    source: 'sinkholes-src',
    paint: { 'fill-color': SINK_COLOR, 'fill-opacity': 0.55 }
  })
  map.addLayer({
    id: 'sinkholes-line',
    type: 'line',
    source: 'sinkholes-src',
    paint: { 'line-color': '#c39bd3', 'line-width': 0.7, 'line-opacity': 0.8 }
  })
  loaded.sinkholes = true
  setCount('cnt-sinkholes', data.features?.length ?? 0)
}

function addRiverineLayers(data: GeoJSON.FeatureCollection) {
  map.addSource('riverine-src', { type: 'geojson', data })
  map.addLayer({
    id: 'riverine-fill',
    type: 'fill',
    source: 'riverine-src',
    paint: { 'fill-color': '#3498db', 'fill-opacity': 0.28 }
  })
  map.addLayer({
    id: 'riverine-line',
    type: 'line',
    source: 'riverine-src',
    paint: { 'line-color': '#5dade2', 'line-width': 1 }
  })
  loaded.riverine = true
  setCount('cnt-riverine', data.features?.length ?? 0)
}

function addRoadsLayer(data: GeoJSON.FeatureCollection) {
  map.addSource('roads-src', { type: 'geojson', data })
  map.addLayer({
    id: 'roads-line',
    type: 'line',
    source: 'roads-src',
    paint: {
      'line-color': ROAD_RISK_COLOR,
      'line-width': ['interpolate', ['linear'], ['zoom'], 11, 2, 16, 5],
      'line-opacity': 0.9,
      'line-cap': 'round',
      'line-join': 'round'
    }
  })
  loaded.roads = true
  setCount('cnt-roads', data.features?.length ?? 0)
}

function addParcelsLayer(data: GeoJSON.FeatureCollection) {
  map.addSource('all-parcels-src', { type: 'geojson', data })
  map.addLayer({
    id: 'all-parcels-fill',
    type: 'fill',
    source: 'all-parcels-src',
    layout: { visibility: 'none' },
    paint: {
      'fill-color': '#2ecc71',
      'fill-opacity': 0.4,
      'fill-outline-color': 'rgba(255,255,255,0.05)'
    }
  })
  loaded['all-parcels'] = true
  setCount('cnt-all-parcels', data.features?.length ?? 0)
}

function addBuildingsLayer(data: GeoJSON.FeatureCollection) {
  map.addSource('buildings-src', { type: 'geojson', data })
  map.addLayer({
    id: 'buildings-ext',
    type: 'fill-extrusion',
    source: 'buildings-src',
    paint: {
      'fill-extrusion-color': buildingVulnColor(0),
      'fill-extrusion-height': ['coalesce', ['get', 'height_m'], 5.5],
      'fill-extrusion-base': 0,
      'fill-extrusion-opacity': 0.85,
      'fill-extrusion-vertical-gradient': true
    }
  })
  loaded.buildings = true
  setCount('cnt-buildings', data.features?.length ?? 0)
}

// Hillshade layer (added/removed dynamically)
function addHillshadeLayer(azimuth = 315) {
  if (map.getLayer('hillshade')) map.removeLayer('hillshade')
  map.addLayer({
    id: 'hillshade',
    type: 'hillshade',
    source: 'terrain',
    paint: {
      'hillshade-exaggeration': 0.45,
      'hillshade-highlight-color': '#f0f0f0',
      'hillshade-shadow-color': '#1a1a2e',
      'hillshade-accent-color': '#888',
      'hillshade-illumination-direction': azimuth,
      'hillshade-illumination-anchor': 'map'
    }
  }, 'osm')   // insert below OSM tiles so terrain shows through
}

// ─── State Updates ────────────────────────────────────────────────────────────

function applyFloodState(rain: number, elevFt: number) {
  currentRain = rain
  currentElevFt = elevFt

  // Update UI labels
  const rainEl = document.getElementById('rain-val')
  const elevEl = document.getElementById('elev-val')
  if (rainEl) rainEl.textContent = rain.toFixed(2) + '"'
  if (elevEl) elevEl.textContent = elevFt.toFixed(0) + ' ft'

  // Sync the other slider if it's linked
  const rainSlider = document.getElementById('rain-slider') as HTMLInputElement | null
  const elevSlider = document.getElementById('elev-slider') as HTMLInputElement | null

  // Flood polygon filter
  if (loaded.flood) {
    const floodCb = document.querySelector('[data-layer="flood"]') as HTMLInputElement | null
    const show = rain > 0 && (floodCb?.checked ?? true)
    map.setLayoutProperty('flood-fill', 'visibility', show ? 'visible' : 'none')
    if (show) map.setFilter('flood-fill', ['<=', ['get', 'min_rain_in'], rain])
  }

  // Recolor buildings
  if (loaded.buildings) {
    map.setPaintProperty('buildings-ext', 'fill-extrusion-color', buildingVulnColor(rain))
    // Slightly raise flooded buildings for visual pop
    const h: ExpressionSpecification = rain === 0
      ? ['coalesce', ['get', 'height_m'], 5.5]
      : ['case',
          ['<=', ['coalesce', ['get', 'flood_at_rain_in'], 9], rain],
          ['max', ['coalesce', ['get', 'height_m'], 5.5], 9],
          ['coalesce', ['get', 'height_m'], 5.5]
        ]
    map.setPaintProperty('buildings-ext', 'fill-extrusion-height', h)
  }

  // Update stats
  updateStats(rain, elevFt)
}

function updateStats(rain: number, elevFt: number) {
  let flooded = 0, atRisk = 0, maxDepth = 0

  for (const f of (buildingsData?.features ?? [])) {
    const fa = (f.properties?.flood_at_rain_in ?? 9) as number
    if (fa <= rain) {
      flooded++
      const ge = (f.properties?.ground_elev_ft ?? elevFt) as number
      const d = Math.max(0, elevFt - ge)
      if (d > maxDepth) maxDepth = d
    }
  }
  for (const f of (allParcelsData?.features ?? [])) {
    const fa = (f.properties?.flood_at_rain_in ?? 9) as number
    if (fa <= rain) atRisk++
  }

  const bldgEl  = document.getElementById('stat-bldg')
  const parcEl  = document.getElementById('stat-parcels')
  const depthEl = document.getElementById('stat-depth')
  if (bldgEl)  bldgEl.textContent  = rain === 0 ? '—' : flooded.toLocaleString()
  if (parcEl)  parcEl.textContent  = rain === 0 ? '—' : atRisk.toLocaleString()
  if (depthEl) depthEl.textContent = rain === 0 ? '—' : maxDepth.toFixed(1) + ' ft'
}

// ─── Feature Inspection ──────────────────────────────────────────────────────

const HIDE_PROPS = new Set([
  'height_m','geometry','index','__index_level_0__',
  'damage_pct_2in','damage_usd_2in','sale_price','prop_value_usd','ead_usd','damage_risk_tier'
])

const LABEL_MAP: Record<string, string> = {
  PVA_PARCEL: 'Parcel ID', ADDRESS: 'Address', SiteAddress: 'Site Address',
  ACRES: 'Acreage', LAND_USE: 'Land Use', ZONING: 'Zoning',
  year_purchase: 'Year Purchased', Jurisdiction: 'Jurisdiction',
  flood_at_rain_in: 'Floods At (rain)', FLD_ZONE: 'FEMA Zone',
  SFHA_TF: 'Special Flood Hazard', risk_tier: 'Risk Tier',
  name: 'Name', building: 'Building Type', amenity: 'Amenity',
}

const RETURN_PERIOD: Record<number, number> = {
  0.25:1, 0.50:2, 0.75:3, 1.00:5, 1.25:7, 1.50:10, 2.00:25, 2.50:50, 3.00:100
}

function fmtValue(k: string, v: unknown): string | null {
  if (v == null || v === '') return null
  if (k === 'ACRES') return typeof v === 'number' ? v.toFixed(3) + ' ac' : String(v)
  if (k === 'flood_at_rain_in') {
    const n = Number(v)
    if (n >= 9) return 'Does not flood at ≤ 3"'
    const rounded = Math.round(n * 100) / 100
    const yr = RETURN_PERIOD[rounded]
    return n.toFixed(2) + '"' + (yr ? ` (${yr}-yr event)` : '')
  }
  return String(v)
}

function showInfo(feature: maplibregl.MapGeoJSONFeature, lngLat: maplibregl.LngLat) {
  const p = feature.properties ?? {}
  const layerNames: Record<string, string> = {
    'buildings-ext':   'Building',
    'fema-ae-fill':    'FEMA 100-yr Zone',
    'fema-x-fill':     'FEMA 500-yr Zone',
    'sinkholes-fill':  'Karst Sinkhole',
    'riverine-fill':   'Riverine Zone',
    'roads-line':      'Flood-Prone Road',
    'flood-fill':      'Flood Zone',
    'all-parcels-fill':'Parcel',
  }
  const titleEl = document.getElementById('info-title')
  if (titleEl) titleEl.textContent = layerNames[feature.layer.id] ?? 'Feature'

  // Terrain elevation from MapLibre
  const terrainM = map.queryTerrainElevation(lngLat, { exaggerated: false })
  const terrainFt = terrainM != null && isFinite(terrainM) ? terrainM * 3.28084 : null

  // Water depth if slider is active
  const waterDepth = (currentElevFt > ELEV_MIN && terrainFt != null)
    ? Math.max(0, currentElevFt - terrainFt)
    : null

  // Vulnerability badge
  const fa = p.flood_at_rain_in as number | undefined
  let badge = ''
  if (fa != null) {
    if (fa <= 1.0)      badge = '<span class="risk-badge risk-crimson">100-YR FLOOD ZONE</span>'
    else if (fa <= 3.0) badge = '<span class="risk-badge risk-amber">500-YR / KARST ZONE</span>'
    else                badge = '<span class="risk-badge risk-safe">ABOVE FLOOD LINE</span>'
  }

  const elevRow = terrainFt != null
    ? `<tr><td>Ground Elevation</td><td><span class="elev-badge">${terrainFt.toFixed(0)} ft (${(terrainFt/3.28084).toFixed(0)} m) MSL</span></td></tr>`
    : ''
  const depthRow = waterDepth != null && waterDepth > 0
    ? `<tr><td>Water Depth</td><td><span class="depth-badge">${waterDepth.toFixed(1)} ft above ground</span></td></tr>`
    : ''

  const rows = Object.entries(p)
    .filter(([k]) => !HIDE_PROPS.has(k))
    .map(([k, v]) => {
      const d = fmtValue(k, v)
      if (!d) return ''
      const label = LABEL_MAP[k] ?? k.replace(/_/g, ' ')
      return `<tr><td>${label}</td><td>${d}</td></tr>`
    })
    .filter(Boolean)
    .join('')

  const tableEl = document.getElementById('info-table')
  if (tableEl) {
    tableEl.innerHTML =
      (badge ? `<tr><td colspan="2">${badge}</td></tr>` : '') +
      elevRow + depthRow +
      (rows || '<tr><td colspan="2" style="color:#555">No attributes</td></tr>')
  }

  const infoEl = document.getElementById('info')
  if (infoEl) infoEl.style.display = 'block'
}

// ─── UI Wiring ────────────────────────────────────────────────────────────────

function wireUI() {

  // ── Rainfall slider ──────────────────────────────────────────────────────
  const rainSlider = document.getElementById('rain-slider') as HTMLInputElement
  rainSlider.addEventListener('input', () => {
    const rain = parseFloat(rainSlider.value)
    // Sync elevation slider to nearest threshold
    const matchElev = ELEV_THRESHOLDS.find(([,r]) => r >= rain)?.[0] ?? ELEV_MAX
    const elevSlider = document.getElementById('elev-slider') as HTMLInputElement
    elevSlider.value = String(matchElev)
    applyFloodState(rain, matchElev)
  })

  // ── Flood elevation slider (new) ─────────────────────────────────────────
  const elevSlider = document.getElementById('elev-slider') as HTMLInputElement
  elevSlider.addEventListener('input', () => {
    const elev = parseFloat(elevSlider.value)
    const rain = elevToRain(elev)
    // Sync rain slider
    rainSlider.value = String(rain)
    applyFloodState(rain, elev)
  })

  // ── Vertical exaggeration ────────────────────────────────────────────────
  const exSlider = document.getElementById('exag-slider') as HTMLInputElement
  exSlider.addEventListener('input', () => {
    const v = parseFloat(exSlider.value)
    const label = document.getElementById('exag-val')
    if (label) label.textContent = v.toFixed(2) + '×'
    map.setTerrain({ source: 'terrain', exaggeration: v })
  })

  // ── Basemap selector ─────────────────────────────────────────────────────
  const basemapSel = document.getElementById('basemap-sel') as HTMLSelectElement
  basemapSel.addEventListener('change', () => {
    const tiles: Record<string, string> = {
      topo:      'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      dark:      'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}.png',
    }
    const url = tiles[basemapSel.value]
    if (url && map.getSource('osm')) {
      // @ts-expect-error — MapLibre allows tile URL update on raster sources
      (map.getSource('osm') as maplibregl.RasterTileSource).setTiles([url])
    }
  })

  // ── Hillshade toggle + direction ─────────────────────────────────────────
  const hillshadeToggle = document.getElementById('hillshade-toggle') as HTMLInputElement
  const hillshadeDirWrap = document.getElementById('hillshade-dir-wrap') as HTMLElement
  const hillshadeDirSel  = document.getElementById('hillshade-dir') as HTMLSelectElement

  hillshadeToggle.addEventListener('change', () => {
    hillshadeDirWrap.style.display = hillshadeToggle.checked ? 'flex' : 'none'
    if (hillshadeToggle.checked) {
      addHillshadeLayer(parseInt(hillshadeDirSel.value))
    } else {
      if (map.getLayer('hillshade')) map.removeLayer('hillshade')
    }
  })
  hillshadeDirSel.addEventListener('change', () => {
    if (hillshadeToggle.checked) addHillshadeLayer(parseInt(hillshadeDirSel.value))
  })

  // ── Layer visibility toggles ─────────────────────────────────────────────
  const layerGroups: Record<string, string[]> = {
    flood:        ['flood-fill'],
    'fema-ae':    ['fema-ae-fill', 'fema-ae-line'],
    'fema-x':     ['fema-x-fill',  'fema-x-line'],
    sinkholes:    ['sinkholes-fill', 'sinkholes-line'],
    riverine:     ['riverine-fill',  'riverine-line'],
    roads:        ['roads-line'],
    buildings:    ['buildings-ext'],
    'all-parcels':['all-parcels-fill'],
  }

  document.querySelectorAll<HTMLInputElement>('[data-layer]').forEach(cb => {
    cb.addEventListener('change', () => {
      const vis = cb.checked ? 'visible' : 'none'
      const ids = layerGroups[cb.dataset.layer ?? ''] ?? []
      ids.forEach(id => {
        if (!map.getLayer(id)) return
        if (id === 'flood-fill' && currentRain === 0 && cb.checked) return  // don't show at rain=0
        map.setLayoutProperty(id, 'visibility', vis)
      })
    })
  })

  // ── Per-layer opacity controls ───────────────────────────────────────────
  document.querySelectorAll<HTMLInputElement>('[data-opacity-layer]').forEach(slider => {
    slider.addEventListener('input', () => {
      const layerId = slider.dataset.opacityLayer!
      const val = parseFloat(slider.value)
      if (!map.getLayer(layerId)) return
      if (layerId === 'buildings-ext') {
        map.setPaintProperty(layerId, 'fill-extrusion-opacity', val)
      } else if (map.getLayer(layerId)) {
        // For fill layers, use fill-opacity
        try { map.setPaintProperty(layerId, 'fill-opacity', val) } catch {}
      }
    })
  })

  document.querySelectorAll<HTMLInputElement>('[data-opacity-group]').forEach(slider => {
    slider.addEventListener('input', () => {
      const group = slider.dataset.opacityGroup!
      const val = parseFloat(slider.value)
      const ids = layerGroups[group] ?? []
      ids.forEach(id => {
        if (!map.getLayer(id)) return
        if (id.endsWith('-fill')) {
          try { map.setPaintProperty(id, 'fill-opacity', val) } catch {}
        } else if (id.endsWith('-line')) {
          try { map.setPaintProperty(id, 'line-opacity', val) } catch {}
        }
      })
    })
  })

  // ── Feature inspection ───────────────────────────────────────────────────
  const CLICKABLE = [
    'buildings-ext','all-parcels-fill',
    'fema-ae-fill','fema-x-fill',
    'sinkholes-fill','riverine-fill',
    'roads-line','flood-fill'
  ]

  CLICKABLE.forEach(id => {
    if (!map.getLayer(id)) return
    map.on('mouseenter', id, () => { map.getCanvas().style.cursor = 'pointer' })
    map.on('mouseleave', id, () => { map.getCanvas().style.cursor = '' })
  })

  map.on('click', e => {
    const active = CLICKABLE.filter(id => map.getLayer(id))
    const feats = map.queryRenderedFeatures(e.point, { layers: active })
    if (!feats.length) {
      const infoEl = document.getElementById('info')
      if (infoEl) infoEl.style.display = 'none'
      return
    }
    showInfo(feats[0], e.lngLat)
  })

  document.getElementById('info-close')?.addEventListener('click', () => {
    const infoEl = document.getElementById('info')
    if (infoEl) infoEl.style.display = 'none'
  })

  // ── Methodology modal ────────────────────────────────────────────────────
  document.getElementById('sci-btn')?.addEventListener('click', () => {
    document.getElementById('sci-modal')?.classList.add('open')
  })
  document.getElementById('sci-close')?.addEventListener('click', () => {
    document.getElementById('sci-modal')?.classList.remove('open')
  })
  document.getElementById('sci-modal')?.addEventListener('click', e => {
    if (e.target === document.getElementById('sci-modal'))
      document.getElementById('sci-modal')?.classList.remove('open')
  })
}

// ─── Main Load ────────────────────────────────────────────────────────────────

map.on('load', async () => {
  setMsg('Loading flood simulation…', 8)

  const [floodData, femaData, sinkholeData, riverineData, roadsData] = await Promise.all([
    fetchGeoJSON(LAYER_FILES.flood),
    fetchGeoJSON(LAYER_FILES.fema),
    fetchGeoJSON(LAYER_FILES.sinkholes),
    fetchGeoJSON(LAYER_FILES.riverine),
    fetchGeoJSON(LAYER_FILES.roads),
  ])

  setMsg('Loading buildings & parcels…', 42)

  const [bldgData, parcelData] = await Promise.all([
    fetchGeoJSON(LAYER_FILES.buildings),
    fetchGeoJSON(LAYER_FILES.parcels),
  ])

  buildingsData = bldgData
  allParcelsData = parcelData

  setMsg('Rendering layers…', 80)

  if (floodData)    addFloodLayer(floodData)
  if (femaData)     addFemaLayers(femaData)
  if (sinkholeData) addSinkholeLayers(sinkholeData)
  if (riverineData) addRiverineLayers(riverineData)
  if (roadsData)    addRoadsLayer(roadsData)
  if (parcelData)   addParcelsLayer(parcelData)
  if (bldgData)     addBuildingsLayer(bldgData)

  setMsg('Wiring controls…', 95)
  wireUI()

  document.getElementById('loading')!.style.display = 'none'
  setMsg('Ready', 100)
})

map.on('error', e => console.warn('MapLibre error:', (e as any).error?.message ?? e))
