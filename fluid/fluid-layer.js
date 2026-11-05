// fluid-layer.js - MapLibre integration of the fluid view (see FLOW_FORMAT.md).
//
// Drapes the WebGL fluid renderer on a MapLibre map (3D terrain on or off) and plays a storm in time.
// The renderer draws one REGION of the flow grid (visible area clipped to the grid) into its own canvas;
// that canvas is a MapLibre 'canvas' source on a raster layer, so it is draped on the terrain like any
// raster. Velocities in the grids are DERIVED from the water-surface slope (Manning), not solver output:
// show `layer.source.label` / `.note` in the UI.
//
// ----------------------------------------------------------------------------- API
//   FluidLayer.supported()                 WebGL2 + DecompressionStream + fetch present (no map needed)
//   new FluidLayer(map, opts)              nothing is added to the map until the first setStorm resolves
//   await layer.setStorm('25yr')           loads meta, prepares the renderer, shows the first picture.
//        {force:true} reloads the same storm. {base:'./layers/flow_county'} loads the storm from another flow folder (a
//        DOMAIN such as the whole county) and makes it the layer's base from then on; the same storm key in another
//        base is a different storm (it is reloaded, nothing is shared with the old one). Resolves to the meta object.
//        Rejects with StormNotFoundError (HTTP 404 on meta.json), FlowFormatError (bad meta or grid),
//        GridMemoryError, any other Error (network, GPU) or an AbortError when a newer setStorm
//        superseded it. A failed request leaves the previous storm untouched: it keeps its picture and
//        playback and the status returns to ready/buffering ('error' event, phase 'storm'); with no previous
//        storm the layer stays clean (status 'error', no storm, nothing on the map). Once the new storm is
//        accepted the old one is disposed and its water hidden until the new storm has drawn.
//   layer.setTime(hourFloat)               continuous time, clamped to layer.timeRange. Hour h is the
//        snapshot at t = h; between hours two grids are blended; hours missing from meta.hours are dry,
//        so the water fades in from dry before the first wet hour; after the last hour the last grid stays.
//        If the grids of the requested time are not decoded yet the picture holds (status 'buffering')
//        and switches when they are there; superseded downloads are aborted.
//   layer.play(msPerHour = 1500) / pause() / togglePlay() / isPlaying
//        rAF driven with real elapsed time, loops to timeRange.loopStart after the last hour (which is held
//        for one hour of playback time). Playback never advances into grids that are not decoded: time
//        holds, status is 'buffering', no hour is skipped. Water and streaks keep moving while paused
//        unless the `freeze` option is set.
//   layer.setOptions({...}) / getOptions()  renderer options (showDepth, showFlow, showParticles, showArrows,
//        showFlowpaths, opacity, flowSpeed, particleCount, particleSpeed, particleLife, trailFade,
//        arrowSpacingPx, arrowScale, quality, minDepthShown, ...) plus the layer options
//        freeze, autoQuality, frameBudgetMs, msPerHour, pixelScale, maxPixels, loopStart.
//        getOptions() returns what was REQUESTED (the quality governor may run fewer particles, see qualityInfo).
//        showFlowpaths loads flowpaths.rgba.gz lazily, the first time it is turned on.
//        Further layer options: governorOverMs, governorRecoverMs, statsTopN, statsMinDepthM, qualityMode.
//   layer.setVisible(bool) / visible       hidden = nothing drawn, loop stopped
//   layer.setBeforeId(id|null)             draw the water directly below that layer (null = default anchors)
//   layer.on(name, fn) / off(name, fn) / once(name, fn)
//   layer.updateRegion(force = false)      recompute the drawn region now (normally done on moveend/zoomend/resize)
//   layer.dispose()                        removes layer, source, listeners, timers, GL resources, grids. Afterwards every
//        method is a safe no-op (probe invalid, statistics null, attach* resolve 0, refreshStats null) except setStorm, which rejects.
//   layer.setQualityLevel(n) / setQualityMode('auto'|'high'|'medium'|'low') / qualityMode   manual quality override (section 8)
//   layer.probe(lngLat, out?) / attachProbe(cb)    water at a point / hover helper (section 9)
//   layer.attachBuildings(fc) / attachRoads(fc) / getBuildingFlood() / getRoadFlood() / getFloodStats() / refreshStats() /
//        buildingLngLat(i) / getIndexInfo()        flood statistics (section 9, event 'stats')
//
// Properties: storm (key or null), meta, source ({kind, label, note} or null), status ({state, message}),
//   time (displayed hour), targetTime, timeRange ({min, max, loopStart, hours}), qualityLevel, qualityInfo,
//   region ([xmin, ymin, xmax, ymax] mercator or null), canvas, renderer, hasPicture, getState().
//   currentGrids() -> {hourA, hourB, mix, rgbaA, rgbaB, meta}: the grids behind the picture (null = dry),
//   for probes and statistics.
//
// Events (callbacks get one object):
//   'status'  {state: 'idle'|'loading'|'buffering'|'ready'|'error', message}
//   'time'    {hour, hourA, hourB, mix, target, buffering}  displayed time; the SAME object is reused, copy what you keep
//   'hour'    {hour, grid, dry, storm, meta}  the integer hour nearest to the displayed time changed;
//             grid is that hour's RGBA Uint8Array (read-only, do not keep it: it is evicted from the cache)
//             or null for an hour without data (dry: all zero)
//   'storm'   {storm, meta, source}         a storm was accepted and drew its first picture
//   'region'  {region, width, height}       the region/canvas was replaced
//   'quality' {level, maxLevel, pixelScale, particleFactor, reason, avgMs}
//   'stats'   {hour, storm, floodedCount, tierCounts, ranked, roads, grid, ...}  the snapshot nearest to the time (round) changed; see section 9
//   'error'   {error, phase, storm, hour}   phase: 'storm' | 'grid' | 'flowpaths' | 'render' | 'stats'
//             An hour grid that fails to load keeps the last good picture; playback waits at that hour (status 'error', retry
//             after 5 s) and skips it after two failures.
//
// Constructor options (all optional): base ('./layers/flow'), beforeId, layerId, sourceId, fetch, fetchInit,
//   scheduler ({raf, caf, now}: tests drive frames with it), autoStart, plus every option listed under setOptions.
//   beforeId defaults to the first of roads-line, all-parcels-fill, buildings-ext that exists, so the water
//   sits above the base map and the FEMA overlays and below the roads and buildings; the position is
//   corrected when those layers are added later.
//
// Section map: 1 constants/errors/emitter, 2 class core, 3 storm loading, 4 time and grid scheduler, 5 region and
// map integration (incl. map context loss), 6 options, 7 frame loop, 8 quality governor, 9 probes/statistics,
// 10 teardown.

import { FluidRenderer } from './fluid-renderer.js';
import { loadStorm, FlowFormatError, GridMemoryError, LRU_HOURS, PREFETCH_CONCURRENCY, mercToLngLat, lngLatToMerc } from './fluid-data.js';
import { buildBuildingIndex, buildRoadIndex, evaluateBuildings, evaluateRoads, summarizeGrid, tierOfDepth } from './fluid-stats.js';

export { FlowFormatError, GridMemoryError };

// ------------------------------------------------------------------ 1. constants

const DEFAULT_BASE = './layers/flow';
const DEFAULT_LAYER_ID = 'fluid-water';
const BEFORE_CANDIDATES = ['roads-line', 'all-parcels-fill', 'buildings-ext'];
const MERC_WORLD = 2 * Math.PI * 6378137;      // mercator metres around the world
const TILE_PX = 512;                           // MapLibre world size is TILE_PX * 2^zoom CSS pixels

const SOURCE_LABELS = {
  rasters: 'Full-precision (solver depth rasters)',
  tiers: 'Preview from depth tiers',
};

// Keys handled by the layer itself; every other key of setOptions goes to the renderer.
const LAYER_DEFAULTS = Object.freeze({
  freeze: false,         // true: water and streaks stand still (time changes still redraw)
  autoQuality: true,     // let the frame-time governor lower canvas resolution and particle count
  frameBudgetMs: 28,     // average frame time above this for ~2 s lowers quality
  msPerHour: 1500,       // playback speed
  pixelScale: 1,         // canvas pixels per CSS pixel (times devicePixelRatio) at quality level 0
  maxPixels: 4.2e6,      // canvas pixel cap at quality level 0 (2048 x 2048)
  loopStart: null,       // hour playback loops to; null = one hour before the first wet hour
  governorOverMs: 2000,  // frame time above the budget for this long lowers quality (half of it when far above)
  governorRecoverMs: 10000,   // headroom for this long (times the flap back-off) raises quality one level
  statsTopN: 200,        // length of the ranked building list of the 'stats' event (the full order is always given)
  statsMinDepthM: 0.1,   // "flooded" threshold of the building and road statistics
});
const LAYER_KEYS = Object.keys(LAYER_DEFAULTS);

// Quality ladder, level 0 = best. The canvas pixel scale drops first, then the particle count, then the
// renderer's own shader quality. `px` multiplies pixelScale and maxPixels at that level.
export const QUALITY_LEVELS = Object.freeze([
  { px: 1.00, particles: 1.00, shader: 2 },
  { px: 0.80, particles: 1.00, shader: 2 },
  { px: 0.64, particles: 1.00, shader: 2 },
  { px: 0.50, particles: 1.00, shader: 2 },
  { px: 0.50, particles: 0.50, shader: 2 },
  { px: 0.40, particles: 0.50, shader: 1 },
  { px: 0.32, particles: 0.25, shader: 1 },
  { px: 0.25, particles: 0.12, shader: 0 },
].map((l) => Object.freeze(l)));

const MAX_SIDE = 4096;             // longest canvas side asked of the renderer
const STALL_MS = 1000;            // a longer gap between frames is a stall (debugger, suspended page), not slow rendering
const REGION_DEBOUNCE_MS = 140;
const REGION_MOVE_THROTTLE_MS = 300;
const DENS_CAP_PX_PER_CELL = 16;
const REGION_MARGIN = 0.22;       // visible area is enlarged by this fraction per side before drawing
const MIN_MARGIN_PX = 64;          // CSS pixels of the view kept around the visible area
const DENSITY_LO = 0.7, DENSITY_HI = 1.6;    // canvas density vs wanted before the region is replaced
const GOVERNOR_SETTLE_MS = 1200;
const HEADROOM_FRACTION = 0.75;    // average frame time below this share of the budget counts as headroom (a 60 Hz display idles at 16.7 ms)
const CADENCE_MARGIN = 1.35;        // budget over a detected display cadence
const CADENCE_MAX_MS = 45;         // slower than about 22 Hz is not a display cadence, it is load
const CADENCE_CHAIN_MS = 20000;     // step-downs closer together than this form one chain (see _govern)
const FLAP_FORGET_MS = 120000;     // this long without a step down right after a step up resets the recovery back-off
const FAIL_COOLDOWN_MS = 5000;
const SKIP_AFTER_FAILS = 2;        // playback skips an hour whose grid failed this many times
const SLICE_TARGET_MS = 25;        // index builds yield to the UI about this often
const QUALITY_MODES = Object.freeze({ high: 0, medium: 3, low: 6 });   // levels of the manual overrides
const CARDINALS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const STILL_MS = 0.02;             // slower water has no direction worth showing

const clamp = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));

// ------------------------------------------------------------------ errors

export class StormNotFoundError extends Error {
  constructor(storm, url, message) {
    super(message || `Storm "${storm}" was not found (${url})`);
    this.name = 'StormNotFoundError';
    this.storm = storm;
    this.url = url;
  }
}

function abortError(message) {
  return new DOMException(message || 'The operation was aborted.', 'AbortError');
}

// ------------------------------------------------------------------ emitter

class Emitter {
  constructor() { this._ls = Object.create(null); }
  on(name, fn) {
    if (typeof fn !== 'function') throw new TypeError('listener must be a function');
    const a = this._ls[name];
    this._ls[name] = a ? a.concat([fn]) : [fn];   // copy on write: emit() iterates without copying
    return this;
  }
  off(name, fn) {
    const a = this._ls[name];
    if (!a) return this;
    const b = a.filter((f) => f !== fn && f._orig !== fn);
    if (b.length) this._ls[name] = b; else delete this._ls[name];
    return this;
  }
  once(name, fn) {
    const w = (e) => { this.off(name, w); fn(e); };
    w._orig = fn;
    return this.on(name, w);
  }
  emit(name, payload) {
    const a = this._ls[name];
    if (!a) return;
    for (let i = 0; i < a.length; i++) {
      try { a[i](payload); } catch (e) { console.error(`[FluidLayer] '${name}' listener threw`, e); }
    }
  }
  clear() { this._ls = Object.create(null); }
}

// ------------------------------------------------------------------ statistics helpers

// Yield to the event loop without the 4 ms clamp and the background throttling of setTimeout.
const _yieldQueue = [];
let _yieldChannel = null;
function yieldNow() {
  return new Promise((resolve) => {
    if (typeof MessageChannel === 'undefined') { setTimeout(resolve, 0); return; }
    if (!_yieldChannel) {
      _yieldChannel = new MessageChannel();
      _yieldChannel.port1.onmessage = () => { const f = _yieldQueue.shift(); if (f) f(); };
    }
    _yieldQueue.push(resolve);
    _yieldChannel.port2.postMessage(0);
  });
}

function gridSignature(meta) {
  return `${meta.bounds_merc.join(',')}|${meta.cell_m}|${meta.width}|${meta.height}`;
}

function concatTyped(Ctor, arrays, total) {
  const out = new Ctor(total);
  let o = 0;
  for (const a of arrays) { out.set(a, o); o += a.length; }
  return out;
}

// Joins the indices of consecutive feature slices into the index fluid-stats would have built in one go.
function mergeIndexParts(kind, parts) {
  const first = parts[0];
  let n = 0;
  for (const p of parts) n += p.count;
  if (kind === 'buildings') {
    const offsets = new Int32Array(n + 1);
    let shift = 0, at = 0, np = 0;
    for (const p of parts) {
      for (let j = 0; j < p.count; j++) offsets[at + j] = shift + p.offsets[j];
      shift += p.offsets[p.count]; at += p.count; np += p.pixels.length;
    }
    offsets[n] = shift;
    return {
      ...first, count: n, offsets, pixels: concatTyped(Int32Array, parts.map((p) => p.pixels), np),
      centroids: concatTyped(Float64Array, parts.map((p) => p.centroids), 2 * n),
      inGrid: concatTyped(Uint8Array, parts.map((p) => p.inGrid), n), _maxCode: new Uint8Array(n),
    };
  }
  const featOffsets = new Int32Array(n + 1);
  let shift = 0, at = 0, inGrid = 0;
  for (const p of parts) {
    for (let j = 0; j < p.count; j++) featOffsets[at + j] = shift + p.featOffsets[j];
    shift += p.featOffsets[p.count]; at += p.count; inGrid += p.inGridLengthM;
  }
  featOffsets[n] = shift;
  const lengthM = concatTyped(Float64Array, parts.map((p) => p.lengthM), n);
  let total = 0;                                   // summed feature by feature like the one-piece build, so the bits agree
  for (let i = 0; i < n; i++) total += lengthM[i];
  return {
    ...first, count: n, featOffsets, pixel: concatTyped(Int32Array, parts.map((p) => p.pixel), shift),
    weightM: concatTyped(Float64Array, parts.map((p) => p.weightM), shift), lengthM, totalLengthM: total, inGridLengthM: inGrid,
  };
}

function featuresOf(fc, what) {
  if (Array.isArray(fc)) return fc;
  if (fc && Array.isArray(fc.features)) return fc.features;
  throw new TypeError(`${what}: expected a GeoJSON FeatureCollection or an array of features`);
}

const ZERO_GRID = Object.freeze({ isZeroGrid: true });   // fluid-stats answers it without reading a pixel (a dry hour)

function defaultScheduler() {
  return {
    raf: (fn) => requestAnimationFrame(fn),
    caf: (id) => cancelAnimationFrame(id),
    now: () => performance.now(),
  };
}

function docHidden() {
  return typeof document !== 'undefined' && (document.visibilityState === 'hidden' || document.hidden === true);
}

// ------------------------------------------------------------------ 2. class core

export class FluidLayer {
  static supported() {
    try {
      return Boolean(FluidRenderer.supported()) && typeof DecompressionStream !== 'undefined' && typeof fetch === 'function';
    } catch (e) {
      return false;
    }
  }

  constructor(map, opts = {}) {
    if (!map || typeof map.addSource !== 'function' || typeof map.addLayer !== 'function') {
      throw new TypeError('FluidLayer: the first argument must be a MapLibre map');
    }
    if (!FluidLayer.supported()) throw new Error('FluidLayer: WebGL2 or DecompressionStream is not available in this browser');
    const { base, beforeId, layerId, sourceId, fetch: fetchImpl, fetchInit, scheduler, autoStart, qualityMode, ...rest } = opts;
    this._map = map;
    this._base = base === undefined ? DEFAULT_BASE : base;
    this._beforeId = beforeId || null;
    this._layerId = layerId || DEFAULT_LAYER_ID;
    this._sourceId = sourceId || `${this._layerId}-src`;
    this._fetch = typeof fetchImpl === 'function' ? fetchImpl : null;
    this._fetchInit = fetchInit || {};
    this._sched = { ...defaultScheduler(), ...(scheduler || {}) };
    this._autoStart = autoStart !== false;
    this._em = new Emitter();
    this._disposed = false;

    this._lopt = { ...LAYER_DEFAULTS };
    const ropt = {};
    for (const k of Object.keys(rest)) {
      if (LAYER_KEYS.includes(k)) this._lopt[k] = rest[k]; else ropt[k] = rest[k];
    }
    this._renderer = new FluidRenderer(ropt);
    const ro = this._renderer.getOptions();
    this._req = { particleCount: ro.particleCount, quality: ro.quality };   // what the caller asked for; the governor scales it

    // storm / data
    this._storm = null; this._stormKey = null; this._meta = null; this._source = null;
    this._seq = 0; this._committedSeq = -1; this._metaCtrl = null; this._first = null; this._noOverlap = false;
    this._vp = null; this._vb = null; this._visSet = null; this._styleCheckPending = false; this._wantBuf = []; this._primeHandler = null;
    this._tMax = 0; this._firstWet = 0;
    this._have = new Map();       // hour -> decoded rgba, only hours the loader still caches or the picture shows
    this._jobs = new Map();       // hour -> {ctrl}
    this._failed = new Map();     // hour -> ms of the failure
    this._retryTimer = 0;
    this._fpCtrl = null; this._fpSet = false; this._fpLoading = null;

    // time
    this._time = 0; this._target = 0; this._dir = 1;
    this._playing = false; this._endHold = 0;
    this._buffering = false;
    this._gA = null; this._gB = null; this._dA = -1; this._dB = -1; this._dMix = 0;
    this._displayedOnce = false; this._shown = false; this._lastHourEvt = -1;
    this._pairBuf = { hA: 0, hB: 0, mix: 0 };
    this._candBuf = { hA: 0, hB: 0, mix: 0 };
    this._evtTime = { hour: 0, hourA: 0, hourB: 0, mix: 0, target: 0, buffering: false };

    // status
    this._status = { state: 'idle', message: '' };

    // map / region
    this._visible = true;
    this._regionApplied = false; this._region = null; this._regionPlan = null; this._active = false;
    this._regionTimer = 0; this._lastMoveCheck = 0; this._pxW = 0; this._pxH = 0;
    this._mapHooked = false; this._retryAddTimer = 0;

    // loop
    this._raf = 0; this._running = false; this._dirty = true; this._lastNow = null; this._srcPlaying = false;
    this._frames = 0;

    // governor
    this._level = 0; this._avg = 0; this._overSince = -1; this._underSince = -1; this._settleUntil = 0;
    this._cadence = 0; this._chainAvg = 0; this._chainRun = 0; this._chainStamp = -1e9;
    this._backoff = 1; this._lastUpStamp = -1e9; this._lastQualityReason = 'init';
    this._pinnedMode = null; this._mapLost = false;

    // probes and statistics
    this._probers = new Set();
    this._pbMerc = [0, 0]; this._pbSample = { depth: 0, u: 0, v: 0, speed: 0, valid: false };
    const kind = () => ({ features: null, ver: 0, cache: new Map(), waiters: [] });
    this._idx = { buildings: kind(), roads: kind() };
    this._flood = null;               // latest 'stats' payload of the current storm
    this._statsReq = null; this._statsBusy = false; this._statsIdle = null; this._statsIdleResolve = null;
    this._failCount = new Map();      // hour -> consecutive failed loads
    this._failMsg = new Map();        // hour -> message of its last failure

    this._loop = (now) => this._onFrame(now);
    this._onMoveEnd = () => this._scheduleRegion();
    this._onMove = () => this._moveCheck();
    this._onResize = () => this._scheduleRegion();
    this._onStyleData = () => this._styleCheck();
    this._onVis = () => this._visibilityChanged();
    this._onMapRemove = () => this.dispose();
    this._onMapLost = () => this._contextLost();
    this._onMapRestored = () => this._contextRestored();
    this._onRendererRestored = () => { this._dirty = true; this._kick(); };
    this._hook();
    this._applyQualityToRenderer();
    if (qualityMode !== undefined) this.setQualityMode(qualityMode);
  }

  // ---------------------------------------------------------------- accessors

  get storm() { return this._stormKey; }
  get base() { return this._storm ? this._storm.base : this._base; }   // flow folder of the current storm (the domain)
  get meta() { return this._meta; }
  get source() { return this._source; }
  get status() { return { ...this._status }; }
  get time() { return this._time; }
  get targetTime() { return this._target; }
  get isPlaying() { return this._playing; }
  get visible() { return this._visible; }
  // Anchor layer the water sits directly below (null = the default roads-line / all-parcels-fill / buildings-ext). The viewer points it at
  // its real-depth overlay while that is on, so the overlay is drawn above the water (see cog/cog-depth.js).
  setBeforeId(id) { this._beforeId = id || null; this._fixOrder(); }
  get hasPicture() { return this._shown; }
  get qualityLevel() { return this._level; }
  get region() { return this._region ? this._region.slice() : null; }
  get canvas() { return this._renderer.canvas; }
  get renderer() { return this._renderer; }
  get disposed() { return this._disposed; }

  get timeRange() {
    const hours = this._meta ? this._meta.hours.slice().sort((a, b) => a - b) : [];
    return { min: 0, max: this._tMax, loopStart: this._loopStart(), hours };
  }

  get qualityInfo() {
    const l = QUALITY_LEVELS[this._level];
    return {
      level: this._level, maxLevel: QUALITY_LEVELS.length - 1, pixelScale: l.px, particleFactor: l.particles, shaderQuality: Math.min(this._req.quality, l.shader),
      avgMs: this._avg, auto: this._lopt.autoQuality, canvasWidth: this._pxW, canvasHeight: this._pxH, reason: this._lastQualityReason, cadenceMs: this._cadence,
    };
  }

  on(name, fn) { if (!this._disposed) this._em.on(name, fn); return this; }
  off(name, fn) { this._em.off(name, fn); return this; }
  once(name, fn) { if (!this._disposed) this._em.once(name, fn); return this; }

  getState() {
    return {
      storm: this._stormKey, status: { ...this._status }, time: this._time, target: this._target, playing: this._playing,
      buffering: this._buffering, shown: this._shown, visible: this._visible, active: this._active, running: this._running,
      frames: this._frames, region: this.region, canvas: [this._pxW, this._pxH], level: this._level, avgMs: this._avg,
      cached: this._storm ? this._storm.cachedHours() : [], have: Array.from(this._have.keys()), inflight: Array.from(this._jobs.keys()),
      displayed: [this._dA, this._dB, this._dMix], layerOnMap: this._onMap(), docHidden: docHidden(),
    };
  }

  // Grids behind the picture (null = dry hour), for probes and statistics. Do not keep them.
  currentGrids() {
    return { hourA: this._dA, hourB: this._dB, mix: this._dMix, rgbaA: this._gA, rgbaB: this._gB, meta: this._meta };
  }

  _setStatus(state, message = '') {
    const s = this._status;
    if (s.state === state && s.message === message) return;
    s.state = state; s.message = message;
    this._em.emit('status', { state, message });
  }

  _emitError(error, phase, hour) {
    this._em.emit('error', { error, phase, storm: this._stormKey, hour });
  }

  // ------------------------------------------------------------------ 3. storm loading

  async setStorm(key, opts = {}) {
    if (this._disposed) throw new Error('FluidLayer is disposed');
    if (typeof key !== 'string' || key === '') throw new TypeError('setStorm: storm key must be a non-empty string');
    const base = opts.base === undefined || opts.base === null ? this._base : String(opts.base);
    if (this._storm && key === this._stormKey && base === this._storm.base && !opts.force && this._seq === this._committedSeq) return this._meta;
    const seq = ++this._seq;
    if (this._metaCtrl) this._metaCtrl.abort();
    if (this._first) { this._first.reject(abortError('setStorm superseded')); this._first = null; }
    const ctrl = new AbortController();
    this._metaCtrl = ctrl;
    this._setStatus('loading', `Loading ${key}`);
    let storm;
    try {
      storm = await loadStorm(base, key, { signal: ctrl.signal, fetch: this._fetch || undefined, fetchInit: this._fetchInit });
    } catch (e) {
      if (seq !== this._seq || this._disposed) throw abortError('setStorm superseded');
      throw this._failStorm(key, e, false, base);
    }
    if (seq !== this._seq || this._disposed) {
      storm.dispose();
      throw abortError('setStorm superseded');
    }
    this._metaCtrl = null;
    try {
      this._commitStorm(storm, key);
      this._base = base;
    } catch (e) {
      storm.dispose();
      throw this._failStorm(key, e, true, base);
    }
    this._committedSeq = seq;
    return this._awaitFirstPicture(seq);
  }

  _failStorm(key, e, committing, base = this._base) {
    let err = e;
    if (e && e.name !== 'AbortError' && !(e instanceof FlowFormatError) && !(e instanceof GridMemoryError)) {
      const m = /HTTP (404|410)\b/.exec(String(e.message || ''));
      if (m) err = new StormNotFoundError(key, `${base}/${key}/meta.json`, `Storm "${key}" is not available (HTTP ${m[1]}): ${e.message}`);
    }
    // a request that failed before anything was changed leaves the previous storm in place
    if (this._storm && !committing) this._setStatus(this._buffering ? 'buffering' : 'ready', '');
    else this._setStatus('error', err && err.message ? err.message : String(err));
    this._emitError(err, 'storm');
    return err;
  }

  _commitStorm(storm, key) {
    this._teardownStorm();
    this._renderer.setStorm(storm.meta);
    this._storm = storm; this._stormKey = key; this._meta = storm.meta;
    const hours = storm.hours;
    this._tMax = hours.length ? hours[hours.length - 1] : 0;
    this._firstWet = hours.length ? hours[0] : 0;
    this._source = {
      kind: storm.meta.source || 'unknown',
      label: SOURCE_LABELS[storm.meta.source] || 'Unknown data source',
      note: typeof storm.meta.source_note === 'string' ? storm.meta.source_note : '',
    };
    this._time = clamp(this._time, 0, this._tMax);
    this._target = this._time;
    this._shown = false; this._displayedOnce = false; this._regionApplied = false;
    this._applyVisibility();
    this._applyQualityToRenderer();
    this._applyFlowpaths();
    this._ensureIndexes();
    this._updateRegion(true);
    this._tryApply();
    this._pumpLoads();
    this._kick();
  }

  _awaitFirstPicture(seq) {
    if (this._shown || this._noOverlap) return Promise.resolve(this._meta);
    return new Promise((resolve, reject) => {
      this._first = { seq, resolve: () => resolve(this._meta), reject };
    });
  }

  _resolveFirst() {
    if (this._first) { const f = this._first; this._first = null; f.resolve(); }
  }

  // Everything that belongs to the current storm; the water is hidden until the next storm has drawn.
  _teardownStorm() {
    for (const j of this._jobs.values()) j.ctrl.abort();
    this._jobs.clear(); this._failed.clear(); this._failCount.clear(); this._failMsg.clear();
    this._flood = null; this._statsReq = null;
    if (this._fpCtrl) { this._fpCtrl.abort(); this._fpCtrl = null; }
    this._fpLoading = null; this._fpSet = false;
    if (this._retryTimer) { clearTimeout(this._retryTimer); this._retryTimer = 0; }
    if (this._storm) this._storm.dispose();
    this._storm = null; this._stormKey = null; this._meta = null; this._source = null;
    this._have.clear();
    this._gA = null; this._gB = null; this._dA = -1; this._dB = -1; this._dMix = 0;
    this._shown = false; this._displayedOnce = false; this._lastHourEvt = -1; this._buffering = false;
    this._applyVisibility();
  }

  _applyFlowpaths() {
    const want = Boolean(this._renderer.getOptions().showFlowpaths) && this._storm !== null;
    if (!want) {
      if (this._fpCtrl) { this._fpCtrl.abort(); this._fpCtrl = null; }
      this._fpLoading = null;
      if (this._fpSet) {
        this._renderer.setFlowpaths(null);
        this._fpSet = false;
        this._dirty = true;
      }
      return;
    }
    if (this._fpSet || this._fpLoading === this._storm) return;
    const st = this._storm;
    const ctrl = new AbortController();
    this._fpCtrl = ctrl; this._fpLoading = st;
    st.getFlowpaths({ signal: ctrl.signal }).then((fp) => {
      if (this._storm !== st || this._disposed) return;
      this._fpLoading = null;
      if (!this._renderer.getOptions().showFlowpaths) return;
      this._renderer.setFlowpaths(fp.rgba);
      this._fpSet = true; this._dirty = true;
      this._kick();
    }, (err) => {
      if (this._fpLoading === st) this._fpLoading = null;
      if (err && err.name === 'AbortError') return;
      if (this._storm === st) this._emitError(err, 'flowpaths');
    });
  }

  // ------------------------------------------------------------------ 4. time and grid scheduler

  _loopStart() {
    const o = this._lopt.loopStart;
    const v = Number.isFinite(o) ? o : Math.max(0, this._firstWet - 1);
    return clamp(v, 0, this._tMax);
  }

  _pair(t, out) {
    const tm = this._tMax;
    if (!(t < tm)) { out.hA = tm; out.hB = tm; out.mix = 0; return out; }
    const f = Math.floor(t);
    out.hA = f; out.hB = f + 1; out.mix = t - f;
    return out;
  }

  // Grid of an integer hour: null = dry (no file), undefined = not decoded yet.
  _gridFor(h) {
    if (!this._storm.hasHour(h)) return null;
    return this._have.get(h);
  }

  setTime(h) {
    if (this._disposed) return this;
    const n = Number(h);
    if (!Number.isFinite(n)) throw new TypeError(`setTime: hour must be a finite number (got ${h})`);
    const t = clamp(n, 0, Math.max(this._tMax, 0) || 24);
    if (t > this._target) this._dir = 1; else if (t < this._target) this._dir = -1;
    this._target = t;
    this._endHold = 0;
    if (!this._storm) { this._time = t; return this; }
    this._tryApply();
    this._pumpLoads();
    this._kick();
    return this;
  }

  play(msPerHour) {
    if (this._disposed) return this;
    if (msPerHour !== undefined) {
      const m = Number(msPerHour);
      if (!(m > 0) || !Number.isFinite(m)) throw new TypeError(`play: msPerHour must be a positive number (got ${msPerHour})`);
      this._lopt.msPerHour = m;
    }
    if (this._playing) return this;
    this._playing = true;
    this._dir = 1;
    this._endHold = this._storm && this._time >= this._tMax ? this._lopt.msPerHour : 0;
    this._lastNow = null;
    this._kick();
    return this;
  }

  pause() {
    if (!this._playing) return this;
    this._playing = false;
    this._dirty = true;
    return this;
  }

  togglePlay() { return this._playing ? this.pause() : this.play(); }

  // Hours to hold decoded for the current target: the pair itself, then the ones playback or a scrub reaches next.
  _wanted(out) {
    out.length = 0;
    const st = this._storm;
    const p = this._pair(this._target, this._candBuf);
    if (st.hasHour(p.hA)) out.push(p.hA);
    if (p.mix > 0 && p.hB !== p.hA && st.hasHour(p.hB)) out.push(p.hB);
    const lo = Math.max(1, Math.floor(this._loopStart()));
    let h = this._dir > 0 ? (p.mix > 0 ? p.hB : p.hA) : p.hA;
    let wrapped = 0;
    for (let k = 0; k < 8 && out.length < LRU_HOURS; k++) {
      h += this._dir;
      if (h > this._tMax) {
        if (!this._playing || this._dir < 0) break;
        h = lo + wrapped; wrapped++;
      }
      if (h < 1) break;
      if (st.hasHour(h) && !out.includes(h)) out.push(h);
    }
    return out;
  }

  _pumpLoads() {
    const st = this._storm;
    if (!st || this._disposed) return;
    const want = this._wanted(this._wantBuf || (this._wantBuf = []));
    const held = st.cachedHours();
    for (const h of want) if (held.includes(h)) st.getHour(h);   // a hit refreshes its place in the loader's LRU
    for (const [h, job] of this._jobs) {
      if (!want.includes(h)) { job.ctrl.abort(); this._jobs.delete(h); }
    }
    const now = this._sched.now();
    for (const h of want) {
      if (this._have.has(h) || this._jobs.has(h)) continue;
      const f = this._failed.get(h);
      if (f !== undefined && now - f < FAIL_COOLDOWN_MS) continue;
      if (this._jobs.size >= PREFETCH_CONCURRENCY) break;
      this._startJob(st, h);
    }
  }

  _startJob(st, h) {
    const ctrl = new AbortController();
    const job = { ctrl };
    this._jobs.set(h, job);
    st.getHour(h, { signal: ctrl.signal }).then((e) => {
      if (this._jobs.get(h) !== job || this._storm !== st) return;
      this._jobs.delete(h);
      this._failed.delete(h); this._failCount.delete(h); this._failMsg.delete(h);
      this._have.set(h, e.rgba);
      this._trimHave();
      this._onGrid();
    }, (err) => {
      if (this._jobs.get(h) === job) this._jobs.delete(h);
      if ((err && err.name === 'AbortError') || this._storm !== st || this._disposed) return;
      this._onGridError(h, err);
    });
  }

  _trimHave() {
    const keep = this._storm.cachedHours();
    for (const h of this._have.keys()) {
      if (!keep.includes(h) && h !== this._dA && h !== this._dB) this._have.delete(h);
    }
  }

  _onGrid() {
    if (this._disposed || !this._storm) return;
    this._tryApply();
    this._pumpLoads();
    this._dirty = true;
    this._kick();
  }

  _onGridError(h, err) {
    this._failed.set(h, this._sched.now());
    this._failMsg.set(h, `Hour ${h}: ${err && err.message ? err.message : err}`);
    this._failCount.set(h, (this._failCount.get(h) || 0) + 1);
    const msg = `Hour ${h}: ${err && err.message ? err.message : err}`;
    this._emitError(err, 'grid', h);
    this._setStatus('error', msg);
    if (this._first) { const f = this._first; this._first = null; f.reject(err); }
    if (!this._retryTimer) {
      this._retryTimer = setTimeout(() => { this._retryTimer = 0; this._pumpLoads(); }, FAIL_COOLDOWN_MS + 50);
    }
  }

  // Show the target time if its grids are decoded; otherwise keep the current picture and report 'buffering'.
  _tryApply() {
    if (!this._storm || this._disposed) return false;
    const t = this._target;
    const p = this._pair(t, this._pairBuf);
    const gA = this._gridFor(p.hA);
    const gB = p.mix > 0 ? this._gridFor(p.hB) : gA;
    if (gA === undefined || gB === undefined) {
      const bad = gA === undefined ? p.hA : p.hB;
      this._buffering = true;
      // an hour that already failed is reported as the error it is, not as "loading"
      if (this._failMsg.has(bad)) this._setStatus('error', this._failMsg.get(bad));
      else if (this._displayedOnce && this._status.state !== 'error') this._setStatus('buffering', `Loading hour ${bad}`);
      return false;
    }
    if (gA !== this._gA || gB !== this._gB || p.mix !== this._dMix) {
      try {
        this._renderer.setHours(gA, gB, p.mix);
      } catch (e) {
        this._emitError(e, 'render');
        this._setStatus('error', e.message);
        return false;
      }
      this._dirty = true;
    }
    this._gA = gA; this._gB = gB; this._dA = p.hA; this._dB = p.hB; this._dMix = p.mix;
    const moved = t !== this._time;
    this._time = t;
    this._displayedOnce = true;
    const wasBuffering = this._buffering;
    this._buffering = false;
    if (this._status.state === 'buffering' || this._status.state === 'error') this._setStatus(this._shown ? 'ready' : 'loading', '');
    this._tryShow();
    if (moved || wasBuffering) this._emitTime();
    const hr = Math.round(t);
    if (hr !== this._lastHourEvt) {
      this._lastHourEvt = hr;
      const grid = hr === p.hA ? gA : gB;
      this._em.emit('hour', { hour: hr, grid, dry: grid === null, storm: this._stormKey, meta: this._meta });
    }
    this._statsMaybe();
    return true;
  }

  _emitTime() {
    const e = this._evtTime;
    e.hour = this._time; e.hourA = this._dA; e.hourB = this._dB; e.mix = this._dMix; e.target = this._target; e.buffering = this._buffering;
    this._em.emit('time', e);
  }

  // The picture is drawn once before the layer is made visible, so a new storm never shows a stale canvas.
  _tryShow() {
    if (this._shown || !this._displayedOnce || !this._regionApplied || !this._storm) return;
    this._renderer.renderOnce();
    this._shown = true;
    this._applyVisibility();
    this._setStatus('ready', '');
    this._em.emit('storm', { storm: this._stormKey, meta: this._meta, source: this._source });
    this._resolveFirst();
    this._kick();
  }

  // Playback clock: advance by real elapsed time, but never into grids that are not decoded.
  _advance(dtMs) {
    if (!this._playing || !this._storm || !this._shown) return;
    if (this._buffering) { this._skipFailedHour(); return; }
    const msph = this._lopt.msPerHour;
    let cand;
    if (this._time >= this._tMax) {
      this._endHold += dtMs;
      if (this._endHold < msph) return;
      this._endHold = 0;
      cand = this._loopStart();
    } else {
      cand = this._time + Math.min(dtMs / msph, 0.5);
      if (cand > this._tMax) cand = this._tMax;
    }
    this._target = cand;
    this._tryApply();
    if (this._buffering) this._pumpLoads();
  }

  // An hour whose grid failed repeatedly would hold playback forever: jump to the next hour that can be shown.
  // The picture stays on the last good grid until then; the failure itself was reported as an 'error' event.
  _skipFailedHour() {
    const p = this._pair(this._target, this._pairBuf);
    let bad = -1;
    if (this._gridFor(p.hA) === undefined) bad = p.hA;
    else if (p.mix > 0 && this._gridFor(p.hB) === undefined) bad = p.hB;
    if (bad < 0 || (this._failCount.get(bad) || 0) < SKIP_AFTER_FAILS) return;
    let t = bad + 1;
    if (t > this._tMax) t = this._loopStart();
    this._target = t;
    this._endHold = 0;
    this._tryApply();
    this._pumpLoads();
  }

  // ------------------------------------------------------------------ 5. region and map integration

  _hook() {
    const m = this._map;
    m.on('moveend', this._onMoveEnd);
    m.on('zoomend', this._onMoveEnd);
    m.on('resize', this._onResize);
    m.on('move', this._onMove);
    m.on('styledata', this._onStyleData);
    m.on('remove', this._onMapRemove);
    m.on('webglcontextlost', this._onMapLost);
    m.on('webglcontextrestored', this._onMapRestored);
    this._renderer.canvas.addEventListener('webglcontextrestored', this._onRendererRestored, false);
    if (typeof document !== 'undefined') document.addEventListener('visibilitychange', this._onVis);
    this._mapHooked = true;
  }

  _unhook() {
    if (!this._mapHooked) return;
    this._mapHooked = false;
    const m = this._map;
    try {
      m.off('moveend', this._onMoveEnd);
      m.off('zoomend', this._onMoveEnd);
      m.off('resize', this._onResize);
      m.off('move', this._onMove);
      m.off('styledata', this._onStyleData);
      m.off('remove', this._onMapRemove);
      m.off('webglcontextlost', this._onMapLost);
      m.off('webglcontextrestored', this._onMapRestored);
      if (this._primeHandler) m.off('sourcedata', this._primeHandler);
    } catch (e) { /* map already removed */ }
    this._primeHandler = null;
    try { this._renderer.canvas.removeEventListener('webglcontextrestored', this._onRendererRestored, false); } catch (e) { /* ignore */ }
    if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', this._onVis);
  }

  // The map lost its GL context: nothing can be shown, so the loop stops. When MapLibre has rebuilt its resources the
  // canvas source and layer are checked, the region is re-applied and the water is drawn again.
  _contextLost() {
    if (this._disposed) return;
    this._mapLost = true;
    this._stopLoop();
  }

  _contextRestored() {
    if (this._disposed) return;
    this._mapLost = false;
    // the canvas source's texture and buffers belong to the lost context: the source is replaced, not repaired
    try { if (this._map.getLayer(this._layerId)) this._map.removeLayer(this._layerId); } catch (e) { /* ignore */ }
    try { if (this._map.getSource(this._sourceId)) this._map.removeSource(this._sourceId); } catch (e) { /* ignore */ }
    this._visSet = null; this._srcPlaying = false; this._regionApplied = false;
    this._dirty = true;
    if (this._storm) this._updateRegion(true);
    this._applyVisibility();
    this._kick();
    setTimeout(() => {      // MapLibre finishes re-creating its sources after the event
      if (this._disposed || !this._storm) return;
      try { if (!this._map.getSource(this._sourceId)) this._updateRegion(true); } catch (e) { return; }
      this._visSet = null; this._applyVisibility(); this._invalidateRtt();
      this._dirty = true; this._kick();
      try { this._map.triggerRepaint(); } catch (e) { /* ignore */ }
    }, 250);
  }

  _onMap() {
    try { return Boolean(this._map.getLayer(this._layerId)); } catch (e) { return false; }
  }

  _hasLayer(id) {
    try { return Boolean(this._map.getLayer(id)); } catch (e) { return false; }
  }

  _resolveBefore() {
    if (this._beforeId) return this._hasLayer(this._beforeId) ? this._beforeId : undefined;
    for (const id of BEFORE_CANDIDATES) if (this._hasLayer(id)) return id;
    return undefined;
  }

  // Visible ground area in mercator metres [xmin, ymin, xmax, ymax] (axis-aligned box of the viewport corners).
  _visibleMerc() {
    const b = this._map.getBounds();
    const sw = b.getSouthWest(), ne = b.getNorthEast();
    const a = lngLatToMerc(sw.lng, sw.lat), c = lngLatToMerc(ne.lng, ne.lat);
    const v = [Math.min(a[0], c[0]), Math.min(a[1], c[1]), Math.max(a[0], c[0]), Math.max(a[1], c[1])];
    return v.every(Number.isFinite) && v[0] < v[2] && v[1] < v[3] ? v : null;
  }

  // Mercator metres per CSS pixel at the bottom of the screen, which is the finest ground scale on screen
  // when the view is pitched (limited to twice the centre's detail so a grazing view stays affordable).
  _nearMpp(mppCenter) {
    try {
      const c = this._map.getContainer();
      const w = c.clientWidth, h = c.clientHeight;
      if (w > 2 && h > 2) {
        const a = this._map.unproject([w / 2, h - 1]), b = this._map.unproject([w / 2 + 1, h - 1]);
        const ma = lngLatToMerc(a.lng, a.lat), mb = lngLatToMerc(b.lng, b.lat);
        const d = Math.hypot(mb[0] - ma[0], mb[1] - ma[1]);
        if (d > 0 && Number.isFinite(d)) return clamp(d, 0.5 * mppCenter, mppCenter);
      }
    } catch (e) { /* fall through */ }
    return mppCenter;
  }

  // Region to draw and the canvas size for it: null = map not measurable yet, {empty:true} = grid not in view.
  _computePlan() {
    const g = this._meta.bounds_merc;
    const vis = this._visibleMerc();
    if (!vis) return null;
    const need = [Math.max(vis[0], g[0]), Math.max(vis[1], g[1]), Math.min(vis[2], g[2]), Math.min(vis[3], g[3])];
    if (!(need[0] < need[2] && need[1] < need[3])) return { empty: true };
    const mppC = MERC_WORLD / (TILE_PX * Math.pow(2, this._map.getZoom()));
    const mx = Math.max((vis[2] - vis[0]) * REGION_MARGIN, MIN_MARGIN_PX * mppC);
    const my = Math.max((vis[3] - vis[1]) * REGION_MARGIN, MIN_MARGIN_PX * mppC);
    const region = [Math.max(vis[0] - mx, g[0]), Math.max(vis[1] - my, g[1]), Math.min(vis[2] + mx, g[2]), Math.min(vis[3] + my, g[3])];
    const near = this._nearMpp(mppC);
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    const lvl = QUALITY_LEVELS[this._level];
    // canvas pixels per mercator metre; capped at DENS_CAP_PX_PER_CELL per grid cell, because the data has no finer
    // detail and a zoom-in past that point then needs no new canvas (every canvas swap is a chance to flash)
    const dens = Math.min(dpr * this._lopt.pixelScale * lvl.px / near, DENS_CAP_PX_PER_CELL / (this._meta.cell_m || 6.4));
    const rw = region[2] - region[0], rh = region[3] - region[1];
    let W = rw * dens, H = rh * dens;
    const cap = this._lopt.maxPixels * lvl.px * lvl.px;
    const f = Math.min(1, MAX_SIDE / Math.max(W, H), Math.sqrt(cap / (W * H)));
    W = Math.max(8, Math.round(W * f)); H = Math.max(8, Math.round(H * f));
    return { region, need, W, H, dens: W / rw };
  }

  _planOk(plan) {
    const cur = this._regionPlan, r = cur.region, n = plan.need;
    const eps = 1e-6;
    if (!(r[0] <= n[0] + eps && r[1] <= n[1] + eps && r[2] >= n[2] - eps && r[3] >= n[3] - eps)) return false;
    const ratio = cur.dens / plan.dens;
    return ratio >= DENSITY_LO && ratio <= DENSITY_HI;
  }

  _scheduleRegion() {
    if (this._disposed) return;
    this._syncView();
    if (this._regionTimer) clearTimeout(this._regionTimer);
    this._regionTimer = setTimeout(() => { this._regionTimer = 0; this._updateRegion(false); }, REGION_DEBOUNCE_MS);
  }

  // While the camera moves: replace the region at once (throttled) if the visible area runs out of it.
  _moveCheck() {
    if (!this._storm || !this._regionApplied) return;
    const now = this._sched.now();
    if (now - this._lastMoveCheck < REGION_MOVE_THROTTLE_MS) return;
    this._lastMoveCheck = now;
    this._syncView();
    try {
      const vis = this._visibleMerc();
      if (!vis) return;
      const g = this._meta.bounds_merc, r = this._region;
      const n = [Math.max(vis[0], g[0]), Math.max(vis[1], g[1]), Math.min(vis[2], g[2]), Math.min(vis[3], g[3])];
      const covered = n[0] >= r[0] && n[1] >= r[1] && n[2] <= r[2] && n[3] <= r[3];
      if (!covered || !this._active) this._updateRegion(false);
    } catch (e) { /* map mid-teardown */ }
  }

  _syncView() {
    const m = this._map, r = this._renderer;
    const pitch = m.getPitch(), bearing = m.getBearing();
    if (Math.abs(pitch - (this._vp ?? -99)) > 0.4 || Math.abs(bearing - (this._vb ?? -999)) > 0.4) {
      this._vp = pitch; this._vb = bearing;
      r.setOptions({ viewPitchDeg: pitch, viewBearingDeg: bearing });
      this._dirty = true;
    }
  }

  updateRegion(force = false) {
    if (this._disposed) return false;
    return this._updateRegion(Boolean(force));
  }

  _updateRegion(force) {
    if (!this._storm || this._disposed) return false;
    let plan;
    try { plan = this._computePlan(); } catch (e) { return false; }
    if (plan === null) return false;
    if (plan.empty) {
      this._noOverlap = true;
      if (this._active) { this._active = false; this._applyVisibility(); this._stopLoop(); }
      this._resolveFirst();
      return false;
    }
    this._noOverlap = false;
    if (!force && this._regionApplied && this._planOk(plan)) {
      if (!this._active) { this._active = true; this._applyVisibility(); this._dirty = true; this._kick(); }
      return false;
    }
    return this._applyPlan(plan);
  }

  _coords(region) {
    const nw = mercToLngLat(region[0], region[3]), ne = mercToLngLat(region[2], region[3]);
    const se = mercToLngLat(region[2], region[1]), sw = mercToLngLat(region[0], region[1]);
    return [nw, ne, se, sw];
  }

  // Resize the canvas, draw the new region and hand it to the map in one task, so the map never sees
  // an empty canvas or the new picture with the old corners.
  _applyPlan(plan) {
    const r = this._renderer;
    this._syncView();
    try {
      r.setRegion(plan.region, plan.W, plan.H);
    } catch (e) {
      this._emitError(e, 'render');
      return false;
    }
    this._pxW = r.canvas.width; this._pxH = r.canvas.height;
    if (this._displayedOnce) r.renderOnce();
    const coords = this._coords(plan.region);
    if (!this._pushCoordinates(coords)) return false;
    this._region = plan.region.slice();
    this._regionPlan = { region: plan.region.slice(), need: plan.need.slice(), dens: this._pxW / (plan.region[2] - plan.region[0]) };
    this._regionApplied = true;
    this._active = true;
    this._settleUntil = this._sched.now() + GOVERNOR_SETTLE_MS;
    this._dirty = true;
    this._applyVisibility();
    this._tryShow();
    this._em.emit('region', { region: this._region.slice(), width: this._pxW, height: this._pxH });
    this._kick();
    try { this._map.triggerRepaint(); } catch (e) { /* ignore */ }
    return true;
  }

  _pushCoordinates(coords) {
    const m = this._map;
    let src = null;
    try { src = m.getSource(this._sourceId); } catch (e) { src = null; }
    if (src) {
      try {
        src.setCoordinates(coords);
        this._fixOrder();
        return true;
      } catch (e) {
        this._emitError(e, 'render');
        return false;
      }
    }
    try {
      m.addSource(this._sourceId, { type: 'canvas', canvas: this._renderer.canvas, coordinates: coords, animate: true });
      this._srcPlaying = true;
      m.addLayer({
        id: this._layerId, type: 'raster', source: this._sourceId,
        layout: { visibility: 'none' },
        paint: { 'raster-fade-duration': 0, 'raster-resampling': 'linear', 'raster-opacity': 1 },
      }, this._resolveBefore());
      this._visSet = 'none';
      this._watchSource();
      return true;
    } catch (e) {
      // the style may not be loaded yet: undo a half-finished add and try again shortly
      try { if (m.getLayer(this._layerId)) m.removeLayer(this._layerId); } catch (e2) { /* ignore */ }
      try { if (m.getSource(this._sourceId)) m.removeSource(this._sourceId); } catch (e2) { /* ignore */ }
      this._srcPlaying = false;
      if (!this._retryAddTimer && !this._disposed) {
        this._retryAddTimer = setTimeout(() => { this._retryAddTimer = 0; this._updateRegion(true); }, 300);
      }
      this._emitError(e, 'render');
      return false;
    }
  }

  // With 3D terrain MapLibre draws rasters into cached per-tile textures ("render to texture") and redraws a cached
  // texture only when the set of source tiles under it changes or a source tile announces itself. A canvas source
  // never announces new content, so without help the water would freeze at its first picture (and stay invisible
  // when its tile arrived after the cache was filled). Dropping the cached textures under the water makes the next
  // map render redraw them from the current canvas. This reaches into MapLibre 4.x internals (map.terrain); if they
  // are not there the water only refreshes on camera moves.
  _invalidateRtt(sourceId) {
    const m = this._map;
    const terr = m.terrain;
    if (!terr || !terr.sourceCache || typeof terr.sourceCache.freeRtt !== 'function') return;
    try {
      const sc = m.style && m.style.sourceCaches && m.style.sourceCaches[sourceId || this._sourceId];
      if (!sc) return;
      const ids = sc.getVisibleCoordinates();
      for (let i = 0; i < ids.length; i++) terr.sourceCache.freeRtt(ids[i]);
    } catch (e) { /* internals changed */ }
  }

  // A source tile (re)loaded: the cached terrain textures under it were drawn without it.
  _watchSource() {
    const m = this._map, id = this._sourceId;
    if (this._primeHandler) m.off('sourcedata', this._primeHandler);
    this._primeHandler = (e) => {
      if (e.sourceId === id && e.sourceDataType === 'idle') this._invalidateRtt();
    };
    m.on('sourcedata', this._primeHandler);
  }

  _applyVisibility() {
    if (!this._onMap()) return;
    const want = this._visible && this._shown && this._active ? 'visible' : 'none';
    if (this._visSet === want) return;
    try {
      this._map.setLayoutProperty(this._layerId, 'visibility', want);
      this._visSet = want;
    } catch (e) { /* style mid-change */ }
  }

  // Keep the water below the first of beforeId / roads-line / all-parcels-fill / buildings-ext, also when they are added later.
  _fixOrder() {
    const m = this._map;
    if (!this._onMap()) return;
    try {
      const order = typeof m.getLayersOrder === 'function' ? m.getLayersOrder() : m.getStyle().layers.map((l) => l.id);
      const i = order.indexOf(this._layerId);
      const want = this._resolveBefore();
      if (i < 0 || want === undefined) return;   // no anchor layer: stay where the layer was added
      const next = i + 1 < order.length ? order[i + 1] : undefined;
      if (next !== want) m.moveLayer(this._layerId, want);
    } catch (e) { /* ignore */ }
  }

  _styleCheck() {
    if (this._disposed || !this._storm || this._styleCheckPending) return;
    this._styleCheckPending = true;
    Promise.resolve().then(() => {
      this._styleCheckPending = false;
      if (this._disposed || !this._storm) return;
      let gone = false;
      try { gone = this._regionApplied && !this._map.getSource(this._sourceId); } catch (e) { return; }
      if (gone) { this._visSet = null; this._updateRegion(true); } else this._fixOrder();
    });
  }

  setVisible(on) {
    if (this._disposed) return this;
    const v = Boolean(on);
    if (v === this._visible) return this;
    this._visible = v;
    this._applyVisibility();
    if (v) { this._dirty = true; this._kick(); } else this._stopLoop();
    return this;
  }

  _setSourcePlaying(on) {
    if (on === this._srcPlaying) return;
    let s = null;
    try { s = this._map.getSource(this._sourceId); } catch (e) { return; }
    if (!s) return;
    this._srcPlaying = on;
    try {
      if (on && typeof s.play === 'function') s.play();
      else if (!on && typeof s.pause === 'function') s.pause();
    } catch (e) { /* source not loaded yet */ }
  }

  // ------------------------------------------------------------------ 6. options

  _setLayerOpt(k, v) {
    const o = this._lopt;
    switch (k) {
      case 'freeze': o[k] = Boolean(v); break;
      case 'autoQuality': o[k] = Boolean(v); if (o[k]) this._pinnedMode = null; break;
      case 'statsTopN': {
        const n = Number(v);
        if (!(n >= 0)) throw new TypeError(`FluidLayer: statsTopN must be a number >= 0 (got ${v})`);
        o[k] = Math.floor(Math.min(n, 1e9));
        break;
      }
      case 'statsMinDepthM': {
        const n = Number(v);
        if (!Number.isFinite(n) || n < 0) throw new TypeError(`FluidLayer: statsMinDepthM must be a number >= 0 (got ${v})`);
        o[k] = n;
        break;
      }
      case 'frameBudgetMs': case 'msPerHour': case 'pixelScale': case 'maxPixels': case 'governorOverMs': case 'governorRecoverMs': {
        const n = Number(v);
        if (!(n > 0) || !Number.isFinite(n)) throw new TypeError(`FluidLayer: option ${k} must be a positive number (got ${v})`);
        o[k] = k === 'pixelScale' ? clamp(n, 0.1, 2) : (k === 'maxPixels' ? clamp(n, 1e4, 1 << 24) : n);
        break;
      }
      case 'loopStart':
        if (v === null || v === undefined) o.loopStart = null;
        else if (Number.isFinite(Number(v))) o.loopStart = Number(v);
        else throw new TypeError(`FluidLayer: loopStart must be a number or null (got ${v})`);
        break;
      default: break;
    }
  }

  setOptions(opts = {}) {
    if (this._disposed) return this;
    const ro = {};
    let regionChanged = false, statsChanged = false;
    for (const k of Object.keys(opts)) {
      if (k === 'qualityMode') {
        this.setQualityMode(opts[k]);
      } else if (LAYER_KEYS.includes(k)) {
        const before = this._lopt[k];
        this._setLayerOpt(k, opts[k]);
        if ((k === 'pixelScale' || k === 'maxPixels') && before !== this._lopt[k]) regionChanged = true;
        if ((k === 'statsTopN' || k === 'statsMinDepthM') && before !== this._lopt[k]) statsChanged = true;
      } else if (k === 'particleCount' || k === 'quality') {
        const n = Number(opts[k]);
        if (!Number.isFinite(n)) throw new TypeError(`FluidLayer: option ${k} must be a finite number (got ${opts[k]})`);
        this._req[k] = k === 'particleCount' ? Math.max(0, Math.round(n)) : clamp(Math.round(n), 0, 2);
      } else {
        ro[k] = opts[k];
      }
    }
    if (Object.keys(ro).length) this._renderer.setOptions(ro);
    this._applyQualityToRenderer();
    if ('showFlowpaths' in opts) this._applyFlowpaths();
    if (regionChanged && this._storm) this._updateRegion(true);
    if (statsChanged) { this._statsReq = null; this._statsMaybe(); }
    this._dirty = true;
    if (this._lopt.freeze === false && !this._running) this._lastNow = null;
    this._kick();
    return this;
  }

  getOptions() {
    return { ...this._renderer.getOptions(), particleCount: this._req.particleCount, quality: this._req.quality, ...this._lopt };
  }

  _applyQualityToRenderer() {
    const l = QUALITY_LEVELS[this._level];
    this._renderer.setOptions({
      particleCount: Math.round(this._req.particleCount * l.particles),
      quality: Math.min(this._req.quality, l.shader),
    });
  }

  // ------------------------------------------------------------------ 7. frame loop

  _shouldRun() {
    if (this._disposed || !this._visible || !this._active || !this._shown || this._mapLost) return false;
    if (docHidden()) return false;
    if (this._lopt.freeze && !this._dirty && !this._playing) return false;
    return true;
  }

  _kick() {
    if (this._running || this._disposed || !this._autoStart) return;
    if (!this._shouldRun()) return;
    this._running = true;
    this._lastNow = null;
    this._setSourcePlaying(true);
    this._raf = this._sched.raf(this._loop);
  }

  _stopLoop() {
    this._running = false;
    if (this._raf) { this._sched.caf(this._raf); this._raf = 0; }
    this._lastNow = null;
    this._overSince = -1; this._underSince = -1;
    this._setSourcePlaying(false);
  }

  _onFrame(now) {
    this._raf = 0;
    if (!this._running) return;
    if (!this._shouldRun()) { this._stopLoop(); return; }
    this._raf = this._sched.raf(this._loop);
    this._step(now);
  }

  // One frame: playback clock, renderer frame, map repaint, governor. Public as tick() so tests can drive it.
  _step(now) {
    const dt = this._lastNow === null ? 0 : now - this._lastNow;
    this._lastNow = now;
    const stall = dt > STALL_MS || dt < 0;
    this._advance(stall ? 0 : dt);
    const r = this._renderer;
    try {
      if (this._lopt.freeze) { if (this._dirty) r.renderOnce(); } else r.frame(now);
    } catch (e) {
      this._emitError(e, 'render');
    }
    this._dirty = false;
    this._frames++;
    this._invalidateRtt();
    this._map.triggerRepaint();
    this._govern(now, dt, stall);
  }

  tick(now) {
    if (this._disposed) return;
    this._step(Number.isFinite(now) ? now : this._sched.now());
  }

  _visibilityChanged() {
    if (this._disposed) return;
    if (docHidden()) this._stopLoop();
    else { this._lastNow = null; this._dirty = true; this._kick(); }
  }

  // ------------------------------------------------------------------ 8. quality governor

  // Frame interval, not GPU time, is what the browser lets us measure, and a display caps it at its refresh
  // period: with headroom the interval says nothing about how much more load would fit. So recovery is a probe:
  // after governorRecoverMs below HEADROOM_FRACTION of the budget one level is given back, and a step down soon
  // after a step up doubles the wait for the next probe (flap back-off, up to 8x, forgotten after FLAP_FORGET_MS).
  _govern(now, dt, stall) {
    if (stall || dt <= 0 || now < this._settleUntil) return;
    const a = 1 - Math.exp(-dt / 400);
    this._avg = this._avg === 0 ? dt : this._avg + (dt - this._avg) * a;
    if (!this._lopt.autoQuality) return;
    // A display or a throttled tab that delivers frames every 33 ms is not overload. When several levels were given up
    // (two in a row) and the interval did not move, it is a cadence: the budget follows it (CADENCE_MARGIN) so quality can recover,
    // and real load on top of it still shows as an interval above that budget.
    if (this._cadence > 0 && this._avg < this._cadence * 0.6) this._cadence = 0;
    const budget = Math.max(this._lopt.frameBudgetMs, this._cadence * CADENCE_MARGIN);
    if (this._avg > budget) {
      this._underSince = -1;
      const heavy = this._avg > 2 * budget;
      if (this._overSince < 0) this._overSince = now;
      else if (now - this._overSince >= this._lopt.governorOverMs * (heavy ? 0.5 : 1)) {
        // a step down that did not make the interval shorter than the one before it counts as "no effect"
        if (now - this._chainStamp > CADENCE_CHAIN_MS || this._avg < this._chainAvg * 0.85 || this._avg > CADENCE_MAX_MS) this._chainRun = 0;
        else this._chainRun++;
        if (this._chainRun >= 2) {
          this._cadence = this._avg;
          this._chainRun = 0; this._chainStamp = -1e9;
          this._overSince = -1;
          this._lastQualityReason = 'cadence';
          return;
        }
        this._chainAvg = this._avg; this._chainStamp = now;
        const steps = this._avg > 3 * budget ? 3 : (this._avg > 1.6 * budget ? 2 : 1);
        this.setQualityLevel(this._level + steps, 'frame time');
      }
    } else {
      this._overSince = -1;
      if (this._level > 0 && this._avg < budget * HEADROOM_FRACTION) {
        if (this._underSince < 0) this._underSince = now;
        else if (now - this._underSince >= this._lopt.governorRecoverMs * this._backoff) this.setQualityLevel(this._level - 1, 'headroom');
      } else {
        this._underSince = -1;
      }
    }
  }

  // Normally the governor calls this; callers may pin a level (set autoQuality false to keep it).
  setQualityLevel(level, reason = 'manual') {
    if (this._disposed) return this;
    const l = clamp(Math.round(level), 0, QUALITY_LEVELS.length - 1);
    if (l === this._level) return this;
    const now = this._sched.now();
    if (l > this._level && now - this._lastUpStamp < 20000) this._backoff = Math.min(8, this._backoff * 2);
    if (l < this._level) {
      this._chainRun = 0;
      if (now - this._lastUpStamp > FLAP_FORGET_MS) this._backoff = 1;
      this._lastUpStamp = now;
    }
    const pxChanged = QUALITY_LEVELS[l].px !== QUALITY_LEVELS[this._level].px;
    this._level = l;
    this._lastQualityReason = reason;
    this._overSince = -1; this._underSince = -1;
    this._settleUntil = now + GOVERNOR_SETTLE_MS;
    this._avg = 0;
    this._applyQualityToRenderer();
    if (pxChanged && this._storm) this._updateRegion(true);
    this._dirty = true;
    const q = QUALITY_LEVELS[l];
    this._em.emit('quality', { level: l, maxLevel: QUALITY_LEVELS.length - 1, pixelScale: q.px, particleFactor: q.particles, reason, avgMs: this._avg });
    this._kick();
    return this;
  }

  // 'auto' lets the governor choose; 'high' / 'medium' / 'low' pin a level (0 / 3 / 6 of QUALITY_LEVELS) and switch
  // the governor off until 'auto' is set again.
  setQualityMode(mode) {
    if (this._disposed) return this;
    if (mode === 'auto') {
      this._pinnedMode = null;
      this._lopt.autoQuality = true;
      this._overSince = -1; this._underSince = -1;
      this._kick();
      return this;
    }
    if (!Object.prototype.hasOwnProperty.call(QUALITY_MODES, mode)) throw new TypeError(`setQualityMode: mode must be 'auto', 'high', 'medium' or 'low' (got ${mode})`);
    this._lopt.autoQuality = false;
    this._pinnedMode = mode;
    this.setQualityLevel(QUALITY_MODES[mode], `mode ${mode}`);
    return this;
  }

  // 'auto', the pinned 'high' | 'medium' | 'low', or 'fixed' (autoQuality turned off with a level set by hand).
  get qualityMode() {
    if (this._lopt.autoQuality) return 'auto';
    return this._pinnedMode && QUALITY_MODES[this._pinnedMode] === this._level ? this._pinnedMode : 'fixed';
  }

  // ------------------------------------------------------------------ 9. probes and statistics

  // Water at a point at the displayed time: bilinear between pixel centres and linear between the two hours
  // (the same blend the renderer draws, see fluid-data sample()). `lngLat` is [lng, lat] or {lng, lat}. Pass `out` to
  // reuse an object; the returned fields are
  //   valid       false outside the grid, outside the model domain, before a storm has drawn or after dispose (all numbers 0)
  //   depthM, speedMs, u (m/s toward east), v (m/s toward north)
  //   bearingDeg  compass heading the water moves TOWARD, 0 = north, 90 = east, clockwise, 0..360 (0 when still)
  //   cardinal    8-point name of that heading ('NE'), '' when the water is still (speed below 0.02 m/s)
  //   moving      speed >= 0.02 m/s
  //   hour        the displayed continuous time
  // Speeds are derived from the water-surface slope (Manning), not solver output.
  probe(lngLat, out) {
    const o = out || { valid: false, depthM: 0, speedMs: 0, u: 0, v: 0, bearingDeg: 0, cardinal: '', moving: false, hour: 0 };
    o.valid = false; o.depthM = 0; o.speedMs = 0; o.u = 0; o.v = 0; o.bearingDeg = 0; o.cardinal = ''; o.moving = false;
    o.hour = this._time;
    if (this._disposed || !this._storm || !this._shown || !lngLat) return o;
    const lng = Array.isArray(lngLat) ? lngLat[0] : lngLat.lng;
    const lat = Array.isArray(lngLat) ? lngLat[1] : lngLat.lat;
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) return o;
    const m = lngLatToMerc(lng, lat, this._pbMerc);
    const s = this._storm.sample(this._gA, this._gB, this._dMix, m[0], m[1], this._pbSample);
    if (!s.valid) return o;
    o.valid = true; o.depthM = s.depth; o.speedMs = s.speed; o.u = s.u; o.v = s.v;
    if (s.speed >= STILL_MS) {
      let b = Math.atan2(s.u, s.v) * 180 / Math.PI;
      if (b < 0) b += 360;
      o.bearingDeg = b; o.moving = true;
      o.cardinal = CARDINALS[Math.round(b / 45) % 8];
    }
    return o;
  }

  // Hover probe without per-event allocation: `cb(result, event)` is called on every mousemove with the SAME result
  // object (copy what you keep) and `cb(null, null)` when the pointer leaves the map. Returns a function that detaches.
  attachProbe(cb) {
    if (typeof cb !== 'function') throw new TypeError('attachProbe: callback must be a function');
    if (this._disposed) return () => {};
    const res = this.probe(null);
    const ll = [0, 0];
    const map = this._map;
    const move = (e) => { ll[0] = e.lngLat.lng; ll[1] = e.lngLat.lat; this.probe(ll, res); cb(res, e); };
    const leave = () => cb(null, null);
    map.on('mousemove', move);
    map.on('mouseout', leave);
    const detach = () => {
      if (!this._probers.delete(detach)) return;
      try { map.off('mousemove', move); map.off('mouseout', leave); } catch (e) { /* map removed */ }
    };
    this._probers.add(detach);
    return detach;
  }

  // ---- flood statistics (fluid-stats.js semantics)
  //
  // attachBuildings(fc) / attachRoads(fc) take a GeoJSON FeatureCollection (or a feature array; null detaches). The
  // pixel indices are built for the grid of the current storm, in slices of about 25 ms that yield to the UI, and are
  // cached by grid geometry, so storms with the same grid share them. The returned promise resolves to the feature
  // count once the index of the current grid exists (0 when replaced or disposed meanwhile).
  // 'stats' fires when the snapshot nearest to the displayed time changes: its hour is round(time), the hour of the
  // 'hour' event, and the numbers are those of that snapshot's grid (the water is a blend of two snapshots, so at
  // half-way they can differ slightly from the picture); it does not fire while the time moves inside a half-hour band; it also fires once a storm has drawn and
  // when an index becomes ready or statsMinDepthM changes. Nothing is computed unless a 'stats' listener or an
  // attached data set exists (or refreshStats() is called). If the computation falls behind a very fast playback the
  // hours in between are skipped, never queued. A payload:
  //   {hour, storm, floodedCount, tierCounts, belowTierCount, buildingCount, ranked, order, depthM, speedMs,
  //    roads: {pctFlooded, floodedLengthM, totalLengthM, floodedFeatures, perFeaturePct} | null,
  //    grid: {wetCells, wetAreaKm2, maxDepthM, meanDepthM, maxSpeedMs, volumeM3}, computeMs, cpuMs}
  //   computeMs is wall time including the yields, cpuMs the time inside fluid-stats
  //   floodedCount/tierCounts/ranked are null/null/[] without attached buildings; ranked is the first statsTopN of
  //   `order` as {index, depthM, tier, speedMs}; order, depthM, speedMs are the typed arrays of evaluateBuildings.
  attachBuildings(fc) { return this._attach('buildings', fc); }
  attachRoads(fc) { return this._attach('roads', fc); }

  _attach(kind, fc) {
    const k = this._idx[kind];
    const features = fc === null || fc === undefined ? null : featuresOf(fc, kind === 'buildings' ? 'attachBuildings' : 'attachRoads');
    if (this._disposed) return Promise.resolve(0);
    for (const e of k.cache.values()) e.cancel = true;
    k.cache.clear();
    k.features = features; k.ver++;
    for (const r of k.waiters.splice(0)) r();
    this._statsReq = null;
    if (!features) { this._statsMaybe(); return Promise.resolve(0); }
    const ver = k.ver;
    this._ensureIndexes();
    return this._indexReady(kind, ver);
  }

  async _indexReady(kind, ver) {
    const k = this._idx[kind];
    for (;;) {
      if (this._disposed || k.ver !== ver) return 0;
      const e = this._ensureIndex(kind);
      if (!e) { await new Promise((r) => k.waiters.push(r)); continue; }
      const idx = await e.promise;
      if (idx) return idx.count;
      if (e.cancel && !this._disposed && k.ver === ver && this._meta && gridSignature(this._meta) === e.sig) return 0;
    }
  }

  _ensureIndexes() {
    for (const kind of ['buildings', 'roads']) {
      const k = this._idx[kind];
      if (this._meta) {
        const sig = gridSignature(this._meta);
        for (const [s, e] of Array.from(k.cache)) if (s !== sig && !e.index) { e.cancel = true; k.cache.delete(s); }
        this._ensureIndex(kind);
      }
      for (const r of k.waiters.splice(0)) r();
    }
  }

  _ensureIndex(kind) {
    const k = this._idx[kind];
    if (!k.features || !this._meta) return null;
    const sig = gridSignature(this._meta);
    let e = k.cache.get(sig);
    if (e) return e;
    e = { sig, index: null, cancel: false, promise: null, ms: 0, slices: 0, maxSliceMs: 0 };
    k.cache.set(sig, e);
    while (k.cache.size > 2) {          // grids of two storms at most; the oldest one goes
      const [os, oe] = k.cache.entries().next().value;
      oe.cancel = true; k.cache.delete(os);
    }
    const meta = this._meta, ver = k.ver;
    e.promise = this._buildIndex(kind, e, k.features, meta).then((idx) => {
      if (!idx || e.cancel || this._disposed) return null;
      e.index = idx;
      if (k.ver === ver) this._statsMaybe();
      return idx;
    }, (err) => {
      if (k.cache.get(sig) === e) k.cache.delete(sig);
      if (!this._disposed) this._emitError(err, 'stats');
      return null;
    });
    return e;
  }

  // Builds the index slice by slice so that no task blocks the UI for much longer than SLICE_TARGET_MS.
  async _buildIndex(kind, e, features, meta) {
    const build = kind === 'buildings' ? buildBuildingIndex : buildRoadIndex;
    const parts = [];
    let n = 0, step = 64;
    const t00 = performance.now();
    do {
      if (e.cancel || this._disposed) return null;
      const t0 = performance.now();
      parts.push(build(meta, features.slice(n, n + step)));
      const dt = performance.now() - t0;
      e.slices++; e.maxSliceMs = Math.max(e.maxSliceMs, dt);
      n += step;
      step = clamp(Math.round(step * clamp(SLICE_TARGET_MS / Math.max(dt, 0.2), 0.5, 2)), 16, 4096);
      if (n < features.length) await yieldNow();
    } while (n < features.length);
    e.ms = performance.now() - t00;
    return parts.length === 1 ? parts[0] : mergeIndexParts(kind, parts);
  }

  _statsActive() {
    return Boolean(this._em._ls.stats) || Boolean(this._idx.buildings.features) || Boolean(this._idx.roads.features);
  }

  // Called whenever the displayed pair changed or something the statistics depend on did. Cheap when nothing did.
  _statsMaybe(force = false) {
    if (this._disposed || !this._storm || !this._displayedOnce) return;
    if (!force && !this._statsActive()) return;
    const bi = this._currentIndex('buildings'), ri = this._currentIndex('roads');
    // the water is a blend of two snapshots; the numbers belong to the nearer one, the hour the 'hour' event names
    const near = this._dMix >= 0.5 ? 1 : 0;
    const req = { storm: this._storm, hour: near ? this._dB : this._dA, grid: near ? this._gB : this._gA, bIndex: bi, rIndex: ri, bVer: this._idx.buildings.ver, rVer: this._idx.roads.ver };
    const old = this._statsReq;
    if (old && old.storm === req.storm && old.hour === req.hour && old.bIndex === bi && old.rIndex === ri && old.bVer === req.bVer && old.rVer === req.rVer) return;
    this._statsReq = req;
    if (!this._statsBusy) this._statsLoop();
  }

  _currentIndex(kind) {
    const k = this._idx[kind];
    if (!k.features || !this._meta) return null;
    const e = k.cache.get(gridSignature(this._meta));
    return e && e.index ? e.index : null;
  }

  // One computation at a time. A request that arrives meanwhile replaces the waiting one, so a playback faster than
  // the computation skips hours instead of building a queue.
  async _statsLoop() {
    this._statsBusy = true;
    if (!this._statsIdle) this._statsIdle = new Promise((r) => { this._statsIdleResolve = r; });
    try {
      let done = null;
      while (this._statsReq && this._statsReq !== done && !this._disposed) {
        const req = this._statsReq;
        done = req;
        const res = await this._computeStats(req);
        if (this._disposed) break;
        if (req.storm === this._storm && req.bVer === this._idx.buildings.ver && req.rVer === this._idx.roads.ver) {
          this._flood = res;
          this._em.emit('stats', res);
        }
      }
    } catch (err) {
      if (!this._disposed) this._emitError(err, 'stats');
    } finally {
      this._statsBusy = false;
      const r = this._statsIdleResolve;
      this._statsIdle = null; this._statsIdleResolve = null;
      if (r) r();
    }
  }

  async _computeStats(req) {
    const t0 = performance.now();
    const grid = req.grid || ZERO_GRID;
    const o = { minDepthM: this._lopt.statsMinDepthM };
    let b = null, r = null, cpu = 0;
    await yieldNow();
    let t = performance.now();
    if (req.bIndex) b = evaluateBuildings(req.bIndex, grid, o);
    cpu += performance.now() - t;
    await yieldNow();
    t = performance.now();
    if (req.rIndex) r = evaluateRoads(req.rIndex, grid, o);
    cpu += performance.now() - t;
    await yieldNow();
    t = performance.now();
    const g = summarizeGrid(req.storm.meta, grid);
    cpu += performance.now() - t;
    const ranked = [];
    if (b) {
      const top = Math.min(b.order.length, this._lopt.statsTopN);
      for (let i = 0; i < top; i++) {
        const idx = b.order[i];
        ranked.push({ index: idx, depthM: b.depthM[idx], tier: tierOfDepth(b.depthM[idx]), speedMs: b.speedMs[idx] });
      }
    }
    return {
      hour: req.hour, storm: req.storm.storm,
      floodedCount: b ? b.floodedCount : null, tierCounts: b ? b.tierCounts : null, belowTierCount: b ? b.belowTierCount : null,
      buildingCount: req.bIndex ? req.bIndex.count : 0, ranked, order: b ? b.order : new Int32Array(0),
      depthM: b ? b.depthM : null, speedMs: b ? b.speedMs : null, meanDepthM: b ? b.meanDepthM : null,
      roads: r ? { pctFlooded: r.pctFlooded, floodedLengthM: r.floodedLengthM, totalLengthM: r.totalLengthM, floodedFeatures: r.floodedFeatures, perFeaturePct: r.perFeaturePct } : null,
      grid: g, wetAreaKm2: g.wetAreaKm2, maxDepthM: g.maxDepthM, maxSpeedMs: g.maxSpeedMs,
      minDepthM: o.minDepthM, computeMs: performance.now() - t0, cpuMs: cpu,
    };
  }

  // Latest values of the current storm, null until the first 'stats' payload exists (or without the data set).
  getFloodStats() { return this._flood; }
  getBuildingFlood() {
    const f = this._flood;
    if (!f || f.floodedCount === null) return null;
    return {
      hour: f.hour, storm: f.storm, count: f.buildingCount, floodedCount: f.floodedCount, tierCounts: f.tierCounts, belowTierCount: f.belowTierCount,
      ranked: f.ranked, order: f.order, depthM: f.depthM, speedMs: f.speedMs, meanDepthM: f.meanDepthM,
    };
  }
  getRoadFlood() {
    const f = this._flood;
    if (!f || !f.roads) return null;
    return { hour: f.hour, storm: f.storm, ...f.roads };
  }

  // Computes the statistics of the displayed whole hour now (also without listeners) and resolves to the payload
  // (null after dispose or without a drawn storm). The 'stats' event fires for it as usual.
  async refreshStats() {
    if (this._disposed || !this._storm || !this._displayedOnce) return null;
    this._statsReq = null;
    this._statsMaybe(true);
    while (this._statsIdle) await this._statsIdle;
    return this._disposed ? null : this._flood;
  }

  // [lng, lat] of an attached building (index in the feature array), null when unknown or not yet indexed.
  buildingLngLat(index, out = [0, 0]) {
    const idx = this._currentIndex('buildings');
    if (!idx || !(index >= 0 && index < idx.count)) return null;
    const x = idx.centroids[2 * index], y = idx.centroids[2 * index + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    out[0] = x; out[1] = y;
    return out;
  }

  // Cost of the last index builds, for tests and diagnostics.
  getIndexInfo() {
    const info = {};
    for (const kind of ['buildings', 'roads']) {
      const k = this._idx[kind];
      const e = this._meta ? k.cache.get(gridSignature(this._meta)) : null;
      info[kind] = {
        attached: Boolean(k.features), count: k.features ? k.features.length : 0, ready: Boolean(e && e.index),
        ms: e ? e.ms : 0, slices: e ? e.slices : 0, maxSliceMs: e ? e.maxSliceMs : 0, cachedGrids: k.cache.size,
      };
    }
    return info;
  }

  // ------------------------------------------------------------------ 10. teardown

  // MapLibre 4.7 does not free the texture, vertex buffer and vertex array of an image/canvas source when it is removed
  // (one of each per source stays in the map's GL context for good; measured with a bare canvas source and no layer
  // of ours). They are released here; guarded because these are MapLibre internals.
  _releaseSourceGpu(src) {
    if (!src) return;
    try {
      for (const k of ['texture', 'boundsBuffer', 'boundsSegments']) {
        const o = src[k];
        if (o && typeof o.destroy === 'function') o.destroy();
        if (o) src[k] = null;
      }
    } catch (e) { /* internals changed */ }
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this._playing = false;
    this._stopLoop();
    for (const t of [this._regionTimer, this._retryTimer, this._retryAddTimer]) if (t) clearTimeout(t);
    this._regionTimer = this._retryTimer = this._retryAddTimer = 0;
    if (this._metaCtrl) { this._metaCtrl.abort(); this._metaCtrl = null; }
    if (this._first) { const f = this._first; this._first = null; f.reject(abortError('FluidLayer disposed')); }
    for (const j of this._jobs.values()) j.ctrl.abort();
    this._jobs.clear();
    if (this._fpCtrl) { this._fpCtrl.abort(); this._fpCtrl = null; }
    if (this._storm) this._storm.dispose();
    this._storm = null; this._have.clear(); this._failed.clear(); this._failCount.clear();
    this._gA = null; this._gB = null;
    for (const k of Object.values(this._idx)) {
      for (const e of k.cache.values()) e.cancel = true;
      k.cache.clear(); k.features = null; k.ver++;
      for (const r of k.waiters.splice(0)) r();
    }
    for (const d of Array.from(this._probers)) d();
    this._flood = null; this._statsReq = null;
    this._unhook();
    let src = null;
    try { if (this._map.getLayer(this._layerId)) this._map.removeLayer(this._layerId); } catch (e) { /* map gone */ }
    try { src = this._map.getSource(this._sourceId) || null; } catch (e) { src = null; }
    try { if (src) this._map.removeSource(this._sourceId); } catch (e) { /* map gone */ }
    this._releaseSourceGpu(src);
    this._renderer.dispose();
    this._setStatus('idle', 'disposed');
    this._em.clear();
  }
}
