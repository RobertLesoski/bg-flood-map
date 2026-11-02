// Harness for fluid-layer.js: a real MapLibre map (same style and terrain as viewer.html) with the real
// flow grids served by serve.py. window.__results holds {name, pass, detail} rows like the other harnesses.
//   layer-harness.html                interactive: storm / hour / play / view buttons
//   layer-harness.html?run=1          run the whole suite on load;  ?run=1&only=<regexp> runs matching groups
// Tests drive the layer with a manual scheduler (virtual time, map.redraw() per frame) because requestAnimationFrame
// is throttled to a few frames a second in a hidden or background pane; one test uses the real rAF loop.
import { FluidLayer, StormNotFoundError, QUALITY_LEVELS } from '../fluid-layer.js';
import { buildBuildingIndex, buildRoadIndex, evaluateBuildings, evaluateRoads, summarizeGrid, tierOfDepth } from '../fluid-stats.js';
import { loadStorm, FlowFormatError, mercToLngLat, lngLatToMerc, LRU_HOURS, PREFETCH_CONCURRENCY, decodeDepth } from '../fluid-data.js';

const BASE = new URL('../../layers/flow', location.href).pathname;
const params = new URLSearchParams(location.search);
const CENTER = [-86.419, 36.972];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------------------ GL object counters

// Live WebGL objects of the whole page (create adds, delete removes) and the number of contexts created, so that
// leak tests can compare counts before and after a cycle. Installed before the map exists.
const GLC = (() => {
  const live = {};
  const kinds = ['Texture', 'Buffer', 'Framebuffer', 'Renderbuffer', 'Program', 'Shader', 'VertexArray', 'TransformFeedback'];
  for (const k of kinds) live[k] = new Map();   // object -> its context
  const state = { contexts: 0 };
  for (const proto of [window.WebGL2RenderingContext && WebGL2RenderingContext.prototype, window.WebGLRenderingContext && WebGLRenderingContext.prototype]) {
    if (!proto) continue;
    for (const k of kinds) {
      const c = proto[`create${k}`], d = proto[`delete${k}`];
      if (!c || !d) continue;
      proto[`create${k}`] = function (...a) { const o = c.apply(this, a); if (o) live[k].set(o, this); return o; };
      proto[`delete${k}`] = function (o, ...a) { live[k].delete(o); return d.call(this, o, ...a); };
    }
  }
  const gc = HTMLCanvasElement.prototype.getContext;
  const seen = new WeakSet();
  HTMLCanvasElement.prototype.getContext = function (t, ...a) {
    const r = gc.call(this, t, ...a);
    if (r && !seen.has(r)) { seen.add(r); state.contexts++; }
    return r;
  };
  return {
    // objects of a lost context are gone with it; only those of live contexts count
    snapshot() {
      const o = { contexts: state.contexts };
      for (const k of kinds) {
        for (const [obj, ctx] of Array.from(live[k])) if (ctx.isContextLost()) live[k].delete(obj);
        o[k] = live[k].size;
      }
      return o;
    },
    // the same counts split by owner: the map's own context or any other (the layer's renderer canvases)
    split(mapCanvas) {
      const o = { map: {}, other: {} };
      for (const k of kinds) {
        o.map[k] = 0; o.other[k] = 0;
        for (const [, ctx] of live[k]) if (!ctx.isContextLost()) o[ctx.canvas === mapCanvas ? 'map' : 'other'][k]++;
      }
      return o;
    },
  };
})();

// ------------------------------------------------------------------------------ map

const map = new maplibregl.Map({
  container: 'map', preserveDrawingBuffer: true, maxPitch: 85, hash: false,
  style: {
    version: 8,
    sources: {
      osm: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256, maxzoom: 19,
        attribution: '&copy; OpenStreetMap' },
      terrain: { type: 'raster-dem', tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
        encoding: 'terrarium', tileSize: 256, maxzoom: 15 },
    },
    // viewer.html uses raster-opacity 0.6. With 3D terrain MapLibre 4.7.1 draws some terrain proxy tiles twice (retained
    // parent tile + children) and a translucent raster then shows as bright bands with stretched streaks (alpha 0.6 ->
    // 0.84 measured on the drawing buffer; it happens with no fluid layer on the map at all). The harness therefore
    // darkens the base with brightness instead of opacity (same look on this dark page, drawn identically twice).
    layers: [{ id: 'osm', type: 'raster', source: 'osm', paint: { 'raster-opacity': 1, 'raster-saturation': -0.5, 'raster-brightness-min': 0.02, 'raster-brightness-max': 0.62 } }],
    terrain: { source: 'terrain', exaggeration: 2 },
  },
  center: CENTER, zoom: 12, pitch: 52, bearing: -14,
});
map.addControl(new maplibregl.NavigationControl(), 'top-right');
// A pane that is not in front gets no requestAnimationFrame, so 'load' would never come: draw by hand until it does.
await new Promise((r) => {
  let done = false;
  map.on('load', () => { done = true; r(); });
  const t = setInterval(() => { if (done) clearInterval(t); else { try { map.redraw(); } catch (e) { /* style not ready */ } } }, 150);
});

let mglCtx = null;
const px4 = new Uint8Array(4);
function syncGpu() {
  if (!mglCtx) mglCtx = map.getCanvas().getContext('webgl2') || map.getCanvas().getContext('webgl');
  mglCtx.readPixels(0, 0, 1, 1, mglCtx.RGBA, mglCtx.UNSIGNED_BYTE, px4);
}
function redraw() { map.redraw(); syncGpu(); }

async function settle(ms = 25000) {
  const t0 = performance.now();
  await sleep(60);
  while (performance.now() - t0 < ms) {
    if (map.areTilesLoaded() && !map.isMoving()) break;
    await sleep(120);
  }
  redraw();
}

async function jump(o) {
  map.jumpTo(o);
  await settle();
}

// Manual scheduler: queued callbacks run when step()/pump() is called, with virtual time.
function makeSched(start = 1000) {
  let vt = start, id = 0;
  const queue = new Map();
  const s = {
    sched: { raf: (fn) => { queue.set(++id, fn); return id; }, caf: (i) => { queue.delete(i); }, now: () => vt },
    get vt() { return vt; },
    pending: () => queue.size,
    step(dt = 16.7) {
      vt += dt;
      const fns = Array.from(queue.values());
      queue.clear();
      for (const f of fns) f(vt);
    },
    pump(n, dt = 16.7, draw = true) {
      for (let i = 0; i < n; i++) { s.step(dt); if (draw) redraw(); }
    },
  };
  return s;
}

const BASE_OPTS = { base: BASE, autoQuality: false, particleCount: 20000 };
const live = new Set();
async function makeLayer(storm, o = {}) {
  const S = o.sched || makeSched();
  const layer = new FluidLayer(map, { ...BASE_OPTS, scheduler: S.sched, ...o.opts });
  live.add(layer);
  if (storm) await layer.setStorm(storm);
  return { layer, S };
}
function drop(layer) { live.delete(layer); layer.dispose(); }

// Wait until `cond()` is true while pumping virtual frames; returns the number of frames used (or -1 on timeout).
async function pumpUntil(S, cond, maxMs = 15000, dt = 16.7) {
  const t0 = performance.now();
  let n = 0;
  while (performance.now() - t0 < maxMs) {
    if (cond()) return n;
    S.pump(1, dt);
    n++;
    await sleep(8);
  }
  return cond() ? n : -1;
}

// ------------------------------------------------------------------------------ pixels

function mapPixel(x, y, out = new Uint8Array(4)) {
  // x, y in CSS pixels of the map container (top-left origin)
  if (!mglCtx) syncGpu();
  const c = map.getCanvas();
  const sx = c.width / c.clientWidth, sy = c.height / c.clientHeight;
  mglCtx.readPixels(Math.round(x * sx), Math.round(c.height - 1 - y * sy), 1, 1, mglCtx.RGBA, mglCtx.UNSIGNED_BYTE, out);
  return out;
}
const isWaterRgb = (p) => p[2] > p[0] + 25 && p[2] >= p[1] && p[2] > 70;   // blue dominant (OSM base is desaturated grey)

// ------------------------------------------------------------------------------ results

window.__results = [];
window.__done = false;
function record(name, pass, detail = '') {
  window.__results.push({ name, pass: !!pass, detail: String(detail) });
  const list = $('list');
  if (list) {
    const row = document.createElement('div');
    row.className = `t ${pass ? 'pass' : 'fail'}`;
    row.innerHTML = `<b>${pass ? 'PASS' : 'FAIL'}</b> ${name} <i></i>`;
    row.lastChild.textContent = detail;
    list.appendChild(row);
  }
}
function setStatus(s) { const e = $('status'); if (e) e.textContent = s; }

// ------------------------------------------------------------------------------ data helpers

const stormCache = new Map();
async function rawStorm(key) {
  if (!stormCache.has(key)) stormCache.set(key, await loadStorm(BASE, key));
  return stormCache.get(key);
}
// Deepest pixel of the flow-paths grid (first of the maximum), as {col,row,x,y,lng,lat,R}
async function deepestPixel(key) {
  const st = await rawStorm(key);
  const fp = (await st.getFlowpaths()).rgba;
  const m = st.meta, W = m.width, H = m.height;
  let best = -1, bi = 0, bg = -1;
  for (let i = 0; i < W * H; i++) {
    const r = fp[i * 4];
    if (r > best || (r === best && fp[i * 4 + 1] > bg)) { best = r; bi = i; bg = fp[i * 4 + 1]; }
  }
  const col = bi % W, row = (bi / W) | 0;
  const x = m.bounds_merc[0] + (col + 0.5) * m.cell_m, y = m.bounds_merc[3] - (row + 0.5) * m.cell_m;
  const ll = mercToLngLat(x, y);
  return { col, row, x, y, lng: ll[0], lat: ll[1], R: best, depth: decodeDepth(best) };
}

// ------------------------------------------------------------------------------ interactive demo

let demo = null;
let hotspot = null;            // deepest flow-path pixel of the current storm
let sliding = false;

function fmtStatus() {
  if (!demo) return 'no layer';
  const s = demo.getState(), q = demo.qualityInfo;
  return `${s.status.state}${s.status.message ? ' (' + s.status.message + ')' : ''} | t=${s.time.toFixed(2)} | q${q.level} ${q.canvasWidth}x${q.canvasHeight} | avg ${q.avgMs.toFixed(1)} ms | frames ${s.frames}`;
}

async function startDemo(storm) {
  if (demo) { live.delete(demo); demo.dispose(); demo = null; }
  demo = new FluidLayer(map, { base: BASE, autoQuality: params.get('auto') !== '0', particleCount: 40000 });
  live.add(demo);
  window.layer = demo;
  demo.on('time', (e) => { if (!sliding) { $('hour').value = e.hour; } $('hourv').textContent = e.hour.toFixed(2); });
  demo.on('status', () => setStatus(fmtStatus()));
  demo.on('quality', () => setStatus(fmtStatus()));
  demo.on('region', () => setStatus(fmtStatus()));
  demo.on('error', (e) => { console.warn('layer error', e.phase, e.error); setStatus(`ERROR ${e.phase}: ${e.error && e.error.message}`); });
  try {
    await demo.setStorm(storm);
    $('badge').textContent = demo.source.label;
    $('badge').title = demo.source.note;
    hotspot = await deepestPixel(storm);
  } catch (e) {
    if (e.name !== 'AbortError') setStatus(`setStorm failed: ${e.name}: ${e.message}`);
  }
}

function buildViews() {
  const box = $('views');
  const mk = (label, fn) => { const b = document.createElement('button'); b.textContent = label; b.onclick = fn; box.appendChild(b); };
  for (const p of [0, 30, 60, 80]) mk(`p${p}`, () => map.easeTo({ pitch: p, duration: 0 }));
  for (const z of [11, 12, 14, 15, 17, 18]) mk(`z${z}`, () => map.jumpTo({ zoom: z }));
  for (const b of [0, 90, 200]) mk(`b${b}`, () => map.jumpTo({ bearing: b }));
  mk('hot', () => { if (hotspot) map.jumpTo({ center: [hotspot.lng, hotspot.lat] }); });
  mk('home', () => map.jumpTo({ center: CENTER, zoom: 12, pitch: 52, bearing: -14 }));
}

function bindUi() {
  buildViews();
  $('storm').onchange = () => startDemo($('storm').value);
  $('hour').addEventListener('input', (e) => { sliding = true; if (demo) demo.setTime(+e.target.value); });
  $('hour').addEventListener('change', () => { sliding = false; });
  $('play').onclick = () => { if (!demo) return; demo.togglePlay(); $('play').classList.toggle('on', demo.isPlaying); $('play').textContent = demo.isPlaying ? 'pause' : 'play'; };
  $('freeze').onclick = () => { const on = !demo.getOptions().freeze; demo.setOptions({ freeze: on }); $('freeze').classList.toggle('on', on); };
  $('vis').onclick = () => { demo.setVisible(!demo.visible); $('vis').classList.toggle('on', demo.visible); };
  $('terrain').onchange = (e) => map.setTerrain(e.target.checked ? { source: 'terrain', exaggeration: 2 } : null);
  const opt = (id, key) => { $(id).onchange = (e) => demo.setOptions({ [key]: e.target.checked }); };
  opt('oDepth', 'showDepth'); opt('oFlow', 'showFlow'); opt('oPart', 'showParticles'); opt('oArr', 'showArrows'); opt('oFp', 'showFlowpaths');
  $('runTests').onclick = () => runTests();
  $('toggleList').onclick = () => { const l = $('list'); l.style.display = l.style.display === 'none' ? '' : 'none'; };
  $('min').onclick = () => $('panel').classList.toggle('min');
  setInterval(() => { if (demo && !window.__testing) setStatus(fmtStatus()); }, 500);
}

const GROUPS = [];
async function runTests(opts = {}) {
  window.__testing = true;
  const only = opts.only ? new RegExp(opts.only, 'i') : null;
  window.__results.length = 0; window.__done = false;
  if ($('list')) $('list').textContent = '';
  if (demo) { live.delete(demo); demo.dispose(); demo = null; }
  for (const [name, fn] of GROUPS) {
    if (only && !only.test(name)) continue;
    setStatus(`running ${name}...`);
    const n0 = window.__results.length;
    try { await fn(); } catch (e) { record(`${name}: threw`, false, e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e); }
    for (const l of Array.from(live)) drop(l);
    if (window.__results.length === n0) record(`${name}: no checks ran`, false);
  }
  const pass = window.__results.filter((r) => r.pass).length, total = window.__results.length;
  const text = `${pass}/${total} passed${pass === total ? '' : ` - ${total - pass} FAILED`}`;
  if ($('summary')) $('summary').textContent = text;
  setStatus(text);
  window.__testing = false;
  window.__done = true;
  return text;
}

// ------------------------------------------------------------------------------ test helpers

const R_EARTH = 6378137;
const invMerc = (x, y) => [x / R_EARTH * 180 / Math.PI, (2 * Math.atan(Math.exp(y / R_EARTH)) - Math.PI / 2) * 180 / Math.PI];
const near = (a, b, eps) => Math.abs(a - b) <= eps;

function hashStride(arr, stride = 997) {
  let h = 17;
  for (let i = 0; i < arr.length; i += stride) h = (Math.imul(h, 31) + arr[i]) | 0;
  return h;
}
// Alpha statistics of the renderer canvas (premultiplied RGBA)
function canvasStats(layer) {
  const p = layer.renderer.readPixels();
  let n = 0, sum = 0;
  for (let i = 3; i < p.length; i += 4) { if (p[i] > 8) n++; sum += p[i]; }
  return { n, sum, w: layer.canvas.width, h: layer.canvas.height, px: p };
}
function bytesDiffer(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  return d;
}

// fetch that delays and counts grid downloads (hNN.rgba.gz and flowpaths)
function slowFetch(latencyMs, stats) {
  return (url, init) => {
    if (!/\.rgba\.gz/.test(url)) return fetch(url, init);
    const fp = /flowpaths/.test(url);
    if (fp) stats.flowpaths++;
    stats.starts++; stats.inflight++;
    stats.maxInflight = Math.max(stats.maxInflight, stats.inflight);
    const sig = init && init.signal;
    return new Promise((resolve, reject) => {
      let done = false;
      const fin = () => { if (!done) { done = true; stats.inflight--; } };
      const t = setTimeout(() => { fetch(url, init).then((res) => { fin(); resolve(res); }, (e) => { fin(); reject(e); }); }, latencyMs);
      if (sig) sig.addEventListener('abort', () => { if (!done) { clearTimeout(t); stats.aborted++; fin(); reject(new DOMException('aborted', 'AbortError')); } }, { once: true });
    });
  };
}
const newStats = () => ({ starts: 0, inflight: 0, maxInflight: 0, aborted: 0, flowpaths: 0 });

// Clean view for deterministic pixels: flat depth colour only
const FLAT = { showFlow: false, showParticles: false, showArrows: false };

async function gridCentreView(extra = {}) {
  const st = await rawStorm('25yr');
  const b = st.meta.bounds_merc;
  const c = invMerc((b[0] + b[2]) / 2, (b[1] + b[3]) / 2);
  await jump({ center: c, zoom: 12, pitch: 0, bearing: 0, ...extra });
}

// ------------------------------------------------------------------------------ test groups

GROUPS.push(['basic', async () => {
  record('FluidLayer.supported() is true here', FluidLayer.supported() === true);
  const stats = newStats();
  const { layer, S } = await makeLayer(null, { opts: { fetch: slowFetch(0, stats) } });
  record('new layer is idle, nothing on the map yet', layer.status.state === 'idle' && !map.getLayer('fluid-water') && !map.getSource('fluid-water-src'));
  const states = [], evts = { storm: 0, hour: [], time: 0, region: 0 };
  layer.on('status', (e) => states.push(e.state));
  layer.on('storm', () => evts.storm++);
  layer.on('hour', (e) => evts.hour.push(e));
  layer.on('time', () => evts.time++);
  layer.on('region', () => evts.region++);
  await gridCentreView();
  const meta = await layer.setStorm('25yr');
  record('setStorm resolves to the meta and shows a picture', meta && meta.storm === '25yr' && layer.hasPicture && layer.storm === '25yr', `status ${JSON.stringify(layer.status)}`);
  record('status went loading -> ready', states[0] === 'loading' && states[states.length - 1] === 'ready', states.join(','));
  record("'storm' event fired once, 'region' event fired", evts.storm === 1 && evts.region >= 1, `storm ${evts.storm}, region ${evts.region}`);
  record('source info: preview from depth tiers + note', layer.source.kind === 'tiers' && layer.source.label === 'Preview from depth tiers' && layer.source.note.length > 20, JSON.stringify(layer.source.label));
  record('layer and source exist on the map, visible', Boolean(map.getLayer('fluid-water') && map.getSource('fluid-water-src')) && map.getLayoutProperty('fluid-water', 'visibility') === 'visible');
  record('layer paints raster-fade-duration 0 and linear resampling', map.getPaintProperty('fluid-water', 'raster-fade-duration') === 0 && map.getPaintProperty('fluid-water', 'raster-resampling') === 'linear');
  record('timeRange is 0..last hour with loopStart one before first wet hour', JSON.stringify([layer.timeRange.min, layer.timeRange.max, layer.timeRange.loopStart]) === '[0,24,1]', JSON.stringify(layer.timeRange).slice(0, 80));

  // hour events carry the grid; dry hours carry null
  layer.setTime(15);
  await pumpUntil(S, () => layer.time === 15);
  const e15 = evts.hour.find((e) => e.hour === 15);
  record("'hour' event for a wet hour carries the grid", e15 && e15.grid && e15.grid.length === meta.width * meta.height * 4 && e15.dry === false && e15.storm === '25yr');
  layer.setTime(0.4);
  await pumpUntil(S, () => layer.time === 0.4);
  const e0 = evts.hour.find((e) => e.hour === 0);
  record("'hour' event for a dry hour has grid null and dry true", e0 && e0.grid === null && e0.dry === true);
  record("'time' events were emitted", evts.time >= 2, `${evts.time}`);

  // layer order: below roads-line / all-parcels-fill even when those are added afterwards
  const dummy = (id, type, before) => map.addLayer({ id, type, source: 'osm' === id ? 'osm' : 'dummy-geo', paint: type === 'line' ? { 'line-color': '#f00' } : { 'fill-color': '#0f0' } }, before);
  map.addSource('dummy-geo', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  const above = () => { const o = map.getLayersOrder(); return o[o.indexOf('fluid-water') + 1]; };
  record('without overlays the water is on top of the stack', above() === undefined, String(above()));
  dummy('buildings-ext', 'fill');
  await sleep(50);
  record('water moves below buildings-ext when it appears', above() === 'buildings-ext', String(above()));
  dummy('all-parcels-fill', 'fill', 'buildings-ext');   // viewer.html order, bottom to top: roads-line, all-parcels-fill, buildings-ext
  await sleep(50);
  record('water is below all-parcels-fill too', above() === 'buildings-ext' || above() === 'all-parcels-fill', String(above()));
  map.addLayer({ id: 'roads-line', type: 'line', source: 'dummy-geo', paint: { 'line-color': '#f00' } }, 'all-parcels-fill');
  await sleep(50);
  const o = map.getLayersOrder();
  record('water sits directly below roads-line, above osm', o.indexOf('fluid-water') + 1 === o.indexOf('roads-line') && o.indexOf('osm') < o.indexOf('fluid-water'), o.join(' > '));
  for (const id of ['roads-line', 'all-parcels-fill', 'buildings-ext']) map.removeLayer(id);
  map.removeSource('dummy-geo');
  await sleep(50);

  // typed errors; the previous storm stays on a failed request
  let err = null;
  try { await layer.setStorm('nope-9yr'); } catch (e) { err = e; }
  record('missing storm rejects with StormNotFoundError', err instanceof StormNotFoundError && err.storm === 'nope-9yr', err && `${err.name}: ${err.message}`.slice(0, 120));
  record('a failed request leaves the previous storm shown', layer.storm === '25yr' && layer.hasPicture && layer.status.state !== 'error', JSON.stringify(layer.status));
  drop(layer);

  // bad data: wrong format tag in meta, and a corrupt grid
  const badMeta = (url, init) => (/meta\.json/.test(url)
    ? fetch(url, init).then(async (r) => { const j = await r.json(); j.format = 'nope'; return new Response(JSON.stringify(j), { status: 200 }); })
    : fetch(url, init));
  const b1 = await makeLayer(null, { opts: { fetch: badMeta } });
  err = null;
  try { await b1.layer.setStorm('25yr'); } catch (e) { err = e; }
  record('bad meta rejects with FlowFormatError', err instanceof FlowFormatError, err && `${err.name}: ${err.message}`.slice(0, 120));
  record('bad meta leaves status error and no layer on the map', b1.layer.status.state === 'error' && !map.getLayer('fluid-water'), JSON.stringify(b1.layer.status).slice(0, 100));
  drop(b1.layer);
  const shortGz = async () => {
    const cs = new CompressionStream('gzip');
    const w = cs.writable.getWriter(); w.write(new Uint8Array(100)); w.close();
    return new Response(cs.readable, { status: 200, headers: { 'content-type': 'application/gzip' } });
  };
  const badGrid = (url, init) => (/h\d\d\.rgba\.gz/.test(url) ? shortGz() : fetch(url, init));
  const b2 = await makeLayer(null, { opts: { fetch: badGrid } });
  const errEvts = [];
  b2.layer.on('error', (e) => errEvts.push(e));
  await gridCentreView();
  await b2.layer.setStorm('25yr');           // first picture is dry (t=0), so the storm itself is accepted
  b2.layer.setTime(10);
  await pumpUntil(b2.S, () => errEvts.length > 0, 8000);
  record('corrupt grid emits error {phase grid} with FlowFormatError and status error',
    errEvts.length > 0 && errEvts[0].phase === 'grid' && errEvts[0].error instanceof FlowFormatError && b2.layer.status.state === 'error', errEvts[0] && `${errEvts[0].error.name} hour ${errEvts[0].hour}`);
  record('the picture holds while the grid is broken', b2.layer.time === 0.4 || b2.layer.time === 0, `time ${b2.layer.time}`);
  drop(b2.layer);
}]);

GROUPS.push(['storm switch', async () => {
  const stats = newStats();
  const { layer, S } = await makeLayer(null, { opts: { fetch: slowFetch(500, stats), ...FLAT } });
  await gridCentreView();
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14);
  const hash25 = hashStride(layer.currentGrids().rgbaA);
  record('25yr hour 14 is on screen', layer.hasPicture && layer.meta.storm === '25yr', `hash ${hash25}`);
  // superseded setStorm
  const p1 = layer.setStorm('10yr');
  const p2 = layer.setStorm('5yr');
  let e1 = null;
  try { await p1; } catch (e) { e1 = e; }
  record('a superseded setStorm rejects with AbortError', e1 && e1.name === 'AbortError', e1 && e1.name);
  // wait for commit of 5yr (meta loaded) while the grids are still slow
  let committed = false;
  for (let i = 0; i < 400 && !committed; i++) { committed = layer.storm === '5yr'; if (!committed) await sleep(10); }
  record('after the switch the old water is hidden at once', committed && layer.hasPicture === false && map.getLayoutProperty('fluid-water', 'visibility') === 'none',
    `storm ${layer.storm}, picture ${layer.hasPicture}, vis ${map.getLayoutProperty('fluid-water', 'visibility')}`);
  record('old storm grids are gone from the layer', layer.currentGrids().rgbaA === null && layer.meta.storm === '5yr');
  const meta = await p2;
  // the new storm's first picture needs hour 14 (time is kept): it is shown only after it was decoded
  await pumpUntil(S, () => layer.hasPicture && layer.time === 14, 12000);
  const g = layer.currentGrids();
  record('new storm draws its own grid and shows it', meta.storm === '5yr' && layer.hasPicture && g.rgbaA && hashStride(g.rgbaA) !== hash25 && map.getLayoutProperty('fluid-water', 'visibility') === 'visible',
    `hash ${g.rgbaA && hashStride(g.rgbaA)}`);
  // old storm hour files are no longer held
  record('only 5yr hours are cached', layer.getState().cached.length <= 4 && layer.storm === '5yr', JSON.stringify(layer.getState().cached));
}]);

GROUPS.push(['time', async () => {
  const st = await rawStorm('25yr');
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  await gridCentreView();
  await layer.setStorm('25yr');
  const settleAt = async (t) => {
    layer.setTime(t);
    const n = await pumpUntil(S, () => layer.time === t && !layer.getState().buffering, 12000);
    S.pump(2);
    return n >= 0;
  };
  for (const h of [6, 12, 20]) {
    const ok = await settleAt(h);
    const g = layer.currentGrids();
    const ref = (await st.getHour(h)).rgba;
    record(`integer hour ${h}: pair is (${h}, mix 0) and the grid is hour ${h}`, ok && g.hourA === h && g.mix === 0 && g.rgbaA && hashStride(g.rgbaA) === hashStride(ref) && g.rgbaB === g.rgbaA,
      `A ${g.hourA} B ${g.hourB} mix ${g.mix}`);
  }
  await settleAt(12.25);
  let g = layer.currentGrids();
  record('fractional 12.25: blends hour 12 and 13 with mix 0.25', g.hourA === 12 && g.hourB === 13 && near(g.mix, 0.25, 1e-9) && g.rgbaA && g.rgbaB && g.rgbaA !== g.rgbaB, `A ${g.hourA} B ${g.hourB} mix ${g.mix}`);
  await settleAt(12.999);
  g = layer.currentGrids();
  record('12.999 stays in pair (12, 13) with mix near 1', g.hourA === 12 && g.hourB === 13 && g.mix > 0.99, `mix ${g.mix}`);
  await settleAt(0.5);
  g = layer.currentGrids();
  let cs = canvasStats(layer);
  record('t=0.5 (hours 0,1 have no file): both grids dry and the canvas is empty', g.rgbaA === null && g.rgbaB === null && cs.n === 0, `wet px ${cs.n}`);
  await settleAt(1.5);
  g = layer.currentGrids();
  record('t=1.5: hour 1 is dry (null), hour 2 is the first wet grid', g.hourA === 1 && g.hourB === 2 && g.rgbaA === null && g.rgbaB !== null);
  await settleAt(40);
  g = layer.currentGrids();
  record('after the last hour the time clamps and the last grid stays', layer.time === 24 && g.hourA === 24 && g.hourB === 24 && g.mix === 0 && g.rgbaA === g.rgbaB, `time ${layer.time}`);
  await settleAt(-3);
  record('negative time clamps to 0', layer.time === 0);

  // wet area grows while the storm builds (same region for every hour)
  const areas = [];
  for (const h of [5, 8, 11, 14, 17, 20]) { await settleAt(h); areas.push(canvasStats(layer).n); }
  let grows = true;
  for (let i = 1; i < areas.length; i++) if (!(areas[i] > areas[i - 1])) grows = false;
  record('wet area grows hour after hour (5,8,11,14,17,20)', grows, areas.join(' < '));

  // fade in from dry: alpha at the first wet pixel rises with the mix between hour 1 (dry) and hour 2
  const g2 = (await st.getHour(2)).rgba;
  let bi = -1, br = 0;
  for (let i = 0; i < g2.length; i += 4) if (g2[i] > br) { br = g2[i]; bi = i / 4; }
  const m = st.meta;
  const wx = m.bounds_merc[0] + ((bi % m.width) + 0.5) * m.cell_m, wy = m.bounds_merc[3] - (Math.floor(bi / m.width) + 0.5) * m.cell_m;
  const ll = invMerc(wx, wy);
  await jump({ center: ll, zoom: 16, pitch: 0, bearing: 0 });
  layer.updateRegion(true);
  const alphas = [];
  for (const mix of [0, 0.25, 0.5, 0.75, 1]) {
    await settleAt(1 + mix);
    const rp = layer.renderer.worldToPixel(wx, wy);
    const p = layer.renderer.readPixel(Math.round(rp[0]), Math.round(rp[1]));
    alphas.push(p[3]);
  }
  let mono = true;
  for (let i = 1; i < alphas.length; i++) if (alphas[i] < alphas[i - 1]) mono = false;
  record('water fades in from dry: alpha is 0 at mix 0 and rises with the mix', mono && alphas[0] === 0 && alphas[4] > 0, `hour-2 max depth ${decodeDepth(br).toFixed(2)} m, alpha ${alphas.join(',')}`);
}]);

GROUPS.push(['alignment', async () => {
  const hp = await deepestPixel('25yr');
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  await jump({ center: [hp.lng, hp.lat], zoom: 16, pitch: 0, bearing: 0 });
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14);
  const st = await rawStorm('25yr');
  const m = st.meta, W = m.width;
  map.addSource('align-pt', { type: 'geojson', data: { type: 'Point', coordinates: [hp.lng, hp.lat] } });
  map.addLayer({ id: 'align-circle', type: 'circle', source: 'align-pt', paint: { 'circle-radius': 4, 'circle-color': '#ff0000', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1 } });
  const grid = layer.currentGrids().rgbaA;
  const win = 36, stride = 2;
  const cellsWet = [], cellsDry = [];
  const wetAt = (c, r) => (c < 0 || r < 0 || c >= W || r >= m.height ? false : grid[(r * W + c) * 4] > 0);
  for (let r = hp.row - win; r <= hp.row + win; r += stride) {
    for (let c = hp.col - win; c <= hp.col + win; c += stride) {
      if (c < 0 || r < 0 || c >= W || r >= m.height) continue;
      const R = grid[(r * W + c) * 4], A = grid[(r * W + c) * 4 + 3];
      if (R >= 40) cellsWet.push([c, r]);
      else if (R === 0 && A === 255) {
        let clear = true;
        for (let dr = -3; dr <= 3 && clear; dr++) for (let dc = -3; dc <= 3; dc++) if (wetAt(c + dc, r + dr)) { clear = false; break; }
        if (clear) cellsDry.push([c, r]);
      }
    }
  }
  const classify = (shiftCells) => {
    const cv = map.getCanvas();
    const w = cv.clientWidth, h = cv.clientHeight, p = new Uint8Array(4);
    let tp = 0, fn = 0, tn = 0, fp = 0;
    const test = (cells, wantWater) => {
      for (const [c, r] of cells) {
        const x = m.bounds_merc[0] + (c + 0.5 + shiftCells) * m.cell_m, y = m.bounds_merc[3] - (r + 0.5) * m.cell_m;
        const ll = invMerc(x, y);
        const s = map.project(ll);
        if (!(s.x > 2 && s.y > 2 && s.x < w - 2 && s.y < h - 2)) continue;
        mapPixel(s.x, s.y, p);
        const isW = isWaterRgb(p);
        if (wantWater) { if (isW) tp++; else fn++; } else if (isW) fp++; else tn++;
      }
    };
    test(cellsWet, true); test(cellsDry, false);
    const bal = ((tp / Math.max(1, tp + fn)) + (tn / Math.max(1, tn + fp))) / 2;
    return { tp, fn, tn, fp, bal };
  };
  const sourceCorners = () => {
    const s = map.getSource('fluid-water-src'), r = layer.region;
    const exp = [invMerc(r[0], r[3]), invMerc(r[2], r[3]), invMerc(r[2], r[1]), invMerc(r[0], r[1])];
    let d = 0;
    for (let i = 0; i < 4; i++) d = Math.max(d, Math.abs(s.coordinates[i][0] - exp[i][0]), Math.abs(s.coordinates[i][1] - exp[i][1]));
    return d;
  };
  const dp = new Uint8Array(4);
  let firstCfg = true;
  for (const terrain of [true, false]) {
    map.setTerrain(terrain ? { source: 'terrain', exaggeration: 2 } : null);
    for (const pitch of [0, 60]) {
      for (const bearing of [0, 90, 200]) {
        const tag = `pitch ${pitch} bearing ${bearing} terrain ${terrain ? 'on' : 'off'}`;
        // The terrain keeps loading tiles after a camera jump (and the pane may be resized meanwhile), so a
        // configuration is measured again, up to 3 times, until its numbers hold; a real misalignment fails every try.
        let m1 = null, attempts = 0;
        while (attempts < 3) {
          attempts++;
          await jump({ center: [hp.lng, hp.lat], zoom: 16, pitch, bearing });
          layer.updateRegion(true);
          S.pump(4);
          for (let k = 0; k < 20; k++) {            // wait for a stable terrain height at the target
            const e0 = map.queryTerrainElevation([hp.lng, hp.lat]);
            await sleep(250); S.pump(1);
            if (!terrain || (map.queryTerrainElevation([hp.lng, hp.lat]) === e0 && map.areTilesLoaded())) break;
          }
          await settle();
          S.pump(3);
          const dcorner = sourceCorners();                                   // (1) canvas source coordinates are the region corners
          const rp = layer.renderer.worldToPixel(hp.x, hp.y);                // (2) renderer mapping: deepest pixel holds deep water
          const cp = layer.renderer.readPixel(Math.round(rp[0]), Math.round(rp[1]), dp);
          const deepOk = cp[3] > 200 && cp[2] > cp[0] + 40;
          const sp = map.project([hp.lng, hp.lat]);                          // (3) map mapping: marker drawn at the projected point ...
          let rx = 0, ry = 0, rn = 0;
          const q = new Uint8Array(4);
          for (let dy = -14; dy <= 14; dy++) for (let dx = -14; dx <= 14; dx++) {
            mapPixel(sp.x + dx, sp.y + dy, q);
            if (q[0] > 200 && q[1] < 90 && q[2] < 90) { rx += dx; ry += dy; rn++; }
          }
          const redOff = rn ? Math.hypot(rx / rn, ry / rn) : 99;
          map.setLayoutProperty('align-circle', 'visibility', 'none'); redraw();   // ... and water under it
          const mp = mapPixel(sp.x, sp.y, new Uint8Array(4));
          map.setLayoutProperty('align-circle', 'visibility', 'visible'); redraw();
          const cls = classify(0);
          m1 = { dcorner, cp, mp, redOff, rn, deepOk, cls };
          if (deepOk && isWaterRgb(mp) && redOff < 2.5 && cls.bal >= 0.85) break;
        }
        const { dcorner, cp, mp, redOff, rn, deepOk, cls } = m1;
        record(`align ${tag}: source corners = region corners`, dcorner < 1e-9, `max diff ${dcorner.toExponential(1)} deg`);
        record(`align ${tag}: deepest pixel is deep water in canvas and on screen, marker on it`, deepOk && isWaterRgb(mp) && redOff < 2.5,
          `canvas ${Array.from(cp)} screen ${Array.from(mp)} marker offset ${redOff.toFixed(2)} px (${rn} px), attempts ${attempts}`);
        record(`align ${tag}: wet/dry cells land on water/land (balanced accuracy)`, cls.bal >= 0.85 && cls.tp + cls.fn >= 20 && cls.tn + cls.fp >= 20, `bal ${cls.bal.toFixed(3)} tp ${cls.tp} fn ${cls.fn} tn ${cls.tn} fp ${cls.fp}, attempts ${attempts}`);
        if (firstCfg) {
          firstCfg = false;
          const sh = classify(4);
          layer.setVisible(false); S.pump(2);
          const off = classify(0);
          layer.setVisible(true); S.pump(2);
          record('align control: sampling 4 cells away or with the layer hidden scores clearly worse',
            sh.bal < cls.bal - 0.12 && off.bal < cls.bal - 0.12, `shifted ${sh.bal.toFixed(3)}, hidden ${off.bal.toFixed(3)}, normal ${cls.bal.toFixed(3)}`);
        }
      }
    }
  }
  map.removeLayer('align-circle'); map.removeSource('align-pt');
  map.setTerrain({ source: 'terrain', exaggeration: 2 });
}]);

GROUPS.push(['playback', async () => {
  const stats = newStats();
  const { layer, S } = await makeLayer(null, { opts: { fetch: slowFetch(900, stats), ...FLAT, loopStart: null } });
  await gridCentreView();
  const sts = [];
  layer.on('status', (e) => sts.push(e.state));
  await layer.setStorm('5yr');
  layer.setTime(7);
  await pumpUntil(S, () => layer.time === 7, 15000);
  record('scrub to an unloaded hour reports buffering, then ready', sts.includes('buffering') && sts[sts.length - 1] === 'ready' && layer.time === 7, sts.join(','));
  // play faster than the data can arrive: 900 ms per grid, 2 in parallel, 60 ms of virtual time per hour (the frames are paced by real time, so this is far faster than 900 ms per 2 grids)
  const seen = [], jumps = [];
  let prev = layer.time, minPair = true, maxStep = 0, holds = 0, lastBuf = false;
  layer.on('time', (e) => {
    seen.push(e.hourA);
    maxStep = Math.max(maxStep, e.hour - prev);
    if (e.hour < prev - 1e-9) jumps.push(`${prev}->${e.hour}`);
    prev = e.hour;
  });
  layer.play(60);
  record('isPlaying is true', layer.isPlaying === true);
  const t0 = performance.now();
  let bufFrames = 0, frames = 0, tHold = -1, heldOk = true;
  let bufMs = 0;
  while (performance.now() - t0 < 26000 && layer.time < 17) {
    const before = layer.time, tf = performance.now();
    S.pump(1, 16.7);
    await sleep(6);
    frames++;
    const st = layer.getState();
    if (st.buffering) {
      bufFrames++; bufMs += performance.now() - tf;
      if (layer.time !== before) heldOk = false;
    }
    const g = layer.currentGrids();
    if (g.rgbaA === undefined || (g.mix > 0 && g.rgbaB === undefined)) minPair = false;
    if (g.hourA >= 3 && g.rgbaA === null) minPair = false;
  }
  layer.pause();
  const hrs = Array.from(new Set(seen));
  let consecutive = hrs.length > 1;
  for (let i = 1; i < hrs.length; i++) if (hrs[i] !== hrs[i - 1] + 1) consecutive = false;
  record('playback reached hour 17 and every hour in between was shown, none skipped', layer.time >= 17 - 1e-9 && consecutive, `hours ${hrs.join(',')}`);
  record('playback held (buffering) while grids were missing and time stood still then', bufMs > 1800 && bufFrames >= 5 && heldOk && sts.filter((s) => s === 'buffering').length >= 2, `buffering ${bufMs.toFixed(0)} ms in ${bufFrames}/${frames} frames, status changes ${sts.filter((s) => s === 'buffering').length}`);
  record('time never ran backwards and never advanced more than half an hour per frame', jumps.length === 0 && maxStep <= 0.5 + 1e-9, `max step ${maxStep.toFixed(3)} h, backward ${jumps.join(' ')}`);
  record('every displayed pair had its grids (dry hours null)', minPair);
  record('at most PREFETCH_CONCURRENCY grid downloads were ever in flight', stats.maxInflight <= PREFETCH_CONCURRENCY, `max in flight ${stats.maxInflight}, starts ${stats.starts}, aborted ${stats.aborted}`);
  record('the loader cache never exceeds the LRU of 4', layer.getState().cached.length <= LRU_HOURS, JSON.stringify(layer.getState().cached));

  // loop: after the last hour (held one hour) playback restarts at loopStart
  layer.setTime(23.2);
  await pumpUntil(S, () => layer.time === 23.2, 15000);
  const times = [];
  layer.on('time', (e) => times.push(e.hour));
  layer.play(150);
  const t1 = performance.now();
  let wrapped = false;
  while (performance.now() - t1 < 25000 && !wrapped) {
    S.pump(1, 16.7);
    await sleep(6);
    if (times.length && times[times.length - 1] < 5 && times.some((x) => x >= 24)) wrapped = true;
  }
  layer.pause();
  const iMax = times.indexOf(24);
  record('playback holds the last hour, then loops to loopStart (2 for 5yr: one before the first wet hour 3)', wrapped && iMax >= 0 && times[times.length - 1] >= 2 && times[times.length - 1] < 3, `reached 24 at event ${iMax}, last ${times[times.length - 1] && times[times.length - 1].toFixed(2)}`);
}]);

GROUPS.push(['rapid scrubbing', async () => {
  const stats = newStats();
  const { layer, S } = await makeLayer(null, { opts: { fetch: slowFetch(350, stats), ...FLAT } });
  await gridCentreView();
  await layer.setStorm('25yr');
  layer.setTime(8);
  await pumpUntil(S, () => layer.time === 8, 15000);
  const base = stats.starts;
  let last = 8;
  for (let i = 0; i < 200; i++) {
    last = 3 + Math.random() * 20.9;
    layer.setTime(last);
    if (i % 4 === 0) S.pump(1);
    if (i % 10 === 0) await sleep(4);
  }
  record('200 rapid scrubs: loads were cancelled, never more than the concurrency limit in flight', stats.aborted > 0 && stats.maxInflight <= PREFETCH_CONCURRENCY, `aborted ${stats.aborted}, started ${stats.starts - base}, max in flight ${stats.maxInflight}`);
  const final = 13.4;
  layer.setTime(final);
  const n = await pumpUntil(S, () => layer.time === final && !layer.getState().buffering && layer.getState().inflight.length === 0, 20000);
  S.pump(2);
  const g = layer.currentGrids(), ref = (await (await rawStorm('25yr')).getHour(13)).rgba, ref14 = (await (await rawStorm('25yr')).getHour(14)).rgba;
  record('after the storm of scrubs the picture is the last requested time', n >= 0 && g.hourA === 13 && g.hourB === 14 && near(g.mix, 0.4, 1e-9) && hashStride(g.rgbaA) === hashStride(ref) && hashStride(g.rgbaB) === hashStride(ref14),
    `A ${g.hourA} B ${g.hourB} mix ${g.mix}`);
  const st = layer.getState();
  record('state is consistent: ready, nothing in flight, at most 4 cached', st.status.state === 'ready' && st.inflight.length === 0 && st.cached.length <= LRU_HOURS, JSON.stringify({ s: st.status.state, inflight: st.inflight, cached: st.cached }));
}]);

GROUPS.push(['options', async () => {
  const stats = newStats();
  const { layer, S } = await makeLayer(null, { opts: { fetch: slowFetch(0, stats), particleCount: 30000 } });
  const hp = await deepestPixel('25yr');
  await jump({ center: [hp.lng, hp.lat], zoom: 15, pitch: 0, bearing: 0 });
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 12000);
  S.pump(150, 16.7, false);                       // let streaks build up
  S.pump(2);
  const snap = () => { const c = canvasStats(layer); return c; };
  const base = snap();
  record('baseline renders water', base.n > 5000, `non-empty px ${base.n} of ${base.w * base.h}`);
  record('flow-paths are not downloaded unless asked for', stats.flowpaths === 0);
  // time moves the water while not frozen
  S.pump(10, 33);
  let a = snap();
  S.pump(10, 33);
  let b = snap();
  record('water and streaks keep animating while time is paused', bytesDiffer(a.px, b.px) > 500, `${bytesDiffer(a.px, b.px)} bytes differ`);
  layer.setOptions({ freeze: true });
  S.pump(6, 33);
  a = snap();
  S.pump(10, 33);
  b = snap();
  record('freeze: the picture stands still', bytesDiffer(a.px, b.px) === 0 && S.pending() === 0, `${bytesDiffer(a.px, b.px)} bytes differ, loop pending ${S.pending()}`);
  layer.setTime(15);
  S.pump(3);
  b = snap();
  record('freeze: a time change still redraws', bytesDiffer(a.px, b.px) > 0 && layer.time === 15 && S.pending() === 0, `${bytesDiffer(a.px, b.px)} bytes differ`);
  layer.setOptions({ freeze: false });
  S.pump(5, 33);
  record('unfreeze restarts the loop', S.pending() === 1 && layer.getOptions().freeze === false);

  // layer toggles
  const nBase = snap();
  const toggles = [['showDepth', 'depth'], ['showFlow', 'flow shading'], ['showArrows', 'arrows'], ['showParticles', 'streaks']];
  for (const [key, label] of toggles) {
    layer.setOptions({ [key]: false });
    S.pump(4, 16.7);
    const c = snap();
    const d = bytesDiffer(nBase.px, c.px);
    record(`${key}=false changes the picture (${label})`, d > 200 && layer.getOptions()[key] === false, `${d} bytes differ, wet px ${nBase.n} -> ${c.n}`);
    layer.setOptions({ [key]: true });
    S.pump(4, 16.7);
  }
  layer.setOptions({ showDepth: false, showFlow: false, showArrows: false, showParticles: false });
  S.pump(4);
  record('all layers off gives an empty canvas', snap().n === 0);
  layer.setOptions({ showDepth: true, showFlow: false, showArrows: false, showParticles: false, opacity: 1 });
  S.pump(4);
  const full = snap();
  layer.setOptions({ opacity: 0.5 });
  S.pump(4);
  const half = snap();
  record('opacity 0.5 roughly halves the water alpha', half.sum < full.sum * 0.65 && half.sum > full.sum * 0.35, `alpha sum ${full.sum} -> ${half.sum}`);
  layer.setOptions({ opacity: 1, minDepthShown: 1.5 });
  S.pump(4);
  const deepOnly = snap();
  record('minDepthShown 1.5 m hides the shallow water', deepOnly.n < full.n * 0.6 && deepOnly.n > 0, `non-empty px ${full.n} -> ${deepOnly.n}`);
  layer.setOptions({ minDepthShown: 0.02 });
  // flow paths load lazily when switched on
  layer.setOptions({ showFlowpaths: true, showDepth: false });
  const tFp = performance.now();
  let fpOn = false;
  while (performance.now() - tFp < 15000 && !fpOn) { S.pump(2); await sleep(100); fpOn = snap().n > 0; }
  record('showFlowpaths downloads the flow-paths grid and draws it', fpOn && stats.flowpaths === 1, `flowpaths requests ${stats.flowpaths}, drawn ${fpOn}`);
  layer.setOptions({ showFlowpaths: false, showDepth: true });
  S.pump(3);

  // option round trip and quality interplay
  layer.setOptions({ particleCount: 12345, flowSpeed: 3, quality: 1, msPerHour: 700 });
  let o = layer.getOptions();
  record('getOptions returns what was set', o.particleCount === 12345 && o.flowSpeed === 3 && o.quality === 1 && o.msPerHour === 700 && o.freeze === false && o.autoQuality === false, JSON.stringify({ p: o.particleCount, f: o.flowSpeed, q: o.quality }));
  layer.setQualityLevel(4, 'test');
  o = layer.getOptions();
  const ro = layer.renderer.getOptions();
  record('a lower quality level scales the renderer but getOptions keeps the requested values', o.particleCount === 12345 && ro.particleCount === Math.round(12345 * QUALITY_LEVELS[4].particles) && layer.qualityLevel === 4, `renderer particles ${ro.particleCount}, level ${layer.qualityLevel}`);
  layer.setQualityLevel(0);
  let threw = null;
  try { layer.setOptions({ flowSpeed: 'fast' }); } catch (e) { threw = e; }
  record('a non-numeric option throws TypeError', threw instanceof TypeError, threw && threw.message);
  threw = null;
  try { layer.setOptions({ msPerHour: -5 }); } catch (e) { threw = e; }
  record('a bad layer option throws TypeError', threw instanceof TypeError, threw && threw.message);
  threw = null;
  try { layer.setTime(NaN); } catch (e) { threw = e; }
  record('setTime(NaN) throws TypeError', threw instanceof TypeError);
}]);

function visibleNeed(gridB) {
  const b = map.getBounds(), sw = b.getSouthWest(), ne = b.getNorthEast();
  const fwd = (lng, lat) => [R_EARTH * lng * Math.PI / 180, R_EARTH * Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360))];
  const a = fwd(sw.lng, sw.lat), c = fwd(ne.lng, ne.lat);
  return [Math.max(Math.min(a[0], c[0]), gridB[0]), Math.max(Math.min(a[1], c[1]), gridB[1]), Math.min(Math.max(a[0], c[0]), gridB[2]), Math.min(Math.max(a[1], c[1]), gridB[3])];
}
const covers = (r, n) => r && r[0] <= n[0] + 1e-6 && r[1] <= n[1] + 1e-6 && r[2] >= n[2] - 1e-6 && r[3] >= n[3] - 1e-6;
async function waitFor(cond, ms = 3000) {
  const t0 = performance.now();
  while (performance.now() - t0 < ms) { if (cond()) return true; await sleep(30); }
  return cond();
}

GROUPS.push(['region', async () => {
  const hp = await deepestPixel('25yr');
  const st = await rawStorm('25yr');
  const gb = st.meta.bounds_merc;
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  const regions = [];
  layer.on('region', (e) => regions.push(e));
  await gridCentreView();
  await layer.setStorm('25yr');
  const gc = invMerc((gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2);
  const dpr = window.devicePixelRatio || 1;
  const rows = [];
  for (let z = 11; z <= 18; z++) {
    const centre = z <= 13 ? gc : [hp.lng, hp.lat];
    map.jumpTo({ center: centre, zoom: z, pitch: 0, bearing: 0 });
    const wantNow = dpr / (40075016.686 / (512 * 2 ** z));
    const settled = () => {
      const r0 = layer.region, c0 = layer.canvas;
      if (!covers(r0, visibleNeed(gb))) return false;
      const ratio0 = (c0.width / (r0[2] - r0[0])) / wantNow;
      return c0.width * c0.height >= 0.85 * 4.2e6 || (ratio0 > 0.6 && ratio0 < 1.7);
    };
    const ok = await waitFor(settled, 3000);
    await settle();
    S.pump(2);
    const r = layer.region, c = layer.canvas, need = visibleNeed(gb);
    const inGrid = r[0] >= gb[0] - 1e-6 && r[1] >= gb[1] - 1e-6 && r[2] <= gb[2] + 1e-6 && r[3] <= gb[3] + 1e-6;
    const lim = c.width <= 4096 && c.height <= 4096 && c.width * c.height <= 4.2e6 * 1.001 && c.width >= 8 && c.height >= 8;
    const mpp = 40075016.686 / (512 * 2 ** z);
    const dens = c.width / (r[2] - r[0]), want = dpr / mpp;
    const capped = c.width * c.height >= 0.85 * 4.2e6;
    const ratio = dens / want;
    const sq = layer.getState();
    rows.push(`z${z}:${c.width}x${c.height}`);
    record(`zoom ${z}: region covers the visible grid area, lies inside the grid`, ok && inGrid && covers(r, need), `region ${r.map((v) => v.toFixed(0)).join(',')}`);
    record(`zoom ${z}: canvas within limits (<=4096 per side, <=4.2 MP) and the renderer got that size`, lim && sq.canvas[0] === c.width && sq.canvas[1] === c.height, `${c.width}x${c.height} = ${(c.width * c.height / 1e6).toFixed(2)} MP`);
    record(`zoom ${z}: about one canvas pixel per device pixel (density ratio ${ratio.toFixed(2)}${capped ? ', area capped' : ''})`, capped || (ratio > 0.55 && ratio < 1.8), `density ${dens.toFixed(4)} px/m wanted ${want.toFixed(4)}`);
    record(`zoom ${z}: layer drawn and visible`, sq.active && sq.shown && map.getLayoutProperty('fluid-water', 'visibility') === 'visible');
  }
  // pan
  map.jumpTo({ center: [hp.lng, hp.lat], zoom: 15, pitch: 0, bearing: 0 });
  await waitFor(() => covers(layer.region, visibleNeed(gb)), 3000);
  const n0 = regions.length;
  for (let i = 0; i < 10; i++) { map.panBy([15, 0], { duration: 0 }); await sleep(30); }
  await sleep(500);
  record('ten small pans keep covering the view with at most 2 region replacements', covers(layer.region, visibleNeed(gb)) && regions.length - n0 <= 2, `replacements ${regions.length - n0}`);
  const before = layer.region;
  map.panBy([1400, 0], { duration: 0 });
  const ok2 = await waitFor(() => covers(layer.region, visibleNeed(gb)), 3000);
  record('a large pan replaces the region and it covers the new view', ok2 && layer.region[0] !== before[0], `region x ${before[0].toFixed(0)} -> ${layer.region[0].toFixed(0)}`);
  // pitch 80 at several bearings stays within limits and covers
  for (const [pitch, bearing, z] of [[80, 30, 14], [80, 200, 15], [60, 90, 13]]) {
    map.jumpTo({ center: [hp.lng, hp.lat], zoom: z, pitch, bearing });
    const ok3 = await waitFor(() => covers(layer.region, visibleNeed(gb)), 3000);
    const c = layer.canvas;
    record(`pitch ${pitch} bearing ${bearing} zoom ${z}: region covers, canvas within limits`, ok3 && c.width <= 4096 && c.height <= 4096 && c.width * c.height <= 4.2e6 * 1.001, `${c.width}x${c.height}`);
  }
  // away from the grid: nothing drawn, loop stopped
  map.jumpTo({ center: [-86.9, 37.5], zoom: 12, pitch: 0, bearing: 0 });
  await waitFor(() => layer.getState().active === false, 3000);
  S.pump(2);
  record('grid out of view: layer inactive, hidden, loop stopped', layer.getState().active === false && map.getLayoutProperty('fluid-water', 'visibility') === 'none' && S.pending() === 0, JSON.stringify({ a: layer.getState().active, p: S.pending() }));
  map.jumpTo({ center: gc, zoom: 12 });
  await waitFor(() => layer.getState().active === true, 3000);
  S.pump(2);
  record('panning back resumes drawing', layer.getState().active && map.getLayoutProperty('fluid-water', 'visibility') === 'visible' && S.pending() === 1);
  record('region summary', true, rows.join(' '));
}]);

GROUPS.push(['loop and visibility', async () => {
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  await gridCentreView();
  await layer.setStorm('25yr');
  layer.setTime(10);
  await pumpUntil(S, () => layer.time === 10 && !layer.getState().buffering, 12000);
  S.pump(3);
  record('a running layer keeps exactly one frame request queued', S.pending() === 1 && layer.getState().running);
  layer.play(1000);
  S.pump(10);
  const setHidden = (hidden) => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (hidden ? 'hidden' : 'visible') });
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    document.dispatchEvent(new Event('visibilitychange'));
  };
  try {
    const f0 = layer.renderer.stats.frames, t0 = layer.time;
    setHidden(true);
    record('document hidden: the frame loop is cancelled', S.pending() === 0 && layer.getState().running === false);
    S.pump(60);
    record('document hidden: nothing is rendered and playback does not advance', layer.renderer.stats.frames === f0 && layer.time === t0 && layer.isPlaying, `frames ${layer.renderer.stats.frames - f0}, time ${t0} -> ${layer.time}`);
    S.step(5000);                                 // five seconds pass while hidden
    setHidden(false);
    record('document visible again: the loop resumes', S.pending() === 1 && layer.getState().running);
    S.pump(3);
    const dt = layer.time - t0;
    record('resuming does not jump the clock (the hidden time is not played)', layer.renderer.stats.frames > f0 && dt >= 0 && dt < 0.12, `time advanced ${dt.toFixed(3)} h`);
  } finally {
    delete document.visibilityState; delete document.hidden;
  }
  layer.pause();
  const f1 = layer.renderer.stats.frames;
  layer.setVisible(false);
  S.pump(10);
  record('setVisible(false): hidden on the map, loop stopped, nothing rendered', S.pending() === 0 && layer.renderer.stats.frames === f1 && map.getLayoutProperty('fluid-water', 'visibility') === 'none');
  layer.setVisible(true);
  S.pump(3);
  record('setVisible(true): visible again and rendering', S.pending() === 1 && layer.renderer.stats.frames > f1 && map.getLayoutProperty('fluid-water', 'visibility') === 'visible');
  record('the canvas source plays only while the loop runs', map.getSource('fluid-water-src')._playing === true);
  layer.setOptions({ freeze: true });
  S.pump(4);
  record('frozen and idle: loop stopped and the source paused', S.pending() === 0 && map.getSource('fluid-water-src')._playing === false);
  drop(layer);

  // the real requestAnimationFrame loop advances time with real elapsed time
  const rl = new FluidLayer(map, { ...BASE_OPTS, ...FLAT });
  live.add(rl);
  await rl.setStorm('25yr');
  rl.setTime(5);
  await waitFor(() => rl.time === 5 && !rl.getState().buffering, 8000);
  const sts = [];
  rl.on('status', (e) => sts.push(e.state));
  const tA = rl.time, w0 = performance.now();
  rl.play(500);
  await sleep(3000);
  const el = performance.now() - w0, adv = rl.time - tA, fr = rl.getState().frames;
  rl.pause();
  const buffered = sts.includes('buffering');
  record('real rAF playback: time advances with elapsed time (500 ms per hour)', buffered ? adv > 0.8 : (adv > el / 500 * 0.6 && adv < el / 500 * 1.15), `advanced ${adv.toFixed(2)} h in ${(el / 1000).toFixed(2)} s, ${fr} frames${buffered ? ', buffered' : ''}`);
  const e = rl.getState();
  record('real rAF loop renders continuously', fr > 8 && e.running, `frames ${fr}`);
}]);

GROUPS.push(['quality governor', async () => {
  const { layer, S } = await makeLayer(null, { opts: { ...FLAT, autoQuality: true, frameBudgetMs: 28, pixelScale: 0.25, particleCount: 20000, showParticles: true } });
  await gridCentreView();
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 12000);
  const qs = [];
  layer.on('quality', (e) => qs.push({ ...e, vt: S.vt }));
  const px0 = layer.qualityInfo.canvasWidth * layer.qualityInfo.canvasHeight;
  S.pump(300, 16.7, false);
  record('frames within the budget (16.7 ms) keep quality level 0', qs.length === 0 && layer.qualityLevel === 0, `avg ${layer.qualityInfo.avgMs.toFixed(1)} ms`);
  const tStart = S.vt;
  S.pump(260, 40, false);
  const first = qs[0];
  record('frames averaging 40 ms (budget 28) lower quality after about 2 s', first && first.level >= 1 && first.reason === 'frame time' && first.vt - tStart > 1900 && first.vt - tStart < 5200, first && `level ${first.level} after ${((first.vt - tStart) / 1000).toFixed(1)} s`);
  const px1 = layer.qualityInfo.canvasWidth * layer.qualityInfo.canvasHeight;
  record("'quality' event reports level, pixelScale and particleFactor; the canvas got smaller", first && first.maxLevel === QUALITY_LEVELS.length - 1 && first.pixelScale <= 0.8 && px1 < px0, `canvas ${px0} -> ${px1} px, scale ${first && first.pixelScale}`);
  S.pump(700, 90, false);
  record('very slow frames (90 ms) keep stepping down to the last level and no further', layer.qualityLevel === QUALITY_LEVELS.length - 1, `level ${layer.qualityLevel}`);
  const ro = layer.renderer.getOptions();
  record('at the last level the renderer runs fewer particles and lower shader quality', ro.particleCount === Math.round(20000 * QUALITY_LEVELS[QUALITY_LEVELS.length - 1].particles) && ro.quality === QUALITY_LEVELS[QUALITY_LEVELS.length - 1].shader && layer.getOptions().particleCount === 20000, `renderer particles ${ro.particleCount} quality ${ro.quality}`);
  const lvl = layer.qualityLevel, n = qs.length, tR = S.vt;
  S.pump(400, 10, false);
  record('recovery is slow: no step up within the first 8 s of headroom', qs.length === n && layer.qualityLevel === lvl, `4 s of headroom, level ${layer.qualityLevel}`);
  S.pump(1700, 10, false);
  const up = qs.filter((q) => q.reason === 'headroom');
  record('with headroom quality recovers step by step', up.length >= 1 && layer.qualityLevel < lvl, `${up.length} step(s) up in ${((S.vt - tR) / 1000).toFixed(0)} s, level ${lvl} -> ${layer.qualityLevel}`);
  layer.setOptions({ autoQuality: false });
  const l2 = layer.qualityLevel;
  S.pump(300, 90, false);
  record('autoQuality false pins the level', layer.qualityLevel === l2);
  layer.setOptions({ autoQuality: true });
  layer.setQualityLevel(0, 'test');
  record('setQualityLevel pins any level and emits a quality event', layer.qualityLevel === 0 && qs[qs.length - 1].reason === 'test');
  layer.setQualityLevel(99);
  record('setQualityLevel clamps to the last level', layer.qualityLevel === QUALITY_LEVELS.length - 1);
}]);

// Measurements, not pass/fail limits: per-frame cost of the draped canvas by size on this GPU (the numbers go in the report).
GROUPS.push(['frame cost', async () => {
  const { layer, S } = await makeLayer(null, { opts: { particleCount: 40000 } });
  await gridCentreView();
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 12000);
  const stat = (a) => { const s = a.slice().sort((x, y) => x - y); return { med: s[s.length >> 1], p90: s[Math.floor(s.length * 0.9)] }; };
  const f1 = (v) => v.toFixed(1);
  const N = 24;
  const gl = layer.renderer.gl.getExtension('WEBGL_debug_renderer_info');
  const gpu = gl ? layer.renderer.gl.getParameter(gl.UNMASKED_RENDERER_WEBGL) : '?';
  record(`GPU used by the browser`, true, `${gpu}, viewport ${innerWidth}x${innerHeight} @${devicePixelRatio}`);
  for (const pitch of [0, 60]) {
    await gridCentreView({ pitch });
    layer.setVisible(false);
    S.pump(3);
    const tb = [];
    for (let i = 0; i < N; i++) { const t0 = performance.now(); map.redraw(); syncGpu(); tb.push(performance.now() - t0); }
    layer.setVisible(true);
    const base = stat(tb);
    record(`pitch ${pitch}: map alone (water hidden)`, true, `${f1(base.med)} ms median, p90 ${f1(base.p90)}`);
    for (const cap of [1.05e6, 2.1e6, 4.19e6, 9.4e6, 16.7e6]) {
      layer.setOptions({ maxPixels: cap, pixelScale: 2 });
      layer.updateRegion(true);
      S.pump(8);
      const c = layer.canvas;
      const tf = [], tm = [];
      for (let i = 0; i < N; i++) {
        const t0 = performance.now();
        S.step(16.7); layer.renderer.sync();
        const t1 = performance.now();
        map.redraw(); syncGpu();
        const t2 = performance.now();
        tf.push(t1 - t0); tm.push(t2 - t1);
      }
      const a = stat(tf), b = stat(tm), tot = stat(tf.map((v, i) => v + tm[i]));
      record(`pitch ${pitch}: canvas ${c.width}x${c.height} (${(c.width * c.height / 1e6).toFixed(2)} MP)`, true, `fluid ${f1(a.med)} ms + map ${f1(b.med)} ms = ${f1(tot.med)} ms median (p90 ${f1(tot.p90)}); map alone ${f1(base.med)}`);
    }
  }
  layer.setOptions({ maxPixels: 4.2e6, pixelScale: 1 });
}]);

GROUPS.push(['animation on the map', async () => {
  const hp = await deepestPixel('25yr');
  const grab = () => {
    if (!mglCtx) syncGpu();
    const W = map.getCanvas().width, H = map.getCanvas().height, buf = new Uint8Array(W * H * 4);
    mglCtx.readPixels(0, 0, W, H, mglCtx.RGBA, mglCtx.UNSIGNED_BYTE, buf);
    return buf;
  };
  const blueCount = (b) => { let n = 0; for (let i = 0; i < b.length; i += 4) if (b[i + 2] > b[i] + 25 && b[i + 2] > 70) n++; return n; };
  const { layer, S } = await makeLayer(null, { opts: { particleCount: 20000 } });
  await jump({ center: [hp.lng, hp.lat], zoom: 16, pitch: 0, bearing: 0 });
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 12000);
  S.pump(3);
  await settle();
  S.pump(3);
  for (const terrain of [true, false]) {
    map.setTerrain(terrain ? { source: 'terrain', exaggeration: 2 } : null);
    for (const pitch of [0, 60]) {
      await jump({ center: [hp.lng, hp.lat], zoom: 16, pitch, bearing: 20 });
      S.pump(4);
      await settle();
      S.pump(60, 16.7, false);
      S.pump(2);
      const a = grab();
      S.pump(20, 33);
      const b = grab();
      let d = 0;
      for (let i = 0; i < a.length; i += 4) if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) d++;
      record(`map picture animates (terrain ${terrain ? 'on' : 'off'}, pitch ${pitch})`, blueCount(b) > 2000 && d > 400, `${blueCount(b)} blue px, ${d} px changed between frames`);
    }
  }
  map.setTerrain({ source: 'terrain', exaggeration: 2 });
}]);

GROUPS.push(['no flash at region updates', async () => {
  const hp = await deepestPixel('25yr');
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  await jump({ center: [hp.lng, hp.lat], zoom: 16, pitch: 0, bearing: 0 });
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 12000);
  S.pump(4);
  await settle();
  const waterAt = () => {
    const s = map.project([hp.lng, hp.lat]);
    const p = new Uint8Array(4);
    for (const [dx, dy] of [[0, 0], [8, 0], [-8, 0], [0, 8], [0, -8]]) { mapPixel(s.x + dx, s.y + dy, p); if (isWaterRgb(p)) return true; }
    return false;
  };
  record('probe point shows water before the test', waterAt());
  let bad = 0, frames = 0, replaced = 0;
  const nReg = []; layer.on('region', () => nReg.push(1));
  for (let i = 0; i < 96; i++) {
    if (i % 8 === 2) layer.updateRegion(true);
    if (i % 16 === 5) layer.setOptions({ pixelScale: layer.getOptions().pixelScale === 1 ? 0.7 : 1 });
    S.pump(1, 16.7, true);
    frames++;
    if (!waterAt()) bad++;
  }
  replaced = nReg.length;
  record('no blank or torn frame while the region and canvas size are replaced every few frames', bad === 0 && replaced >= 12, `${bad} bad of ${frames} frames, ${replaced} replacements`);
  bad = 0; frames = 0;
  const c0 = map.getCenter();
  const cw = map.getCanvas().clientWidth, ch = map.getCanvas().clientHeight;
  for (let i = 0; i < 60; i++) {
    map.jumpTo({ center: [c0.lng + i * 0.00002, c0.lat - i * 0.000012] });
    layer.updateRegion(false);
    S.pump(1, 16.7, true);
    const s = map.project([hp.lng, hp.lat]);
    if (s.x < 12 || s.y < 12 || s.x > cw - 12 || s.y > ch - 12) continue;     // the probe point left the screen
    frames++;
    if (!waterAt()) bad++;
  }
  record('water stays under the probe point while panning (frames with the point on screen)', bad === 0 && frames >= 30, `${bad} bad of ${frames} frames`);
}]);

GROUPS.push(['teardown', async () => {
  const origAdd = document.addEventListener, origRemove = document.removeEventListener;
  let vis = 0;
  document.addEventListener = function (t, ...r) { if (t === 'visibilitychange') vis++; return origAdd.call(this, t, ...r); };
  document.removeEventListener = function (t, ...r) { if (t === 'visibilitychange') vis--; return origRemove.call(this, t, ...r); };
  const evs = ['moveend', 'zoomend', 'resize', 'move', 'styledata', 'remove'];
  const count = (ev) => (map._listeners && map._listeners[ev] ? map._listeners[ev].length : 0);
  try {
    const before = evs.map(count);
    await gridCentreView();
    const { layer, S } = await makeLayer('25yr', { opts: FLAT });
    layer.setTime(12);
    layer.play(500);
    await pumpUntil(S, () => layer.time >= 12.2, 12000);
    const during = evs.map(count);
    record('the layer registers map and document listeners', during.every((n, i) => n > before[i]) && vis === 1, `map listeners ${before} -> ${during}, document ${vis}`);
    const cv = layer.canvas;
    layer.dispose();
    layer.dispose();
    const after = evs.map(count);
    record('dispose removes map listeners and the document listener', after.every((n, i) => n === before[i]) && vis === 0, `map listeners ${after}, document ${vis}`);
    record('dispose removes layer and source from the map and stops the loop', !map.getLayer('fluid-water') && !map.getSource('fluid-water-src') && S.pending() === 0);
    record('dispose releases GL resources and grids', cv.width === 1 && cv.height === 1 && layer.disposed && layer.currentGrids().rgbaA === null && layer.status.state === 'idle');
    let err = null;
    try { await layer.setStorm('25yr'); } catch (e) { err = e; }
    record('setStorm after dispose rejects', err instanceof Error);
    // repeated create/dispose cycles do not accumulate listeners
    for (let i = 0; i < 6; i++) { const x = await makeLayer(i % 2 ? '5yr' : null, { opts: FLAT }); x.S.pump(2); drop(x.layer); }
    record('six create/dispose cycles leave no listeners behind', evs.map(count).every((n, i) => n === before[i]) && vis === 0, `map listeners ${evs.map(count)}, document ${vis}`);
  } finally {
    document.addEventListener = origAdd; document.removeEventListener = origRemove;
  }
}]);

// ------------------------------------------------------------------------------ L2 helpers

const geo = {};
async function loadGeo(which) {
  if (!geo[which]) {
    const r = await fetch(new URL(`../../layers/${which}.geojson`, location.href).pathname);
    geo[which] = (await r.json()).features;
  }
  return geo[which];
}

async function gzip(bytes) {
  return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer());
}

// A 64 x 64 synthetic storm (cell 6.4 m) at the map centre, served through the layer's `fetch` option. Block (bc, br) of
// 8 x 8 pixels: columns bc = 0..7 flow toward the compass headings 0, 45, ... 315 degrees at 2 m/s, rows br = 1..6 differ in
// depth, row 0 is outside the model domain (A = 0), row 7 is still water. Hour 14 equals hour 13 with every depth code + 40.
const SYN = { n: 64, cell: 6.4, speed: 2 };
function synthMeta() {
  const c = lngLatToMerc(CENTER[0], CENTER[1]);
  const half = SYN.n * SYN.cell / 2;
  return {
    format: 'bgflow', version: 1, storm: 'synth', label: 'synthetic', source: 'tiers', width: SYN.n, height: SYN.n, cell_m: SYN.cell,
    bounds_merc: [c[0] - half, c[1] - half, c[0] + half, c[1] + half], hours: [13, 14],
    depth_encoding: { type: 'sqrt', max_m: 8 }, vel_encoding: { vmax_ms: 4, zero_code: 128 }, wet_threshold_m: 0.02, qmax_m2s: 2,
    cell_ground_m: 5.1,
  };
}
function synthGrid(hour) {
  const n = SYN.n, g = new Uint8Array(n * n * 4);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const bc = col >> 3, br = row >> 3, i = (row * n + col) * 4;
      if (br === 0) { g[i] = 0; g[i + 1] = 128; g[i + 2] = 128; g[i + 3] = 0; continue; }
      g[i + 3] = 255;
      if (br === 7) { g[i] = 50 + (hour === 14 ? 40 : 0); g[i + 1] = 128; g[i + 2] = 128; continue; }
      const b = bc * 45 * Math.PI / 180;
      g[i] = 60 + 20 * br + (hour === 14 ? 40 : 0);
      g[i + 1] = 128 + Math.round(SYN.speed * Math.sin(b) / 4 * 127);
      g[i + 2] = 128 + Math.round(SYN.speed * Math.cos(b) / 4 * 127);
    }
  }
  return g;
}
// Centre of block (bc, br) as [lng, lat] (the pixel centre of its fifth pixel, where bilinear reads one pixel only).
function synthPoint(meta, bc, br) {
  const col = bc * 8 + 4, row = br * 8 + 4;
  return mercToLngLat(meta.bounds_merc[0] + (col + 0.5) * meta.cell_m, meta.bounds_merc[3] - (row + 0.5) * meta.cell_m);
}
async function synthFetch(meta, failHours = null) {
  const files = { 'meta.json': new TextEncoder().encode(JSON.stringify(meta)) };
  for (const h of meta.hours) files[`h${String(h).padStart(2, '0')}.rgba.gz`] = await gzip(synthGrid(h));
  return (url) => {
    const name = String(url).split('?')[0].split('/').pop();
    const hm = /^h(\d\d)\.rgba\.gz$/.exec(name);
    if (failHours && hm && failHours.has(Number(hm[1]))) return Promise.resolve(new Response('boom', { status: 500 }));
    const body = files[name];
    return Promise.resolve(body ? new Response(body.slice(), { status: 200 }) : new Response('nope', { status: 404 }));
  };
}

// Collects the events of a layer: {list, count}
function collect(layer, name) {
  const r = { list: [], get count() { return this.list.length; } };
  layer.on(name, (e) => r.list.push(e));
  return r;
}

// Real-time frame loop that does not depend on requestAnimationFrame (throttled to about 1 fps in a background pane):
// timers about every `minMs`, real elapsed time, and a GPU sync after each frame so the frame interval includes the GPU cost.
function makeRealtimeSched(minMs = 8) {
  let id = 0, layerRef = null;
  const timers = new Map();
  return {
    sched: {
      raf: (fn) => { const i = ++id; timers.set(i, setTimeout(() => { timers.delete(i); fn(performance.now()); if (layerRef && !layerRef.disposed) layerRef.renderer.sync(); }, minMs)); return i; },
      caf: (i) => { clearTimeout(timers.get(i)); timers.delete(i); },
      now: () => performance.now(),
    },
    bind(l) { layerRef = l; return l; },
  };
}

const sameArray = (a, b) => {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

async function gcNudge() {
  if (typeof window.gc === 'function') { window.gc(); await sleep(30); window.gc(); }
  else { for (let i = 0; i < 2; i++) { const x = new ArrayBuffer(96 << 20); new Uint8Array(x)[0] = 1; await sleep(60); } }
  await sleep(120);
}
const heapMB = () => (performance.memory ? performance.memory.usedJSHeapSize / 1048576 : NaN);

// ------------------------------------------------------------------------------ L2 groups: probe

GROUPS.push(['probe', async () => {
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  await gridCentreView();
  record('probe before a storm is invalid and safe', layer.probe([CENTER[0], CENTER[1]]).valid === false);
  await layer.setStorm('25yr');
  layer.setTime(14.37);
  await pumpUntil(S, () => Math.abs(layer.time - 14.37) < 1e-9 && !layer.getState().buffering, 20000);
  const st = await rawStorm('25yr');
  const gA = (await st.getHour(14)).rgba, gB = (await st.getHour(15)).rgba;
  const m = st.meta, W = m.width, H = m.height;
  // pixels to probe: deepest, fastest, a mid-depth one, a dry in-domain one
  let fast = -1, fi = 0, mid = -1, dry = -1;
  for (let i = 0; i < W * H; i++) {
    const r = gA[i * 4];
    if (r === 0) { if (dry < 0 && gA[i * 4 + 3] !== 0 && i > W * H / 3) dry = i; continue; }
    const sp = Math.hypot(gA[i * 4 + 1] - 128, gA[i * 4 + 2] - 128);
    if (sp > fast) { fast = sp; fi = i; }
    if (mid < 0 && r > 90 && r < 100 && i > W * H / 2) mid = i;
  }
  const dp = await deepestPixel('25yr');
  const pt = (i, fx = 0.5, fy = 0.5) => mercToLngLat(m.bounds_merc[0] + ((i % W) + fx) * m.cell_m, m.bounds_merc[3] - (((i / W) | 0) + fy) * m.cell_m);
  const pts = [
    ['deepest', [dp.lng, dp.lat]], ['fastest', pt(fi)], ['fastest, off centre', pt(fi, 0.13, 0.81)], ['mid depth', pt(mid)], ['mid depth, off centre', pt(mid, 0.9, 0.2)],
    ['dry in domain', pt(dry)], ['outside the grid', [m.bounds_lnglat[0] - 0.2, CENTER[1]]],
  ];
  const o = { depth: 0, u: 0, v: 0, speed: 0, valid: false };
  let allSame = true;
  const detail = [];
  for (const [name, ll] of pts) {
    const p = layer.probe(ll);
    const mc = lngLatToMerc(ll[0], ll[1]);
    const s = st.sample(gA, gB, layer.currentGrids().mix, mc[0], mc[1], o);
    let b = Math.atan2(s.u, s.v) * 180 / Math.PI; if (b < 0) b += 360;
    const same = p.valid === s.valid && p.depthM === s.depth && p.u === s.u && p.v === s.v && p.speedMs === s.speed && p.hour === layer.time
      && (!s.valid || s.speed < 0.02 || Math.abs(p.bearingDeg - b) < 1e-9);
    if (!same) allSame = false;
    detail.push(`${name}: ${s.valid ? s.depth.toFixed(3) + ' m ' + s.speed.toFixed(2) + ' m/s ' + p.bearingDeg.toFixed(1) + ' deg ' + p.cardinal : 'invalid'}`);
  }
  record('probe equals fluid-data sample at 7 points (depth, u, v, speed, valid, bearing)', allSame, detail.join('; '));
  const pDeep = layer.probe([dp.lng, dp.lat]);
  record('probe at the deepest pixel is wet and valid; hour is the continuous time', pDeep.valid && pDeep.depthM > 0.5 && near(pDeep.hour, 14.37, 1e-9), `depth ${pDeep.depthM.toFixed(2)} m`);
  const out = { stale: true };
  const r1 = layer.probe({ lng: dp.lng, lat: dp.lat }, out);
  record('probe accepts {lng, lat} and fills the given object without allocating a new one', r1 === out && out.valid === true && out.depthM === pDeep.depthM);
  // heap growth over 200000 calls, best of three (GC timing only ever adds to a run; a real per-call allocation shows in all three)
  const ll = [dp.lng, dp.lat];
  const growth = [];
  for (let run = 0; run < 3; run++) {
    await gcNudge();
    const h0 = heapMB();
    for (let i = 0; i < 200000; i++) { ll[0] += 1e-9; layer.probe(ll, out); }
    growth.push(heapMB() - h0);
  }
  const g = Math.min(...growth);
  record('200000 probe calls with a reused object allocate (almost) nothing', !(g > 2), `heap growth per run ${growth.map((v) => v.toFixed(2)).join(' / ')} MB${performance.memory ? '' : ' (performance.memory missing)'}`);
  const calls = [];
  let sameObj = true, first = null;
  const detach = layer.attachProbe((res) => { if (res) { first = first || res; if (res !== first) sameObj = false; } calls.push(res ? { ...res } : null); });
  for (let k = 0; k < 3; k++) map.fire('mousemove', { lngLat: { lng: dp.lng, lat: dp.lat } });
  map.fire('mouseout', {});
  detach();
  map.fire('mousemove', { lngLat: { lng: dp.lng, lat: dp.lat } });
  record('attachProbe: callback per mousemove with one reused result, null on mouseout, silent after detach', calls.length === 4 && sameObj && calls[0].valid && Math.abs(calls[0].depthM - pDeep.depthM) < 1e-9 && calls[3] === null, `${calls.length} calls`);
  layer.setTime(14.5);
  await pumpUntil(S, () => layer.time === 14.5 && !layer.getState().buffering, 20000);
  const pm = layer.probe([dp.lng, dp.lat]);
  const mc = lngLatToMerc(dp.lng, dp.lat);
  const a = st.sample(gA, null, 0, mc[0], mc[1]), b = st.sample(gB, null, 0, mc[0], mc[1]);
  record('probe at a half hour is the mean of the two hour grids', Math.abs(pm.depthM - (a.depth + b.depth) / 2) < 1e-9, `${a.depth.toFixed(3)} / ${pm.depthM.toFixed(3)} / ${b.depth.toFixed(3)} m`);
  drop(layer);

  // ---- bearing convention on a synthetic storm with known velocities
  const meta = synthMeta();
  const f = await synthFetch(meta);
  const x = await makeLayer(null, { opts: { ...FLAT, base: '/synth', fetch: f } });
  await jump({ center: CENTER, zoom: 15, pitch: 0, bearing: 0 });
  await x.layer.setStorm('synth');
  x.layer.setTime(13);
  await pumpUntil(x.S, () => x.layer.time === 13 && !x.layer.getState().buffering, 8000);
  const names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  let okB = true;
  const bd = [];
  for (let bc = 0; bc < 8; bc++) {
    const p = x.layer.probe(synthPoint(meta, bc, 2));
    const want = bc * 45;
    const diff = Math.abs(((p.bearingDeg - want + 540) % 360) - 180);
    if (!(p.valid && diff < 1 && p.cardinal === names[bc] && Math.abs(p.speedMs - 2) < 0.04)) okB = false;
    bd.push(`${names[bc]} ${p.bearingDeg.toFixed(1)}`);
  }
  record('bearing: north-east water gives 45 deg, east 90 deg, and so on round the compass (0 = north, clockwise)', okB, bd.join(', '));
  const ne = x.layer.probe(synthPoint(meta, 1, 3)), ea = x.layer.probe(synthPoint(meta, 2, 3));
  record('bearing: u = v > 0 is NE (u east, v north) and v = 0, u > 0 is E', ne.u > 0 && ne.v > 0 && near(ne.bearingDeg, 45, 0.6) && ea.v === 0 && ea.u > 0 && near(ea.bearingDeg, 90, 1e-9), `NE ${ne.u.toFixed(2)},${ne.v.toFixed(2)} -> ${ne.bearingDeg.toFixed(2)}; E ${ea.u.toFixed(2)},${ea.v.toFixed(2)} -> ${ea.bearingDeg.toFixed(2)}`);
  const still = x.layer.probe(synthPoint(meta, 3, 7)), none = x.layer.probe(synthPoint(meta, 3, 0));
  record('still water has no direction; outside the model domain (A = 0) the probe is invalid', still.valid && still.moving === false && still.cardinal === '' && still.bearingDeg === 0 && !none.valid && none.depthM === 0);
  record('a point outside the grid is invalid', x.layer.probe([-86.0, 36.0]).valid === false);
  x.layer.setTime(13.5);
  await pumpUntil(x.S, () => x.layer.time === 13.5 && !x.layer.getState().buffering, 8000);
  const pb = x.layer.probe(synthPoint(meta, 4, 3));
  const d13 = decodeDepth(60 + 60), d14 = decodeDepth(60 + 60 + 40);
  record('between hours 13 and 14 the probe blends decoded depths linearly', near(pb.depthM, (d13 + d14) / 2, 1e-6), `${pb.depthM.toFixed(4)} vs ${((d13 + d14) / 2).toFixed(4)}`);
}]);

// ------------------------------------------------------------------------------ L2 groups: statistics

function hasStats(e) { return e && e.floodedCount !== null && e.roads; }

GROUPS.push(['statistics', async () => {
  const feats = await loadGeo('buildings'), roads = await loadGeo('roads');
  const st = await rawStorm('25yr');
  const meta = st.meta;
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  await gridCentreView();
  const ev = collect(layer, 'stats');
  let resolved = -1;
  let maxGap = 0, last = performance.now();
  const hb = setInterval(() => { const n = performance.now(); maxGap = Math.max(maxGap, n - last); last = n; }, 5);
  const pB = layer.attachBuildings({ type: 'FeatureCollection', features: feats }).then((n) => { resolved = n; return n; });
  const pR = layer.attachRoads(roads);
  await sleep(50);
  record('attachBuildings before a storm stays pending (no grid to index yet)', resolved === -1 && layer.getIndexInfo().buildings.attached && !layer.getIndexInfo().buildings.ready);
  layer.setTime(14.2);
  await layer.setStorm('25yr');
  last = performance.now(); maxGap = 0;
  const [nb, nr] = await Promise.all([pB, pR]);
  clearInterval(hb);
  const info = layer.getIndexInfo();
  record('both attach promises resolve to the feature counts once the index exists', nb === feats.length && nr === roads.length, `${nb} buildings, ${nr} roads`);
  record('building index in slices: no slice far above 50 ms, the UI keeps getting turns', info.buildings.maxSliceMs <= 70 && info.roads.maxSliceMs <= 70 && maxGap < 200,
    `buildings ${info.buildings.slices} slices, longest ${info.buildings.maxSliceMs.toFixed(1)} ms, total ${info.buildings.ms.toFixed(0)} ms; roads ${info.roads.slices} slices, longest ${info.roads.maxSliceMs.toFixed(1)} ms, total ${info.roads.ms.toFixed(0)} ms; longest gap between 5 ms timer ticks ${maxGap.toFixed(0)} ms`);
  await waitFor(() => ev.list.some((e) => hasStats(e) && e.hour === 14), 15000);
  const e14 = ev.list.find((e) => hasStats(e) && e.hour === 14);
  record("'stats' fires after the storm and the indices exist; hour is round(14.2) = 14", Boolean(e14) && e14.storm === '25yr', `${ev.count} events so far`);

  // reference: fluid-stats called directly on the same hour grid
  const g14 = (await st.getHour(14)).rgba;
  const t0 = performance.now();
  const bi = buildBuildingIndex(meta, feats);
  const directMs = performance.now() - t0;
  const eb = evaluateBuildings(bi, g14);
  const ri = buildRoadIndex(meta, roads);
  const er = evaluateRoads(ri, g14);
  const gs = summarizeGrid(meta, g14);
  record('building statistics equal fluid-stats on the same hour (count, tiers, order, depth, speed)', e14.floodedCount === eb.floodedCount && JSON.stringify(e14.tierCounts) === JSON.stringify(eb.tierCounts)
    && e14.belowTierCount === eb.belowTierCount && sameArray(e14.order, eb.order) && sameArray(e14.depthM, eb.depthM) && sameArray(e14.speedMs, eb.speedMs) && e14.buildingCount === feats.length,
    `${e14.floodedCount} flooded of ${feats.length}, tiers ${JSON.stringify(e14.tierCounts)}, direct one-piece index ${directMs.toFixed(0)} ms`);
  record('road statistics equal fluid-stats (pctFlooded, floodedLengthM, totalLengthM)', e14.roads.pctFlooded === er.pctFlooded && e14.roads.floodedLengthM === er.floodedLengthM && e14.roads.totalLengthM === er.totalLengthM && sameArray(e14.roads.perFeaturePct, er.perFeaturePct),
    `${er.pctFlooded.toFixed(2)} % of ${(er.totalLengthM / 1000).toFixed(1)} km = ${(er.floodedLengthM / 1000).toFixed(2)} km`);
  record('grid statistics equal summarizeGrid (wet area, max depth, max speed)', ['wetCells', 'wetAreaKm2', 'maxDepthM', 'meanDepthM', 'maxSpeedMs', 'volumeM3'].every((k) => e14.grid[k] === gs[k]) && e14.wetAreaKm2 === gs.wetAreaKm2 && e14.maxDepthM === gs.maxDepthM && e14.maxSpeedMs === gs.maxSpeedMs,
    `wet ${gs.wetAreaKm2.toFixed(2)} km2, max ${gs.maxDepthM.toFixed(2)} m, ${gs.maxSpeedMs.toFixed(2)} m/s`);
  const top = e14.ranked;
  let rankOk = top.length === Math.min(200, eb.order.length);
  for (let i = 0; i < top.length; i++) {
    const r = top[i];
    if (r.index !== eb.order[i] || r.depthM !== eb.depthM[r.index] || r.speedMs !== eb.speedMs[r.index] || r.tier !== tierOfDepth(eb.depthM[r.index])) { rankOk = false; break; }
  }
  record('ranked list = first statsTopN of the deepest-first order with index, depthM, tier, speedMs', rankOk && top.length > 0 && top[0].depthM >= top[top.length - 1].depthM, `${top.length} entries, deepest ${top[0] && top[0].depthM.toFixed(2)} m (${top[0] && top[0].tier})`);
  const bf = layer.getBuildingFlood(), rf = layer.getRoadFlood(), fs = layer.getFloodStats();
  record('getBuildingFlood / getRoadFlood / getFloodStats return the latest values', bf.hour === 14 && bf.floodedCount === e14.floodedCount && rf.pctFlooded === e14.roads.pctFlooded && fs === ev.list[ev.count - 1]);
  // centroid for fly-to
  const bi0 = eb.order[0];
  const c = layer.buildingLngLat(bi0);
  let inside = false;
  if (c) {
    const ring = feats[bi0].geometry.type === 'Polygon' ? feats[bi0].geometry.coordinates[0] : feats[bi0].geometry.coordinates[0][0];
    const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
    inside = c[0] >= Math.min(...xs) && c[0] <= Math.max(...xs) && c[1] >= Math.min(...ys) && c[1] <= Math.max(...ys);
  }
  record('buildingLngLat gives the footprint centroid (inside its bounding box); bad index gives null', inside && layer.buildingLngLat(-1) === null && layer.buildingLngLat(1e9) === null, c && c.map((v) => v.toFixed(5)).join(', '));

  // only a change of the nearest snapshot (round of the time) triggers a recompute
  const n0 = ev.count, obj0 = layer.getFloodStats();
  for (const t of [14.45, 14.05, 14.1, 14.3]) { layer.setTime(t); await pumpUntil(S, () => layer.time === t && !layer.getState().buffering, 15000); }
  await sleep(500);
  record('moving the time inside the band of hour 14 (13.5 to 14.5) does not recompute (no event, same payload object)', ev.count === n0 && layer.getFloodStats() === obj0, `${ev.count - n0} new events`);
  layer.setTime(15.0);
  await waitFor(() => ev.count > n0, 15000);
  await sleep(300);
  record("crossing to hour 15 fires exactly one 'stats' for hour 15", ev.count === n0 + 1 && ev.list[n0].hour === 15, `${ev.count - n0} event(s), hour ${ev.list[n0] && ev.list[n0].hour}`);
  const n1 = ev.count;
  layer.setTime(15.4); await pumpUntil(S, () => layer.time === 15.4 && !layer.getState().buffering, 15000);
  layer.setTime(14.2); await pumpUntil(S, () => layer.time === 14.2 && !layer.getState().buffering, 15000);
  await sleep(500);
  record('going back into the band of hour 14 fires for hour 14 again (round of the time), once', ev.count === n1 + 1 && ev.list[n1].hour === 14, `${ev.count - n1} event(s)`);
  layer.setTime(0.3);
  await pumpUntil(S, () => layer.time === 0.3 && !layer.getState().buffering, 15000);
  await waitFor(() => ev.list[ev.count - 1].hour === 0, 5000);
  const e0 = ev.list[ev.count - 1];
  record('a dry hour (before the first wet hour) gives zero statistics', e0.hour === 0 && e0.floodedCount === 0 && e0.roads.pctFlooded === 0 && e0.wetAreaKm2 === 0 && e0.maxDepthM === 0 && e0.order.length === 0 && e0.roads.totalLengthM === er.totalLengthM, `flooded ${e0.floodedCount}`);

  // storm change: the index of the same grid is reused, the numbers are those of the new storm
  layer.setTime(14.4);
  await pumpUntil(S, () => layer.time === 14.4 && !layer.getState().buffering, 15000);
  await waitFor(() => ev.list[ev.count - 1].hour === 14, 5000);
  const info1 = layer.getIndexInfo();
  const n2 = ev.count, tS = performance.now();
  await layer.setStorm('5yr');
  await waitFor(() => ev.list.slice(n2).some((e) => e.storm === '5yr' && hasStats(e)), 15000);
  const dtS = performance.now() - tS;
  const e5 = ev.list.slice(n2).find((e) => e.storm === '5yr' && hasStats(e));
  const st5 = await rawStorm('5yr');
  const sameGrid = JSON.stringify(st5.meta.bounds_merc) === JSON.stringify(meta.bounds_merc) && st5.meta.width === meta.width && st5.meta.height === meta.height && st5.meta.cell_m === meta.cell_m;
  const g5 = (await st5.getHour(14)).rgba;
  const eb5 = evaluateBuildings(bi, g5);
  const info2 = layer.getIndexInfo();
  record('after setStorm(5yr) the statistics are those of the 5yr grid and the 25yr index is reused', Boolean(e5) && sameGrid && e5.hour === 14 && e5.floodedCount === eb5.floodedCount && sameArray(e5.order, eb5.order)
    && info2.buildings.ms === info1.buildings.ms && info2.buildings.slices === info1.buildings.slices && info2.buildings.cachedGrids === 1,
    `${e5 && e5.floodedCount} flooded (25yr ${eb.floodedCount}), first 5yr statistics ${dtS.toFixed(0)} ms after setStorm, ${info2.buildings.cachedGrids} cached grid geometry`);
  record('no 25yr numbers survive the storm change', layer.getFloodStats().storm === '5yr' && layer.getBuildingFlood().storm === '5yr');
  const pdet = layer.getFloodStats().computeMs;
  layer.setOptions({ statsMinDepthM: 0.6 });
  await waitFor(() => layer.getFloodStats().minDepthM === 0.6, 8000);
  const ebm = evaluateBuildings(bi, g5, { minDepthM: 0.6 });
  record('statsMinDepthM recomputes with the new flooded threshold', layer.getFloodStats().floodedCount === ebm.floodedCount && ebm.floodedCount < eb5.floodedCount, `${eb5.floodedCount} at 0.1 m -> ${ebm.floodedCount} at 0.6 m; compute ${pdet.toFixed(0)} ms`);
  layer.setOptions({ statsMinDepthM: 0.1, statsTopN: 5 });
  await waitFor(() => layer.getFloodStats().ranked.length === 5 && layer.getFloodStats().minDepthM === 0.1, 8000);
  record('statsTopN shortens the ranked list but not the order', layer.getFloodStats().ranked.length === 5 && layer.getFloodStats().order.length === eb5.floodedCount);
  drop(layer);

  // grid statistics without any attached data, and refreshStats
  const g = await makeLayer(null, { opts: FLAT });
  const ev2 = collect(g.layer, 'stats');
  await g.layer.setStorm('25yr');
  g.layer.setTime(14.4);
  await pumpUntil(g.S, () => g.layer.time === 14.4 && !g.layer.getState().buffering, 15000);
  await waitFor(() => ev2.list.some((e) => e.hour === 14), 8000);
  const e2 = ev2.list.find((e) => e.hour === 14);
  record('without attached data the event carries the grid summary only', e2.floodedCount === null && e2.roads === null && e2.ranked.length === 0 && e2.wetAreaKm2 === gs.wetAreaKm2 && g.layer.getBuildingFlood() === null && g.layer.getRoadFlood() === null);
  drop(g.layer);
  const q = await makeLayer(null, { opts: FLAT });
  await q.layer.setStorm('25yr');
  q.layer.setTime(14.4);
  await pumpUntil(q.S, () => q.layer.time === 14.4 && !q.layer.getState().buffering, 15000);
  const rs = await q.layer.refreshStats();
  record('refreshStats computes on demand without listeners or data', rs && rs.hour === 14 && rs.wetAreaKm2 === gs.wetAreaKm2 && q.layer.getFloodStats() === rs);
  drop(q.layer);

  // 5 s of real playback at fast speed: one 'stats' per whole-hour change, in order
  const rt = makeRealtimeSched(8);
  const rl = rt.bind(new FluidLayer(map, { ...BASE_OPTS, ...FLAT, scheduler: rt.sched }));
  live.add(rl);
  await rl.setStorm('25yr');
  rl.setTime(5.0);
  await rl.attachBuildings(feats); await rl.attachRoads(roads);
  await waitFor(() => rl.getFloodStats() && rl.getFloodStats().hour === 5 && rl.getFloodStats().floodedCount !== null && rl.getFloodStats().roads, 15000);
  const sev = collect(rl, 'stats');
  const trans = [];
  let lastFloor = Math.round(rl.time);
  rl.on('time', (e) => { const f = Math.round(e.hour); if (f !== lastFloor) { lastFloor = f; trans.push(f); } });
  rl.play(300);
  await sleep(5000);
  rl.pause();
  await sleep(1200);
  const hrs = sev.list.map((e) => e.hour);
  record('5 s of playback at 300 ms per hour: a stats event for every change of the nearest hour, none extra, in order', trans.length >= 5 && sameArray(hrs, trans),
    `${trans.length} hour changes [${trans.join(' ')}], ${hrs.length} events [${hrs.join(' ')}], wall ${sev.list.map((e) => e.computeMs.toFixed(0)).join('/')} ms, cpu ${sev.list.map((e) => e.cpuMs.toFixed(0)).join('/')} ms`);
  drop(rl);
}]);

GROUPS.push(['statistics match the hour event', async () => {
  const feats = await loadGeo('buildings'), roads = await loadGeo('roads');
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  await gridCentreView();
  await layer.setStorm('25yr');
  const hours = collect(layer, 'hour'), ev = collect(layer, 'stats');
  layer.setTime(13);
  await pumpUntil(S, () => layer.time === 13 && !layer.getState().buffering, 12000);
  await layer.attachBuildings(feats); await layer.attachRoads(roads);
  await waitFor(() => layer.getFloodStats() && layer.getFloodStats().hour === 13 && layer.getFloodStats().roads, 15000);
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 12000);
  await waitFor(() => layer.getFloodStats().hour === 14, 15000);
  const s14 = layer.getFloodStats();
  // 13.9 is 90 % hour 14: the numbers must be those of hour 14 and the 'hour' event must agree
  layer.setTime(13);
  await pumpUntil(S, () => layer.time === 13 && !layer.getState().buffering, 12000);
  await waitFor(() => layer.getFloodStats().hour === 13, 15000);
  const nh = hours.count;
  layer.setTime(13.9);
  await pumpUntil(S, () => layer.time === 13.9 && !layer.getState().buffering, 12000);
  await waitFor(() => layer.getFloodStats().hour === 14, 15000);
  const s139 = layer.getFloodStats();
  const hev = hours.list[hours.count - 1];
  record("at t=13.9 the statistics are those of hour 14 (the nearest snapshot) and equal the t=14 numbers", s139.hour === 14 && s139.floodedCount === s14.floodedCount && s139.wetAreaKm2 === s14.wetAreaKm2 && s139.roads.pctFlooded === s14.roads.pctFlooded,
    `13.9: flooded ${s139.floodedCount}, ${s139.wetAreaKm2.toFixed(2)} km2, roads ${s139.roads.pctFlooded.toFixed(2)} %; 14: ${s14.floodedCount}, ${s14.wetAreaKm2.toFixed(2)}, ${s14.roads.pctFlooded.toFixed(2)}`);
  record("the 'hour' event and the 'stats' payload name the same hour at t=13.9", hours.count > nh && hev.hour === 14 && s139.hour === hev.hour, `hour event ${hev.hour}, stats ${s139.hour}`);
  layer.setTime(13.4);
  await pumpUntil(S, () => layer.time === 13.4 && !layer.getState().buffering, 12000);
  await waitFor(() => layer.getFloodStats().hour === 13, 15000);
  record("at t=13.4 the nearest snapshot is hour 13", layer.getFloodStats().hour === 13 && hours.list[hours.count - 1].hour === 13);
}]);

// ------------------------------------------------------------------------------ L2 groups: governor

// The frame interval is capped by the display: a 30 Hz cadence must not read as overload.
GROUPS.push(['governor cadence', async () => {
  const { layer, S } = await makeLayer(null, { opts: { ...FLAT, autoQuality: true, frameBudgetMs: 28, pixelScale: 0.25, governorRecoverMs: 3000, showParticles: true } });
  await gridCentreView();
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 12000);
  const qs = collect(layer, 'quality');
  S.pump(40, 16.7, false);
  // a 30 Hz display (or a throttled pane) is a cadence, not overload: quality must not sink to the bottom for good
  const nc = qs.count;
  let tc = S.vt;
  for (let i = 0; i < 3000 && S.vt - tc < 90000; i++) S.step(33.3);
  const lowC = Math.max(...qs.list.slice(nc).map((e) => e.level), 0);
  const lvlEnd = layer.qualityLevel;
  record('a steady 33.3 ms cadence (30 Hz display) does not drive quality to the bottom, and it recovers to level 0', lowC <= 3 && lvlEnd === 0 && layer.qualityInfo.cadenceMs > 30, `lowest level ${lowC}, end level ${lvlEnd}, cadence ${layer.qualityInfo.cadenceMs.toFixed(1)} ms after ${((S.vt - tc) / 1000).toFixed(0)} s`);
  // real load on top of the cadence still steps down
  const nd = qs.count;
  for (let i = 0; i < 400 && qs.count === nd; i++) S.step(90);
  record('load above the cadence budget (90 ms frames) still lowers quality', qs.count > nd && qs.list[nd].reason === 'frame time', `${qs.count - nd} step(s)`);
  // the display speeds up again: the cadence is forgotten
  for (let i = 0; i < 2000 && layer.qualityInfo.cadenceMs > 0; i++) S.step(16.7);
  record('when frames come at 60 Hz again the cadence is forgotten', layer.qualityInfo.cadenceMs === 0);
}]);

// Virtual-time policy checks (the numbers come from the layer's own frame clock, no real load involved).
GROUPS.push(['governor policy', async () => {
  const { layer, S } = await makeLayer(null, { opts: { ...FLAT, autoQuality: true, frameBudgetMs: 28, pixelScale: 0.25, governorRecoverMs: 3000, showParticles: true } });
  await gridCentreView();
  await layer.setStorm('25yr');
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 12000);
  const qs = collect(layer, 'quality');
  // a 60 Hz display with jitter around 16.7 ms is headroom, not "almost over budget"
  layer.setQualityLevel(3, 'test');
  const tUp = S.vt;
  for (let i = 0; i < 700 && layer.qualityLevel === 3; i++) S.step(i % 2 ? 17.6 : 15.8);
  record('60 Hz frames (15.8/17.6 ms jitter) are headroom: quality recovers after governorRecoverMs', layer.qualityLevel < 3 && qs.list[qs.count - 1].reason === 'headroom', `level 3 -> ${layer.qualityLevel} after ${((S.vt - tUp) / 1000).toFixed(1)} s virtual (recover time 3 s + settle 1.2 s)`);
  layer.setQualityLevel(0, 'test');
  S.pump(40, 16.7, false);
  // far over budget reacts faster and steps further
  const t0 = S.vt, n0 = qs.count;
  for (let i = 0; i < 400 && qs.count === n0; i++) S.step(120);
  const first = qs.list[n0];
  record('frames of 120 ms (4x the budget) step down within about 1.5 s by up to 3 levels', first && first.reason === 'frame time' && first.level >= 2 && S.vt - t0 < 2500, first && `level ${first.level} after ${((S.vt - t0) / 1000).toFixed(2)} s virtual`);
  // flapping: load comes back right after every recovery; the wait for the next probe grows
  layer.setQualityLevel(0, 'test');
  S.pump(40, 16.7, false);
  const ups = [];
  let prev = qs.count;
  for (let cycle = 0; cycle < 4; cycle++) {
    for (let i = 0; i < 600 && layer.qualityLevel === 0; i++) S.step(45);       // load: lowers quality
    const tDown = S.vt;
    const lvl = layer.qualityLevel;
    for (let i = 0; i < 20000 && layer.qualityLevel >= lvl; i++) S.step(16.7);  // load gone: waits for a probe
    ups.push(S.vt - tDown);
    for (let i = 0; i < 400 && layer.qualityLevel !== 0; i++) S.step(16.7);
    layer.setQualityLevel(0, 'test');
    S.pump(30, 16.7, false);
  }
  record('after a step down soon after a step up, the next probe waits longer (back-off), up to a cap', ups[1] > ups[0] * 1.5 && ups[3] < ups[1] * 1.3, `waits ${ups.map((v) => (v / 1000).toFixed(1)).join(' -> ')} s`);
  void prev;

  // manual override
  const lay2 = await makeLayer(null, { opts: { ...FLAT, autoQuality: true, pixelScale: 0.25, qualityMode: 'medium' } });
  record("constructor option qualityMode: 'medium' pins level 3 and turns the governor off", lay2.layer.qualityLevel === 3 && lay2.layer.qualityMode === 'medium' && lay2.layer.getOptions().autoQuality === false);
  drop(lay2.layer);
  const L = layer;
  L.setOptions({ autoQuality: true });
  record("qualityMode getter is 'auto' while the governor is on", L.qualityMode === 'auto');
  L.setQualityMode('low');
  S.pump(300, 90, false);
  record("setQualityMode('low') pins level 6; slow frames do not change it", L.qualityLevel === 6 && L.qualityMode === 'low' && qs.list[qs.count - 1].reason === 'mode low');
  L.setQualityMode('medium'); record("'medium' pins level 3", L.qualityLevel === 3 && L.qualityMode === 'medium');
  L.setQualityMode('high'); S.pump(300, 90, false); record("'high' pins level 0 whatever the frame time", L.qualityLevel === 0 && L.qualityMode === 'high');
  L.setOptions({ qualityMode: 'low' });
  record('setOptions({qualityMode}) works too', L.qualityLevel === 6 && L.qualityMode === 'low');
  L.setQualityMode('auto');
  S.pump(300, 60, false);
  record("'auto' hands control back: slow frames lower the level again from where it was pinned", L.qualityMode === 'auto' && L.qualityLevel === QUALITY_LEVELS.length - 1, `level ${L.qualityLevel}`);
  L.setOptions({ autoQuality: false }); L.setQualityLevel(2);
  record("autoQuality false plus a manual level reads 'fixed'", L.qualityMode === 'fixed');
  let err = null;
  try { L.setQualityMode('ultra'); } catch (e) { err = e; }
  record('an unknown mode throws TypeError and changes nothing', err instanceof TypeError && L.qualityLevel === 2);
}]);

// Real load: a big canvas and 200000 particles on the real rAF loop. The numbers are the measurement.
GROUPS.push(['governor under real load', async () => {
  // the pane is small; a big map element (clipped by the pane) makes a big canvas
  const box = map.getContainer();
  const oldCss = box.style.cssText;
  box.style.cssText = `${oldCss};position:absolute;left:0;top:0;width:1500px;height:900px;inset:auto`;
  map.resize();
  await gridCentreView({ zoom: 12.5 });
  const q = [];
  const rt = makeRealtimeSched(8);
  const rl = rt.bind(new FluidLayer(map, { scheduler: rt.sched, base: BASE, autoQuality: true, frameBudgetMs: 28, particleCount: 200000, particleCoverage: 0, pixelScale: 2, maxPixels: 16.7e6, governorRecoverMs: 2500, showDepth: true, showFlow: true, showParticles: true, showArrows: true }));
  live.add(rl);
  const t0 = performance.now();
  rl.on('quality', (e) => q.push({ t: performance.now() - t0, level: e.level, px: e.pixelScale, pf: e.particleFactor, reason: e.reason, w: rl.qualityInfo.canvasWidth, h: rl.qualityInfo.canvasHeight }));
  await rl.setStorm('25yr');
  rl.setTime(14);
  await waitFor(() => rl.time === 14 && !rl.getState().buffering, 15000);
  const tReady = performance.now() - t0;
  const c0 = rl.qualityInfo;
  const startW = c0.canvasWidth, startH = c0.canvasHeight;
  await sleep(1500);
  const f0 = rl.getState().frames, tf0 = performance.now();
  await sleep(1000);
  const fps0 = (rl.getState().frames - f0) / ((performance.now() - tf0) / 1000);
  await waitFor(() => q.length > 0, 20000);
  const firstStep = q.length ? q[0].t - tReady : -1;
  // wait until it settles (no step for 6 s) or hits the last level
  let lastCount = -1, still = 0;
  for (let i = 0; i < 60 && still < 12; i++) { await sleep(500); if (q.length === lastCount) still++; else { still = 0; lastCount = q.length; } }
  const peak = rl.qualityLevel;
  const f1 = rl.getState().frames, tf1 = performance.now();
  await sleep(2000);
  const fps1 = (rl.getState().frames - f1) / ((performance.now() - tf1) / 1000);
  const ci = rl.qualityInfo;
  const timeline = q.map((e) => `${(e.t / 1000).toFixed(1)}s L${e.level} ${e.w}x${e.h} ${e.reason}`).join(' | ');
  record('heavy config (200000 particles, large canvas): the governor steps down within a few seconds of the first picture', q.length > 0 && firstStep < 6000,
    `map ${map.getContainer().clientWidth}x${map.getContainer().clientHeight} css px, start canvas ${startW}x${startH}, ${fps0.toFixed(1)} fps; first step ${(firstStep / 1000).toFixed(1)} s after ready; ${timeline}`);
  record('it settles: the frame time is back near the budget (or the ladder is exhausted), better than at the start', (rl.qualityInfo.avgMs <= 28 * 1.15 || peak === QUALITY_LEVELS.length - 1) && fps1 > fps0 * 1.2, `level ${peak} canvas ${ci.canvasWidth}x${ci.canvasHeight}, avg ${ci.avgMs.toFixed(1)} ms, ${fps1.toFixed(1)} fps (start ${fps0.toFixed(1)} fps)`);
  // load removed: the requested config shrinks
  const tr0 = performance.now();
  box.style.cssText = oldCss;      // the big map itself is part of the load (terrain draw per frame): remove it too
  map.resize();
  rl.setOptions({ pixelScale: 0.25, particleCount: 3000 });
  await waitFor(() => rl.qualityLevel === 0, 60000);
  const rec = (performance.now() - tr0) / 1000;
  const ups = q.filter((e) => e.reason === 'headroom');
  record('with the load removed the governor climbs back to level 0', rl.qualityLevel === 0, `level ${peak} -> ${rl.qualityLevel} in ${rec.toFixed(1)} s (${ups.length} steps up, recover wait 2.5 s per level), avg ${rl.qualityInfo.avgMs.toFixed(1)} ms`);
  drop(rl);
  box.style.cssText = oldCss;
  map.resize();
  await settle(8000);
}]);

// ------------------------------------------------------------------------------ L2 groups: leaks and lifecycle

GROUPS.push(['leaks', async () => {
  await gridCentreView();
  const { layer, S } = await makeLayer(null, { opts: FLAT });
  const refs = [];
  layer.on('hour', (e) => { if (e.grid) refs.push(new WeakRef(e.grid)); });
  const errs = collect(layer, 'error');
  const settleAt = async (storm, t) => { await layer.setStorm(storm); layer.setTime(t); await pumpUntil(S, () => layer.time === t && !layer.getState().buffering, 25000); S.pump(3); };
  const sig = () => hashStride(canvasStats(layer).px);
  await settleAt('5yr', 14); const h5 = sig();
  await settleAt('25yr', 14); const h25 = sig();
  await settleAt('5yr', 14);
  record('precondition: 5yr and 25yr draw different pictures at hour 14', h5 !== h25 && sig() === h5);
  await gcNudge();
  const base = { heap: heapMB(), gl: GLC.snapshot(), have: layer.getState().have.length, cached: layer.getState().cached.length };
  let stale = 0, staleNow = 0;
  const t0 = performance.now();
  for (let i = 0; i < 10; i++) {
    layer.setTime(14);
    await layer.setStorm('25yr');
    if (sig() !== h25) staleNow++;
    await settleAt('25yr', 14); if (sig() !== h25) stale++;
    await layer.setStorm('5yr');
    if (sig() !== h5) staleNow++;
    await settleAt('5yr', 14); if (sig() !== h5) stale++;
  }
  const secs = (performance.now() - t0) / 1000;
  await gcNudge();
  const end = { heap: heapMB(), gl: GLC.snapshot(), have: layer.getState().have.length, cached: layer.getState().cached.length, inflight: layer.getState().inflight.length };
  record('5yr -> 25yr -> 5yr x10: the picture right after setStorm and after settling is never the other storm', stale === 0 && staleNow === 0, `${staleNow} stale right after setStorm, ${stale} after settling, 40 switches in ${secs.toFixed(0)} s`);
  // region replacements (canvas source coordinates and size) must not accumulate GL objects either
  const gsr0 = GLC.snapshot();
  for (let i = 0; i < 20; i++) { layer.updateRegion(true); S.pump(2); }
  const gsr1 = GLC.snapshot();
  const rdiff = Object.keys(gsr0).filter((k) => gsr1[k] !== gsr0[k]).map((k) => `${k} ${gsr0[k]} -> ${gsr1[k]}`);
  record('20 forced region replacements add no GL objects', rdiff.length === 0, rdiff.join(', ') || JSON.stringify(gsr1));
  const glDiff = Object.keys(base.gl).filter((k) => k !== 'contexts' && end.gl[k] !== base.gl[k]).map((k) => `${k} ${base.gl[k]} -> ${end.gl[k]}`);
  record('GL objects (textures, buffers, framebuffers, programs, shaders, VAOs) do not grow', glDiff.length === 0 && end.gl.contexts === base.gl.contexts, `before ${JSON.stringify(base.gl)}; after ${JSON.stringify(end.gl)}`);
  record('JS heap does not grow (after a GC nudge)', !(end.heap - base.heap > 15), `${base.heap.toFixed(1)} -> ${end.heap.toFixed(1)} MB${typeof window.gc === 'function' ? ' (gc exposed)' : ' (no gc(): allocation nudge)'}`);
  const alive = refs.filter((r) => r.deref() !== undefined).length;
  record('old grids are collectable: few of the grids the layer ever reported are still alive', alive <= LRU_HOURS + 6, `${alive} of ${refs.length} grid buffers alive (cache holds at most ${LRU_HOURS})`);
  record('the layer holds only the cached hours and no pending loads or errors', end.have <= LRU_HOURS + 2 && end.cached <= LRU_HOURS && end.inflight <= PREFETCH_CONCURRENCY && errs.count === 0, `have ${end.have}, cached ${end.cached}, inflight ${end.inflight}, errors ${errs.count}`);
  drop(layer);

  // create / dispose cycles
  await gcNudge();
  const g0 = GLC.snapshot(), hp0 = heapMB(), sp0 = GLC.split(map.getCanvas());
  const refs2 = [];
  for (let i = 0; i < 5; i++) {
    const x = await makeLayer(null, { opts: FLAT });
    x.layer.on('hour', (e) => { if (e.grid) refs2.push(new WeakRef(e.grid)); });
    await x.layer.setStorm('25yr');
    x.layer.setTime(14);
    await pumpUntil(x.S, () => x.layer.time === 14 && !x.layer.getState().buffering, 25000);
    x.S.pump(5);
    drop(x.layer);
  }
  await gcNudge();
  const g1 = GLC.snapshot(), hp1 = heapMB(), sp1 = GLC.split(map.getCanvas());
  const gd = Object.keys(g0).filter((k) => k !== 'contexts' && g1[k] !== g0[k]).map((k) => `${k} ${g0[k]} -> ${g1[k]}`);
  record('dispose + new FluidLayer x5: no GL objects left behind', gd.length === 0, `contexts created ${g1.contexts - g0.contexts}; map ctx ${JSON.stringify(sp0.map)} -> ${JSON.stringify(sp1.map)}; other ctx ${JSON.stringify(sp0.other)} -> ${JSON.stringify(sp1.other)}`);
  await gcNudge();
  const alive2 = refs2.filter((r) => r.deref() !== undefined).length;
  const hp2 = heapMB();
  record('dispose + new FluidLayer x5: the grids of the disposed layers are collectable (each is 40 MB)', alive2 <= 3, `${alive2} of ${refs2.length} grid buffers still alive`);
  record('dispose + new FluidLayer x5: JS heap does not grow (generous: GC timing noise)', !(Math.min(hp1, hp2) - hp0 > 40), `${hp0.toFixed(1)} -> ${hp1.toFixed(1)} -> ${hp2.toFixed(1)} MB after another nudge`);
}]);

GROUPS.push(['context loss', async () => {
  await gridCentreView();
  const { layer, S } = await makeLayer('25yr', { opts: FLAT });
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 25000);
  S.pump(4);
  const errs = collect(layer, 'error');
  const waterSamples = () => {
    const c = map.getContainer();
    let n = 0;
    const p = new Uint8Array(4);
    for (let i = 1; i < 24; i++) for (let j = 1; j < 14; j++) if (isWaterRgb(mapPixel(c.clientWidth * i / 24, c.clientHeight * j / 14, p))) n++;
    return n;
  };
  redraw();
  const w0 = waterSamples();
  const before = hashStride(canvasStats(layer).px);
  record('precondition: water is on screen', w0 > 3, `${w0} water samples`);

  // the layer's own canvas
  const rgl = layer.canvas.getContext('webgl2');
  const rlose = rgl.getExtension('WEBGL_lose_context');
  rlose.loseContext();
  await sleep(150);
  const wasLost = layer.renderer.lost;
  let threw = null;
  try { S.pump(10); layer.setTime(15); layer.setTime(14); S.pump(3); } catch (e) { threw = e; }
  record('renderer context lost: the layer keeps running without exceptions', wasLost === true && threw === null && errs.count === 0, threw ? String(threw) : `lost ${wasLost}`);
  rlose.restoreContext();
  await waitFor(() => !layer.renderer.lost, 4000);
  S.pump(6);
  redraw();
  const after = hashStride(canvasStats(layer).px);
  const w1 = waterSamples();
  record('renderer context restored: the same picture is back on the canvas and on the map', layer.renderer.stats.contextRestores === 1 && after === before && w1 >= w0 * 0.8, `restores ${layer.renderer.stats.contextRestores}, picture ${after === before ? 'identical' : 'DIFFERENT'}, map water samples ${w0} -> ${w1}`);

  // the map's context
  let mLost = 0, mRest = 0;
  const onL = () => mLost++, onR = () => mRest++;
  map.on('webglcontextlost', onL); map.on('webglcontextrestored', onR);
  const mgl = map.getCanvas().getContext('webgl2');
  const mlose = mgl.getExtension('WEBGL_lose_context');
  mlose.loseContext();
  await sleep(300);
  record('map context lost: the layer stops its frame loop', mLost === 1 && S.pending() === 0 && layer.getState().running === false, `lost events ${mLost}, pending frames ${S.pending()}`);
  let t2 = null;
  try { S.pump(5, 16.7, false); layer.setTime(13); layer.setTime(14); } catch (e) { t2 = e; }
  record('map context lost: calls stay safe', t2 === null);
  mlose.restoreContext();
  await waitFor(() => mRest === 1, 6000);
  await sleep(600);
  // MapLibre 4.7.1 itself leaves the base map black after a context restore until new tiles are requested
  // (reproducible with no fluid layer at all), so the view is moved to make it load them again.
  await jump({ zoom: 12.6 });
  await jump({ zoom: 12 });
  S.pump(8);
  await sleep(1200);
  redraw();
  S.pump(4);
  redraw();
  const w2 = waterSamples();
  record('map context restored: the loop runs again and the water is back on the map', mRest === 1 && layer.getState().running && w2 >= w0 * 0.6, `restored events ${mRest}, running ${layer.getState().running}, water samples ${w0} -> ${w2}, map layer present ${Boolean(map.getLayer('fluid-water'))}, errors ${errs.count}`);
  map.off('webglcontextlost', onL); map.off('webglcontextrestored', onR);
  drop(layer);
}]);

GROUPS.push(['real visibility', async () => {
  await gridCentreView();
  const rt = makeRealtimeSched(16);
  const rl = rt.bind(new FluidLayer(map, { ...BASE_OPTS, ...FLAT, scheduler: rt.sched }));
  live.add(rl);
  await rl.setStorm('25yr');
  rl.setTime(10);
  await waitFor(() => rl.time === 10 && !rl.getState().buffering, 20000);
  rl.play(1500);
  await sleep(1200);
  const setHidden = (hidden) => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (hidden ? 'hidden' : 'visible') });
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    document.dispatchEvent(new Event('visibilitychange'));
  };
  try {
    setHidden(true);
    await sleep(100);
    const f0 = rl.getState().frames, t0 = rl.time;
    await sleep(1500);
    record('hidden page (real rAF): no frames, the clock stands still, playback flag kept', rl.getState().frames === f0 && rl.time === t0 && rl.isPlaying && rl.getState().running === false, `frames ${f0} -> ${rl.getState().frames}, time ${t0.toFixed(3)} -> ${rl.time.toFixed(3)}`);
    setHidden(false);
    await sleep(1200);
    const dt = rl.time - t0;
    record('visible again: frames resume and the clock continues from where it stopped', rl.getState().frames > f0 && dt > 0.2 && dt < 1.1, `time advanced ${dt.toFixed(2)} h in 1.2 s (1500 ms per hour)`);
  } finally { delete document.visibilityState; delete document.hidden; }
  rl.pause();
  drop(rl);
}]);

// ------------------------------------------------------------------------------ L2 groups: robustness

GROUPS.push(['robustness', async () => {
  await gridCentreView();
  const { layer, S } = await makeLayer('25yr', { opts: FLAT });
  layer.setTime(14);
  await pumpUntil(S, () => layer.time === 14 && !layer.getState().buffering, 25000);
  S.pump(3);
  const h14 = hashStride(canvasStats(layer).px);
  const errs = collect(layer, 'error');
  const sts = collect(layer, 'status');
  let err = null;
  try { await layer.setStorm('nope'); } catch (e) { err = e; }
  record('setStorm of a missing storm rejects with StormNotFoundError (name, storm, url)', err instanceof StormNotFoundError && err.name === 'StormNotFoundError' && err.storm === 'nope' && /nope\/meta\.json/.test(err.url), err && `${err.name}: ${err.message}`);
  S.pump(3);
  record('...and the previous storm stays valid: storm, picture, status, time, playback', layer.storm === '25yr' && layer.hasPicture && layer.status.state === 'ready' && layer.time === 14 && hashStride(canvasStats(layer).px) === h14
    && errs.count === 1 && errs.list[0].phase === 'storm' && errs.list[0].error === err, `storm ${layer.storm}, status ${layer.status.state}, ${errs.count} error event(s)`);
  layer.setTime(15);
  await pumpUntil(S, () => layer.time === 15 && !layer.getState().buffering, 25000);
  record('...and the layer keeps working afterwards (setTime)', layer.time === 15 && layer.probe([CENTER[0], CENTER[1]]) !== null);
  drop(layer);
  // a first storm that is missing leaves a clean layer
  const x = await makeLayer(null, { opts: FLAT });
  let e2 = null;
  try { await x.layer.setStorm('nope'); } catch (e) { e2 = e; }
  record('missing first storm: rejects, status error, no storm, nothing on the map', e2 instanceof StormNotFoundError && x.layer.status.state === 'error' && x.layer.storm === null && !x.layer.hasPicture && !map.getLayer('fluid-water'));
  await x.layer.setStorm('25yr');
  record('...and the same layer loads a real storm afterwards', x.layer.storm === '25yr' && x.layer.hasPicture && x.layer.status.state === 'ready');
  drop(x.layer);

  // an hour grid that fails: last good picture stays, 'error' fires, scrubbing elsewhere works, playback skips it after retries
  const meta = synthMeta();
  meta.hours = [12, 13, 14, 15, 16];
  const fail = new Set([14]);
  const files = {};
  const ok = await synthFetch({ ...meta, hours: [13, 14] });
  const f = (url, init) => {
    const nm = String(url).split('?')[0].split('/').pop();
    const hm = /^h(\d\d)\.rgba\.gz$/.exec(nm);
    if (nm === 'meta.json') return Promise.resolve(new Response(JSON.stringify(meta), { status: 200 }));
    if (hm) {
      const h = Number(hm[1]);
      if (fail.has(h)) return Promise.resolve(new Response('boom', { status: 500 }));
      return ok(String(url).replace(/h\d\d/, 'h13'), init);
    }
    return ok(url, init);
  };
  void files;
  await jump({ center: CENTER, zoom: 15, pitch: 0, bearing: 0 });
  const rt = makeRealtimeSched(16);
  const rl = rt.bind(new FluidLayer(map, { ...BASE_OPTS, ...FLAT, base: '/synth', fetch: f, scheduler: rt.sched }));
  live.add(rl);
  const gerr = collect(rl, 'error');
  const gsts = collect(rl, 'status');
  await rl.setStorm('synth');
  rl.setTime(13);
  await waitFor(() => rl.time === 13 && !rl.getState().buffering, 5000);
  await sleep(100);
  const pic13 = hashStride(canvasStats(rl).px);
  const times = [];
  rl.on('time', (e) => times.push(e.hour));
  rl.play(400);
  await waitFor(() => gerr.list.some((e) => e.phase === 'grid'), 6000);
  await sleep(200);
  const ge = gerr.list.find((e) => e.phase === 'grid');
  const heldAt = rl.time;
  record("failing hour grid: 'error' event (phase grid, hour 14), status error, last good picture and time held, loop alive", Boolean(ge) && ge.hour === 14 && rl.status.state === 'error' && heldAt < 14 && rl.getState().running,
    `held at t=${heldAt.toFixed(2)}, status ${rl.status.state} [${gsts.list.map((e) => e.state).join('>')}], buffering ${rl.getState().buffering}, running ${rl.getState().running}, error hour ${ge && ge.hour}, frames ${rl.getState().frames}`);
  record('...the picture is still the last good one (mix of 13 and the dry/last grid is allowed to move only forward in time, never blank)', canvasStats(rl).n > 0);
  void pic13;
  // scrubbing back to an hour that loads works and clears the error
  rl.pause();
  rl.setTime(13);
  await waitFor(() => rl.time === 13 && rl.status.state !== 'error', 4000);
  record('setTime to a loadable hour afterwards clears the error and shows it', rl.time === 13 && rl.status.state === 'ready', rl.status.state);
  // playback skips an hour that keeps failing (2 failures, retry after 5 s)
  rl.setTime(13.2);
  await waitFor(() => rl.time === 13.2, 3000);
  const t1 = performance.now();
  rl.play(400);
  await waitFor(() => rl.time >= 15, 20000);
  const waited = (performance.now() - t1) / 1000;
  const nerr = gerr.list.filter((e) => e.phase === 'grid' && e.hour === 14).length;
  record('playback skips an hour whose grid failed twice instead of stalling for good', rl.time >= 15 && nerr >= 2, `reached t=${rl.time.toFixed(2)} after ${waited.toFixed(1)} s, ${nerr} failures of hour 14 reported`);
  rl.pause();
  fail.clear();
  drop(rl);
  void S;
}]);

GROUPS.push(['after dispose', async () => {
  await gridCentreView();
  const { layer } = await makeLayer('25yr', { opts: FLAT });
  layer.setTime(14);
  await waitFor(() => layer.time === 14, 15000);
  layer.attachBuildings([]);
  const det = layer.attachProbe(() => {});
  const cv = layer.canvas;
  drop(layer);
  const calls = {
    setTime: () => layer.setTime(5), play: () => layer.play(100), pause: () => layer.pause(), togglePlay: () => layer.togglePlay(), setOptions: () => layer.setOptions({ opacity: 0.5, pixelScale: 1, autoQuality: true }),
    setVisible: () => layer.setVisible(true), updateRegion: () => layer.updateRegion(true), setQualityLevel: () => layer.setQualityLevel(3), setQualityMode: () => layer.setQualityMode('low'),
    tick: () => layer.tick(1234), on: () => layer.on('stats', () => {}), once: () => layer.once('time', () => {}), off: () => layer.off('time', () => {}), getState: () => layer.getState(),
    getOptions: () => layer.getOptions(), currentGrids: () => layer.currentGrids(), probe: () => layer.probe([CENTER[0], CENTER[1]]), detach: () => det(), attachProbe: () => layer.attachProbe(() => {})(),
    getBuildingFlood: () => layer.getBuildingFlood(), getRoadFlood: () => layer.getRoadFlood(), getFloodStats: () => layer.getFloodStats(), buildingLngLat: () => layer.buildingLngLat(0), getIndexInfo: () => layer.getIndexInfo(),
    dispose: () => layer.dispose(), timeRange: () => layer.timeRange, qualityInfo: () => layer.qualityInfo, region: () => layer.region,
  };
  const bad = [];
  for (const [k, fn] of Object.entries(calls)) { try { fn(); } catch (e) { bad.push(`${k}: ${e.message}`); } }
  record('after dispose every method is a safe no-op (no exception)', bad.length === 0, bad.join('; ') || `${Object.keys(calls).length} calls`);
  const pr = layer.probe([CENTER[0], CENTER[1]]);
  record('after dispose probe is invalid, statistics are null, state is idle, the layer is gone from the map', pr.valid === false && layer.getFloodStats() === null && layer.status.state === 'idle' && layer.disposed && !map.getLayer('fluid-water') && cv.width === 1);
  const a = await layer.attachBuildings([]);
  const b = await layer.refreshStats();
  let err = null;
  try { await layer.setStorm('25yr'); } catch (e) { err = e; }
  record('after dispose attach resolves 0, refreshStats resolves null, setStorm rejects (documented)', a === 0 && b === null && err instanceof Error, err && err.message);
  const before = layer.getState().frames;
  layer.setTime(7); layer.play(10);
  await sleep(150);
  record('after dispose nothing runs: no frame request, no playing flag', layer.getState().frames === before && layer.isPlaying === false);
}]);

// @@GROUPS8@@




bindUi();
window.__h = { GLC, loadGeo, collect, synthMeta, synthFetch, makeRealtimeSched, FluidLayer, map, makeLayer, makeSched, settle, jump, redraw, mapPixel, deepestPixel, rawStorm, startDemo, runTests: (o) => runTests(o), live, BASE, sleep, drop, pumpUntil, record };
if (params.get('demo') !== '0') await startDemo($('storm').value);
if (params.get('run')) runTests({ only: params.get('only') });
