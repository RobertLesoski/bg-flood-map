// Harness for fluid-data.js and fluid-renderer.js. Everything here is synthetic: a storm is
// built in JavaScript following FLOW_FORMAT.md, gzipped with CompressionStream, served from
// memory through a stubbed fetch and loaded by the real loader. Nothing touches the network.
import {
  loadStorm, FlowStorm, validateMeta, sample, decodeDepth, decodeVel, lngLatToMerc, mercToLngLat, LRU_HOURS,
} from '../fluid-data.js';
import * as DATA from '../fluid-data.js';
const PREFETCH_SLOTS = DATA.PREFETCH_CONCURRENCY || 2;   // an older loader without the export is held to the same limit
// ?renderer=<file next to harness.js> runs the whole suite against a modified copy of the renderer
// (mutation testing: a deliberately broken renderer must make the matching tests fail)
const rendererParam = new URLSearchParams(location.search).get('renderer');
const { FluidRenderer } = await import(rendererParam ? new URL(rendererParam, import.meta.url).href : '../fluid-renderer.js');

// ----------------------------------------------------------------------------- utilities

const clamp = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function hash2(i, j) {
  let h = (Math.imul(i, 374761393) + Math.imul(j, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function vnoise(x, y) {
  const i = Math.floor(x), j = Math.floor(y);
  const fx = x - i, fy = y - j;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  return lerp(lerp(hash2(i, j), hash2(i + 1, j), ux), lerp(hash2(i, j + 1), hash2(i + 1, j + 1), ux), uy) * 2 - 1;
}

const PAL = [[143, 211, 244], [58, 160, 224], [21, 96, 184], [10, 47, 122]];
function paletteColor(d) {
  const mixc = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  let c = mixc(PAL[0], PAL[1], clamp((d - 0.1) / 0.2, 0, 1));
  c = mixc(c, PAL[2], clamp((d - 0.3) / 0.3, 0, 1));
  return mixc(c, PAL[3], clamp((d - 0.6) / 0.6, 0, 1));
}

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// ------------------------------------------------------------------------ synthetic storm

const GW = 1024, GH = 768, CELL = 5.0;
const HOURS = [3, 4, 5, 6, 7, 8];
const DOMAIN_R = 2350;
const CTRL = [[-2050, -1000], [-1600, -940], [-900, -1150], [-150, -1000], [550, -1060], [1050, -820],
  [1330, -330], [1300, 300], [1250, 880], [1420, 1500], [1500, 2100]];
const BASIN = { x: 1750, y: 450, r: 520 };
const POND = { x: -1250, y: 850, r: 250 };
const MOUND = { x: 150, y: 700 };

// encoder following FLOW_FORMAT.md
function encodeCell(buf, o, depth, u, v, valid) {
  let R = 0;
  if (valid && depth >= 0.02) R = Math.round(255 * Math.sqrt(clamp(depth / 8, 0, 1)));
  buf[o] = R;
  if (R === 0) { buf[o + 1] = 128; buf[o + 2] = 128; } else {
    buf[o + 1] = 128 + Math.round(clamp(u / 4, -1, 1) * 127);
    buf[o + 2] = 128 + Math.round(clamp(v / 4, -1, 1) * 127);
  }
  buf[o + 3] = valid ? 255 : 0;
}

function streamGeometry() {
  const pts = [];
  const P = [CTRL[0], ...CTRL, CTRL[CTRL.length - 1]];
  for (let i = 1; i < P.length - 2; i++) {
    const p0 = P[i - 1], p1 = P[i], p2 = P[i + 1], p3 = P[i + 2];
    const steps = Math.ceil(Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 2);
    for (let s = 0; s < steps; s++) {
      const t = s / steps, t2 = t * t, t3 = t2 * t;
      const f = (k) => 0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3);
      pts.push([f(0), f(1)]);
    }
  }
  pts.push(CTRL[CTRL.length - 1]);
  const N = GW * GH;
  const dist = new Float32Array(N).fill(1e9), tx = new Float32Array(N), ty = new Float32Array(N), sArc = new Float32Array(N);
  const RAD = 280, rc = Math.ceil(RAD / CELL);
  let arc = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
    let dx = b[0] - a[0], dy = b[1] - a[1];
    const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
    if (i > 0) arc += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    const col0 = (pts[i][0] + GW * CELL / 2) / CELL - 0.5, row0 = (GH * CELL / 2 - pts[i][1]) / CELL - 0.5;
    for (let r = Math.max(0, Math.floor(row0) - rc); r <= Math.min(GH - 1, Math.ceil(row0) + rc); r++) {
      for (let c = Math.max(0, Math.floor(col0) - rc); c <= Math.min(GW - 1, Math.ceil(col0) + rc); c++) {
        const X = (c + 0.5) * CELL - GW * CELL / 2, Y = GH * CELL / 2 - (r + 0.5) * CELL;
        const d = Math.hypot(X - pts[i][0], Y - pts[i][1]);
        const idx = r * GW + c;
        if (d < dist[idx]) { dist[idx] = d; tx[idx] = dx; ty[idx] = dy; sArc[idx] = arc; }
      }
    }
  }
  return { dist, tx, ty, sArc };
}

// Physical truth for hour index hi (0..5): depth m, u east m/s, v north m/s, domain flag.
function buildTruth(geom, hi) {
  const t = hi / 5;
  const N = GW * GH;
  const depth = new Float32Array(N), U = new Float32Array(N), V = new Float32Array(N), valid = new Uint8Array(N);
  const Wch = 28 + 30 * t, Dch = 0.45 + 1.1 * t;
  const sigM = 130 + 90 * t;
  for (let row = 0; row < GH; row++) {
    for (let col = 0; col < GW; col++) {
      const idx = row * GW + col;
      const X = (col + 0.5) * CELL - GW * CELL / 2, Y = GH * CELL / 2 - (row + 0.5) * CELL;
      if (X * X + Y * Y > DOMAIN_R * DOMAIN_R) continue;
      valid[idx] = 1;
      let d = 0, mu = 0, mv = 0;
      const dist = geom.dist[idx];
      if (dist < 300) {
        const lump = 1 + 0.16 * vnoise(X / 55, Y / 55);
        const ds = (Dch * Math.exp(-((dist / Wch) ** 2)) + (0.03 + 0.2 * t) * Math.exp(-((dist / 260) ** 2))) * lump;
        const sp = ((0.7 + 1.9 * t) * Math.exp(-((dist / (Wch * 1.3)) ** 2)) * (1 + 0.25 * Math.sin(geom.sArc[idx] / 180)) + 0.12 * Math.exp(-((dist / 260) ** 2)));
        d += ds; mu += ds * sp * geom.tx[idx]; mv += ds * sp * geom.ty[idx];
      }
      {
        const dx = X - BASIN.x, dy = Y - BASIN.y, r = Math.hypot(dx, dy);
        if (r < BASIN.r) {
          const b = 1 - (r / BASIN.r) ** 2;
          const db = (0.05 + 1.5 * t) * Math.pow(b, 0.7);
          const th = Math.atan2(dy, dx);
          const sw = 0.35 * (4 * (r / BASIN.r) * (1 - r / BASIN.r)) * (0.5 + t);
          d += db; mu += db * (sw * Math.sin(th) + 0.12 * b); mv += db * (-sw * Math.cos(th));
        }
      }
      {
        const r = Math.hypot(X - POND.x, Y - POND.y);
        if (r < POND.r) { d += (0.25 + 0.9 * t) * Math.sqrt(1 - (r / POND.r) ** 2); }
      }
      {
        const dx = X - MOUND.x, dy = Y - MOUND.y, r = Math.hypot(dx, dy);
        if (r < 600) {
          const dm = (1.0 - 0.45 * t) * Math.exp(-((r / sigM) ** 2)) * (130 / sigM);
          const rs = r / sigM;
          const sp = 1.1 * rs * Math.exp(0.5 - 0.5 * rs * rs * 2);
          if (r > 1e-6) { d += dm; mu += dm * sp * dx / r; mv += dm * sp * dy / r; }
        }
      }
      if (d > 0) {
        let u = mu / d, v = mv / d;
        const cap = Math.sqrt(9.81 * d), sp = Math.hypot(u, v);
        if (sp > cap) { u *= cap / sp; v *= cap / sp; }
        depth[idx] = d; U[idx] = u; V[idx] = v;
      }
    }
  }
  return { depth, u: U, v: V, valid };
}

function encodeHour(truth) {
  const buf = new Uint8Array(GW * GH * 4);
  for (let i = 0; i < GW * GH; i++) encodeCell(buf, i * 4, truth.depth[i], truth.u[i], truth.v[i], truth.valid[i] === 1);
  return buf;
}

function encodeFlowpaths(truths) {
  const N = GW * GH;
  const maxD = new Float32Array(N), sumQ = new Float32Array(N), cnt = new Uint8Array(N);
  for (const tr of truths) {
    for (let i = 0; i < N; i++) {
      const d = tr.depth[i];
      if (d >= 0.02) { cnt[i]++; sumQ[i] += d * Math.hypot(tr.u[i], tr.v[i]); }
      if (d > maxD[i]) maxD[i] = d;
    }
  }
  const buf = new Uint8Array(N * 4);
  for (let i = 0; i < N; i++) {
    const valid = truths[0].valid[i] === 1;
    const o = i * 4;
    buf[o] = valid && maxD[i] >= 0.02 ? Math.round(255 * Math.sqrt(clamp(maxD[i] / 8, 0, 1))) : 0;
    const q = sumQ[i] / truths.length;
    buf[o + 1] = valid ? Math.round(255 * clamp(Math.sqrt(q / 2.0), 0, 1)) : 0;
    buf[o + 2] = valid ? Math.round(255 * cnt[i] / 24) : 0;
    buf[o + 3] = valid ? 255 : 0;
  }
  return buf;
}

async function gzip(u8) {
  const stream = new Blob([u8]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const CENTRE = lngLatToMerc(-86.4417, 36.9903);
const GRID_MERC = [CENTRE[0] - GW * CELL / 2, CENTRE[1] - GH * CELL / 2, CENTRE[0] + GW * CELL / 2, CENTRE[1] + GH * CELL / 2];

function makeMeta() {
  const ll0 = mercToLngLat(GRID_MERC[0], GRID_MERC[1]), ll1 = mercToLngLat(GRID_MERC[2], GRID_MERC[3]);
  return {
    format: 'bgflow', version: 1, storm: 'synthetic', label: 'Synthetic test storm', source: 'rasters',
    source_note: 'generated in the browser by the fluid test harness',
    velocity_method: 'analytic test field (not solver output)', crs: 'EPSG:3857',
    width: GW, height: GH, cell_m: CELL, cell_ground_m: CELL * Math.cos(36.9903 * Math.PI / 180),
    bounds_merc: GRID_MERC.slice(), bounds_lnglat: [ll0[0], ll0[1], ll1[0], ll1[1]],
    depth_encoding: { type: 'sqrt', max_m: 8.0 }, vel_encoding: { vmax_ms: 4.0, zero_code: 128 },
    wet_threshold_m: 0.02, qmax_m2s: 2.0, hours: HOURS.slice(), storm_depth_in: 4.2,
    built: new Date().toISOString(),
  };
}

const hourName = (h) => `h${String(h).padStart(2, '0')}.rgba.gz`;

// The file name of a request URL: the query string (?v=<built>) is ignored, as the real server does.
const urlName = (url) => String(url).split('?')[0].split('/').pop();

// A fetch() stand-in that serves a Map<name, Uint8Array|string> and logs every request.
// fn.peak is the most requests that were waiting out their `delay` at the same time.
function makeFetch(files, opts = {}) {
  const log = [];
  const delay = opts.delay || 0;
  let active = 0;
  const fn = async (url, init = {}) => {
    const name = urlName(url);
    const entry = { url: String(url), name, init, signal: init.signal || null, aborted: false };
    log.push(entry);
    active++;
    if (active > fn.peak) fn.peak = active;
    try {
      if (delay) {
        await new Promise((resolve, reject) => {
          const t = setTimeout(resolve, delay);
          if (init.signal) {
            const onAbort = () => { clearTimeout(t); entry.aborted = true; reject(new DOMException('Aborted', 'AbortError')); };
            if (init.signal.aborted) onAbort(); else init.signal.addEventListener('abort', onAbort, { once: true });
          }
        });
      }
    } finally {
      active--;
    }
    const data = files.get(name);
    if (data === undefined) return new Response('not found', { status: 404, statusText: 'Not Found' });
    return new Response(data, { status: 200 });
  };
  fn.log = log;
  fn.peak = 0;
  return fn;
}

let SYN = null;
async function buildSynthetic() {
  if (SYN) return SYN;
  const t0 = performance.now();
  const geom = streamGeometry();
  const truths = HOURS.map((_, hi) => buildTruth(geom, hi));
  const rgba = truths.map(encodeHour);
  const flow = encodeFlowpaths(truths);
  const meta = makeMeta();
  const files = new Map();
  files.set('meta.json', JSON.stringify(meta));
  for (let i = 0; i < HOURS.length; i++) files.set(hourName(HOURS[i]), await gzip(rgba[i]));
  files.set('flowpaths.rgba.gz', await gzip(flow));
  SYN = { meta, truths, rgba, flow, files, buildMs: performance.now() - t0, geom };
  return SYN;
}

// local metres (east, north from the grid centre) <-> Mercator
const localToMerc = (X, Y) => [CENTRE[0] + X, CENTRE[1] + Y];
const gridRegion = () => GRID_MERC.slice();
function viewRegion(cx, cy, widthM, aspect) {
  const h = widthM / aspect;
  return [CENTRE[0] + cx - widthM / 2, CENTRE[1] + cy - h / 2, CENTRE[0] + cx + widthM / 2, CENTRE[1] + cy + h / 2];
}
const VIEWS = {
  full: [0, 0, 5120],
  stream: [-900, -1050, 900],
  close: [-860, -1130, 240],
  bend: [1200, -350, 1300],
  basin: [1750, 450, 1300],
  pond: [-1250, 850, 700],
  mound: [150, 700, 520],
};

// ------------------------------------------------------------------------------ backdrop

function drawBackdrop(canvas, region, kind) {
  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext('2d');
  if (kind === 'none') { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h); return; }
  if (kind === 'dark') { ctx.fillStyle = '#181f27'; ctx.fillRect(0, 0, w, h); return; }
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const mppX = (region[2] - region[0]) / w, mppY = (region[3] - region[1]) / h;
  for (let py = 0; py < h; py++) {
    const Y = (region[3] - (py + 0.5) * mppY) - CENTRE[1];
    for (let px = 0; px < w; px++) {
      const X = (region[0] + (px + 0.5) * mppX) - CENTRE[0];
      const o = (py * w + px) * 4;
      const n = vnoise(X / 180, Y / 180) * 0.5 + vnoise(X / 45, Y / 45) * 0.25;
      let r = 176 + n * 26, g = 190 + n * 22, b = 150 + n * 20;
      const bx = Math.floor(X / 140), by = Math.floor(Y / 140);
      const fx = X / 140 - bx, fy = Y / 140 - by;
      if (fx < 0.075 || fx > 0.925 || fy < 0.075 || fy > 0.925) { r = 228; g = 228; b = 220; } else {
        const hb = hash2(bx, by);
        if (hb > 0.3 && fx > 0.2 && fx < 0.8 && fy > 0.2 && fy < 0.8) {
          const sub = hash2(Math.floor(X / 28), Math.floor(Y / 28));
          if (sub > 0.15) { const s = 128 + hb * 36; r = s; g = s - 6; b = s - 12; }
        }
      }
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// -------------------------------------------------------------------------------- stage

const LAYER_PRESETS = {
  composite: {},
  water: { showFlow: false, showParticles: false, showArrows: false },
  surface: { showDepth: false, showParticles: false, showArrows: false },
  'depth+surface': { showParticles: false, showArrows: false },
  particles: { showDepth: false, showFlow: false, showArrows: false },
  arrows: { showDepth: false, showFlow: false, showParticles: false },
  flowpaths: { showDepth: false, showFlow: false, showParticles: false, showArrows: false, showFlowpaths: true },
};

const stage = {
  el: null, bg: null, renderer: null, view: 'full', layer: 'composite', backdrop: 'map', hourT: 4.2,
  playing: false, raf: 0, hours: null, fp: null, storm: null, simMs: 0, W: 0, Hh: 0, sizeKey: '',
};

function layoutStage() {
  const header = document.querySelector('header');
  const bare = new URLSearchParams(location.search).has('bare');
  if (bare) header.style.display = 'none';
  const availH = Math.max(window.innerHeight - (bare ? 0 : header.offsetHeight) - 2, 300);
  const availW = Math.max(window.innerWidth - (document.getElementById('results').classList.contains('hidden') ? 0 : 320), 400);
  let h = availH, w = Math.round(h * 4 / 3);
  if (w > availW) { w = availW; h = Math.round(w * 3 / 4); }
  const el = stage.el;
  el.style.width = `${w}px`; el.style.height = `${h}px`;
  stage.W = w; stage.Hh = h;
}

function applyStageHours() {
  const t = clamp(stage.hourT, 0, HOURS.length - 1);
  const i0 = Math.min(Math.floor(t), HOURS.length - 2);
  stage.renderer.setHours(stage.hours[i0], stage.hours[i0 + 1], t - i0);
}

const stageSizeKey = () => `${stage.W}x${stage.Hh}@${window.devicePixelRatio}`;

function applyStageView() {
  layoutStage();
  stage.sizeKey = stageSizeKey();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(stage.W * dpr), h = Math.round(stage.Hh * dpr);
  const [cx, cy, wm] = VIEWS[stage.view];
  stage.bg.width = w; stage.bg.height = h;
  const region = viewRegion(cx, cy, wm, w / h);
  stage.renderer.setRegion(region, w, h);
  drawBackdrop(stage.bg, region, stage.backdrop);
}

function setStatus(msg) { document.getElementById('status').textContent = msg; }

async function initStage() {
  const syn = await buildSynthetic();
  const fetchStub = makeFetch(syn.files);
  const storm = await loadStorm('/synthetic', 'synthetic', { fetch: fetchStub });
  stage.storm = storm;
  stage.hours = [];
  for (const h of HOURS) stage.hours.push((await storm.getHour(h)).rgba);
  stage.fp = (await storm.getFlowpaths()).rgba;
  stage.el = document.getElementById('stage');
  stage.bg = document.createElement('canvas');
  stage.el.appendChild(stage.bg);
  layoutStage();
  stage.renderer = new FluidRenderer({ antialias: true });
  stage.el.appendChild(stage.renderer.canvas);
  stage.renderer.setStorm(storm.meta);
  stage.renderer.setFlowpaths(stage.fp);
  applyStageHours();
  applyStageView();
  stage.renderer.setOptions(LAYER_PRESETS.composite);
  stage.renderer.frame(0);
  setStatus(`ready - GPU: ${stage.renderer.gpu.renderer || 'unknown'}`);
}

function stageFrame(nowMs) {
  stage.renderer.frame(nowMs);
}

function loop(now) {
  if (!stage.playing) return;
  stage.simMs = now;
  stageFrame(now);
  stage.raf = requestAnimationFrame(loop);
}

const H = {
  stage,
  async ready() { if (!stage.renderer) await initStage(); return true; },
  show(cfg = {}) {
    if (cfg.layer) {
      stage.layer = cfg.layer;
      stage.renderer.setOptions({ showDepth: true, showFlow: true, showParticles: true, showArrows: true, showFlowpaths: false, ...LAYER_PRESETS[cfg.layer] });
      stage.renderer.clearTrails();
    }
    if (cfg.options) stage.renderer.setOptions(cfg.options);
    if (cfg.backdrop) stage.backdrop = cfg.backdrop;
    if (cfg.hour !== undefined) { stage.hourT = cfg.hour; applyStageHours(); document.getElementById('hour').value = String(cfg.hour); }
    if (cfg.view || cfg.backdrop || cfg.resize) { if (cfg.view) stage.view = cfg.view; applyStageView(); }
    stage.renderer.renderOnce();
    updateButtons();
  },
  // advance the simulation deterministically: n frames of dtMs each, starting after the last time
  step(n = 60, dtMs = 1000 / 60) {
    for (let i = 0; i < n; i++) { stage.simMs += dtMs; stageFrame(stage.simMs); }
    stage.renderer.sync();
  },
  play() { if (stage.playing) return; stage.playing = true; stage.raf = requestAnimationFrame(loop); document.getElementById('play').classList.add('on'); document.getElementById('play').textContent = 'pause'; },
  pause() { stage.playing = false; cancelAnimationFrame(stage.raf); document.getElementById('play').classList.remove('on'); document.getElementById('play').textContent = 'play'; },
  // magnified nearest-neighbour crop of the renderer canvas in the top-left corner (for screenshots)
  zoom(sx, sy, sw, sh, k = 4) {
    let c = document.getElementById('zoomcv');
    if (sw <= 0) { if (c) c.remove(); return; }
    if (!c) { c = document.createElement('canvas'); c.id = 'zoomcv'; c.style.cssText = 'position:fixed;left:0;top:0;z-index:99;image-rendering:pixelated;border:2px solid #f0f'; document.body.appendChild(c); }
    c.width = sw * k; c.height = sh * k;
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(stage.bg, sx, sy, sw, sh, 0, 0, sw * k, sh * k);
    ctx.drawImage(stage.renderer.canvas, sx, sy, sw, sh, 0, 0, sw * k, sh * k);
  },
  results() { return window.__results; },
  runTests,
  data: { buildSynthetic, makeFetch, viewRegion, localToMerc },
};
window.H = H;

function updateButtons() {
  for (const b of document.querySelectorAll('#layerBtns button')) b.classList.toggle('on', b.dataset.k === stage.layer);
  for (const b of document.querySelectorAll('#viewBtns button')) b.classList.toggle('on', b.dataset.k === stage.view);
  for (const b of document.querySelectorAll('#bgBtns button')) b.classList.toggle('on', b.dataset.k === stage.backdrop);
}

function buildUi() {
  const mk = (id, keys, fn) => {
    const host = document.getElementById(id);
    for (const k of keys) {
      const b = document.createElement('button');
      b.textContent = k; b.dataset.k = k;
      b.addEventListener('click', () => fn(k));
      host.appendChild(b);
    }
  };
  mk('layerBtns', Object.keys(LAYER_PRESETS), (k) => H.show({ layer: k }));
  mk('viewBtns', Object.keys(VIEWS), (k) => H.show({ view: k }));
  mk('bgBtns', ['map', 'dark', 'none'], (k) => H.show({ backdrop: k }));
  document.getElementById('hour').addEventListener('input', (e) => H.show({ hour: Number(e.target.value) }));
  document.getElementById('play').addEventListener('click', () => (stage.playing ? H.pause() : H.play()));
  document.getElementById('toggleResults').addEventListener('click', () => {
    document.getElementById('results').classList.toggle('hidden');
    if (stage.renderer) H.show({ resize: true });
  });
  document.getElementById('runTests').addEventListener('click', async () => {
    document.getElementById('results').classList.remove('hidden');
    H.pause();
    await runTests();
    if (stage.renderer) H.show({ resize: true });
  });
  // setRegion clears the trails, so only react to a real change of the stage size (a screenshot
  // capture can fire resize events without changing anything)
  window.addEventListener('resize', () => {
    if (!stage.renderer) return;
    layoutStage();
    if (stageSizeKey() === stage.sizeKey) return;
    H.show({ resize: true });
  });
}

// -------------------------------------------------------------------------------- tests

window.__results = [];
window.__done = false;

function record(name, pass, detail = '') {
  window.__results.push({ name, pass: !!pass, detail: String(detail) });
  const list = document.getElementById('list');
  if (list) {
    const row = document.createElement('div');
    row.className = `row ${pass ? 'pass' : 'fail'}`;
    const b = document.createElement('b'); b.textContent = pass ? 'PASS ' : 'FAIL ';
    row.appendChild(b);
    row.appendChild(document.createTextNode(name));
    if (detail) { const i = document.createElement('i'); i.textContent = `  ${detail}`; row.appendChild(i); }
    list.appendChild(row);
  }
  console.log(`${pass ? 'PASS' : 'FAIL'} ${name}${detail ? ' :: ' + detail : ''}`);
}

async function test(name, fn) {
  if (window.__only && !window.__only.test(name)) return;
  try {
    const r = await fn();
    if (r === true || r === undefined) record(name, true);
    else if (typeof r === 'object') record(name, r.pass, r.detail);
    else record(name, !!r);
  } catch (e) {
    record(name, false, `threw ${e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e}`);
  }
}

async function rejects(promise) {
  try { await promise; } catch (e) { return e; }
  return null;
}

// renderer factory used by the tests. The particle coverage cap is off unless a test asks for it, so
// that tests which count or follow particles see the particleCount they set.
function makeRenderer(syn, o = {}) {
  const w = o.w || GW, h = o.h || GH;
  const r = new FluidRenderer({ antialias: o.antialias !== false, particleCoverage: 0, ...(o.options || {}) });
  r.setStorm(o.meta || syn.meta);
  r.setHours(o.a || syn.rgba[2], o.b || syn.rgba[3], o.mix === undefined ? 0.37 : o.mix);
  if (o.flowpaths !== false) r.setFlowpaths(o.fp || syn.flow);
  r.setRegion(o.region || gridRegion(), w, h);
  return r;
}

function stepFrames(r, n, startMs = 0, dtMs = 1000 / 60) {
  let t = startMs;
  for (let i = 0; i < n; i++) { r.frame(t); t += dtMs; }
  return t;
}

const px = (img, w, x, y) => { const o = (y * w + x) * 4; return [img[o], img[o + 1], img[o + 2], img[o + 3]]; };

function diffCount(a, b, tol = 3) {
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (Math.abs(a[i] - b[i]) > tol || Math.abs(a[i + 1] - b[i + 1]) > tol || Math.abs(a[i + 2] - b[i + 2]) > tol || Math.abs(a[i + 3] - b[i + 3]) > tol) n++;
  }
  return n;
}

function alphaSum(img) { let s = 0; for (let i = 3; i < img.length; i += 4) s += img[i]; return s; }

const QUIET = { showFlow: false, showParticles: false, showArrows: false, showFlowpaths: false };

async function dataTests(syn) {
  const meta = syn.meta;
  const mk = (extra = {}) => { const f = makeFetch(syn.files, extra); return f; };

  await test('loadStorm: loads synthetic meta, hours sorted, encoding constants', async () => {
    const f = mk();
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const ok = s instanceof FlowStorm && s.width === GW && s.height === GH && s.hours.join() === HOURS.join()
      && s.encoding.depthMax === 8 && s.encoding.vmax === 4 && f.log.length === 1 && /\/synthetic\/synthetic\/meta\.json$/.test(f.log[0].url);
    s.dispose();
    return { pass: ok, detail: `url=${f.log[0].url}` };
  });

  await test('loadStorm: rejects bad meta with clear messages', async () => {
    const bad = {
      'wrong format': [{ ...meta, format: 'nope' }, /format/],
      'wrong version': [{ ...meta, version: 2 }, /version/],
      'non-integer width': [{ ...meta, width: 10.5 }, /width/],
      'bounds disagree with size': [{ ...meta, width: GW + 7 }, /bounds_merc/],
      'inverted bounds': [{ ...meta, bounds_merc: [meta.bounds_merc[2], meta.bounds_merc[1], meta.bounds_merc[0], meta.bounds_merc[3]] }, /bounds_merc/],
      'bad hours': [{ ...meta, hours: [0, 3] }, /hours/],
      'bad depth encoding': [{ ...meta, depth_encoding: { type: 'linear', max_m: 8 } }, /depth_encoding/],
      'grid too large': [{ ...meta, width: 9000, bounds_merc: [0, 0, 9000 * CELL, GH * CELL] }, /larger than the supported/],
    };
    const lines = [];
    let all = true;
    for (const [label, [m, re]] of Object.entries(bad)) {
      const files = new Map([['meta.json', JSON.stringify(m)]]);
      const e = await rejects(loadStorm('/x', 's', { fetch: makeFetch(files) }));
      const ok = !!e && re.test(e.message);
      if (!ok) all = false;
      lines.push(`${label}: ${ok ? 'ok' : 'FAILED (' + (e ? e.message : 'no error') + ')'}`);
    }
    const e1 = await rejects(loadStorm('/x', 's', { fetch: makeFetch(new Map([['meta.json', '{not json']])) }));
    const ok1 = !!e1 && /not valid JSON/.test(e1.message);
    const e2 = await rejects(loadStorm('/x', 's', { fetch: makeFetch(new Map()) }));
    const ok2 = !!e2 && /HTTP 404/.test(e2.message);
    let e3 = null; try { validateMeta(null); } catch (x) { e3 = x; }
    // the real fetch against a path the server does not have
    const e4 = await rejects(loadStorm('/fluid/test/does-not-exist', 'nope'));
    const ok4 = !!e4 && /HTTP 404/.test(e4.message);
    return { pass: all && ok1 && ok2 && ok4 && !!e3 && /JSON object/.test(e3.message), detail: `${lines.join('; ')}; json:${ok1}; 404:${ok2}; real-fetch-404:${ok4} (${e4 ? e4.message : ''})` };
  });

  await test('getHour: gunzip output equals the encoder bytes (all hours) and length is w*h*4', async () => {
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: mk() });
    let ok = true;
    const bad = [];
    for (let i = 0; i < HOURS.length; i++) {
      const g = await s.getHour(HOURS[i]);
      if (g.hour !== HOURS[i] || g.rgba.length !== GW * GH * 4 || !bytesEqual(g.rgba, syn.rgba[i])) { ok = false; bad.push(HOURS[i]); }
    }
    s.dispose();
    return { pass: ok, detail: ok ? `${HOURS.length} hours byte-identical` : `mismatch in hours ${bad}` };
  });

  await test('getHour: quantisation error of decoded depth/velocity vs analytic truth is within half a code', async () => {
    const hi = 3, tr = syn.truths[hi], rgba = syn.rgba[hi];
    let worstD = 0, worstV = 0, n = 0;
    for (let i = 0; i < GW * GH; i++) {
      if (!tr.valid[i] || tr.depth[i] < 0.03) continue;
      n++;
      const R = rgba[i * 4];
      const dd = Math.abs(decodeDepth(R) - Math.min(tr.depth[i], 8));
      const tol = 8 * (R + 0.25) / 65025 + 1e-6;
      worstD = Math.max(worstD, dd / tol);
      const du = Math.abs(decodeVel(rgba[i * 4 + 1]) - clamp(tr.u[i], -4, 4)), dv = Math.abs(decodeVel(rgba[i * 4 + 2]) - clamp(tr.v[i], -4, 4));
      worstV = Math.max(worstV, du / (0.5 * 4 / 127 + 1e-6), dv / (0.5 * 4 / 127 + 1e-6));
    }
    return { pass: n > 10000 && worstD <= 1.001 && worstV <= 1.001, detail: `${n} wet cells, worst depth err ${worstD.toFixed(3)} codes, worst velocity err ${worstV.toFixed(3)} codes` };
  });

  await test('getHour: hours outside meta.hours return one shared all-zero grid, no network', async () => {
    const f = mk();
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const n0 = f.log.length;
    const a = await s.getHour(1), b = await s.getHour(24), c = await s.getHour(0);
    let zero = true;
    for (let i = 0; i < a.rgba.length; i++) if (a.rgba[i] !== 0) { zero = false; break; }
    const ok = a.rgba === b.rgba && b.rgba === c.rgba && a.rgba.length === GW * GH * 4 && zero && a.rgba.isZeroGrid === true && f.log.length === n0 && a.hour === 1 && b.hour === 24;
    s.dispose();
    return { pass: ok, detail: `shared=${a.rgba === b.rgba}, allZero=${zero}, fetches=${f.log.length - n0}` };
  });

  await test('getHour: concurrent requests for one hour share a single fetch and result', async () => {
    const f = mk({ delay: 40 });
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const n0 = f.log.length;
    const [a, b, c] = await Promise.all([s.getHour(4), s.getHour(4), s.getHour(4)]);
    const n = f.log.length - n0;
    s.dispose();
    return { pass: n === 1 && a === b && b === c, detail: `${n} fetch(es), same object=${a === b && b === c}` };
  });

  await test('getHour: aborting the only waiter cancels the fetch and a retry works', async () => {
    const f = mk({ delay: 120 });
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const n0 = f.log.length;
    const ctrl = new AbortController();
    const p = s.getHour(5, { signal: ctrl.signal });
    setTimeout(() => ctrl.abort(), 20);
    const e = await rejects(p);
    await sleep(10);
    const aborted = f.log[n0] && f.log[n0].aborted;
    const retry = await s.getHour(5);
    const total = f.log.length - n0;
    const pre = await rejects(s.getHour(5, { signal: AbortSignal.abort() }));
    s.dispose();
    const ok = !!e && e.name === 'AbortError' && aborted && total === 2 && bytesEqual(retry.rgba, syn.rgba[2]) && !!pre && pre.name === 'AbortError';
    return { pass: ok, detail: `err=${e && e.name}, fetchAborted=${aborted}, fetches=${total}, preAborted=${pre && pre.name}` };
  });

  await test('getHour: one of two waiters aborting does not cancel the shared fetch', async () => {
    const f = mk({ delay: 80 });
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const n0 = f.log.length;
    const ctrl = new AbortController();
    const p1 = s.getHour(6, { signal: ctrl.signal });
    const p2 = s.getHour(6);
    setTimeout(() => ctrl.abort(), 20);
    const e = await rejects(p1);
    const r2 = await p2;
    const entry = f.log[n0];
    s.dispose();
    const ok = !!e && e.name === 'AbortError' && !entry.aborted && f.log.length - n0 === 1 && bytesEqual(r2.rgba, syn.rgba[3]);
    return { pass: ok, detail: `p1=${e && e.name}, fetchAborted=${entry.aborted}, fetches=${f.log.length - n0}` };
  });

  await test(`getHour: LRU keeps ${LRU_HOURS} decoded hours, hits do not refetch, eviction refetches`, async () => {
    const f = mk();
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    for (const h of [3, 4, 5, 6, 7]) await s.getHour(h);
    const c1 = s.cachedHours().join();
    const n1 = f.log.filter((x) => x.name.startsWith('h')).length;
    await s.getHour(4);
    const n2 = f.log.filter((x) => x.name.startsWith('h')).length;
    const c2 = s.cachedHours().join();
    await s.getHour(3);
    const n3 = f.log.filter((x) => x.name.startsWith('h')).length;
    const c3 = s.cachedHours().join();
    s.dispose();
    const ok = c1 === '4,5,6,7' && n2 === n1 && c2 === '5,6,7,4' && n3 === n1 + 1 && c3 === '6,7,4,3';
    return { pass: ok, detail: `after 5 loads [${c1}], after hit [${c2}], after evicted refetch [${c3}], fetches ${n1}/${n2}/${n3}` };
  });

  await test('getFlowpaths: decodes to encoder bytes and is cached', async () => {
    const f = mk();
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const a = await s.getFlowpaths(), b = await s.getFlowpaths();
    const n = f.log.filter((x) => x.name.startsWith('flowpaths')).length;
    s.dispose();
    return { pass: bytesEqual(a.rgba, syn.flow) && a === b && n === 1, detail: `fetches=${n}` };
  });

  await test('prefetch: warms the cache, never rejects (even for a failing hour)', async () => {
    const files = new Map(syn.files); files.delete(hourName(7));
    const f = makeFetch(files);
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const res = await s.prefetch([3, 4, 7, 99]);
    const cached = s.cachedHours().join();
    const ok = res.length === 3 && res.filter((r) => r.status === 'fulfilled').length === 2 && res.filter((r) => r.status === 'rejected').length === 1
      && cached === '3,4';
    s.dispose();
    return { pass: ok, detail: `statuses=${res.map((r) => r.status)}, cache=[${cached}]` };
  });

  await test('prefetch: 24 hours ask for 4 in order, at most 2 download at a time, nothing warmed gets evicted', async () => {
    const all = Array.from({ length: 24 }, (_, i) => i + 1);
    const files = new Map([['meta.json', JSON.stringify({ ...meta, hours: all })], ['flowpaths.rgba.gz', syn.files.get('flowpaths.rgba.gz')]]);
    for (const h of all) files.set(hourName(h), syn.files.get(hourName(3)));
    const f = makeFetch(files, { delay: 25 });
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const res = await s.prefetch(all);
    const grids = () => f.log.filter((x) => x.name.startsWith('h')).map((x) => x.name);
    const asked = grids().join();
    const cached = s.cachedHours().slice().sort((a, b) => a - b).join();
    const peak = f.peak;
    // a second call: the three hours already held are refreshed, not downloaded again, and only the new one is fetched
    const n1 = grids().length;
    const res2 = await s.prefetch([5, 1, 2, 3]);
    const newGrids = grids().slice(n1).join();
    const cached2 = s.cachedHours().slice().sort((a, b) => a - b).join();
    // a duplicate and a missing hour in the list take no slot
    const res3 = await s.prefetch([5, 5, 99, 5]);
    // dispose with downloads queued: they settle as rejected and no further download starts
    const f2 = makeFetch(files, { delay: 25 });
    const s2 = await loadStorm('/synthetic', 'synthetic', { fetch: f2 });
    const p = s2.prefetch([1, 2, 3, 4]);
    s2.dispose();
    const res4 = await p;
    const started = f2.log.filter((x) => x.name.startsWith('h')).length;
    s.dispose();
    const ok = res.length === LRU_HOURS && res.every((r) => r.status === 'fulfilled') && asked === 'h01.rgba.gz,h02.rgba.gz,h03.rgba.gz,h04.rgba.gz'
      && cached === '1,2,3,4' && peak <= PREFETCH_SLOTS && peak >= 1
      && res2.length === 4 && newGrids === 'h05.rgba.gz' && cached2 === '1,2,3,5'
      && res3.length === 1 && res4.length === 4 && res4.every((r) => r.status === 'rejected') && started <= PREFETCH_SLOTS;
    return { pass: ok, detail: `prefetch(1..24): ${res.length} results, requested ${asked}, cache [${cached}], peak concurrent requests ${peak} (limit ${PREFETCH_SLOTS}); then [5,1,2,3]: fetched ${newGrids}, cache [${cached2}]; [5,5,99,5]: ${res3.length} result; dispose while queued: ${res4.map((r) => r.status[0]).join('')}, ${started} downloads started` };
  });

  await test('grid requests carry ?v=<meta.built>: hours and flow paths, a rebuild gets new URLs, meta.json is revalidated', async () => {
    const enc = encodeURIComponent(meta.built);
    const f = mk();
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    await s.getHour(4); await s.getFlowpaths(); await s.prefetch([5]);
    const urlOf = (name) => (f.log.find((x) => x.name === name) || {}).url || '';
    const metaReq = f.log[0];
    const okA = urlOf(hourName(4)).endsWith(`/synthetic/synthetic/${hourName(4)}?v=${enc}`) && urlOf(hourName(5)).endsWith(`${hourName(5)}?v=${enc}`)
      && urlOf('flowpaths.rgba.gz').endsWith(`/flowpaths.rgba.gz?v=${enc}`) && !/\?/.test(metaReq.url) && metaReq.init.cache === 'no-cache';
    s.dispose();
    // the same storm rebuilt later: a cache keyed by URL (the browser's, or the server's one-hour max-age) cannot hand out the old grid
    const built2 = '2031-01-02T03:04:05+00:00';
    const rebuilt = new Map(syn.files); rebuilt.set('meta.json', JSON.stringify({ ...meta, built: built2 }));
    const f2 = makeFetch(rebuilt);
    const s2 = await loadStorm('/synthetic', 'synthetic', { fetch: f2 });
    await s2.getHour(4);
    const u1 = urlOf(hourName(4)), u2 = f2.log.find((x) => x.name === hourName(4)).url;
    const okB = u1 !== u2 && u2.endsWith(`?v=${encodeURIComponent(built2)}`) && /%3A/.test(u2) && !/[ &#]/.test(u2.split('?')[1]);
    s2.dispose();
    // no build time (or one that is not a string): plain URLs, no dangling '?'
    const plain = [];
    for (const built of [undefined, 12345, '']) {
      const m = { ...meta }; if (built === undefined) delete m.built; else m.built = built;
      const f3 = makeFetch(new Map([...syn.files, ['meta.json', JSON.stringify(m)]]));
      const s3 = await loadStorm('/synthetic', 'synthetic', { fetch: f3 });
      await s3.getHour(4); await s3.getFlowpaths();
      plain.push(f3.log.slice(1).every((x) => !x.url.includes('?')));
      s3.dispose();
    }
    // characters that would end or split a query string are escaped
    const odd = { ...meta, built: 'a b&c=d#e?f' };
    const f4 = makeFetch(new Map([...syn.files, ['meta.json', JSON.stringify(odd)]]));
    const s4 = await loadStorm('/synthetic', 'synthetic', { fetch: f4 });
    await s4.getHour(4);
    const u4 = new URL(f4.log[1].url, location.href);
    const okC = u4.search === '?v=a%20b%26c%3Dd%23e%3Ff' && u4.hash === '';
    s4.dispose();
    return { pass: okA && okB && plain.every(Boolean) && okC, detail: `hour/flowpaths urls carry ?v=${enc}: ${okA}; rebuild ${u1.split('?')[1] || '(none)'} -> ${u2.split('?')[1]}: ${okB}; no/odd build time gives plain urls: ${plain}; escaping: ${okC} (${u4.search})` };
  });

  await test('getHour: an allocation failure is reported as out of memory, not as a corrupt file', async () => {
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: mk() });
    const Real = Uint8Array;
    const target = GW * GH * 4;
    // a browser that cannot find room for one more grid throws RangeError from the typed-array constructor
    globalThis.Uint8Array = class extends Real {
      constructor(...a) {
        if (a.length === 1 && a[0] === target) throw new RangeError('Array buffer allocation failed');
        super(...a);
      }
    };
    let e;
    try { e = await rejects(s.getHour(4)); } finally { globalThis.Uint8Array = Real; }
    const good = await s.getHour(5);
    s.dispose();
    return { pass: !!e && e.name === 'GridMemoryError' && /Out of memory/.test(e.message) && !/gunzip/.test(e.message) && bytesEqual(good.rgba, syn.rgba[2]), detail: `${e && e.name}: ${e && e.message}; a later hour still loads: ${bytesEqual(good.rgba, syn.rgba[2])}` };
  });

  await test('getHour: the w*h*4 buffer is allocated when decoded bytes arrive, so corrupt, empty and missing grids hold none', async () => {
    const files = new Map(syn.files);
    files.set(hourName(3), new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));   // not gzip
    files.set(hourName(4), await gzip(new Uint8Array(0)));                              // gzip of nothing
    files.delete(hourName(5));                                                          // 404
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: makeFetch(files) });
    const Real = Uint8Array;
    const target = GW * GH * 4;
    let allocs = 0;
    globalThis.Uint8Array = class extends Real {
      constructor(...a) {
        if (a.length === 1 && a[0] === target) allocs++;
        super(...a);
      }
    };
    let e3, e4, e5, good, allocsFailed, allocsGood;
    try {
      e3 = await rejects(s.getHour(3)); e4 = await rejects(s.getHour(4)); e5 = await rejects(s.getHour(5));
      allocsFailed = allocs;
      good = await s.getHour(6);
      allocsGood = allocs - allocsFailed;
    } finally { globalThis.Uint8Array = Real; }
    s.dispose();
    const ok = !!e3 && /gunzip/.test(e3.message) && !!e4 && /decoded 0 bytes/.test(e4.message) && !!e5 && /HTTP 404/.test(e5.message) && allocsFailed === 0 && allocsGood === 1 && bytesEqual(good.rgba, syn.rgba[3]);
    return { pass: ok, detail: `full-size allocations: ${allocsFailed} for the three failed grids, ${allocsGood} for the good one (${e4 && e4.message})` };
  });

  await test('getHour: wrong decoded size and corrupt gzip give clear errors; failed hour is not cached', async () => {
    const short = await gzip(new Uint8Array(GW * GH * 4 - 4));
    const files = new Map(syn.files);
    files.set(hourName(3), short);
    files.set(hourName(4), new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: makeFetch(files) });
    const e1 = await rejects(s.getHour(3));
    const e2 = await rejects(s.getHour(4));
    const e3 = await rejects(s.getHour(2.5));
    s.dispose();
    const e4 = await rejects(s.getHour(5));
    const ok = !!e1 && /decoded \d+ bytes, expected \d+/.test(e1.message) && !!e2 && /gunzip/.test(e2.message) && e3 instanceof TypeError && !!e4 && /disposed/.test(e4.message) && s.cachedHours().length === 0;
    return { pass: ok, detail: `short: ${e1 && e1.message}; corrupt: ${e2 && e2.message}` };
  });

  await test('getHour: a stream longer than w*h*4 is cut off with a clear error; truncated gzip fails', async () => {
    const files = new Map(syn.files);
    files.set(hourName(3), await gzip(new Uint8Array(GW * GH * 4 + 4096)));
    const full = syn.files.get(hourName(4));
    files.set(hourName(4), full.slice(0, full.length >> 1));
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: makeFetch(files) });
    const e1 = await rejects(s.getHour(3));
    const e2 = await rejects(s.getHour(4));
    const good = await s.getHour(5);
    s.dispose();
    const ok = !!e1 && /decoded more than \d+ bytes, expected \d+/.test(e1.message) && !!e2 && /gunzip/.test(e2.message) && bytesEqual(good.rgba, syn.rgba[2]);
    return { pass: ok, detail: `long: ${e1 && e1.message}; truncated: ${e2 && e2.message}` };
  });

  await test('getHour: a body the browser already inflated (Content-Encoding: gzip) is used as it is', async () => {
    const f = async (url) => {
      const name = urlName(url);
      if (name === 'meta.json') return new Response(syn.files.get(name), { status: 200 });
      if (name === hourName(4)) return new Response(syn.rgba[1], { status: 200, headers: { 'Content-Encoding': 'gzip' } });
      return new Response('nope', { status: 404, statusText: 'Not Found' });
    };
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const g = await s.getHour(4);
    const e = await rejects(s.getHour(5));
    s.dispose();
    return { pass: bytesEqual(g.rgba, syn.rgba[1]) && !!e && /HTTP 404/.test(e.message), detail: `bytes=${g.rgba.length}, 404 for hour 5: ${!!e}` };
  });

  await test('getHour: aborting while the body is streaming rejects with AbortError and caches nothing', async () => {
    const gz = syn.files.get(hourName(3));
    const f = async (url, init = {}) => {
      const name = urlName(url);
      if (name === 'meta.json') return new Response(syn.files.get(name), { status: 200 });
      const part = gz.slice(0, 2000);
      const body = new ReadableStream({
        start(c) {
          c.enqueue(part);   // a first chunk arrives, then the connection stalls until aborted
          if (init.signal) init.signal.addEventListener('abort', () => c.error(new DOMException('Aborted', 'AbortError')), { once: true });
        },
      });
      return new Response(body, { status: 200 });
    };
    const s = await loadStorm('/synthetic', 'synthetic', { fetch: f });
    const ctrl = new AbortController();
    const p = s.getHour(3, { signal: ctrl.signal });
    setTimeout(() => ctrl.abort(), 30);
    const e = await rejects(p);
    const cached = s.cachedHours().length;
    s.dispose();
    return { pass: !!e && e.name === 'AbortError' && cached === 0, detail: `err=${e && e.name}: ${e && e.message}, cached=${cached}` };
  });

  // ---- sample() ----
  await test('sample: exact at pixel centres (depth, u, v) for 3000 random wet texels', async () => {
    let rng = 12345;
    const rnd = () => { rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0; return rng / 4294967296; };
    const A = syn.rgba[2];
    let n = 0, worst = 0;
    const out = { depth: 0, u: 0, v: 0, speed: 0, valid: false };
    for (let tries = 0; tries < 100000 && n < 3000; tries++) {
      const col = Math.floor(rnd() * GW), row = Math.floor(rnd() * GH);
      const o = (row * GW + col) * 4;
      if (A[o] === 0) continue;
      const x = meta.bounds_merc[0] + (col + 0.5) * CELL, y = meta.bounds_merc[3] - (row + 0.5) * CELL;
      sample(A, null, 0, x, y, meta, out);
      const e = Math.max(Math.abs(out.depth - decodeDepth(A[o])), Math.abs(out.u - decodeVel(A[o + 1])), Math.abs(out.v - decodeVel(A[o + 2])));
      worst = Math.max(worst, e);
      if (!out.valid) worst = 1;
      n++;
    }
    return { pass: n === 3000 && worst < 1e-9, detail: `n=${n}, worst abs error ${worst.toExponential(2)}` };
  });

  await test('sample: bilinear in code space between texels, temporal mix of decoded hours', async () => {
    const A = syn.rgba[2], B = syn.rgba[3];
    // find a 2x2 block of wet texels in both hours
    let found = null;
    for (let row = 300; row < 500 && !found; row++) {
      for (let col = 100; col < 900; col++) {
        const ok = [0, 1].every((dy) => [0, 1].every((dx) => { const o = ((row + dy) * GW + col + dx) * 4; return A[o] > 5 && B[o] > 5; }));
        if (ok) { found = [col, row]; break; }
      }
    }
    if (!found) return { pass: false, detail: 'no wet 2x2 block found' };
    const [col, row] = found;
    const out = {};
    const x0 = meta.bounds_merc[0] + (col + 0.5) * CELL, y0 = meta.bounds_merc[3] - (row + 0.5) * CELL;
    const tx = 0.3, ty = 0.7;
    const o = (c, r) => (r * GW + c) * 4;
    const code = (arr, ch) => (1 - tx) * (1 - ty) * arr[o(col, row) + ch] + tx * (1 - ty) * arr[o(col + 1, row) + ch] + (1 - tx) * ty * arr[o(col, row + 1) + ch] + tx * ty * arr[o(col + 1, row + 1) + ch];
    const expectHour = (arr) => ({ d: decodeDepth(code(arr, 0)), u: decodeVel(code(arr, 1)), v: decodeVel(code(arr, 2)) });
    const ea = expectHour(A), eb = expectHour(B);
    sample(A, B, 0.25, x0 + tx * CELL, y0 - ty * CELL, meta, out);
    const eD = 0.75 * ea.d + 0.25 * eb.d, eU = 0.75 * ea.u + 0.25 * eb.u, eV = 0.75 * ea.v + 0.25 * eb.v;
    const err = Math.max(Math.abs(out.depth - eD), Math.abs(out.u - eU), Math.abs(out.v - eV));
    sample(A, B, 0, x0 + tx * CELL, y0 - ty * CELL, meta, out);
    const err0 = Math.abs(out.depth - ea.d);
    sample(A, B, 1, x0 + tx * CELL, y0 - ty * CELL, meta, out);
    const err1 = Math.abs(out.depth - eb.d);
    return { pass: err < 1e-9 && err0 < 1e-9 && err1 < 1e-9, detail: `err mix .25=${err.toExponential(1)}, mix0=${err0.toExponential(1)}, mix1=${err1.toExponential(1)}; speed=${out.speed.toFixed(3)}` };
  });

  await test('sample: edges, outside grid, A=0, null hours, all-zero grid (tiny known grid)', async () => {
    const w = 8, h = 6, cell = 10;
    const tm = { width: w, height: h, cell_m: cell, bounds_merc: [1000, 2000, 1000 + w * cell, 2000 + h * cell], depth_encoding: { type: 'sqrt', max_m: 8 }, vel_encoding: { vmax_ms: 4, zero_code: 128 } };
    const g = new Uint8Array(w * h * 4);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) { const o = (r * w + c) * 4; g[o] = 40 + 10 * c + r; g[o + 1] = 128 + 5 * c; g[o + 2] = 128 - 4 * r; g[o + 3] = 255; }
    const hole = (2 * w + 3) * 4; g[hole] = 0; g[hole + 1] = 128; g[hole + 2] = 128; g[hole + 3] = 0;   // A=0 texel at col 3,row 2
    const cx = (c) => tm.bounds_merc[0] + (c + 0.5) * cell, cy = (r) => tm.bounds_merc[3] - (r + 0.5) * cell;
    const out = {};
    const checks = [];
    const chk = (name, cond) => checks.push([name, cond]);
    sample(g, null, 0, tm.bounds_merc[0], tm.bounds_merc[3], tm, out);   // exact NW corner of the grid
    chk('NW corner clamps to texel (0,0)', out.valid && Math.abs(out.depth - decodeDepth(40)) < 1e-9 && Math.abs(out.u - decodeVel(128)) < 1e-9);
    sample(g, null, 0, tm.bounds_merc[2], tm.bounds_merc[1], tm, out);   // exact SE corner
    chk('SE corner clamps to texel (7,5)', out.valid && Math.abs(out.depth - decodeDepth(40 + 70 + 5)) < 1e-9);
    sample(g, null, 0, tm.bounds_merc[0] - 0.001, cy(0), tm, out);
    chk('1 mm west of the grid is invalid and all zero', !out.valid && out.depth === 0 && out.u === 0 && out.speed === 0);
    sample(g, null, 0, cx(0), tm.bounds_merc[3] + 0.001, tm, out);
    chk('1 mm north of the grid is invalid', !out.valid);
    sample(g, null, 0, cx(3), cy(2), tm, out);
    chk('centre of the A=0 texel is invalid', !out.valid && out.depth === 0);
    sample(g, null, 0, cx(3) + 0.25 * cell, cy(2), tm, out);
    chk('a quarter texel from the hole towards its neighbour is still invalid (A=64)', !out.valid && out.depth === 0);
    sample(g, null, 0, cx(3) + 0.75 * cell, cy(2), tm, out);
    chk('three quarters of the way to the valid neighbour is valid (A=191)', out.valid && out.depth > 0);
    sample(g, null, 0, cx(3) - 0.75 * cell, cy(2), tm, out);
    chk('same on the other side of the hole', out.valid);
    sample(null, g, 1, cx(1), cy(1), tm, out);
    chk('A null, B real, mix 1 reads B', out.valid && Math.abs(out.depth - decodeDepth(40 + 10 + 1)) < 1e-9);
    sample(null, null, 0.5, cx(1), cy(1), tm, out);
    chk('both null: invalid, zero', !out.valid && out.depth === 0);
    sample(g, null, 0.5, cx(1), cy(1), tm, out);
    chk('B null at mix 0.5 halves the depth of A', Math.abs(out.depth - 0.5 * decodeDepth(51)) < 1e-9);
    sample(g, g, NaN, cx(1), cy(1), tm, out);
    chk('NaN mix is treated as 0', Math.abs(out.depth - decodeDepth(51)) < 1e-9);
    const z = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) z[i * 4 + 3] = 255;   // R=G=B=0 (G=0 would decode to -4 m/s), A=255
    sample(z, z, 0.3, cx(2), cy(2), tm, out);
    chk('zero-code velocity bytes with R=0 give no phantom velocity', out.valid && out.depth === 0 && out.u === 0 && out.v === 0 && out.speed === 0);
    const bad = checks.filter((c) => !c[1]).map((c) => c[0]);
    return { pass: bad.length === 0, detail: bad.length ? `failed: ${bad.join('; ')}` : `${checks.length} checks` };
  });

  await test('lngLatToMerc / mercToLngLat: known values and round trip', async () => {
    const o = lngLatToMerc(0, 0), e = lngLatToMerc(180, 0), bg = lngLatToMerc(-86.4417, 36.9903);
    const back = mercToLngLat(bg[0], bg[1]);
    const pole = lngLatToMerc(0, 90);
    const ok = Math.abs(o[0]) < 1e-9 && Math.abs(o[1]) < 1e-6 && Math.abs(e[0] - 20037508.342789244) < 1e-3
      && Math.abs(back[0] + 86.4417) < 1e-9 && Math.abs(back[1] - 36.9903) < 1e-9 && Number.isFinite(pole[1]);
    return { pass: ok, detail: `BG = ${bg[0].toFixed(1)}, ${bg[1].toFixed(1)}; round trip err ${Math.abs(back[1] - 36.9903).toExponential(1)} deg` };
  });
}

async function rendererTests(syn) {
  const meta = syn.meta;
  let gpuInfo = '';

  await test('FluidRenderer.supported() and GPU identity', async () => {
    const sup = FluidRenderer.supported();
    const r = makeRenderer(syn, { w: 64, h: 48 });
    const gl = r.gl;
    gpuInfo = `${r.gpu.vendor} / ${r.gpu.renderer}`;
    const info = `${gl.getParameter(gl.VERSION)}; MAX_TEXTURE_SIZE ${gl.getParameter(gl.MAX_TEXTURE_SIZE)}; trail buffer ${r.stats.trailFloat ? 'RGBA16F' : 'RGBA8'}; UNMASKED_RENDERER ${r.gpu.renderer}`;
    const attrs = gl.getContextAttributes();
    const ok = sup === true && attrs.preserveDrawingBuffer === true && attrs.premultipliedAlpha === true && attrs.alpha === true;
    r.dispose();
    return { pass: ok, detail: info };
  });

  await test('depth colour pixel-exact: palette, depth alpha, soft shoreline, nothing where dry or A=0 (all 786k pixels)', async () => {
    const r = makeRenderer(syn, { options: QUIET });
    r.renderOnce();
    const img = r.readPixels();
    const A = syn.rgba[2], B = syn.rgba[3], m = 0.37;
    let bad = 0, worst = 0, wet = 0, outsideNonZero = 0, outside = 0, dryNonZero = 0;
    for (let i = 0; i < GW * GH; i++) {
      const o = i * 4;
      const dA = decodeDepth(A[o]), dB = decodeDepth(B[o]);
      const depth = dA * (1 - m) + dB * m;
      const valid = Math.max(A[o + 3], B[o + 3]) / 255;
      const shore = smoothstep(0.02, 0.10, depth);
      const alpha = lerp(0.42, 0.92, smoothstep(0, 1, depth)) * shore * valid;
      const col = paletteColor(depth);
      const exp = [col[0] * alpha, col[1] * alpha, col[2] * alpha, 255 * alpha];
      let e = 0;
      for (let k = 0; k < 3; k++) e = Math.max(e, Math.abs(img[o + k] - exp[k]));
      e = Math.max(e, Math.abs(img[o + 3] - exp[3]));
      if (e > worst) worst = e;
      if (e > 4) bad++;
      if (alpha > 0.3) wet++;
      if (valid === 0) { outside++; if (img[o + 3] !== 0) outsideNonZero++; }
      else if (depth <= 0.02 && img[o + 3] !== 0) dryNonZero++;
    }
    r.dispose();
    return { pass: bad <= 200 && wet > 20000 && outsideNonZero === 0 && dryNonZero === 0, detail: `wet px ${wet}, outside-domain px ${outside} (non-transparent ${outsideNonZero}), dry non-transparent ${dryNonZero}, max error ${worst.toFixed(1)}/255, px over 4/255: ${bad}` };
  });

  await test('palette stops: exact colour at 0.1 / 0.3 / 0.6 / 1.2 m (flat synthetic grid)', async () => {
    const w = 4, h = 4;
    const tm = { ...meta, width: w, height: h, cell_m: CELL, bounds_merc: [GRID_MERC[0], GRID_MERC[1], GRID_MERC[0] + w * CELL, GRID_MERC[1] + h * CELL] };
    const lines = [];
    let ok = true;
    for (const [d, idx] of [[0.1, 0], [0.3, 1], [0.6, 2], [1.2, 3]]) {
      const g = new Uint8Array(w * h * 4);
      const R = Math.round(255 * Math.sqrt(d / 8));
      for (let i = 0; i < w * h; i++) { g[i * 4] = R; g[i * 4 + 1] = 128; g[i * 4 + 2] = 128; g[i * 4 + 3] = 255; }
      const r = new FluidRenderer(QUIET);
      r.setStorm(tm); r.setHours(g, g, 0); r.setRegion(tm.bounds_merc, 4, 4);
      r.renderOnce();
      const p = r.readPixel(1, 1);
      const a = p[3] / 255;
      const un = [p[0] / a, p[1] / a, p[2] / a];
      const dd = decodeDepth(R);
      const exp = paletteColor(dd);
      const e = Math.max(...un.map((v, k) => Math.abs(v - exp[k])));
      lines.push(`${d} m (code depth ${dd.toFixed(3)}): rgb ${un.map((v) => v.toFixed(0))} vs ${exp.map((v) => v.toFixed(0))}, alpha ${a.toFixed(2)}`);
      if (e > 3) ok = false;
      r.dispose();
    }
    return { pass: ok, detail: lines.join(' | ') };
  });

  await test('orientation: designed features sit where the data says (pond NW, basin E) and mirrored spots are empty', async () => {
    const r = makeRenderer(syn, { options: QUIET, mix: 0 });
    r.renderOnce();
    const img = r.readPixels();
    const tr = syn.truths[2];
    const toPx = (X, Y) => [Math.floor((X + GW * CELL / 2) / CELL), Math.floor((GH * CELL / 2 - Y) / CELL)];
    const at = (X, Y) => { const [c, rw] = toPx(X, Y); return [px(img, GW, c, rw), c, rw]; };
    const lines = [];
    let ok = true;
    for (const [name, X, Y] of [['pond', POND.x, POND.y], ['basin', BASIN.x, BASIN.y], ['mound', MOUND.x, MOUND.y]]) {
      const [p, c, rw] = at(X, Y);
      const truthD = tr.depth[rw * GW + c];
      const wet = p[3] > 120 && p[2] > p[0] && truthD > 0.2;
      lines.push(`${name}@(col ${c},row ${rw}) alpha ${p[3]} depth ${truthD.toFixed(2)}`);
      if (!wet) ok = false;
      for (const [mx, my, label] of [[-X, Y, 'mirror-x'], [X, -Y, 'mirror-y'], [-X, -Y, 'mirror-xy']]) {
        const [pm, cm, rm] = at(mx, my);
        const td = tr.depth[rm * GW + cm];
        if (td < 0.005) {
          if (pm[3] !== 0) { ok = false; lines.push(`${name} ${label} should be empty but alpha=${pm[3]}`); }
        }
      }
    }
    r.dispose();
    return { pass: ok, detail: lines.join('; ') };
  });

  await test('particles advect with the flow: east in the east-flowing reach, north in the north-flowing reach', async () => {
    const r = makeRenderer(syn, { options: { showArrows: false, showDepth: false, showFlow: false, particleCount: 30000, particleSpeed: 40, seed: 7 }, mix: 0.5 });
    let t = stepFrames(r, 4);
    const p0 = r.getParticles();
    const n = p0.count;
    const x0 = Float64Array.from(p0.x), y0 = Float64Array.from(p0.y), a0 = Float32Array.from(p0.age);
    t = stepFrames(r, 45, t);
    const p1 = r.getParticles();
    const out = {};
    const E = { n: 0, dx: 0, dy: 0, right: 0 }, N = { n: 0, dx: 0, dy: 0, right: 0 };
    let moved = 0, dotPos = 0, dotN = 0;
    for (let i = 0; i < n; i++) {
      if (a0[i] < 0 || p1.age[i] < 0 || p1.age[i] < a0[i]) continue;
      const dx = p1.x[i] - x0[i], dy = p1.y[i] - y0[i];
      if (Math.hypot(dx, dy) > 250) continue;   // respawned elsewhere
      sample(syn.rgba[2], syn.rgba[3], 0.5, x0[i], y0[i], meta, out);
      if (out.speed < 0.2) continue;
      moved++;
      const dot = (dx * out.u + dy * out.v) / (Math.hypot(dx, dy) * out.speed + 1e-12);
      dotN++; if (dot > 0.7) dotPos++;
      if (out.u > 1.0 && Math.abs(out.v) < 0.3 * out.u) { E.n++; E.dx += dx; E.dy += dy; if (dx > 0) E.right++; }
      if (out.v > 1.0 && Math.abs(out.u) < 0.3 * out.v) { N.n++; N.dx += dx; N.dy += dy; if (dy > 0) N.right++; }
    }
    r.dispose();
    const meanE = [E.dx / Math.max(E.n, 1), E.dy / Math.max(E.n, 1)], meanN = [N.dx / Math.max(N.n, 1), N.dy / Math.max(N.n, 1)];
    const ok = E.n >= 30 && N.n >= 30 && meanE[0] > 15 && Math.abs(meanE[1]) < 0.6 * meanE[0] && E.right / E.n > 0.95
      && meanN[1] > 15 && Math.abs(meanN[0]) < 0.6 * meanN[1] && N.right / N.n > 0.95 && dotPos / dotN > 0.9;
    return { pass: ok, detail: `east reach: ${E.n} particles, mean move (${meanE[0].toFixed(1)}, ${meanE[1].toFixed(1)}) m, ${(100 * E.right / Math.max(E.n, 1)).toFixed(0)}% eastward; north reach: ${N.n} particles, mean move (${meanN[0].toFixed(1)}, ${meanN[1].toFixed(1)}) m, ${(100 * N.right / Math.max(N.n, 1)).toFixed(0)}% northward; ${(100 * dotPos / Math.max(dotN, 1)).toFixed(0)}% of ${dotN} within 45 deg of the sampled flow` };
  });

  await test('particle step is exact: one frame moves a particle by sampled velocity x particleSpeed / cos(lat) x dt', async () => {
    const r = makeRenderer(syn, { options: { showArrows: false, showDepth: false, showFlow: false, particleCount: 20000, particleSpeed: 40, particleBudgetMs: 1000, seed: 13 }, mix: 0.5 });
    const t = stepFrames(r, 5);
    const p0 = r.getParticles();
    const x0 = Float64Array.from(p0.x), y0 = Float64Array.from(p0.y), a0 = Float32Array.from(p0.age);
    r.frame(t);   // exactly one more frame of 1/60 s
    const p1 = r.getParticles();
    const yMid = (meta.bounds_merc[1] + meta.bounds_merc[3]) / 2;
    const merc = 1 / Math.cos(2 * Math.atan(Math.exp(yMid / 6378137)) - Math.PI / 2);
    const out = {};
    let n = 0, worst = 0, fastN = 0, sumRatio = 0;
    for (let i = 0; i < p0.count; i++) {
      if (a0[i] < 0 || Math.abs(p1.age[i] - a0[i] - 1 / 60) > 1e-4) continue;   // skip particles that respawned
      sample(syn.rgba[2], syn.rgba[3], 0.5, x0[i], y0[i], meta, out);
      const px = out.u * 40 * merc / 60, py = out.v * 40 * merc / 60;
      worst = Math.max(worst, Math.abs(p1.x[i] - x0[i] - px), Math.abs(p1.y[i] - y0[i] - py));
      n++;
      if (out.speed > 1) { fastN++; sumRatio += Math.hypot(p1.x[i] - x0[i], p1.y[i] - y0[i]) / (out.speed * 40 / 60); }
    }
    r.dispose();
    return { pass: n > 5000 && fastN > 100 && worst < 1e-5 && Math.abs(sumRatio / fastN - merc) < 1e-5, detail: `${n} particles stepped once, worst deviation from the predicted move ${worst.toExponential(2)} m; mean (moved metres / (speed x 40 x dt)) = ${(sumRatio / fastN).toFixed(5)} vs 1/cos(lat) = ${merc.toFixed(5)}` };
  });

  await test('particles: only spawn on wet cells, respawn when dry/outside/old, speed scales with particleSpeed', async () => {
    const r = makeRenderer(syn, { options: { showArrows: false, showDepth: false, showFlow: false, particleCount: 8000, particleSpeed: 40, particleLife: 1.0, seed: 3, particleBudgetMs: 1000 }, mix: 0.5 });
    let t = stepFrames(r, 3);
    const out = {};
    let dry = 0, alive = 0;
    // A particle that drifted onto dry ground is only noticed when it is next sampled, so a
    // frame with dt = 0 (nothing moves, every particle is sampled) must leave none on dry cells.
    const check = () => {
      r.frame(t - 1000 / 60);
      const p = r.getParticles();
      for (let i = 0; i < p.count; i++) {
        if (p.age[i] < 0) continue;
        alive++;
        sample(syn.rgba[2], syn.rgba[3], 0.5, p.x[i], p.y[i], meta, out);
        if (!out.valid || out.depth < 0.02) dry++;
      }
    };
    check();
    t = stepFrames(r, 120, t);     // 2 s with a 1 s life: everything has been respawned at least once
    check();
    const p = r.getParticles();
    let maxAge = 0, aliveNow = 0;
    for (let i = 0; i < p.count; i++) if (p.age[i] >= 0) { aliveNow++; maxAge = Math.max(maxAge, p.age[i]); }
    // speed scaling: displacement over 30 frames with particleSpeed 40 vs 0
    const disp = (ps) => {
      const rr = makeRenderer(syn, { options: { showArrows: false, showDepth: false, showFlow: false, particleCount: 3000, particleSpeed: ps, particleLife: 30, seed: 5 }, mix: 0.5 });
      let tt = stepFrames(rr, 3);
      const q = rr.getParticles(); const xs = Float64Array.from(q.x), ys = Float64Array.from(q.y), ag = Float32Array.from(q.age);
      stepFrames(rr, 30, tt);
      const q2 = rr.getParticles();
      let sum = 0, cnt = 0;
      for (let i = 0; i < q.count; i++) if (ag[i] >= 0 && q2.age[i] > ag[i]) { sum += Math.hypot(q2.x[i] - xs[i], q2.y[i] - ys[i]); cnt++; }
      rr.dispose();
      return sum / Math.max(cnt, 1);
    };
    const d40 = disp(40), d0 = disp(0), d80 = disp(80);
    r.dispose();
    const ok = alive > 5000 && dry === 0 && maxAge <= 1.0 * 1.31 && aliveNow > 5000 && d0 === 0 && d40 > 5 && Math.abs(d80 / d40 - 2) < 0.05;
    return { pass: ok, detail: `alive ${alive}, on dry/outside ${dry}, max age ${maxAge.toFixed(2)} s (life 1 s +-30%), mean move/30 frames: speed0 ${d0.toFixed(2)} m, speed40 ${d40.toFixed(1)} m, speed80 ${d80.toFixed(1)} m (ratio ${(d80 / d40).toFixed(3)})` };
  });

  await test('particle coverage cap: fewer active particles when zoomed in, linear in particleCoverage, trails stay readable', async () => {
    const wide = gridRegion();
    const close = viewRegion(-860, -1130, 240, GW / GH);
    const mk = (region, cov) => makeRenderer(syn, { options: { showDepth: false, showFlow: false, showArrows: false, particleCount: 60000, particleCoverage: cov, seed: 21, particleBudgetMs: 1000 }, mix: 0.5, region });
    const capOf = (region, cov) => { const r = mk(region, cov); stepFrames(r, 25); const c = r.stats.particleCap; r.dispose(); return c; };
    const off = capOf(close, 0), cWide = capOf(wide, 0.6), cClose = capOf(close, 0.6), cClose2 = capOf(close, 1.2), cClose3 = capOf(close, 0.3);
    // ink saturation of the trail buffer on the particle-only picture after the trails have built up
    const ink = (region, cov) => {
      const r = mk(region, cov);
      stepFrames(r, 200); r.renderOnce();
      const img = r.readPixels();
      let covered = 0, dense = 0;
      for (let i = 3; i < img.length; i += 4) { if (img[i] > 12) covered++; if (img[i] > 230) dense++; }
      r.dispose();
      return { covered, dense, frac: dense / Math.max(covered, 1) };
    };
    const capped = ink(close, 0.6), uncapped = ink(close, 0);
    const ok = off === 60000 && cClose >= 256 && cClose < cWide && cWide <= 60000
      && Math.abs(cClose2 / cClose - 2) < 0.35 && Math.abs(cClose3 / cClose - 0.5) < 0.2
      && capped.covered > 20000 && capped.frac < 0.12 && uncapped.frac > 3 * capped.frac;
    return { pass: ok, detail: `active particles (60000 allowed): cap off ${off}; wide view ${cWide}; close view ${cClose} (coverage 0.3 -> ${cClose3}, 1.2 -> ${cClose2}); close-up trail pixels that are >90% opaque, among covered pixels: capped ${(100 * capped.frac).toFixed(1)}% of ${capped.covered}, uncapped ${(100 * uncapped.frac).toFixed(1)}% of ${uncapped.covered}` };
  });

  await test('particle budget: when four slices cannot fit particleBudgetMs the active count is shed and the pass stays near the budget', async () => {
    const run = (budgetMs) => {
      const r = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showArrows: false, particleCount: 60000, particleBudgetMs: budgetMs, seed: 31 }, mix: 0.5 });
      const ms = [];
      let t = 0;
      for (let i = 0; i < 300; i++) { r.frame(t); t += 1000 / 60; if (i >= 150) ms.push(r.stats.particleMs); if (i % 6 === 5) r.sync(); }
      const out = { cap: r.stats.particleCap, slices: r.stats.particleSlices, avg: ms.reduce((a, b) => a + b, 0) / ms.length, drawnLast: r.stats.particlesDrawn };
      r.dispose();
      return out;
    };
    const loose = run(1000), tight = run(1);
    return { pass: loose.cap === 60000 && loose.slices === 1 && tight.cap < 40000 && tight.cap >= 256 && tight.slices === 4 && tight.avg < 1.8, detail: `budget 1000 ms: ${loose.cap} active, ${loose.slices} slice, pass ${loose.avg.toFixed(2)} ms/frame; budget 1 ms: ${tight.cap} active, ${tight.slices} slices, pass ${tight.avg.toFixed(2)} ms/frame` };
  });

  await test('water surface pattern streams downstream: measured drift east in the east reach, north in the north reach', async () => {
    const lines = [];
    let ok = true;
    const measure = async (label, cx, cy, probeX, probeY, expectDir) => {
      const region = viewRegion(cx, cy, 900, GW / GH);
      const r = makeRenderer(syn, { options: { showParticles: false, showArrows: false, particleCount: 0 }, mix: 0.5, region });
      const lum = () => {
        r.renderOnce();
        const img = r.readPixels();
        const l = new Float32Array(GW * GH);
        for (let i = 0; i < GW * GH; i++) l[i] = (img[i * 4] + img[i * 4 + 1] + img[i * 4 + 2]) / 3;
        return l;
      };
      r.frame(0);
      const tA = stepFrames(r, 4, 0);
      const l0 = lum();
      const dtS = 0.2;
      stepFrames(r, Math.round(dtS * 60), tA);
      const l1 = lum();
      const mp = localToMerc(probeX, probeY);
      const s = sample(syn.rgba[2], syn.rgba[3], 0.5, mp[0], mp[1], meta);
      const pc = r.worldToPixel(mp[0], mp[1]);
      const px0 = Math.round(pc[0]), py0 = Math.round(pc[1]);
      const hw = expectDir === 'east' ? 48 : 12, hh = expectDir === 'east' ? 12 : 48;
      let best = -2, bdx = 0, bdy = 0;
      for (let dy = -30; dy <= 30; dy++) {
        for (let dx = -30; dx <= 30; dx++) {
          let sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0, n = 0;
          for (let y = py0 - hh; y < py0 + hh; y++) {
            for (let x = px0 - hw; x < px0 + hw; x++) {
              const a = l0[y * GW + x], b = l1[(y + dy) * GW + x + dx];
              sa += a; sb += b; sab += a * b; saa += a * a; sbb += b * b; n++;
            }
          }
          const c = (sab - sa * sb / n) / Math.sqrt(Math.max((saa - sa * sa / n) * (sbb - sb * sb / n), 1e-9));
          if (c > best) { best = c; bdx = dx; bdy = dy; }
        }
      }
      const mpp = (region[2] - region[0]) / GW;
      const merc = 1 / Math.cos(36.9903 * Math.PI / 180);
      const expPx = (expectDir === 'east' ? s.u : s.v) * merc * r.options.flowSpeed * dtS / mpp;
      const got = expectDir === 'east' ? bdx : -bdy;
      const across = expectDir === 'east' ? bdy : bdx;
      const pass = best > 0.5 && got > 0 && got > 0.5 * expPx && got < 1.6 * expPx && Math.abs(across) <= 0.25 * got + 1.5;
      if (!pass) ok = false;
      lines.push(`${label}: flow ${s.u.toFixed(2)},${s.v.toFixed(2)} m/s -> pattern moved (${bdx}, ${bdy}) px in ${dtS} s (corr ${best.toFixed(2)}), expected about ${expPx.toFixed(1)} px ${expectDir === 'east' ? 'right' : 'up'}: ${pass ? 'ok' : 'FAIL'}`);
      r.dispose();
    };
    await measure('east reach', -900, -1050, -900, -1150, 'east');
    await measure('north reach', 1300, 0, 1303, 0, 'north');
    return { pass: ok, detail: lines.join(' | ') };
  });

  await test('composite is valid premultiplied alpha; water layers draw nothing where A=0 (arrow tips and streak caps may poke out)', async () => {
    const r = makeRenderer(syn, { options: { particleCount: 30000, showFlowpaths: true, seed: 5 }, mix: 0.4 });
    stepFrames(r, 90);
    const A = syn.rgba[2], B = syn.rgba[3];
    const scan = () => {
      r.renderOnce();
      const img = r.readPixels();
      let bad = 0, outside = 0;
      for (let i = 0; i < GW * GH; i++) {
        const o = i * 4;
        if (Math.max(img[o], img[o + 1], img[o + 2]) > img[o + 3] + 1) bad++;
        if (A[o + 3] === 0 && B[o + 3] === 0 && img[o + 3] !== 0) outside++;
      }
      return [bad, outside];
    };
    const [badAll, outAll] = scan();
    r.setOptions({ showParticles: false, showArrows: false });
    const [badW, outW] = scan();
    r.dispose();
    return { pass: badAll === 0 && badW === 0 && outW === 0 && outAll < GW * GH * 0.0005, detail: `all layers: ${badAll} px with rgb > alpha, ${outAll} px outside the domain (arrow glyphs and streak caps at the boundary); water + flow-paths only: ${badW} bad, ${outW} outside` };
  });

  await test('A=0 texels are never drawn, even when R holds a depth (water, arrows, particles, flow paths)', async () => {
    // The format writes R=0 wherever A=0, so this is deliberately contract-violating data: it checks that the
    // renderer itself honours the domain flag instead of relying on the builder.
    const [c0, c1, r0, r1] = [250, 420, 540, 690];   // texel box around the east reach of the stream
    const punch = (src) => {
      const g = new Uint8Array(src);
      let wet = 0;
      for (let row = r0; row < r1; row++) for (let col = c0; col < c1; col++) { const o = (row * GW + col) * 4; if (g[o] > 0) wet++; g[o + 3] = 0; }
      return [g, wet];
    };
    const [gA, wetA] = punch(syn.rgba[2]), [gB] = punch(syn.rgba[3]), [gF] = punch(syn.flow);
    const inset = (m) => ({ x0: c0 + m, x1: c1 - m, y0: r0 + m, y1: r1 - m });
    const empty = (img, b) => { let n = 0; for (let y = b.y0; y < b.y1; y++) for (let x = b.x0; x < b.x1; x++) if (img[(y * GW + x) * 4 + 3] !== 0) n++; return n; };
    // water + surface + flow paths + particles; streak ends may overshoot a few px, so look 8 px inside the box
    const r = makeRenderer(syn, { options: { showArrows: false, showFlowpaths: true, particleCount: 30000, seed: 17 }, a: gA, b: gB, fp: gF, mix: 0.5 });
    const tEnd = stepFrames(r, 120);
    r.frame(tEnd - 1000 / 60);   // same time as the last frame: dt = 0, every particle is sampled where it stands
    r.renderOnce();
    const nWater = empty(r.readPixels(), inset(8));
    const live = r.getParticles();
    const bx = (c) => GRID_MERC[0] + c * CELL, by = (rw) => GRID_MERC[3] - rw * CELL;
    let inside = 0, alive = 0;
    for (let i = 0; i < live.count; i++) {
      if (live.age[i] < 0) continue;
      alive++;
      if (live.x[i] > bx(c0 + 3) && live.x[i] < bx(c1 - 3) && live.y[i] < by(r0 + 3) && live.y[i] > by(r1 - 3)) inside++;
    }
    r.dispose();
    // arrows: glyphs reach about 24 px from their centre, so look 24 px inside
    const ra = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showParticles: false, showArrows: true }, a: gA, b: gB, mix: 0.5 });
    ra.renderOnce();
    const nArrow = empty(ra.readPixels(), inset(24));
    const arrowsElsewhere = (() => { const img = ra.readPixels(); let n = 0; for (let i = 3; i < img.length; i += 4) if (img[i] > 0) n++; return n; })();
    ra.dispose();
    return { pass: wetA > 3000 && nWater === 0 && inside === 0 && alive > 2000 && nArrow === 0 && arrowsElsewhere > 3000, detail: `box holds ${wetA} wet texels with A=0; opaque px inside the box (water+particles+flow paths) ${nWater}; live particles inside ${inside} of ${alive}; arrow px inside ${nArrow} (arrow px elsewhere ${arrowsElsewhere}, so arrows do draw)` };
  });

  await test('no seams: rendering the left and right halves of a region separately equals the full render', async () => {
    const full = gridRegion();
    const mid = (full[0] + full[2]) / 2;
    const opt = { showParticles: false, showArrows: false, showFlowpaths: false, particleCount: 0 };
    const rf = makeRenderer(syn, { options: opt, mix: 0.4, region: full });
    stepFrames(rf, 7, 0); rf.renderOnce();
    const f = rf.readPixels();
    const rl = makeRenderer(syn, { options: opt, mix: 0.4, region: [full[0], full[1], mid, full[3]], w: GW / 2, h: GH });
    stepFrames(rl, 7, 0); rl.renderOnce();
    const l = rl.readPixels();
    const rr = makeRenderer(syn, { options: opt, mix: 0.4, region: [mid, full[1], full[2], full[3]], w: GW / 2, h: GH });
    stepFrames(rr, 7, 0); rr.renderOnce();
    const rgt = rr.readPixels();
    let worst = 0, bad = 0;
    const hw = GW / 2;
    for (let y = 0; y < GH; y++) {
      for (let x = 0; x < hw; x++) {
        for (const [half, off] of [[l, 0], [rgt, hw]]) {
          const o = (y * hw + x) * 4, of = (y * GW + x + off) * 4;
          for (let k = 0; k < 4; k++) { const e = Math.abs(half[o + k] - f[of + k]); if (e > worst) worst = e; if (e > 4) bad++; }
        }
      }
    }
    rf.dispose(); rl.dispose(); rr.dispose();
    return { pass: bad === 0, detail: `max difference ${worst}/255 over ${GW * GH * 4} channel values (only the screen-space dither may differ), ${bad} over 4` };
  });

  await test('region partly outside the grid: outside is transparent, inside is unchanged', async () => {
    const full = gridRegion();
    const shifted = [full[0] - 1280, full[1], full[2] - 1280, full[3]];     // 256 px at 5 m/px hang over the west edge
    const opt = { showParticles: false, showArrows: false, showFlowpaths: false, particleCount: 0 };
    const rf = makeRenderer(syn, { options: opt, mix: 0.4, region: full });
    stepFrames(rf, 3, 0); rf.renderOnce(); const f = rf.readPixels();
    const rs = makeRenderer(syn, { options: opt, mix: 0.4, region: shifted });
    stepFrames(rs, 3, 0); rs.renderOnce(); const sft = rs.readPixels();
    let outsideBad = 0, insideBad = 0;
    for (let y = 0; y < GH; y++) {
      for (let x = 0; x < GW; x++) {
        const o = (y * GW + x) * 4;
        if (x < 256) { if (sft[o + 3] !== 0) outsideBad++; } else {
          const of = (y * GW + (x - 256)) * 4;
          for (let k = 0; k < 4; k++) if (Math.abs(sft[o + k] - f[of + k]) > 4) { insideBad++; break; }
        }
      }
    }
    rf.dispose(); rs.dispose();
    return { pass: outsideBad === 0 && insideBad === 0, detail: `${outsideBad} px drawn beyond the west edge, ${insideBad} px inside that differ from the unshifted render` };
  });

  await test('arrow orientation: glyph heads point along the sampled flow (east reach right, north reach up)', async () => {
    const r = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showParticles: false, showFlowpaths: false, showArrows: true, arrowSpacingPx: 40 }, mix: 0.5 });
    r.renderOnce();
    const img = r.readPixels();
    const S = 40, nx = Math.ceil(GW / S), ny = Math.ceil(GH / S);
    const out = {};
    let tested = 0, good = 0, eastN = 0, eastOk = 0, northN = 0, northOk = 0, hidden = 0, shownBad = 0;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const cxp = (ix + 0.5) * S, cyp = (iy + 0.5) * S;
        const wx = GRID_MERC[0] + cxp * CELL, wy = GRID_MERC[3] - cyp * CELL;
        sample(syn.rgba[2], syn.rgba[3], 0.5, wx, wy, meta, out);
        // fill pixels of this cell
        let sx = 0, sy = 0, cnt = 0;
        for (let y = iy * S; y < Math.min((iy + 1) * S, GH); y++) {
          for (let x = ix * S; x < Math.min((ix + 1) * S, GW); x++) {
            const o = (y * GW + x) * 4, a = img[o + 3];
            if (a > 200 && (img[o] + img[o + 1] + img[o + 2]) / (3 * a) > 0.6) { sx += x + 0.5 - cxp; sy += y + 0.5 - cyp; cnt++; }
          }
        }
        const mustHide = !out.valid || out.depth < 0.04 || out.speed < 0.02;
        if (mustHide) { hidden++; if (cnt > 0) shownBad++; continue; }
        // judge only clearly moving water: the glyph samples a few cells around its centre,
        // so slow, strongly curving flow may legitimately differ from the point sample
        if (!(out.valid && out.depth >= 0.1 && out.speed >= 0.5)) continue;
        tested++;
        if (cnt < 8) continue;
        const mx = sx / cnt, my = sy / cnt;
        const dirx = out.u, diry = -out.v, dl = Math.hypot(dirx, diry);
        const along = (mx * dirx + my * diry) / dl;
        const perp = Math.abs(-mx * diry + my * dirx) / dl;
        const L = S * clamp(0.2 + 0.52 * Math.sqrt(out.speed), 0.3, 0.92);   // glyph length rule of the shader
        if (along > 0.015 * L && perp < along) good++;
        if (out.u > 0.6 && Math.abs(out.v) < 0.35 * out.u && wx - CENTRE[0] < 300) { eastN++; if (mx > 0.5) eastOk++; }
        if (out.v > 0.6 && Math.abs(out.u) < 0.35 * out.v && Math.abs(wx - CENTRE[0] - 1300) < 200) { northN++; if (my < -0.5) northOk++; }
      }
    }
    r.dispose();
    const ok = tested >= 20 && good / tested > 0.95 && eastN >= 4 && eastOk === eastN && northN >= 4 && northOk === northN && shownBad === 0;
    return { pass: ok, detail: `${tested} arrows checked, ${good} head along flow (${(100 * good / tested).toFixed(0)}%); east reach ${eastOk}/${eastN} point right; north reach ${northOk}/${northN} point up; ${hidden} cells that must be hidden, ${shownBad} wrongly drawn` };
  });

  await test('arrow culling and spacing: arrow count follows depth/speed rules and arrowSpacingPx', async () => {
    const r = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showParticles: false, showFlowpaths: false, showArrows: true, arrowSpacingPx: 40 }, mix: 0.5 });
    const count = (S) => {
      r.setOptions({ arrowSpacingPx: S }); r.renderOnce();
      const img = r.readPixels();
      let n = 0; for (let i = 3; i < img.length; i += 4) if (img[i] > 0) n++;
      return n;
    };
    const n80 = count(80), n40 = count(40), n20 = count(20);
    r.dispose();
    return { pass: n40 > 5000 && n20 > n40 * 0.9 && n40 > n80 * 0.9, detail: `covered px at spacing 80/40/20: ${n80}/${n40}/${n20}` };
  });

  await test('flow-paths layer: bright channels where discharge is high, transparent elsewhere', async () => {
    const r = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showParticles: false, showArrows: false, showFlowpaths: true } });
    r.renderOnce();
    const img = r.readPixels();
    let strong = 0, covered = 0;
    for (let i = 0; i < GW * GH; i++) {
      const o = i * 4;
      if (img[o + 3] > 0) covered++;
      if (img[o + 3] > 150 && img[o + 2] > 200) strong++;
    }
    // stream centre line (east reach, x=-900) must be strong; far dry corner must be empty
    const X = -900, Y = -1150;   // control point of the stream
    const c = Math.floor((X + GW * CELL / 2) / CELL), rw = Math.floor((GH * CELL / 2 - Y) / CELL);
    let best = 0;
    for (let dy = -6; dy <= 6; dy++) best = Math.max(best, px(img, GW, c, rw + dy)[3]);
    const corner = px(img, GW, 10, 10);
    r.dispose();
    return { pass: strong > 800 && best > 150 && corner[3] === 0 && covered < GW * GH * 0.5, detail: `strong px ${strong}, covered ${covered}, stream alpha ${best}, corner alpha ${corner[3]}` };
  });

  await test('toggling every option works at run time and never recompiles shaders', async () => {
    const zoom = viewRegion(-900, -1050, 900, GW / GH);   // metres per pixel well below one ripple wavelength
    const r = makeRenderer(syn, { options: { particleCount: 20000, seed: 11 }, mix: 0.5, region: zoom });
    const compiles0 = r.stats.shaderCompiles;
    const tEnd = stepFrames(r, 50);
    const W = GW * GH;
    const shot = () => { r.renderOnce(); return r.readPixels(); };
    const base = shot();
    const lines = [];
    let ok = true;
    const expectDiff = (label, opts, minFrac, extra) => {
      const prev = {};
      for (const k of Object.keys(opts)) prev[k] = r.options[k];
      r.setOptions(opts);
      const img = shot();
      const n = diffCount(base, img);
      r.setOptions(prev);
      let pass = n / W >= minFrac;
      let note = `${(100 * n / W).toFixed(2)}% px changed`;
      if (extra) { const e = extra(img); pass = pass && e.pass; note += `, ${e.note}`; }
      if (!pass) ok = false;
      lines.push(`${label}: ${pass ? 'ok' : 'FAIL'} (${note})`);
    };
    expectDiff('showDepth=false', { showDepth: false }, 0.02, (img) => ({ pass: alphaSum(img) < alphaSum(base), note: 'alpha sum dropped' }));
    expectDiff('showFlow=false', { showFlow: false }, 0.01);
    expectDiff('showArrows=false', { showArrows: false }, 0.004);
    expectDiff('showFlowpaths=true', { showFlowpaths: true }, 0.001);
    expectDiff('opacity=0.4', { opacity: 0.4 }, 0.02, (img) => ({ pass: alphaSum(img) < alphaSum(base), note: 'alpha sum dropped' }));
    expectDiff('flowSpeed=0', { flowSpeed: 0 }, 0.005);
    expectDiff('sunAzimuthDeg=45', { sunAzimuthDeg: 45 }, 0.005);
    expectDiff('sunElevationDeg=20', { sunElevationDeg: 20 }, 0.005);
    expectDiff('minDepthShown=0.5', { minDepthShown: 0.5 }, 0.003, (img) => ({ pass: alphaSum(img) < alphaSum(base), note: 'thin water removed' }));
    expectDiff('arrowSpacingPx=100', { arrowSpacingPx: 100 }, 0.002);
    expectDiff('rippleStrength=0', { rippleStrength: 0 }, 0.005);
    expectDiff('quality=0', { quality: 0 }, 0.003);
    expectDiff('rippleScalePx=40', { rippleScalePx: 40 }, 0.003);
    expectDiff('flowCycleSec=7', { flowCycleSec: 7 }, 0.0005);
    expectDiff('arrowScale=0.5', { arrowScale: 0.5 }, 0.001);
    expectDiff('flowpathsOpacity (with flowpaths on)', { showFlowpaths: true, flowpathsOpacity: 0.3 }, 0.001);
    expectDiff('viewPitchDeg=70', { viewPitchDeg: 70 }, 0.01);
    // numeric particle options
    r.setOptions({ particleCount: 1000 }); r.frame(tEnd + 16);
    const drawn1k = r.stats.particlesDrawn;
    // budget off here: with the default budget the active count is shed to what the CPU can advect (tested separately)
    r.setOptions({ particleCount: 200000, particleBudgetMs: 1000 }); stepFrames(r, 40, tEnd + 32);
    const drawn200k = r.stats.particlesDrawn;
    const okCount = drawn1k <= 1000 && drawn200k > 20000 && r.getParticles().count === 200000 && r.stats.particleCap === 200000;
    r.setOptions({ particleCount: 5000, particleBudgetMs: 6 }); const p5 = r.getParticles().count === 5000;
    lines.push(`particleCount 1000 -> drawn ${drawn1k}; 200000 -> drawn ${drawn200k} (array ${200000}); 5000 resize ${p5}: ${okCount && p5 ? 'ok' : 'FAIL'}`);
    if (!(okCount && p5)) ok = false;
    // trailFade: stronger fade -> less trail energy after the same frames
    const energy = (fade) => { const rr = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showArrows: false, particleCount: 20000, seed: 11, trailFade: fade }, mix: 0.5 }); stepFrames(rr, 90); rr.renderOnce(); const a = alphaSum(rr.readPixels()); rr.dispose(); return a; };
    const widthEnergy = (wpx) => { const rr = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showArrows: false, particleCount: 20000, seed: 11, particleWidthPx: wpx }, mix: 0.5 }); stepFrames(rr, 60); rr.renderOnce(); const a = alphaSum(rr.readPixels()); rr.dispose(); return a; };
    const alphaEnergy = (al) => { const rr = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showArrows: false, particleCount: 20000, seed: 11, particleAlpha: al }, mix: 0.5 }); stepFrames(rr, 60); rr.renderOnce(); const a = alphaSum(rr.readPixels()); rr.dispose(); return a; };
    const aLow = alphaEnergy(0.2), aHigh = alphaEnergy(0.9);
    lines.push(`particleAlpha 0.2 -> ${aLow}, 0.9 -> ${aHigh}: ${aHigh > aLow * 1.15 ? 'ok' : 'FAIL'} (dense trails saturate, so the response is sub-linear)`);
    if (!(aHigh > aLow * 1.15)) ok = false;
    const wThin = widthEnergy(1), wThick = widthEnergy(4);
    lines.push(`particleWidthPx 1 -> ${wThin}, 4 -> ${wThick}: ${wThick > wThin * 1.5 ? 'ok' : 'FAIL'}`);
    if (!(wThick > wThin * 1.5)) ok = false;
    const eLow = energy(0.02), eHigh = energy(0.3);
    lines.push(`trailFade 0.02 -> ${eLow}, 0.3 -> ${eHigh}: ${eHigh < eLow * 0.7 ? 'ok' : 'FAIL'}`);
    if (!(eHigh < eLow * 0.7)) ok = false;
    // particleLife: short life -> all ages below it
    const rl = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showArrows: false, particleCount: 4000, particleLife: 0.3, seed: 2 }, mix: 0.5 });
    stepFrames(rl, 60);
    const pl = rl.getParticles(); let maxAge = 0; for (let i = 0; i < pl.count; i++) if (pl.age[i] > maxAge) maxAge = pl.age[i];
    rl.dispose();
    lines.push(`particleLife 0.3 -> max age ${maxAge.toFixed(3)} s: ${maxAge <= 0.3 * 1.3 + 0.02 ? 'ok' : 'FAIL'}`);
    if (!(maxAge <= 0.3 * 1.3 + 0.02)) ok = false;
    // showParticles off then on
    r.setOptions({ particleCount: 20000, showParticles: false });
    const offTrail = r.debugTrailMax();
    r.setOptions({ showParticles: true });
    const compiles1 = r.stats.shaderCompiles;
    const same = compiles1 === compiles0;
    lines.push(`showParticles=false clears trails (max alpha ${offTrail}); shader compiles ${compiles0} -> ${compiles1}: ${offTrail === 0 && same ? 'ok' : 'FAIL'}`);
    if (!(offTrail === 0 && same)) ok = false;
    // bad input
    let threw = false; try { r.setOptions({ opacity: 'abc' }); } catch (e) { threw = e instanceof TypeError; }
    lines.push(`non-numeric option throws TypeError: ${threw ? 'ok' : 'FAIL'}`);
    if (!threw) ok = false;
    r.dispose();
    return { pass: ok, detail: lines.join('; ') };
  });

  await test('setRegion: clears trails, keeps particle positions (Mercator metres), respawns the ones outside', async () => {
    const r = makeRenderer(syn, { options: { showArrows: false, particleCount: 10000, seed: 4 }, mix: 0.5 });
    let t = stepFrames(r, 40);
    const before = r.debugTrailMax();
    const p = r.getParticles();
    const xs = Float64Array.from(p.x), ys = Float64Array.from(p.y), ag = Float32Array.from(p.age);
    const newRegion = viewRegion(-900, -1050, 1200, GW / GH);
    r.setRegion(newRegion, 800, 600);
    const after = r.debugTrailMax();
    r.renderOnce();
    const q = r.getParticles();
    let same = 0, alive0 = 0;
    for (let i = 0; i < q.count; i++) if (ag[i] >= 0) { alive0++; if (q.x[i] === xs[i] && q.y[i] === ys[i]) same++; }
    t = stepFrames(r, 3, t);
    const q2 = r.getParticles();
    let outside = 0, alive1 = 0;
    for (let i = 0; i < q2.count; i++) {
      if (q2.age[i] < 0) continue;
      alive1++;
      if (q2.x[i] < newRegion[0] || q2.x[i] > newRegion[2] || q2.y[i] < newRegion[1] || q2.y[i] > newRegion[3]) outside++;
    }
    const sizeOk = r.canvas.width === 800 && r.canvas.height === 600;
    // a pan keeps the canvas size, so the trail texture is not re-created: setRegion itself must clear it
    stepFrames(r, 40, t + 1000);
    const beforePan = r.debugTrailMax();
    r.setRegion(viewRegion(-800, -1000, 1200, 800 / 600), 800, 600);
    const afterPan = r.debugTrailMax();
    r.dispose();
    return { pass: before > 0.5 && after === 0 && same === alive0 && outside === 0 && alive1 > 1000 && sizeOk && beforePan > 0.5 && afterPan === 0, detail: `trail alpha before ${before.toFixed(2)} -> after ${after}; positions kept ${same}/${alive0}; after 3 frames ${outside} of ${alive1} alive particles outside the new region; canvas ${r.canvas.width}x${r.canvas.height}; same-size pan: trail alpha ${beforePan.toFixed(2)} -> ${afterPan}` };
  });

  await test('setStorm with other bounds after setRegion: water, arrows and flow paths follow the new bounds (same picture as a fresh renderer)', async () => {
    const region = gridRegion();
    const shift = 2000;   // metres east = 400 px at 5 m/px: the grid now starts 400 px into the region
    const meta2 = { ...meta, bounds_merc: [meta.bounds_merc[0] + shift, meta.bounds_merc[1], meta.bounds_merc[2] + shift, meta.bounds_merc[3]] };
    const opt = { showParticles: false, showArrows: true, showFlowpaths: true, particleCount: 0 };
    const r = makeRenderer(syn, { options: opt, mix: 0.4, region });
    r.renderOnce(); const before = r.readPixels();
    // the order a viewer uses when it switches storms and the map has not moved: no setRegion in between
    r.setStorm(meta2); r.setHours(syn.rgba[2], syn.rgba[3], 0.4); r.setFlowpaths(syn.flow);
    r.renderOnce(); const switched = r.readPixels();
    const ref = makeRenderer(syn, { options: opt, meta: meta2, mix: 0.4, region });
    ref.renderOnce(); const fresh = ref.readPixels();
    let leftDrawn = 0;
    for (let y = 0; y < GH; y++) for (let x = 0; x < 360; x++) if (switched[(y * GW + x) * 4 + 3] !== 0) leftDrawn++;   // west of the new grid edge (400 px) minus the arrow glyph reach
    const moved = diffCount(before, switched), vsFresh = diffCount(switched, fresh);
    // the particles use the same bounds: every live one sits on water of the NEW storm
    const rp = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showArrows: false, particleCount: 8000, seed: 3, particleBudgetMs: 1000 }, mix: 0.5, region });
    stepFrames(rp, 3);
    rp.setStorm(meta2); rp.setHours(syn.rgba[2], syn.rgba[3], 0.5);
    stepFrames(rp, 30, 1000);
    const live = rp.getParticles(), out = {};
    let alive = 0, dry = 0;
    for (let i = 0; i < live.count; i++) {
      if (live.age[i] < 0) continue;
      alive++;
      sample(syn.rgba[2], syn.rgba[3], 0.5, live.x[i], live.y[i], meta2, out);
      if (!out.valid || out.depth < 0.02) dry++;
    }
    r.dispose(); ref.dispose(); rp.dispose();
    return { pass: leftDrawn === 0 && moved > GW * GH * 0.05 && vsFresh === 0 && alive > 2000 && dry === 0, detail: `${leftDrawn} px drawn west of the shifted grid, ${moved} px changed by the switch, ${vsFresh} px differ from a renderer that was given the new storm first; particles: ${dry} of ${alive} alive on dry/outside cells of the new storm` };
  });

  await test('particleCount 0 with showParticles on: the streaks left behind are cleared and do not stay frozen', async () => {
    const r = makeRenderer(syn, { options: { showDepth: false, showFlow: false, showArrows: false, particleCount: 20000, seed: 8 }, mix: 0.5, w: 512, h: 384 });
    let t = stepFrames(r, 60);
    r.renderOnce();
    const before = alphaSum(r.readPixels()), trailBefore = r.debugTrailMax();
    r.setOptions({ particleCount: 0 });
    t = stepFrames(r, 600, t);
    r.renderOnce();
    const after = alphaSum(r.readPixels()), trailAfter = r.debugTrailMax();
    const drawn = r.stats.particlesDrawn, cap = r.stats.particleCap;
    r.setOptions({ particleCount: 20000 });
    stepFrames(r, 60, t);
    r.renderOnce();
    const again = alphaSum(r.readPixels());
    r.dispose();
    return { pass: before > 100000 && trailBefore > 0.5 && after <= before * 0.001 && trailAfter < 0.01 && drawn === 0 && cap === 0 && again > before * 0.5, detail: `canvas alpha sum with 20000 particles ${before} (trail max ${trailBefore.toFixed(2)}) -> ${after} (trail max ${trailAfter}) after 600 frames at particleCount 0; particlesDrawn ${drawn}, particleCap ${cap}; streaks return with particles: ${again}` };
  });

  await test('setRegion: a canvas above maxCanvasPixels is scaled down in shape, the drawing buffer matches it and the image stays registered', async () => {
    const region = gridRegion();
    const cap = 100000;
    const mk = (w, h) => {
      const r = new FluidRenderer({ ...QUIET, antialias: false, maxCanvasPixels: cap });
      r.setStorm(meta); r.setHours(syn.rgba[2], syn.rgba[3], 0.37);
      r.setRegion(region, w, h);
      return r;
    };
    const r = mk(1000, 500);   // 500000 px asked for
    const cw = r.canvas.width, ch = r.canvas.height, gl = r.gl;
    const bufOk = gl.drawingBufferWidth === cw && gl.drawingBufferHeight === ch;
    const shapeOk = cw * ch <= cap && cw * ch > 0.9 * cap && Math.abs(cw / ch - 2) < 0.02;
    const corner = r.pixelToWorld(cw, ch);   // bottom-right canvas corner = SE corner of the region
    const regOk = Math.abs(corner[0] - region[2]) < 1e-6 && Math.abs(corner[1] - region[1]) < 1e-6;
    r.renderOnce(); const img = r.readPixels();
    const ref = new FluidRenderer({ ...QUIET, antialias: false });   // asked for exactly that size
    ref.setStorm(meta); ref.setHours(syn.rgba[2], syn.rgba[3], 0.37); ref.setRegion(region, cw, ch);
    ref.renderOnce();
    const n = diffCount(img, ref.readPixels()), covered = alphaSum(img) / (cw * ch * 255);
    r.dispose(); ref.dispose();
    // the default limit on this GPU: 6144 x 6144 used to give a 6144 canvas on a 5760 drawing buffer
    const big = new FluidRenderer({ ...QUIET, antialias: false });
    big.setStorm(meta); big.setHours(syn.rgba[2], syn.rgba[3], 0.37);
    big.setRegion(region, 6144, 6144);
    const bw = big.canvas.width, bh = big.canvas.height, dw = big.gl.drawingBufferWidth, dh = big.gl.drawingBufferHeight;
    const lost = big.gl.isContextLost();
    big.renderOnce();
    const bigOk = !lost && bw * bh <= 16777216 && dw === bw && dh === bh && bw >= 1 && Math.abs(big.pixelToWorld(bw, bh)[0] - region[2]) < 1e-6;
    big.dispose();
    return { pass: bufOk && shapeOk && regOk && n === 0 && covered > 0.05 && bigOk, detail: `1000x500 asked, cap ${cap}: canvas ${cw}x${ch}, drawing buffer ${bufOk ? 'same size' : 'DIFFERENT SIZE'}, ${n} px differ from a renderer built at ${cw}x${ch}, coverage ${(100 * covered).toFixed(0)}%; default cap, 6144x6144 asked: canvas ${bw}x${bh}, buffer ${dw}x${dh}, context lost ${lost}` };
  });

  await test('setRegion: when the browser gives a smaller drawing buffer than asked for, the renderer follows the buffer', async () => {
    const region = viewRegion(-300, -300, 2400, 4 / 3);
    const canvas = document.createElement('canvas');
    const r = new FluidRenderer({ ...QUIET, canvas, antialias: false });
    r.setStorm(meta); r.setHours(syn.rgba[2], syn.rgba[3], 0.37); r.setFlowpaths(syn.flow);
    // stand-in for a browser that cuts the buffer to 700x400 (real ones do it above their pixel limit): it reports the cut size
    Object.defineProperty(r.gl, 'drawingBufferWidth', { get: () => Math.min(canvas.width, 700) });
    Object.defineProperty(r.gl, 'drawingBufferHeight', { get: () => Math.min(canvas.height, 400) });
    r.setRegion(region, 1024, 768);
    const cw = canvas.width, ch = canvas.height;
    const corner = r.pixelToWorld(700, 400);
    const regOk = Math.abs(corner[0] - region[2]) < 1e-6 && Math.abs(corner[1] - region[1]) < 1e-6;
    r.renderOnce(); const img = r.readPixels();
    const ref = new FluidRenderer({ ...QUIET, antialias: false });
    ref.setStorm(meta); ref.setHours(syn.rgba[2], syn.rgba[3], 0.37); ref.setRegion(region, 700, 400);
    ref.renderOnce();
    const n = diffCount(img, ref.readPixels()), covered = alphaSum(img) / (700 * 400 * 255);
    r.dispose(); ref.dispose();
    return { pass: cw === 700 && ch === 400 && regOk && img.length === 700 * 400 * 4 && n === 0 && covered > 0.05, detail: `asked 1024x768, buffer reported 700x400: canvas ${cw}x${ch}, metres per pixel follow the buffer: ${regOk}, ${n} px differ from a renderer built at 700x400, coverage ${(100 * covered).toFixed(0)}%` };
  });

  await test('hour textures upload only when the array identity changes; advancing an hour uploads one grid', async () => {
    const r = makeRenderer(syn, { options: QUIET, a: syn.rgba[0], b: syn.rgba[1], mix: 0 });
    const U = () => r.stats.uploads;
    const seq = [];
    const u0 = U();   // A,B uploaded in setup, plus flowpaths
    r.setHours(syn.rgba[0], syn.rgba[1], 0.5); seq.push(U() - u0);
    r.setHours(syn.rgba[0], syn.rgba[1], 0.9); seq.push(U() - u0);
    r.setHours(syn.rgba[1], syn.rgba[2], 0.1); seq.push(U() - u0);   // +1 (rgba[2] new), rgba[1] reused
    r.setHours(syn.rgba[2], syn.rgba[1], 0.1); seq.push(U() - u0);   // +0 both resident
    const zero = new Uint8Array(GW * GH * 4); Object.defineProperty(zero, 'isZeroGrid', { value: true });
    r.setHours(zero, syn.rgba[2], 0.3); seq.push(U() - u0);          // +0 zero grid never uploaded
    r.setHours(syn.rgba[2], syn.rgba[3], 0.3); seq.push(U() - u0);   // +1
    r.setHours(syn.rgba[2], syn.rgba[2], 0.5); seq.push(U() - u0);   // +0 same array twice
    r.setHours(syn.rgba[0], syn.rgba[4], 0.5); seq.push(U() - u0);   // +2
    r.setFlowpaths(syn.flow); seq.push(U() - u0);                      // +0 same flowpaths array
    r.dispose();
    const exp = [0, 0, 1, 1, 1, 2, 2, 4, 4];
    return { pass: seq.join() === exp.join(), detail: `cumulative uploads after each call: ${seq} (expected ${exp})` };
  });

  await test('no flicker when the hour textures swap: (A,B,mix 1) and (B,C,mix 0) render the same picture', async () => {
    const r = makeRenderer(syn, { options: { particleCount: 5000, seed: 9 }, a: syn.rgba[1], b: syn.rgba[2], mix: 1 });
    stepFrames(r, 20);
    r.renderOnce();
    const a = r.readPixels();
    r.setHours(syn.rgba[2], syn.rgba[3], 0);
    r.renderOnce();
    const b = r.readPixels();
    const n = diffCount(a, b, 2);
    // also a continuous sweep: no pixel row jumps between consecutive mix steps by more than a small bound
    let worstStep = 0, prev = null;
    for (let i = 0; i <= 20; i++) {
      r.setHours(syn.rgba[2], syn.rgba[3], i / 20); r.renderOnce();
      const img = r.readPixels();
      if (prev) { let s = 0; for (let k = 3; k < img.length; k += 4) s += Math.abs(img[k] - prev[k]); worstStep = Math.max(worstStep, s / (GW * GH)); }
      prev = img;
    }
    r.dispose();
    return { pass: n === 0 && worstStep < 4, detail: `px differing across the swap: ${n}; worst mean alpha change per 5% mix step: ${worstStep.toFixed(2)}/255` };
  });

  await test('all-zero (dry) hours: nothing drawn; mixing a dry hour fades the water, no phantom velocity', async () => {
    const zero = new Uint8Array(GW * GH * 4); Object.defineProperty(zero, 'isZeroGrid', { value: true });
    const rawZero = new Uint8Array(GW * GH * 4);   // same bytes but not marked: uploaded as a full texture
    const r = makeRenderer(syn, { options: { particleCount: 4000, seed: 1 }, a: zero, b: zero, mix: 0.5 });
    stepFrames(r, 10); r.renderOnce();
    const e1 = alphaSum(r.readPixels());
    r.setHours(rawZero, rawZero, 0.5); stepFrames(r, 5, 500); r.renderOnce();
    const e2 = alphaSum(r.readPixels());
    r.setOptions(QUIET); r.setHours(zero, syn.rgba[2], 1); r.renderOnce(); const full = alphaSum(r.readPixels());
    r.setHours(zero, syn.rgba[2], 0.5); r.renderOnce(); const half = alphaSum(r.readPixels());
    r.setHours(rawZero, syn.rgba[2], 0.5); r.renderOnce(); const halfRaw = alphaSum(r.readPixels());
    r.dispose();
    return { pass: e1 === 0 && e2 === 0 && half < full * 0.9 && half > full * 0.25 && Math.abs(halfRaw - half) < full * 0.01, detail: `empty frames alpha sum ${e1}/${e2}; water alpha full ${full}, half-mixed ${half}, half-mixed with unmarked zero array ${halfRaw}` };
  });

  await test('context loss: frame() is safe while lost, and everything is rebuilt after restore', async () => {
    const r = makeRenderer(syn, { options: { showParticles: false, showArrows: true, showFlowpaths: true }, mix: 0.42 });
    r.frame(0); r.frame(100);
    r.renderOnce();
    const before = r.readPixels();
    const framesBefore = r.stats.frames;
    const lose = r.gl.getExtension('WEBGL_lose_context');
    if (!lose) { r.dispose(); return { pass: false, detail: 'WEBGL_lose_context unavailable' }; }
    const lost = new Promise((res) => r.canvas.addEventListener('webglcontextlost', res, { once: true }));
    lose.loseContext();
    await lost;
    await sleep(50);   // Chrome allows restoreContext() only after the lost event has finished dispatching
    let safe = true;
    try { r.frame(100); r.renderOnce(); r.setOptions({ opacity: 0.9 }); r.setOptions({ opacity: 1 }); r.setHours(syn.rgba[2], syn.rgba[3], 0.42); r.setRegion(gridRegion(), GW, GH); r.readPixels(); r.readPixel(3, 3); } catch (e) { safe = false; }
    const wasLost = r.lost;
    const restored = new Promise((res) => r.canvas.addEventListener('webglcontextrestored', res, { once: true }));
    lose.restoreContext();
    await restored;
    await sleep(20);
    r.renderOnce();
    const after = r.readPixels();
    const n = diffCount(before, after, 2);
    const wet = alphaSum(after) > 1e6;
    r.frame(300);
    const framed = r.stats.frames === framesBefore + 1;
    r.dispose();
    return { pass: safe && wasLost && n === 0 && wet && framed && r.stats.contextRestores === 1, detail: `safe while lost ${safe}, lost flag ${wasLost}, restores ${r.stats.contextRestores}, px differing from the pre-loss picture: ${n}, frame() works after restore: ${framed}` };
  });

  await test('dispose: releases the context, later calls are harmless', async () => {
    const r = makeRenderer(syn, { mix: 0.5 });
    r.frame(0); r.dispose();
    let ok = true;
    try { r.frame(16); r.renderOnce(); r.dispose(); } catch (e) { ok = false; }
    return { pass: ok && r.gl.isContextLost(), detail: `contextLost=${r.gl.isContextLost()}` };
  });

  await test('guards: setHours/setFlowpaths need setStorm, wrong sizes and bad regions throw', async () => {
    const r = new FluidRenderer();
    const e = [];
    const t = (fn) => { try { fn(); e.push(null); } catch (x) { e.push(x); } };
    t(() => r.setHours(syn.rgba[0], syn.rgba[1], 0));
    r.setStorm(meta);
    t(() => r.setHours(new Uint8Array(100), null, 0));
    t(() => r.setFlowpaths(new Uint8Array(100)));
    t(() => r.setRegion([1, 1, 0, 2], 10, 10));
    t(() => r.setRegion([0, 0, 1, 1], 0, 10));
    t(() => r.setStorm({ width: 0 }));
    t(() => r.setHours({ rgba: syn.rgba[0] }, { rgba: syn.rgba[1] }, 0.5));   // storm results are accepted as they are
    const ok = e.slice(0, 6).every((x) => x instanceof Error) && e[6] === null;
    r.dispose();
    return { pass: ok, detail: e.map((x) => (x ? x.message.slice(0, 48) : 'ok')).join(' | ') };
  });

  await test('production-size grid (4200 x 4100) uploads, renders and samples', async () => {
    const W = 4200, Hh = 4100;
    const x0 = GRID_MERC[0], y1 = GRID_MERC[3];
    const bm = { ...meta, width: W, height: Hh, cell_m: 5, bounds_merc: [x0, y1 - Hh * 5, x0 + W * 5, y1], hours: [3, 4] };
    const a = new Uint8Array(W * Hh * 4), b = new Uint8Array(W * Hh * 4);
    for (let row = 0; row < Hh; row++) {
      const o0 = row * W * 4;
      for (let col = 0; col < W; col++) {
        const o = o0 + col * 4;
        const wet = ((col >> 6) + (row >> 6)) % 3 === 0;
        a[o] = wet ? 60 + (col & 15) : 0; a[o + 1] = wet ? 160 : 128; a[o + 2] = wet ? 150 : 128; a[o + 3] = 255;
        b[o] = a[o]; b[o + 1] = 128; b[o + 2] = wet ? 100 : 128; b[o + 3] = 255;
      }
    }
    const t0 = performance.now();
    const r = new FluidRenderer({ particleCount: 20000 });
    r.setStorm(bm); r.setHours(a, b, 0.5); r.setFlowpaths(a);
    r.setRegion([x0 + 1000, y1 - 4100 * 5 + 1000, x0 + 1000 + 2048, y1 - 4100 * 5 + 1000 + 1536], 1024, 768);
    r.sync();
    const tUp = performance.now() - t0;
    stepFrames(r, 5);
    r.renderOnce();
    const cov = alphaSum(r.readPixels()) / (1024 * 768 * 255);
    const s = sample(a, b, 0.5, x0 + 1100.5, y1 - 4100 * 5 + 1100.5, bm);
    r.dispose();
    return { pass: cov > 0.05 && Number.isFinite(s.depth), detail: `setup+upload of 2 hours + flowpaths (3 x ${(W * Hh * 4 / 1048576).toFixed(0)} MB) took ${tUp.toFixed(0)} ms; coverage ${(cov * 100).toFixed(0)}%` };
  });

  return gpuInfo;
}

async function perfTests(syn) {
  // allWet: worst case for the water shader (every pixel is wet and moving)
  const allWet = new Uint8Array(GW * GH * 4);
  for (let i = 0; i < GW * GH; i++) { allWet[i * 4] = 70; allWet[i * 4 + 1] = 170; allWet[i * 4 + 2] = 140; allWet[i * 4 + 3] = 255; }
  const results = [];
  // coverage cap off: the probes measure the full 60000 particles the task asks about
  const FULL = { particleCoverage: 0 };
  for (const [label, A, B, region, sizes, extra] of [
    ['flood scene', syn.rgba[2], syn.rgba[3], viewRegion(0, 0, 3840, 1), [1024, 2048, 3072], FULL],
    ['all-wet worst case', allWet, allWet, viewRegion(0, 0, 3840, 1), [1024, 2048, 3072], FULL],
    ['flood scene with default options (coverage cap on)', syn.rgba[2], syn.rgba[3], viewRegion(-900, -1050, 900, 1), [2048], {}],
    // the particle pass alone, never sliced or shed (budget 1000 ms), with the GPU nearly idle: its raw cost per frame
    ['particle pass alone, unsliced (water and arrows off)', syn.rgba[2], syn.rgba[3], viewRegion(0, 0, 3840, 1), [1024], { ...FULL, particleBudgetMs: 1000, showDepth: false, showFlow: false, showArrows: false }],
  ]) {
    for (const size of sizes) {
      await sleep(500);   // let the browser reclaim the previous probe's GPU buffers (this machine has 8 GB shared with the iGPU)
      await test(`perf ${label} ${size}x${size}, 60000 particles, 120 frames`, async () => {
        const r = new FluidRenderer({ particleCount: 60000, seed: 1, ...extra });
        r.setStorm(syn.meta); r.setHours(A, B, 0.5); r.setFlowpaths(syn.flow);
        r.setRegion(region, size, size);
        let t = 0;
        r.frame(t); t += 1000 / 60; r.sync();
        const pMs = [], dMs = [], slices = [];
        const t0 = performance.now();
        // a GPU wait every 6 frames keeps the command queue bounded: an unpaced loop can queue seconds of
        // work, which on a weak integrated GPU gets the context killed by the browser's watchdog
        for (let i = 0; i < 120; i++) { r.frame(t); t += 1000 / 60; pMs.push(r.stats.particleMs); dMs.push(r.stats.drawMs); slices.push(r.stats.particleSlices); if (i % 6 === 5) r.sync(); }
        r.sync();
        const total = (performance.now() - t0) / 120;
        if (r.gl.isContextLost()) { r.dispose(); return { pass: false, detail: `the GL context was lost during the probe, so the timings are meaningless (partial: ${total.toFixed(1)} ms/frame)` }; }
        const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
        const q = (a, f) => a.slice().sort((x, y) => x - y)[Math.floor((a.length - 1) * f)];
        const steady = pMs.slice(40);            // the first frames spend the spawn budget filling the particle array
        const fill = pMs.slice(0, 40);
        // water shader alone, then the same frame with particle trails and arrows
        r.setOptions({ showParticles: false, showArrows: false }); r.sync();
        const t1 = performance.now(); for (let i = 0; i < 20; i++) { r.renderOnce(); } r.sync(); const waterOnly = (performance.now() - t1) / 20;
        r.setOptions({ showParticles: true, showArrows: true });
        const drawn = r.stats.particlesDrawn, fullPass = r.stats.particleFullPassMs, sl = r.stats.particleSlices;
        const lost = r.gl.isContextLost();
        const cap = r.stats.particleCap;
        r.dispose();
        if (lost) return { pass: false, detail: 'the GL context was lost during the water-only measurement' };
        results.push({ label, size, total, steadyAvg: avg(steady), steadyP95: q(steady, 0.95), fullPass, sl });
        // with the default budget the typical pass must stay within about 6 ms per frame (by slicing, then by
        // shedding particles). The median is judged, not the mean: one 100 ms OS-level stall (other processes
        // share this 4-core machine) would move a 120-frame mean by 1 ms, and p95 bounds the tail. The unsliced
        // probe only reports the raw cost.
        const median = q(steady, 0.5);
        return { pass: Number.isFinite(total) && (extra.particleBudgetMs >= 1000 || (median < 6 && q(steady, 0.95) < 25)), detail: `active particles ${cap}; median pass ${median.toFixed(2)} ms; ` + `avg ${total.toFixed(2)} ms/frame (${(1000 / total).toFixed(0)} fps) incl. GPU wait; particle JS pass per frame, steady state: avg ${avg(steady).toFixed(2)} ms, p95 ${q(steady, 0.95).toFixed(2)}, max ${Math.max(...steady).toFixed(2)} (first 40 frames while filling: avg ${avg(fill).toFixed(2)}, max ${Math.max(...fill).toFixed(2)}); full pass over all 60000 = ${fullPass.toFixed(2)} ms, run as ${sl} slice(s); CPU draw submit avg ${avg(dMs).toFixed(2)} ms; water layers only ${waterOnly.toFixed(2)} ms/frame; ${drawn} particles drawn in the last frame` };
      });
    }
  }
  return results;
}

async function runTests(opts = {}) {
  window.__only = opts.only ? new RegExp(opts.only, 'i') : null;
  window.__results.length = 0;
  window.__done = false;
  const list = document.getElementById('list'); if (list) list.textContent = '';
  const summary = document.getElementById('summary');
  setStatus('building synthetic storm...');
  const syn = await buildSynthetic();
  record('synthetic storm built and gzipped', true, `${GW}x${GH} cells of ${CELL} m, ${HOURS.length} hours + flowpaths, ${(syn.buildMs).toFixed(0)} ms, gz sizes ${HOURS.map((h) => (syn.files.get(hourName(h)).length / 1024).toFixed(0) + 'k').join('/')}`);
  setStatus('data tests...');
  if (opts.data !== false) await dataTests(syn);
  setStatus('renderer tests...');
  const gpu = opts.renderer !== false ? await rendererTests(syn) : '';
  if (opts.perf !== false) {
    setStatus('performance probes...');
    await perfTests(syn);
  }
  const pass = window.__results.filter((r) => r.pass).length, total = window.__results.length;
  const text = `${pass}/${total} passed${pass === total ? '' : ` - ${total - pass} FAILED`} - GPU ${gpu}`;
  if (summary) summary.textContent = text;
  setStatus(text);
  window.__done = true;
  return { pass, total, gpu };
}

buildUi();
initStage().catch((e) => { console.error(e); setStatus(`init failed: ${e && e.message}`); window.__initError = String(e && e.stack || e); });
if (location.hash.includes('autorun')) setTimeout(() => runTests(), 500);
