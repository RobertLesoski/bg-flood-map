// fluid-data.js - loader, cache and CPU sampler for bgflow grids (see FLOW_FORMAT.md).
// Plain ES module, no dependencies. Browser only (fetch + DecompressionStream).

const EARTH_R = 6378137;
const MAX_LAT = 85.0511287798066;

export const FORMAT = 'bgflow';
export const VERSION = 1;
export const LRU_HOURS = 4;
export const PREFETCH_CONCURRENCY = 2;   // prefetch downloads in flight at once (each is a full w*h*4 buffer)
export const MAX_GRID_SIDE = 8192;
export const MAX_GRID_PIXELS = 36e6;

export const DEFAULT_ENCODING = Object.freeze({
  depthMax: 8.0,   // metres at code 255
  vmax: 4.0,       // m/s at code 255 / 1
  zeroCode: 128,
  qmax: 2.0,       // m2/s at flowpaths G=255
  wetThreshold: 0.02,
});

export class FlowFormatError extends Error {
  constructor(message) {
    super(message);
    this.name = 'FlowFormatError';
  }
}

export class GridMemoryError extends Error {
  constructor(message) {
    super(message);
    this.name = 'GridMemoryError';
  }
}

// ---------------------------------------------------------------- projection

export function lngLatToMerc(lng, lat, out) {
  const o = out || [0, 0];
  const la = Math.max(-MAX_LAT, Math.min(MAX_LAT, lat));
  o[0] = EARTH_R * lng * Math.PI / 180;
  o[1] = EARTH_R * Math.log(Math.tan(Math.PI / 4 + la * Math.PI / 360));
  return o;
}

export function mercToLngLat(x, y, out) {
  const o = out || [0, 0];
  o[0] = x / EARTH_R * 180 / Math.PI;
  o[1] = (2 * Math.atan(Math.exp(y / EARTH_R)) - Math.PI / 2) * 180 / Math.PI;
  return o;
}

// ------------------------------------------------------------------- decode

export function decodeDepth(R, depthMax = DEFAULT_ENCODING.depthMax) {
  const t = R / 255;
  return t * t * depthMax;
}

export function decodeVel(G, vmax = DEFAULT_ENCODING.vmax) {
  return (G - DEFAULT_ENCODING.zeroCode) / 127 * vmax;
}

// Encoding constants of a meta object, with the format defaults for absent fields.
export function encodingOf(meta) {
  const de = meta.depth_encoding, ve = meta.vel_encoding;
  return {
    depthMax: de && de.max_m !== undefined ? de.max_m : DEFAULT_ENCODING.depthMax,
    vmax: ve && ve.vmax_ms !== undefined ? ve.vmax_ms : DEFAULT_ENCODING.vmax,
    qmax: meta.qmax_m2s !== undefined ? meta.qmax_m2s : DEFAULT_ENCODING.qmax,
    wetThreshold: meta.wet_threshold_m !== undefined ? meta.wet_threshold_m : DEFAULT_ENCODING.wetThreshold,
  };
}

// --------------------------------------------------------------- validation

function isPosInt(n) {
  return typeof n === 'number' && Number.isInteger(n) && n > 0;
}

export function validateMeta(meta, where = 'meta.json') {
  if (meta === null || typeof meta !== 'object' || Array.isArray(meta)) {
    throw new FlowFormatError(`${where}: expected a JSON object`);
  }
  if (meta.format !== FORMAT) {
    throw new FlowFormatError(`${where}: format is ${JSON.stringify(meta.format)}, expected "${FORMAT}"`);
  }
  if (meta.version !== VERSION) {
    throw new FlowFormatError(`${where}: unsupported version ${JSON.stringify(meta.version)} (this viewer reads version ${VERSION})`);
  }
  if (!isPosInt(meta.width) || !isPosInt(meta.height)) {
    throw new FlowFormatError(`${where}: width/height must be positive integers (got ${meta.width} x ${meta.height})`);
  }
  if (meta.width > MAX_GRID_SIDE || meta.height > MAX_GRID_SIDE || meta.width * meta.height > MAX_GRID_PIXELS) {
    throw new FlowFormatError(`${where}: grid ${meta.width} x ${meta.height} is larger than the supported ${MAX_GRID_SIDE} px side / ${MAX_GRID_PIXELS / 1e6} M pixels`);
  }
  if (typeof meta.cell_m !== 'number' || !(meta.cell_m > 0) || !Number.isFinite(meta.cell_m)) {
    throw new FlowFormatError(`${where}: cell_m must be a positive number (got ${meta.cell_m})`);
  }
  const b = meta.bounds_merc;
  if (!Array.isArray(b) || b.length !== 4 || !b.every((v) => typeof v === 'number' && Number.isFinite(v))) {
    throw new FlowFormatError(`${where}: bounds_merc must be [xmin, ymin, xmax, ymax] of finite numbers`);
  }
  if (!(b[0] < b[2]) || !(b[1] < b[3])) {
    throw new FlowFormatError(`${where}: bounds_merc is empty or inverted (${b.join(', ')})`);
  }
  if (Math.round((b[2] - b[0]) / meta.cell_m) !== meta.width || Math.round((b[3] - b[1]) / meta.cell_m) !== meta.height) {
    throw new FlowFormatError(`${where}: bounds_merc spans ${((b[2] - b[0]) / meta.cell_m).toFixed(2)} x ${((b[3] - b[1]) / meta.cell_m).toFixed(2)} cells but width/height are ${meta.width} x ${meta.height}`);
  }
  if (!Array.isArray(meta.hours) || !meta.hours.every((h) => Number.isInteger(h) && h >= 1 && h <= 24)) {
    throw new FlowFormatError(`${where}: hours must be an array of integers 1..24`);
  }
  const de = meta.depth_encoding;
  if (de !== undefined && (de === null || de.type !== 'sqrt' || !(de.max_m > 0))) {
    throw new FlowFormatError(`${where}: depth_encoding must be {"type":"sqrt","max_m":>0}`);
  }
  const ve = meta.vel_encoding;
  if (ve !== undefined && (ve === null || !(ve.vmax_ms > 0) || (ve.zero_code !== undefined && ve.zero_code !== DEFAULT_ENCODING.zeroCode))) {
    throw new FlowFormatError(`${where}: vel_encoding must have vmax_ms > 0 and zero_code 128`);
  }
  return meta;
}

// -------------------------------------------------------------------- fetch

function abortError(signal) {
  if (signal && signal.reason !== undefined && signal.reason !== null) return signal.reason;
  return new DOMException('The operation was aborted.', 'AbortError');
}

function joinUrl(base, ...parts) {
  let s = String(base).replace(/\/+$/, '');
  for (const p of parts) s += '/' + p;
  return s;
}

async function doFetch(fetchImpl, url, init) {
  let res;
  try {
    res = await fetchImpl(url, init);
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    throw new Error(`Network error fetching ${url}: ${e && e.message ? e.message : e}`);
  }
  if (!res.ok) throw new Error(`Failed to load ${url}: HTTP ${res.status} ${res.statusText || ''}`.trim());
  return res;
}

// Reads a Response body into one Uint8Array of exactly `expected` bytes. The body is gunzipped
// here unless the server sent Content-Encoding: gzip, in which case the browser already did it.
// Chunks go straight into the final buffer (a 70-140 MB grid never exists twice in memory) and
// a stream that runs past `expected` is cancelled instead of being buffered to the end. The
// buffer is allocated when the first decoded chunk arrives, so a request that fails or is
// aborted before that never holds one, and a failed allocation is reported as what it is.
async function readGrid(res, url, expected, dims) {
  const enc = (res.headers && res.headers.get && res.headers.get('content-encoding')) || '';
  try {
    let stream = res.body || new Blob([await res.arrayBuffer()]).stream();
    if (!/gzip/i.test(enc)) {
      if (typeof DecompressionStream === 'undefined') throw new Error('this browser has no DecompressionStream');
      stream = stream.pipeThrough(new DecompressionStream('gzip'));
    }
    const reader = stream.getReader();
    let out = null;
    let n = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (n + value.length > expected) {
        reader.cancel().catch(() => {});
        throw new FlowFormatError(`${url}: decoded more than ${expected} bytes, expected ${expected} (${dims})`);
      }
      if (out === null) {
        try {
          out = new Uint8Array(expected);
        } catch (e) {
          reader.cancel().catch(() => {});
          throw new GridMemoryError(`Out of memory: could not allocate ${(expected / 1048576).toFixed(0)} MB for ${url} (${dims}); close other tabs or reduce the number of grids held`);
        }
      }
      out.set(value, n);
      n += value.length;
    }
    if (n !== expected) throw new FlowFormatError(`${url}: decoded ${n} bytes, expected ${expected} (${dims})`);
    return out;
  } catch (e) {
    if (e instanceof FlowFormatError || e instanceof GridMemoryError || (e && e.name === 'AbortError')) throw e;
    throw new Error(`Failed to gunzip ${url}: ${e && e.message ? e.message : e} (corrupt, truncated or not gzip data?)`);
  }
}

// -------------------------------------------------------------------- storm

export class FlowStorm {
  constructor(base, storm, meta, opts = {}) {
    this.base = base;
    this.storm = storm;
    this.meta = meta;
    this.url = joinUrl(base, storm);
    this.width = meta.width;
    this.height = meta.height;
    this.hours = meta.hours.slice().sort((a, b) => a - b);
    this.encoding = encodingOf(meta);
    this._hourSet = new Set(this.hours);
    this._fetch = opts.fetch || null;
    this._fetchInit = opts.fetchInit || {};
    this._lru = new Map();          // hour -> {rgba, hour}, insertion order = recency
    this._inflight = new Map();     // key -> {ctrl, waiters, promise}
    this._flowpaths = null;         // {rgba}
    this._zero = null;
    this._disposed = false;
    this._bytes = meta.width * meta.height * 4;
    // grids are cached by the server for an hour; the build time in the query makes a rebuilt storm a new URL
    this._ver = typeof meta.built === 'string' && meta.built !== '' ? `?v=${encodeURIComponent(meta.built)}` : '';
    this._pq = [];                  // prefetch jobs waiting for a download slot
    this._pActive = 0;
  }

  get disposed() { return this._disposed; }

  _gridUrl(name) { return joinUrl(this.url, name) + this._ver; }

  // Moves a held hour to the most recent end of the LRU.
  _touch(h) {
    const hit = this._lru.get(h);
    if (hit) {
      this._lru.delete(h);
      this._lru.set(h, hit);
    }
    return hit;
  }

  hasHour(h) { return this._hourSet.has(h); }

  // Hours currently held decoded in the LRU, oldest first.
  cachedHours() { return Array.from(this._lru.keys()); }

  // Shared all-dry grid, allocated on first use. Treat it as read-only.
  get zeroGrid() {
    if (!this._zero) {
      const z = new Uint8Array(this._bytes);
      Object.defineProperty(z, 'isZeroGrid', { value: true });
      this._zero = z;
    }
    return this._zero;
  }

  getHour(h, opts = {}) {
    if (!Number.isInteger(h)) return Promise.reject(new TypeError(`getHour: hour must be an integer, got ${h}`));
    if (this._disposed) return Promise.reject(new Error('FlowStorm is disposed'));
    const signal = opts.signal;
    if (signal && signal.aborted) return Promise.reject(abortError(signal));
    if (!this._hourSet.has(h)) {
      return Promise.resolve({ rgba: this.zeroGrid, hour: h });
    }
    const hit = this._touch(h);
    if (hit) return Promise.resolve(hit);
    const name = `h${String(h).padStart(2, '0')}.rgba.gz`;
    return this._shared(h, this._gridUrl(name), signal, (rgba) => {
      const entry = { rgba, hour: h };
      this._lru.set(h, entry);
      while (this._lru.size > LRU_HOURS) this._lru.delete(this._lru.keys().next().value);
      return entry;
    });
  }

  getFlowpaths(opts = {}) {
    if (this._disposed) return Promise.reject(new Error('FlowStorm is disposed'));
    const signal = opts.signal;
    if (signal && signal.aborted) return Promise.reject(abortError(signal));
    if (this._flowpaths) return Promise.resolve(this._flowpaths);
    return this._shared('flowpaths', this._gridUrl('flowpaths.rgba.gz'), signal, (rgba) => {
      this._flowpaths = { rgba };
      return this._flowpaths;
    });
  }

  // Fire-and-forget warm-up; resolves to an array of settled results and never rejects.
  // Only the first LRU_HOURS distinct hours of the list that exist are warmed (the cache holds no
  // more, so later ones would only evict the earlier ones), and at most PREFETCH_CONCURRENCY of
  // them download at once: each one in flight is a full w*h*4 buffer (70-140 MB in production).
  prefetch(hours) {
    const list = Array.isArray(hours) ? hours : [hours];
    const want = [];
    for (const h of list) {
      if (want.length >= LRU_HOURS) break;
      if (this._hourSet.has(h) && !want.includes(h)) want.push(h);
    }
    // hours already held become the most recent first, so the downloads below evict other hours
    for (const h of want) this._touch(h);
    return Promise.allSettled(want.map((h) => this._queued(() => this.getHour(h))));
  }

  _queued(task) {
    return new Promise((resolve, reject) => {
      this._pq.push({ task, resolve, reject });
      this._pump();
    });
  }

  _pump() {
    while (this._pActive < PREFETCH_CONCURRENCY && this._pq.length > 0) {
      const job = this._pq.shift();
      this._pActive++;
      let p;
      try { p = Promise.resolve(job.task()); } catch (e) { p = Promise.reject(e); }
      p.then(job.resolve, job.reject).then(() => { this._pActive--; this._pump(); });
    }
  }

  sample(rgbaA, rgbaB, mix, xMerc, yMerc, out) {
    return sample(rgbaA, rgbaB, mix, xMerc, yMerc, this.meta, out);
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    for (const e of this._inflight.values()) e.ctrl.abort();
    this._inflight.clear();
    for (const job of this._pq) job.reject(new Error('FlowStorm is disposed'));
    this._pq.length = 0;
    this._lru.clear();
    this._flowpaths = null;
    this._zero = null;
  }

  // One network+decode job per key, shared by every concurrent caller. The underlying
  // fetch is aborted only when every waiting caller has aborted.
  _shared(key, url, signal, store) {
    let e = this._inflight.get(key);
    if (!e) {
      const ctrl = new AbortController();
      e = { ctrl, waiters: 0, promise: null };
      const mine = e;
      e.promise = this._loadGrid(url, ctrl.signal).then(
        (rgba) => {
          if (this._inflight.get(key) === mine) this._inflight.delete(key);
          if (this._disposed) throw new Error('FlowStorm is disposed');
          return store(rgba);
        },
        (err) => {
          if (this._inflight.get(key) === mine) this._inflight.delete(key);
          throw err;
        },
      );
      this._inflight.set(key, e);
    }
    const entry = e;
    entry.waiters++;
    return new Promise((resolve, reject) => {
      let done = false;
      const onAbort = () => {
        if (done) return;
        done = true;
        entry.waiters--;
        if (entry.waiters <= 0) {
          if (this._inflight.get(key) === entry) this._inflight.delete(key);
          entry.ctrl.abort();
        }
        reject(abortError(signal));
      };
      if (signal) signal.addEventListener('abort', onAbort, { once: true });
      entry.promise.then(
        (v) => {
          if (done) return;
          done = true;
          if (signal) signal.removeEventListener('abort', onAbort);
          resolve(v);
        },
        (err) => {
          if (done) return;
          done = true;
          if (signal) signal.removeEventListener('abort', onAbort);
          reject(err);
        },
      );
    });
  }

  async _loadGrid(url, signal) {
    const doFetchNow = this._fetch || ((u, i) => fetch(u, i));
    const res = await doFetch(doFetchNow, url, { ...this._fetchInit, signal });
    return readGrid(res, url, this._bytes, `${this.width} x ${this.height} x 4`);
  }
}

export async function loadStorm(base, storm, opts = {}) {
  const url = joinUrl(base, storm, 'meta.json');
  const fetchImpl = opts.fetch || ((u, i) => fetch(u, i));
  // meta.json decides which grids are current, so it is always revalidated whatever the server's cache headers say
  const res = await doFetch(fetchImpl, url, { cache: 'no-cache', ...(opts.fetchInit || {}), signal: opts.signal });
  let meta;
  try {
    meta = JSON.parse(await res.text());
  } catch (e) {
    throw new FlowFormatError(`${url}: not valid JSON (${e.message})`);
  }
  validateMeta(meta, url);
  if (meta.storm !== undefined && meta.storm !== storm) {
    console.warn(`[fluid-data] ${url}: meta.storm is "${meta.storm}", requested "${storm}"`);
  }
  return new FlowStorm(base, storm, meta, opts);
}

// ------------------------------------------------------------------- sample

const _acc = new Float64Array(4);

// Bilinear interpolation of the four channel codes of one grid; absent grid reads as zero.
function bilinearCodes(arr, i00, i10, i01, i11, w00, w10, w01, w11, o) {
  if (!arr) { o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0; return; }
  o[0] = arr[i00] * w00 + arr[i10] * w10 + arr[i01] * w01 + arr[i11] * w11;
  o[1] = arr[i00 + 1] * w00 + arr[i10 + 1] * w10 + arr[i01 + 1] * w01 + arr[i11 + 1] * w11;
  o[2] = arr[i00 + 2] * w00 + arr[i10 + 2] * w10 + arr[i01 + 2] * w01 + arr[i11 + 2] * w11;
  o[3] = arr[i00 + 3] * w00 + arr[i10 + 3] * w10 + arr[i01 + 3] * w01 + arr[i11 + 3] * w11;
}

// CPU twin of what the renderer's shaders compute: the channel codes are interpolated
// bilinearly between pixel centres (CLAMP_TO_EDGE at the grid border), each hour is then
// decoded, and the two hours are blended linearly: value = (1-mix)*A + mix*B.
// Velocity is read as 0 wherever the interpolated depth code is 0, so an all-zero grid
// never produces a phantom -4 m/s. `valid` is false outside the grid and where the
// interpolated A channel is below one half (outside the model domain); in that case
// every number is 0. Pass `out` to avoid an allocation.
export function sample(rgbaA, rgbaB, mix, xMerc, yMerc, meta, out) {
  const o = out || { depth: 0, u: 0, v: 0, speed: 0, valid: false };
  o.depth = 0; o.u = 0; o.v = 0; o.speed = 0; o.valid = false;
  const W = meta.width, H = meta.height, cell = meta.cell_m;
  const b = meta.bounds_merc;
  const gx = (xMerc - b[0]) / cell;
  const gy = (b[3] - yMerc) / cell;
  if (!(gx >= 0 && gx <= W && gy >= 0 && gy <= H)) return o;
  const fx = gx - 0.5, fy = gy - 0.5;
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const tx = fx - x0, ty = fy - y0;
  const xa = x0 < 0 ? 0 : x0, xb = x0 + 1 > W - 1 ? W - 1 : x0 + 1;
  const ya = y0 < 0 ? 0 : y0, yb = y0 + 1 > H - 1 ? H - 1 : y0 + 1;
  const i00 = (ya * W + xa) * 4, i10 = (ya * W + xb) * 4;
  const i01 = (yb * W + xa) * 4, i11 = (yb * W + xb) * 4;
  const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
  const enc = encodingOf(meta);
  const m = mix > 0 ? (mix < 1 ? mix : 1) : 0;

  bilinearCodes(rgbaA, i00, i10, i01, i11, w00, w10, w01, w11, _acc);
  const aA = _acc[3];
  const tA = _acc[0] / 255;
  const dA = tA * tA * enc.depthMax;
  const uA = _acc[0] > 0 ? (_acc[1] - 128) / 127 * enc.vmax : 0;
  const vA = _acc[0] > 0 ? (_acc[2] - 128) / 127 * enc.vmax : 0;

  bilinearCodes(rgbaB, i00, i10, i01, i11, w00, w10, w01, w11, _acc);
  const aB = _acc[3];
  const tB = _acc[0] / 255;
  const dB = tB * tB * enc.depthMax;
  const uB = _acc[0] > 0 ? (_acc[1] - 128) / 127 * enc.vmax : 0;
  const vB = _acc[0] > 0 ? (_acc[2] - 128) / 127 * enc.vmax : 0;

  if (Math.max(aA, aB) < 127.5) return o;
  o.valid = true;
  o.depth = dA * (1 - m) + dB * m;
  o.u = uA * (1 - m) + uB * m;
  o.v = vA * (1 - m) + vB * m;
  o.speed = Math.hypot(o.u, o.v);
  return o;
}
