// cog-depth.js - minimal Cloud-Optimized GeoTIFF reader + real-depth queries for the flood viewer. No dependencies.
//
// Reads the files written by scripts/make_cogs.py: EPSG:3857, ONE band, uint16 depth in CENTIMETRES (65535 = outside the model),
// tiled (512 px), DEFLATE (+ horizontal predictor 2), overviews as extra IFDs. Classic TIFF and BigTIFF, little or big endian.
// Everything is fetched with HTTP Range requests: the head of the file (all IFDs, tile offset tables) once, then only the tiles
// that are looked at. Decoded tiles are kept in a small LRU. serve.py (and any static host with Range support) is enough.
//
//   openCog(url, {fetchFn})                      -> Cog
//   cog.sampleMerc(x, y, {level})                -> {status, raw}      status: 'value' | 'nodata' | 'outside'
//   cog.readWindow(level, c0, r0, c1, r1)        -> {data, width, height, c0, r0}   (pixel window, c1/r1 exclusive, clipped to the level)
//   cog.levelFor(mercPerPixel)                   -> index of the coarsest level whose pixels are no larger than that
//   lngLatToMerc / mercToLngLat                  EPSG:4326 <-> EPSG:3857 (spherical, R = 6378137), the same maths as the file's CRS
//   depthResult(raw)                             -> {status:'depth'|'dry'|'nodata', cm, m, ft}
//   maxInPolygon(cog, rings)                     -> max depth over the pixels a footprint covers (centre inside + outline touches)
//   renderDepthTile(cog, z, x, y, opts)          -> RGBA bytes (256 x 256) of one XYZ tile, tier colours
//   CogStore                                     -> finds <base>/<domain>/<storm>/cog_meta.json and opens max_depth / hourly files
//
// Depth semantics (be exact when you label it): max_depth.tif = the largest depth the solver reached at that cell over the whole run;
// depth_<N*3600>s.tif = the depth at the END of simulated hour N (hour 0 = the first save near t = 0). Values are the solver's float
// metres resampled (bilinear) from the 5 m UTM grid onto the 6.25 m mercator grid (about 5 m on the ground) and rounded to whole cm.

export const NODATA_DEFAULT = 65535;
const R_EARTH = 6378137;
const MAX_LAT = 85.0511287798066;
export const M2FT = 3.28084;
export const TIER_EDGES_CM = [10, 30, 60, 120];            // the viewer's tiers: 0.1 / 0.3 / 0.6 / 1.2 m
export const TIER_NAMES = ['shallow', 'moderate', 'deep', 'severe'];

export function lngLatToMerc(lng, lat) {
  const la = Math.max(-MAX_LAT, Math.min(MAX_LAT, lat));
  return [R_EARTH * lng * Math.PI / 180, R_EARTH * Math.log(Math.tan(Math.PI / 4 + la * Math.PI / 360))];
}
export function mercToLngLat(x, y) {
  return [x / R_EARTH * 180 / Math.PI, (2 * Math.atan(Math.exp(y / R_EARTH)) - Math.PI / 2) * 180 / Math.PI];
}

export class CogError extends Error { constructor(msg, kind, status) { super(msg); this.name = 'CogError'; this.kind = kind || 'format'; this.status = status; } }

// ───────────────────────── byte source (Range requests) ─────────────────────────
class RangeSource {
  constructor(url, fetchFn, headBytes) {
    this.url = url; this.fetchFn = fetchFn; this.headBytes = headBytes;
    this.head = null; this.headStart = 0; this.size = null; this.full = null; this.etag = null;
  }
  async init(signal) {
    const res = await this.fetchFn(this.url, { headers: { Range: `bytes=0-${this.headBytes - 1}` }, signal });
    if (res.status === 404) throw new CogError(`not found: ${this.url}`, 'missing', 404);
    if (!(res.status === 200 || res.status === 206)) throw new CogError(`HTTP ${res.status} for ${this.url}`, 'http', res.status);
    const buf = await res.arrayBuffer();
    this.etag = res.headers.get('etag');
    if (res.status === 206) {
      const m = /bytes (\d+)-(\d+)\/(\d+|\*)/.exec(res.headers.get('content-range') || '');
      if (!m || +m[1] !== 0) throw new CogError('unexpected Content-Range', 'http', 206);
      this.size = m[3] === '*' ? null : +m[3];
      this.head = buf;
    } else {                               // the server ignored Range: we hold the whole file (fine for small files)
      this.full = buf; this.size = buf.byteLength; this.head = buf;
    }
    return this;
  }
  async get(offset, length, signal) {
    if (length <= 0) return new ArrayBuffer(0);
    if (this.full) return this._slice(this.full, offset, length);
    if (offset + length <= this.head.byteLength) return this.head.slice(offset, offset + length);
    const res = await this.fetchFn(this.url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` }, signal });
    if (res.status === 206) {
      const m = /bytes (\d+)-(\d+)\//.exec(res.headers.get('content-range') || '');
      if (!m || +m[1] !== offset) throw new CogError('range answered with a different Content-Range', 'http', 206);
      const buf = await res.arrayBuffer();
      if (buf.byteLength !== Math.min(length, (this.size == null ? Infinity : this.size - offset))) throw new CogError('short range read', 'http', 206);
      return buf;
    }
    if (res.status === 200) {              // Range ignored mid-way (a proxy): keep the whole file
      this.full = await res.arrayBuffer();
      return this._slice(this.full, offset, length);
    }
    if (res.status === 404) throw new CogError(`not found: ${this.url}`, 'missing', 404);
    throw new CogError(`HTTP ${res.status} for ${this.url}`, 'http', res.status);
  }
  _slice(buf, offset, length) {
    if (offset + length > buf.byteLength) throw new CogError('read past the end of the file', 'format');
    return buf.slice(offset, offset + length);
  }
}

// ───────────────────────── TIFF parsing ─────────────────────────
const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8 };
const T = { NEW_SUBFILE: 254, WIDTH: 256, HEIGHT: 257, BITS: 258, COMPRESSION: 259, PHOTOMETRIC: 262, SAMPLES: 277, PLANAR: 284, PREDICTOR: 317,
  TILE_W: 322, TILE_H: 323, TILE_OFFSETS: 324, TILE_COUNTS: 325, SAMPLE_FORMAT: 339, PIXEL_SCALE: 33550, TIEPOINT: 33922, TRANSFORM: 34264,
  GEOKEYS: 34735, NODATA: 42113 };

function readValues(dv, pos, type, count, le) {
  const out = new Array(count);
  switch (type) {
    case 1: case 2: case 6: case 7: for (let i = 0; i < count; i++) out[i] = type === 6 ? dv.getInt8(pos + i) : dv.getUint8(pos + i); break;
    case 3: for (let i = 0; i < count; i++) out[i] = dv.getUint16(pos + 2 * i, le); break;
    case 8: for (let i = 0; i < count; i++) out[i] = dv.getInt16(pos + 2 * i, le); break;
    case 4: case 13: for (let i = 0; i < count; i++) out[i] = dv.getUint32(pos + 4 * i, le); break;
    case 9: for (let i = 0; i < count; i++) out[i] = dv.getInt32(pos + 4 * i, le); break;
    case 11: for (let i = 0; i < count; i++) out[i] = dv.getFloat32(pos + 4 * i, le); break;
    case 12: for (let i = 0; i < count; i++) out[i] = dv.getFloat64(pos + 8 * i, le); break;
    case 16: case 18: for (let i = 0; i < count; i++) out[i] = Number(dv.getBigUint64(pos + 8 * i, le)); break;
    case 17: for (let i = 0; i < count; i++) out[i] = Number(dv.getBigInt64(pos + 8 * i, le)); break;
    case 5: case 10: for (let i = 0; i < 2 * count; i += 2) out[i / 2] = dv.getUint32(pos + 4 * i, le) / dv.getUint32(pos + 4 * i + 4, le); break;
    default: throw new CogError('unsupported TIFF field type ' + type);
  }
  return out;
}

async function parseIfd(src, off, le, big, signal) {
  const cs = big ? 8 : 2, es = big ? 20 : 12, ns = big ? 8 : 4;
  const h = new DataView(await src.get(off, cs, signal));
  const n = big ? Number(h.getBigUint64(0, le)) : h.getUint16(0, le);
  if (n > 4096) throw new CogError('implausible IFD entry count');
  const body = await src.get(off + cs, n * es + ns, signal);
  const dv = new DataView(body);
  const tags = new Map(), pending = [];
  for (let i = 0; i < n; i++) {
    const p = i * es, tag = dv.getUint16(p, le), type = dv.getUint16(p + 2, le);
    const count = big ? Number(dv.getBigUint64(p + 4, le)) : dv.getUint32(p + 4, le);
    const size = (TYPE_SIZE[type] || 0) * count, inl = big ? 8 : 4, vp = p + (big ? 12 : 8);
    if (!TYPE_SIZE[type]) continue;
    if (size <= inl) tags.set(tag, type === 2 ? new TextDecoder().decode(new Uint8Array(body, vp, count)).replace(/\0+$/, '') : readValues(dv, vp, type, count, le));
    else {
      const o = big ? Number(dv.getBigUint64(vp, le)) : dv.getUint32(vp, le);
      pending.push(src.get(o, size, signal).then(b => {
        const d2 = new DataView(b);
        tags.set(tag, type === 2 ? new TextDecoder().decode(new Uint8Array(b, 0, count)).replace(/\0+$/, '') : readValues(d2, 0, type, count, le));
      }));
    }
  }
  await Promise.all(pending);
  const next = big ? Number(dv.getBigUint64(n * es, le)) : dv.getUint32(n * es, le);
  return { tags, next };
}

function arrayType(bits, fmt) {
  if (fmt === 3) { if (bits === 32) return Float32Array; if (bits === 64) return Float64Array; }
  else if (fmt === 2) { if (bits === 8) return Int8Array; if (bits === 16) return Int16Array; if (bits === 32) return Int32Array; }
  else { if (bits === 8) return Uint8Array; if (bits === 16) return Uint16Array; if (bits === 32) return Uint32Array; }
  throw new CogError(`unsupported sample type (${bits} bits, format ${fmt})`);
}

function makeLevel(ifd, index) {
  const g = (t, d) => { const v = ifd.tags.get(t); return v === undefined ? d : v; };
  const width = g(T.WIDTH)[0], height = g(T.HEIGHT)[0];
  if (!ifd.tags.has(T.TILE_W)) throw new CogError('the image is not tiled (not a COG)');
  const samples = g(T.SAMPLES, [1])[0];
  if (samples !== 1) throw new CogError('only single-band images are supported');
  const bits = g(T.BITS, [1])[0], fmt = g(T.SAMPLE_FORMAT, [1])[0], comp = g(T.COMPRESSION, [1])[0], pred = g(T.PREDICTOR, [1])[0];
  if (![1, 8, 32946].includes(comp)) throw new CogError('unsupported compression ' + comp + ' (only none and DEFLATE)');
  if (pred !== 1 && pred !== 2) throw new CogError('unsupported predictor ' + pred);
  const Arr = arrayType(bits, fmt);
  if (pred === 2 && fmt === 3) throw new CogError('predictor 2 on floating point data is not supported');
  const tileW = g(T.TILE_W)[0], tileH = g(T.TILE_H)[0];
  const tilesX = Math.ceil(width / tileW), tilesY = Math.ceil(height / tileH);
  const offsets = g(T.TILE_OFFSETS), counts = g(T.TILE_COUNTS);
  if (offsets.length < tilesX * tilesY || counts.length < tilesX * tilesY) throw new CogError('tile offset table too short');
  return { index, width, height, tileW, tileH, tilesX, tilesY, offsets, counts, bits, fmt, comp, pred, Arr, subfile: g(T.NEW_SUBFILE, [0])[0] };
}

async function inflate(bytes) {
  const ds = new DecompressionStream('deflate');                        // zlib wrapper (TIFF Compression 8 / 32946); adler32 is verified
  return new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
}

// ───────────────────────── the COG ─────────────────────────
const TILE_CACHE = { map: new Map(), bytes: 0, max: 64 * 1024 * 1024 };      // shared LRU of decoded tiles: key -> Promise<TypedArray>
export function setTileCacheBytes(n) { TILE_CACHE.max = n; }
export function tileCacheStats() { return { tiles: TILE_CACHE.map.size, bytes: TILE_CACHE.bytes, max: TILE_CACHE.max }; }
function cachePut(key, p, bytes) {
  TILE_CACHE.map.set(key, { p, bytes }); TILE_CACHE.bytes += bytes;
  while (TILE_CACHE.bytes > TILE_CACHE.max && TILE_CACHE.map.size > 1) {
    const k = TILE_CACHE.map.keys().next().value; if (k === key) break;
    TILE_CACHE.bytes -= TILE_CACHE.map.get(k).bytes; TILE_CACHE.map.delete(k);
  }
}

export class Cog {
  constructor(url, src, levels, geo) {
    this.url = url; this.src = src; this.levels = levels; this.epsg = geo.epsg; this.nodata = geo.nodata;
    this.x0 = geo.x0; this.y0 = geo.y0; this.res = geo.res;                 // level 0: left edge, top edge, mercator metres per pixel
    this.width = levels[0].width; this.height = levels[0].height;
    this.bounds = [this.x0, this.y0 - this.height * this.res, this.x0 + this.width * this.res, this.y0];   // [xmin, ymin, xmax, ymax]
  }
  levelRes(i) { const L = this.levels[i]; return [this.res * this.width / L.width, this.res * this.height / L.height]; }
  levelFor(mercPerPixel) {
    let best = 0;
    for (let i = 0; i < this.levels.length; i++) if (this.levelRes(i)[0] <= mercPerPixel * 1.0000001) best = i;
    return best;
  }
  pixelAt(x, y, level = 0) {                                                 // float column/row at that level (can be outside)
    const [rx, ry] = this.levelRes(level);
    return [(x - this.x0) / rx, (this.y0 - y) / ry];
  }
  async tile(level, tx, ty, signal) {
    const L = this.levels[level], key = `${this.url}|${level}|${tx}|${ty}`;
    const hit = TILE_CACHE.map.get(key);
    if (hit) { TILE_CACHE.map.delete(key); TILE_CACHE.map.set(key, hit); return hit.p; }
    const bytes = L.tileW * L.tileH * (L.bits / 8);
    const p = this._decode(L, tx, ty, signal);
    cachePut(key, p, bytes);
    p.catch(() => { const e = TILE_CACHE.map.get(key); if (e && e.p === p) { TILE_CACHE.bytes -= e.bytes; TILE_CACHE.map.delete(key); } });   // failures are not cached
    return p;
  }
  async _decode(L, tx, ty, signal) {
    const idx = ty * L.tilesX + tx, n = L.tileW * L.tileH;
    const off = L.offsets[idx], cnt = L.counts[idx];
    if (!cnt) { const a = new L.Arr(n); if (this.nodata !== null) a.fill(this.nodata); return a; }     // sparse tile
    const raw = await this.src.get(off, cnt, signal);
    const buf = L.comp === 1 ? raw : await inflate(raw);
    if (buf.byteLength !== n * (L.bits / 8)) throw new CogError(`tile ${tx},${ty} decoded to ${buf.byteLength} bytes, expected ${n * L.bits / 8}`);
    const bytesPer = L.bits / 8;
    if (this.bigEndian && bytesPer > 1) {                                   // swap to the platform order (little endian everywhere that matters)
      const u = new Uint8Array(buf);
      for (let i = 0; i < u.length; i += bytesPer) for (let a = 0, b = bytesPer - 1; a < b; a++, b--) { const t = u[i + a]; u[i + a] = u[i + b]; u[i + b] = t; }
    }
    const arr = new L.Arr(buf);
    if (L.pred === 2) for (let r = 0; r < L.tileH; r++) { const o = r * L.tileW; for (let c = 1; c < L.tileW; c++) arr[o + c] = arr[o + c] + arr[o + c - 1]; }
    return arr;
  }
  // raw sample at one mercator point. 'outside' = the point is not inside the file's extent at all.
  async sampleMerc(x, y, { level = 0, signal } = {}) {
    const L = this.levels[level], [cf, rf] = this.pixelAt(x, y, level);
    const c = Math.floor(cf), r = Math.floor(rf);
    if (!(c >= 0 && r >= 0 && c < L.width && r < L.height)) return { status: 'outside', raw: null };
    const t = await this.tile(level, Math.floor(c / L.tileW), Math.floor(r / L.tileH), signal);
    const raw = t[(r % L.tileH) * L.tileW + (c % L.tileW)];
    return { status: this.nodata !== null && raw === this.nodata ? 'nodata' : 'value', raw, col: c, row: r };
  }
  async sampleLngLat(lng, lat, opts) { const [x, y] = lngLatToMerc(lng, lat); return this.sampleMerc(x, y, opts); }
  // pixel window [c0,c1) x [r0,r1) of one level, clipped to the image; cells outside the file stay at fill
  async readWindow(level, c0, r0, c1, r1, { signal, fill } = {}) {
    const L = this.levels[level];
    c0 = Math.max(0, Math.floor(c0)); r0 = Math.max(0, Math.floor(r0)); c1 = Math.min(L.width, Math.ceil(c1)); r1 = Math.min(L.height, Math.ceil(r1));
    const w = Math.max(0, c1 - c0), h = Math.max(0, r1 - r0);
    const data = new L.Arr(w * h);
    if (fill !== undefined) data.fill(fill);
    if (!w || !h) return { data, width: w, height: h, c0, r0 };
    const jobs = [];
    for (let ty = Math.floor(r0 / L.tileH); ty <= Math.floor((r1 - 1) / L.tileH); ty++) {
      for (let tx = Math.floor(c0 / L.tileW); tx <= Math.floor((c1 - 1) / L.tileW); tx++) {
        jobs.push(this.tile(level, tx, ty, signal).then(t => {
          const ca = Math.max(c0, tx * L.tileW), cb = Math.min(c1, (tx + 1) * L.tileW), ra = Math.max(r0, ty * L.tileH), rb = Math.min(r1, (ty + 1) * L.tileH);
          for (let r = ra; r < rb; r++) {
            const src = (r - ty * L.tileH) * L.tileW + (ca - tx * L.tileW), dst = (r - r0) * w + (ca - c0);
            data.set(t.subarray(src, src + (cb - ca)), dst);
          }
        }));
      }
    }
    await Promise.all(jobs);
    return { data, width: w, height: h, c0, r0 };
  }
}

export async function openCog(url, { fetchFn, headBytes = 65536, signal } = {}) {
  const f = fetchFn || ((u, init) => fetch(u, init));
  const src = await new RangeSource(url, f, headBytes).init(signal);
  const head = new DataView(src.head);
  if (head.byteLength < 16) throw new CogError('file too small to be a TIFF');
  const bo = String.fromCharCode(head.getUint8(0), head.getUint8(1));
  if (bo !== 'II' && bo !== 'MM') throw new CogError('not a TIFF file');
  const le = bo === 'II', magic = head.getUint16(2, le);
  if (magic !== 42 && magic !== 43) throw new CogError('not a TIFF file (bad magic)');
  const big = magic === 43;
  let next = big ? Number(head.getBigUint64(8, le)) : head.getUint32(4, le);
  const ifds = [];
  while (next && ifds.length < 40) { const ifd = await parseIfd(src, next, le, big, signal); ifds.push(ifd); next = ifd.next; }
  if (!ifds.length) throw new CogError('no image in the file');
  const levels = ifds.map((d, i) => makeLevel(d, i)).sort((a, b) => b.width - a.width);
  levels.forEach((L, i) => { L.index = i; });
  const t0 = ifds.find(d => !(((d.tags.get(T.NEW_SUBFILE) || [0])[0]) & 1)) || ifds[0];
  if (levels[0].width !== (t0.tags.get(T.WIDTH) || [])[0]) throw new CogError('the full-resolution image is not the first one');
  // georeferencing (level 0)
  const scale = t0.tags.get(T.PIXEL_SCALE), tie = t0.tags.get(T.TIEPOINT), tr = t0.tags.get(T.TRANSFORM);
  let x0, y0, res;
  if (scale && tie && tie.length >= 6) {
    if (Math.abs(scale[0] - scale[1]) > 1e-9 * scale[0]) throw new CogError('non-square pixels are not supported');
    res = scale[0]; x0 = tie[3] - tie[0] * scale[0]; y0 = tie[4] + tie[1] * scale[1];
  } else if (tr && tr.length >= 16) {
    if (tr[1] !== 0 || tr[4] !== 0) throw new CogError('rotated rasters are not supported');
    if (Math.abs(tr[0] + tr[5]) > 1e-9 * Math.abs(tr[0])) throw new CogError('not a north-up raster with square pixels');
    res = tr[0]; x0 = tr[3]; y0 = tr[7];
  } else throw new CogError('no georeferencing in the file');
  // GeoKeys: ProjectedCSTypeGeoKey 3072, GTRasterTypeGeoKey 1025 (1 = pixel is area, 2 = pixel is point)
  let epsg = null, rasterType = 1;
  const gk = t0.tags.get(T.GEOKEYS);
  if (gk && gk.length >= 4) for (let i = 1; i <= gk[3]; i++) { const id = gk[4 * i], loc = gk[4 * i + 1], v = gk[4 * i + 3]; if (loc === 0) { if (id === 3072 || (id === 2048 && epsg === null)) epsg = v; if (id === 1025) rasterType = v; } }
  if (epsg !== 3857 && epsg !== 3785 && epsg !== 900913) throw new CogError('the file is not in EPSG:3857 (found ' + epsg + ')');
  if (rasterType === 2) { x0 -= res / 2; y0 += res / 2; }
  const nd = t0.tags.get(T.NODATA);
  const nodata = nd === undefined ? null : (Number.isFinite(parseFloat(nd)) ? parseFloat(nd) : null);
  const cog = new Cog(url, src, levels, { epsg: 3857, nodata, x0, y0, res });
  cog.bigEndian = !le;
  return cog;
}

// ───────────────────────── depth values ─────────────────────────
// raw sample -> what to show. dry = exactly 0 cm (less than half a centimetre of water); nodata = outside the modelled area.
export function depthResult(r) {
  if (!r || r.status === 'outside' || r.status === 'nodata') return { status: 'nodata', cm: null, m: null, ft: null };
  const cm = r.raw;
  return { status: cm === 0 ? 'dry' : 'depth', cm, m: cm / 100, ft: cm / 100 * M2FT };
}
export function tierIndex(cm) { return cm >= 120 ? 3 : cm >= 60 ? 2 : cm >= 30 ? 1 : cm >= 10 ? 0 : -1; }
export function formatDepth(res) {
  if (res.status === 'nodata') return 'no data (outside the modelled area)';
  if (res.status === 'dry') return 'no water (0 cm)';
  return `${res.cm} cm (${(res.cm / 30.48).toFixed(2)} ft)`;
}

// Footprint: rings = [[ [lng,lat], ... ], ...] (outer ring first, then holes; GeoJSON Polygon coordinates). The pixels used are the
// ones whose CENTRE is inside the polygon plus every pixel the outline passes through (so a thin shed still gets a value).
export async function maxInPolygon(cog, rings, { level = 0, signal } = {}) {
  const pr = rings.map(ring => ring.map(([lng, lat]) => { const [x, y] = lngLatToMerc(lng, lat); return cog.pixelAt(x, y, level); }));
  let cmin = Infinity, cmax = -Infinity, rmin = Infinity, rmax = -Infinity;
  for (const ring of pr) for (const [c, r] of ring) { if (c < cmin) cmin = c; if (c > cmax) cmax = c; if (r < rmin) rmin = r; if (r > rmax) rmax = r; }
  const L = cog.levels[level];
  if (!(cmax >= 0 && rmax >= 0 && cmin < L.width && rmin < L.height)) return { status: 'outside', cm: null, pixels: 0 };
  const c0 = Math.floor(cmin), r0 = Math.floor(rmin), c1 = Math.floor(cmax) + 1, r1 = Math.floor(rmax) + 1;
  if ((c1 - c0) * (r1 - r0) > 4e6) throw new CogError('footprint window too large');
  const w = c1 - c0, h = r1 - r0, mask = new Uint8Array(w * h);
  // centre test: even-odd over all rings, per row
  for (let r = 0; r < h; r++) {
    const yc = r0 + r + 0.5, xs = [];
    for (const ring of pr) for (let i = 0, n = ring.length; i < n; i++) {
      const [ax, ay] = ring[i], [bx, by] = ring[(i + 1) % n];
      if ((ay <= yc && by > yc) || (by <= yc && ay > yc)) xs.push(ax + (yc - ay) / (by - ay) * (bx - ax));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const ca = Math.max(0, Math.ceil(xs[k] - 0.5 - c0)), cb = Math.min(w - 1, Math.ceil(xs[k + 1] - 0.5 - c0) - 1);
      for (let c = ca; c <= cb; c++) mask[r * w + c] = 1;
    }
  }
  // outline: walk each edge in steps of at most a quarter pixel
  for (const ring of pr) for (let i = 0, n = ring.length; i < n; i++) {
    const [ax, ay] = ring[i], [bx, by] = ring[(i + 1) % n];
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) * 4));
    for (let s = 0; s <= steps; s++) {
      const c = Math.floor(ax + (bx - ax) * s / steps) - c0, r = Math.floor(ay + (by - ay) * s / steps) - r0;
      if (c >= 0 && r >= 0 && c < w && r < h) mask[r * w + c] = 1;
    }
  }
  const win = await cog.readWindow(level, c0, r0, c1, r1, { signal, fill: cog.nodata === null ? 0 : cog.nodata });
  let best = -1, n = 0, nodata = 0;
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    if (!mask[r * w + c]) continue;
    const gc = c0 + c, gr = r0 + r;
    if (gc < 0 || gr < 0 || gc >= L.width || gr >= L.height) continue;
    const v = win.data[(gr - win.r0) * win.width + (gc - win.c0)];
    if (cog.nodata !== null && v === cog.nodata) { nodata++; continue; }
    n++; if (v > best) best = v;
  }
  if (!n) return { status: 'nodata', cm: null, pixels: 0, nodataPixels: nodata };
  return { status: 'value', cm: best, pixels: n, nodataPixels: nodata };
}

// ───────────────────────── XYZ tile renderer (tier colours) ─────────────────────────
// colors: four CSS hex strings (shallow..severe), alpha: 0..255. A pixel is transparent below 10 cm and where the file has no data.
export function hexRgb(h) { const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(h); return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [0, 0, 0]; }
export async function renderDepthTile(cog, z, x, y, { colors, alpha = 255, size = 256, signal } = {}) {
  const out = new Uint8ClampedArray(size * size * 4);
  const span = 2 * Math.PI * R_EARTH / Math.pow(2, z), tx0 = -Math.PI * R_EARTH + x * span, ty0 = Math.PI * R_EARTH - y * span, px = span / size;
  const [bx0, by0, bx1, by1] = cog.bounds;
  if (tx0 + span <= bx0 || tx0 >= bx1 || ty0 - span >= by1 || ty0 <= by0) return out;            // tile entirely outside the raster
  const level = cog.levelFor(px), L = cog.levels[level], [rx, ry] = cog.levelRes(level);
  const cA = (Math.max(tx0, bx0) - cog.x0) / rx, cB = (Math.min(tx0 + span, bx1) - cog.x0) / rx;
  const rA = (cog.y0 - Math.min(ty0, by1)) / ry, rB = (cog.y0 - Math.max(ty0 - span, by0)) / ry;
  const win = await cog.readWindow(level, cA, rA, cB, rB, { signal, fill: cog.nodata === null ? 0 : cog.nodata });
  const rgb = colors.map(hexRgb), nd = cog.nodata;
  for (let j = 0; j < size; j++) {
    const yy = ty0 - (j + 0.5) * px, row = Math.floor((cog.y0 - yy) / ry) - win.r0;
    if (row < 0 || row >= win.height) continue;
    for (let i = 0; i < size; i++) {
      const col = Math.floor((tx0 + (i + 0.5) * px - cog.x0) / rx) - win.c0;
      if (col < 0 || col >= win.width) continue;
      const v = win.data[row * win.width + col];
      if (nd !== null && v === nd) continue;
      const t = tierIndex(v);
      if (t < 0) continue;
      const o = (j * size + i) * 4, c = rgb[t];
      out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = alpha;
    }
  }
  return out;
}

// ───────────────────────── store: per domain / storm ─────────────────────────
// <base>/<domain>/<storm>/cog_meta.json lists the files (written by make_cogs.py). Hour N = the file whose seconds / 3600 rounds to N
// (solver save times are not exact: depth_003604s.tif is hour 1, depth_000015s.tif is hour 0).
export function hourFiles(meta) {
  const best = {};
  for (const name of Object.keys(meta.files || {})) {
    const m = /^depth_(\d+)s\.tif$/.exec(name); if (!m) continue;
    const sec = +m[1], h = Math.round(sec / 3600), d = Math.abs(sec - h * 3600);
    if (!(h in best) || d < best[h].d) best[h] = { name, d };
  }
  const out = {}; for (const h of Object.keys(best)) out[h] = best[h].name;
  return out;
}

export function hoursOf(meta) { return Object.keys(meta.hourFiles || {}).map(Number).sort((a, b) => a - b); }

export class CogStore {
  constructor({ base = './layers/cog', fetchFn, headBytes } = {}) {
    this.base = base.replace(/\/$/, ''); this.fetchFn = fetchFn || ((u, i) => fetch(u, i)); this.headBytes = headBytes || 65536;
    this.metas = new Map(); this.cogs = new Map();
  }
  folder(domain, storm) { return `${this.base}/${domain}/${storm}`; }
  async meta(domain, storm) {                                                    // null when there are no COGs for this storm
    const k = domain + '/' + storm;
    if (!this.metas.has(k)) {
      const p = (async () => {
        const res = await this.fetchFn(`${this.folder(domain, storm)}/cog_meta.json`);
        if (res.status === 404) return null;
        if (!res.ok) throw new CogError('HTTP ' + res.status, 'http', res.status);
        const m = await res.json();
        if (!m || !m.files || !Object.keys(m.files).some(n => n.endsWith('.tif'))) return null;
        m.hourFiles = hourFiles(m);
        return m;
      })();
      this.metas.set(k, p);
      p.catch(() => this.metas.delete(k));                                       // a network error is retried next time, a 404 is remembered
    }
    return this.metas.get(k);
  }
  async has(domain, storm) { try { return Boolean(await this.meta(domain, storm)); } catch (e) { return false; } }
  async open(domain, storm, which) {                                             // which: 'max' or an hour number; null when that file is not there
    const m = await this.meta(domain, storm);
    if (!m) return null;
    const name = which === 'max' ? (m.files['max_depth.tif'] ? 'max_depth.tif' : null) : m.hourFiles[String(which)];
    if (!name) return null;
    const k = `${domain}/${storm}/${name}`;
    if (!this.cogs.has(k)) {
      const p = openCog(`${this.folder(domain, storm)}/${name}`, { fetchFn: this.fetchFn, headBytes: this.headBytes });
      this.cogs.set(k, p);
      p.catch(() => this.cogs.delete(k));
    }
    return this.cogs.get(k);
  }
  async depthAt(domain, storm, lng, lat, which = 'max', opts) {                  // -> depthResult + {file} or null (no file)
    const cog = await this.open(domain, storm, which);
    if (!cog) return null;
    const r = depthResult(await cog.sampleLngLat(lng, lat, opts));
    r.which = which;
    return r;
  }
  async footprintMax(domain, storm, rings, which = 'max', opts) {
    const cog = await this.open(domain, storm, which);
    if (!cog) return null;
    return maxInPolygon(cog, rings, opts);
  }
  hours(meta) { return hoursOf(meta); }
}
