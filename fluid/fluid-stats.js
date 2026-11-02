// fluid-stats.js - building / road / grid statistics computed from bgflow pixels (see FLOW_FORMAT.md).
// Plain ES module, no DOM, no map. Builds on the decoders and the projection of fluid-data.js.
//
// SEMANTICS (the numbers shown in the viewer mean exactly this)
// - Pixels: grid conventions of FLOW_FORMAT.md. Pixel (col,row) is the half-open square [col,col+1) x [row,row+1)
//   in grid coordinates gx = (xMerc - xmin)/cell_m, gy = (ymax - yMerc)/cell_m (row 0 = north). Pixel index = row*width + col.
// - Touch rule (GDAL / rasterio all_touched=True): a building owns every pixel its footprint intersects: pixels whose
//   centre lies inside the polygon plus every pixel a ring edge passes through. Holes are respected (a pixel lying wholly
//   inside a hole is not owned; one crossed by the hole's edge is). A footprint smaller than one pixel owns the pixel it
//   sits in. opts.ringPx adds that many square (Chebyshev) dilation rings. Only pixels inside the grid are kept; a polygon
//   edge lying exactly on a pixel edge is a measure-zero case where GDAL and this code may differ by one pixel.
//   Null, non-polygon, non-finite, fewer than 3 distinct vertices or zero-area geometry gives an empty entry
//   (no pixels, inGrid 0, centroid NaN); it never throws. inGrid is 1 exactly when the entry holds at least one pixel.
// - A = 0 pixels are outside the model domain and carry no information: they count as depth 0, never as flooded, and are
//   left out of a building's mean (no valid pixel at all gives a mean of 0).
// - Depth is the decoded depth code, depth_m = (R/255)^2 * depth_max_m. "Flooded" means decoded depth >= opts.minDepthM
//   (default 0.1 m, the lower edge of the first tier). A building's depth is the MAXIMUM over its pixels, meanDepthM is the
//   mean over its valid touched pixels (dry ones count 0), speedMs is the speed at the deepest pixel (the first one in
//   row-major order when several share the maximum code); speed is 0 where R = 0.
// - Tiers (TIER_BINS_M, lower edges inclusive): shallow >= 0.1, moderate >= 0.3, deep >= 0.6, severe >= 1.2 m. Depths below
//   0.1 m have no tier, so with minDepthM < 0.1 floodedCount = sum(tierCounts) + belowTierCount.
// - Roads: lines are sampled every spacingM GROUND metres (ground = mercator metres * cos(latitude); each segment uses the
//   cosine at its own latitude). A line of ground length L gets n = ceil(L/spacingM) samples at the midpoints of n equal
//   parts, each carrying L/n metres and the pixel it sits in (-1 outside the grid). A sample is flooded when its pixel has
//   A != 0 and decoded depth >= minDepthM. totalLengthM counts the whole data set (also the part outside the grid, which is
//   dry by construction), so the denominator does not change from storm to storm. pctFlooded is a PERCENTAGE 0..100.
// - summarizeGrid uses the definitions of meta.hourly (flowgrid.hour_stats): wet = decoded depth >= wet_threshold_m, area
//   and volume use cell_ground_m squared. Pixels are 8-bit codes, so values differ slightly from the unquantised hourly ones.
// - Performance: indexes are built once per grid geometry; evaluate* allocate only the arrays they return, and a shared
//   all-zero grid (rgba.isZeroGrid) is answered without reading a pixel.

import { lngLatToMerc, decodeDepth, decodeVel, encodingOf, FlowFormatError } from './fluid-data.js';

export const TIER_BINS_M = [0.1, 0.3, 0.6, 1.2];
const EARTH_R = 6378137;
const DEFAULT_MIN_DEPTH_M = 0.1;
const MAX_RING_PX = 64;
const MIN_AREA_PX2 = 1e-7;   // below this a ring is a line (coordinate round-off alone makes collinear points about 1e-9 px2); a 1 m2 building is 0.04 px2

export function tierOfDepth(depthM) {
  if (!(depthM >= TIER_BINS_M[0])) return null;   // also NaN and undefined
  if (depthM >= TIER_BINS_M[3]) return 'severe';
  if (depthM >= TIER_BINS_M[2]) return 'deep';
  if (depthM >= TIER_BINS_M[1]) return 'moderate';
  return 'shallow';
}

// ------------------------------------------------------------------ helpers

function gridOf(meta, who) {
  const b = meta && meta.bounds_merc;
  if (!meta || !Number.isInteger(meta.width) || !Number.isInteger(meta.height) || !(meta.width > 0) || !(meta.height > 0) ||
      !(meta.cell_m > 0) || !Array.isArray(b) || b.length !== 4 || !b.every(Number.isFinite)) {
    throw new FlowFormatError(`${who}: meta needs integer width/height, cell_m > 0 and bounds_merc`);
  }
  return { W: meta.width, H: meta.height, cell: meta.cell_m, xmin: b[0], ymax: b[3], key: `${b.join(',')}|${meta.cell_m}|${meta.width}|${meta.height}` };
}

const _lutCache = new Map();

// Per-encoding lookup tables: decoded depth (float32 as returned, float64 for comparisons), velocity per code, tier id per depth code.
function lutsFor(enc) {
  const key = `${enc.depthMax}|${enc.vmax}`;
  let l = _lutCache.get(key);
  if (l) return l;
  l = { depth32: new Float32Array(256), depth64: new Float64Array(256), vel: new Float64Array(256), tier: new Uint8Array(256) };
  for (let c = 0; c < 256; c++) {
    l.depth64[c] = decodeDepth(c, enc.depthMax);
    l.depth32[c] = l.depth64[c];
    l.vel[c] = decodeVel(c, enc.vmax);
    let t = 0;
    for (let k = 0; k < TIER_BINS_M.length; k++) if (l.depth64[c] >= TIER_BINS_M[k]) t = k + 1;
    l.tier[c] = c === 0 ? 0 : t;
  }
  _lutCache.set(key, l);
  return l;
}

// Smallest depth code >= 1 whose decoded depth is >= minDepthM (256 when none is).
function minCodeFor(lut, minDepthM) {
  for (let c = 1; c < 256; c++) if (lut.depth64[c] >= minDepthM) return c;
  return 256;
}

function checkGridBytes(index, rgba, who) {
  if (rgba && rgba.isZeroGrid === true) return;
  const need = index.width * index.height * 4;
  if (!rgba || rgba.length !== need) {
    throw new FlowFormatError(`${who}: grid has ${rgba ? rgba.length : 'no'} bytes, the index was built for ${index.width} x ${index.height} x 4 = ${need}`);
  }
}

function minDepthOf(opts) {
  const m = opts && opts.minDepthM !== undefined ? opts.minDepthM : DEFAULT_MIN_DEPTH_M;
  if (typeof m !== 'number' || Number.isNaN(m)) throw new TypeError(`minDepthM must be a number (got ${m})`);
  return m;
}

// ------------------------------------------------------- polygon rasteriser

// Scratch state shared by all builds (single-threaded): vertices in grid coordinates, ring and part tables, masks.
const S = {
  vx: new Float64Array(512), vy: new Float64Array(512), nv: 0,
  ringS: new Int32Array(32), ringE: new Int32Array(32), nr: 0,
  partS: new Int32Array(16), partE: new Int32Array(16), np: 0,
  xs: new Float64Array(512),
  mask: new Uint8Array(4096), tmp: new Uint8Array(4096),
  mc0: 0, mr0: 0, mc1: 0, mr1: 0, bw: 0,
};
const _mc = [0, 0];

function growF64(a, n) { if (a.length >= n) return a; const b = new Float64Array(Math.max(n, a.length * 2)); b.set(a); return b; }
function growI32(a, n) { if (a.length >= n) return a; const b = new Int32Array(Math.max(n, a.length * 2)); b.set(a); return b; }

// Appends one ring (lng/lat positions) to the scratch vertex arrays in grid coordinates. Returns the signed area in
// pixel^2, or NaN when the ring is unusable (not an array, a non-finite coordinate, fewer than 3 distinct vertices).
function ingestRing(ring, G) {
  if (!Array.isArray(ring)) return NaN;
  const n = ring.length;
  if (S.vx.length < S.nv + n) { S.vx = growF64(S.vx, S.nv + n); S.vy = growF64(S.vy, S.nv + n); }
  const start = S.nv;
  let m = 0;
  for (let j = 0; j < n; j++) {
    const p = ring[j];
    if (!Array.isArray(p)) return NaN;
    const lng = p[0], lat = p[1];
    if (typeof lng !== 'number' || typeof lat !== 'number' || !Number.isFinite(lng) || !Number.isFinite(lat)) return NaN;
    lngLatToMerc(lng, lat, _mc);
    const x = (_mc[0] - G.xmin) / G.cell, y = (G.ymax - _mc[1]) / G.cell;
    if (m > 0 && x === S.vx[start + m - 1] && y === S.vy[start + m - 1]) continue;
    S.vx[start + m] = x; S.vy[start + m] = y; m++;
  }
  if (m > 1 && S.vx[start] === S.vx[start + m - 1] && S.vy[start] === S.vy[start + m - 1]) m--;
  if (m < 3) return NaN;
  const ox = S.vx[start], oy = S.vy[start];
  let a = 0;
  for (let j = 0; j < m; j++) {
    const k = j + 1 < m ? j + 1 : 0;
    a += (S.vx[start + j] - ox) * (S.vy[start + k] - oy) - (S.vx[start + k] - ox) * (S.vy[start + j] - oy);
  }
  S.nv = start + m;
  return a / 2;
}

// Signed area and centroid of a raw lng/lat ring, relative to its first vertex (precision). out = [area, cx, cy].
function ringMoments(ring, out) {
  const n = ring.length, o0 = ring[0][0], o1 = ring[0][1];
  let a = 0, cx = 0, cy = 0;
  for (let j = 0; j < n; j++) {
    const k = j + 1 < n ? j + 1 : 0;
    const x0 = ring[j][0] - o0, y0 = ring[j][1] - o1, x1 = ring[k][0] - o0, y1 = ring[k][1] - o1;
    const cr = x0 * y1 - x1 * y0;
    a += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr;
  }
  a /= 2;
  if (a !== 0) { cx = cx / (6 * a) + o0; cy = cy / (6 * a) + o1; } else { cx = o0; cy = o1; }
  out[0] = a; out[1] = cx; out[2] = cy;
}

// Marks every mask pixel the segment passes through (closed segment, half-open pixels).
function markEdge(x1, y1, x2, y2) {
  if (y1 > y2) { let t = x1; x1 = x2; x2 = t; t = y1; y1 = y2; y2 = t; }
  let r0 = Math.floor(y1), r1 = Math.floor(y2);
  if (r0 < S.mr0) r0 = S.mr0;
  if (r1 > S.mr1) r1 = S.mr1;
  const dy = y2 - y1;
  const slope = dy > 0 ? (x2 - x1) / dy : 0;
  for (let r = r0; r <= r1; r++) {
    let xa, xb;
    if (dy === 0) { xa = x1; xb = x2; } else {
      const ya = y1 > r ? y1 : r, yb = y2 < r + 1 ? y2 : r + 1;
      xa = ya === y1 ? x1 : x1 + (ya - y1) * slope;
      xb = yb === y2 ? x2 : x1 + (yb - y1) * slope;
    }
    if (xa > xb) { const t = xa; xa = xb; xb = t; }
    let c0 = Math.floor(xa), c1 = Math.floor(xb);
    if (c0 < S.mc0) c0 = S.mc0;
    if (c1 > S.mc1) c1 = S.mc1;
    const base = (r - S.mr0) * S.bw - S.mc0;
    for (let c = c0; c <= c1; c++) S.mask[base + c] = 1;
  }
}

// Marks the pixels whose centre lies inside part p (even-odd over all its rings, so holes are excluded).
function fillPart(p) {
  const rs = S.partS[p], re = S.partE[p];
  for (let r = S.mr0; r <= S.mr1; r++) {
    const yc = r + 0.5;
    let nx = 0;
    for (let q = rs; q < re; q++) {
      const s = S.ringS[q], e = S.ringE[q];
      for (let j = s; j < e; j++) {
        const k = j + 1 < e ? j + 1 : s;
        const y1 = S.vy[j], y2 = S.vy[k];
        if ((y1 <= yc) !== (y2 <= yc)) {
          const x1 = S.vx[j];
          S.xs[nx++] = x1 + (yc - y1) * (S.vx[k] - x1) / (y2 - y1);
        }
      }
    }
    for (let i = 1; i < nx; i++) {            // insertion sort, nx is tiny
      const v = S.xs[i]; let j = i - 1;
      while (j >= 0 && S.xs[j] > v) { S.xs[j + 1] = S.xs[j]; j--; }
      S.xs[j + 1] = v;
    }
    const base = (r - S.mr0) * S.bw - S.mc0;
    for (let i = 0; i + 1 < nx; i += 2) {
      let c0 = Math.ceil(S.xs[i] - 0.5), c1 = Math.ceil(S.xs[i + 1] - 0.5) - 1;
      if (c0 < S.mc0) c0 = S.mc0;
      if (c1 > S.mc1) c1 = S.mc1;
      for (let c = c0; c <= c1; c++) S.mask[base + c] = 1;
    }
  }
}

// Square dilation of the mask by R rings (two separable passes through S.tmp).
function dilateMask(R) {
  const bw = S.bw, bh = S.mr1 - S.mr0 + 1, mask = S.mask, tmp = S.tmp;
  tmp.fill(0, 0, bw * bh);
  for (let r = 0; r < bh; r++) {
    const row = r * bw;
    for (let c = 0; c < bw; c++) {
      if (!mask[row + c]) continue;
      const a = c - R < 0 ? 0 : c - R, b = c + R >= bw ? bw - 1 : c + R;
      for (let k = a; k <= b; k++) tmp[row + k] = 1;
    }
  }
  mask.fill(0, 0, bw * bh);
  for (let r = 0; r < bh; r++) {
    const row = r * bw;
    for (let c = 0; c < bw; c++) {
      if (!tmp[row + c]) continue;
      const a = r - R < 0 ? 0 : r - R, b = r + R >= bh ? bh - 1 : r + R;
      for (let k = a; k <= b; k++) mask[k * bw + c] = 1;
    }
  }
}

function polygonsOf(geom) {
  if (!geom || typeof geom !== 'object' || !Array.isArray(geom.coordinates)) return null;
  if (geom.type === 'Polygon') return [geom.coordinates];
  if (geom.type === 'MultiPolygon') return geom.coordinates;
  return null;
}

// ---------------------------------------------------------- building index

// features: GeoJSON Feature[] (Polygon / MultiPolygon, WGS84). Returns the CSR pixel index (one entry per feature, same order):
//   { kind, count, width, height, ringPx, offsets: Int32Array(count+1), pixels: Int32Array (pixel indices of building i are
//     pixels[offsets[i]..offsets[i+1]), ascending), centroids: Float64Array(2*count) [lng0, lat0, lng1, ...], inGrid: Uint8Array(count), ... }
export function buildBuildingIndex(meta, features, opts = {}) {
  const G = gridOf(meta, 'buildBuildingIndex');
  const R = opts.ringPx === undefined ? 0 : opts.ringPx;
  if (!Number.isInteger(R) || R < 0 || R > MAX_RING_PX) throw new RangeError(`ringPx must be an integer 0..${MAX_RING_PX} (got ${R})`);
  if (!Array.isArray(features)) throw new TypeError('buildBuildingIndex: features must be an array');
  const n = features.length;
  const offsets = new Int32Array(n + 1);
  const centroids = new Float64Array(2 * n).fill(NaN);
  const inGrid = new Uint8Array(n);
  let pix = new Int32Array(Math.max(1024, n * 12));
  let np = 0;
  const mom = [0, 0, 0];

  for (let i = 0; i < n; i++) {
    offsets[i] = np;
    const f = features[i];
    const polys = polygonsOf(f && f.geometry);
    if (!polys) continue;
    S.nv = 0; S.nr = 0; S.np = 0;
    let wSum = 0, wx = 0, wy = 0;
    for (let pi = 0; pi < polys.length; pi++) {
      const rings = polys[pi];
      if (!Array.isArray(rings) || rings.length === 0) continue;
      const nv0 = S.nv, nr0 = S.nr;
      let ok = true;
      for (let ri = 0; ri < rings.length; ri++) {
        const s = S.nv;
        const a = ingestRing(rings[ri], G);
        if (!(Math.abs(a) > MIN_AREA_PX2)) {
          S.nv = s;
          if (ri === 0) { ok = false; break; }   // unusable exterior: drop the part; an unusable hole is simply ignored
          continue;
        }
        if (S.nr >= S.ringS.length) { S.ringS = growI32(S.ringS, S.nr + 1); S.ringE = growI32(S.ringE, S.nr + 1); }
        S.ringS[S.nr] = s; S.ringE[S.nr] = S.nv; S.nr++;
        ringMoments(rings[ri], mom);
        const w = Math.abs(mom[0]) * (ri === 0 ? 1 : -1);
        wSum += w; wx += w * mom[1]; wy += w * mom[2];
      }
      if (!ok) {
        S.nv = nv0; S.nr = nr0;
        continue;
      }
      if (S.np >= S.partS.length) { S.partS = growI32(S.partS, S.np + 1); S.partE = growI32(S.partE, S.np + 1); }
      S.partS[S.np] = nr0; S.partE[S.np] = S.nr; S.np++;
    }
    if (S.np === 0) continue;
    if (wSum > 0) { centroids[2 * i] = wx / wSum; centroids[2 * i + 1] = wy / wSum; }
    else {                                       // holes cancel the whole area (invalid input): fall back to the vertex mean
      const v0 = lngLatOfFirstRing(polys);
      centroids[2 * i] = v0[0]; centroids[2 * i + 1] = v0[1];
    }

    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (let j = 0; j < S.nv; j++) {
      const x = S.vx[j], y = S.vy[j];
      if (x < minx) minx = x; if (x > maxx) maxx = x;
      if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    S.mc0 = Math.max(Math.floor(minx) - R, -R); S.mc1 = Math.min(Math.floor(maxx) + R, G.W - 1 + R);
    S.mr0 = Math.max(Math.floor(miny) - R, -R); S.mr1 = Math.min(Math.floor(maxy) + R, G.H - 1 + R);
    if (S.mc0 > S.mc1 || S.mr0 > S.mr1) continue;
    S.bw = S.mc1 - S.mc0 + 1;
    const size = S.bw * (S.mr1 - S.mr0 + 1);
    if (S.mask.length < size) { S.mask = new Uint8Array(Math.max(size, S.mask.length * 2)); S.tmp = new Uint8Array(S.mask.length); }
    S.mask.fill(0, 0, size);
    if (S.xs.length < S.nv) S.xs = new Float64Array(S.nv * 2);

    for (let q = 0; q < S.nr; q++) {
      const s = S.ringS[q], e = S.ringE[q];
      for (let j = s; j < e; j++) {
        const k = j + 1 < e ? j + 1 : s;
        markEdge(S.vx[j], S.vy[j], S.vx[k], S.vy[k]);
      }
    }
    for (let p = 0; p < S.np; p++) fillPart(p);
    if (R > 0) dilateMask(R);

    const r0 = Math.max(S.mr0, 0), r1 = Math.min(S.mr1, G.H - 1), c0 = Math.max(S.mc0, 0), c1 = Math.min(S.mc1, G.W - 1);
    if (np + (r1 - r0 + 1) * (c1 - c0 + 1) > pix.length) {
      const bigger = new Int32Array(Math.max(pix.length * 2, np + (r1 - r0 + 1) * (c1 - c0 + 1)));
      bigger.set(pix.subarray(0, np));
      pix = bigger;
    }
    for (let r = r0; r <= r1; r++) {
      const base = (r - S.mr0) * S.bw - S.mc0;
      for (let c = c0; c <= c1; c++) if (S.mask[base + c]) pix[np++] = r * G.W + c;
    }
    if (np > offsets[i]) inGrid[i] = 1;
  }
  offsets[n] = np;

  return {
    kind: 'buildings', count: n, width: G.W, height: G.H, cellM: G.cell, gridKey: G.key, ringPx: R,
    offsets, pixels: pix.slice(0, np), centroids, inGrid,
    enc: encodingOf(meta),
    _maxCode: new Uint8Array(n),
  };
}

function lngLatOfFirstRing(polys) {
  for (const rings of polys) {
    const r = Array.isArray(rings) && rings[0];
    if (Array.isArray(r) && r.length && Array.isArray(r[0])) return [r[0][0], r[0][1]];
  }
  return [NaN, NaN];
}

// ------------------------------------------------------ building evaluation

const _hist = new Int32Array(258);

// Returns { depthM, meanDepthM, speedMs: Float32Array(count), floodedCount, tierCounts, belowTierCount, order, minDepthM }.
// order = flooded building indices, deepest first, equal depth by ascending index (counting sort on the depth code).
export function evaluateBuildings(index, rgba, opts = {}) {
  const minDepthM = minDepthOf(opts);
  const n = index.count;
  const depthM = new Float32Array(n), meanDepthM = new Float32Array(n), speedMs = new Float32Array(n);
  const tierCounts = { shallow: 0, moderate: 0, deep: 0, severe: 0 };
  checkGridBytes(index, rgba, 'evaluateBuildings');
  if (rgba.isZeroGrid === true) {
    return { depthM, meanDepthM, speedMs, floodedCount: 0, tierCounts, belowTierCount: 0, order: new Int32Array(0), minDepthM };
  }
  const lut = lutsFor(index.enc);
  const minCode = minCodeFor(lut, minDepthM);
  const depth32 = lut.depth32, vel = lut.vel, tier = lut.tier;
  const offs = index.offsets, pix = index.pixels, maxCode = index._maxCode;
  const d = rgba;
  const hist = _hist;
  hist.fill(0);
  let flooded = 0, below = 0;

  for (let i = 0; i < n; i++) {
    const e = offs[i + 1];
    let maxR = 0, maxP = -1, cnt = 0, sum = 0;
    for (let k = offs[i]; k < e; k++) {
      const p = pix[k] << 2;
      if (d[p + 3] === 0) continue;
      const r = d[p];
      cnt++;
      sum += depth32[r];
      if (r > maxR) { maxR = r; maxP = p; }
    }
    maxCode[i] = maxR;
    if (maxR === 0) continue;
    depthM[i] = depth32[maxR];
    meanDepthM[i] = sum / cnt;
    const u = vel[d[maxP + 1]], v = vel[d[maxP + 2]];
    speedMs[i] = Math.sqrt(u * u + v * v);
    if (maxR >= minCode) {
      flooded++;
      hist[maxR]++;
      const t = tier[maxR];
      if (t === 0) below++;
      else if (t === 1) tierCounts.shallow++;
      else if (t === 2) tierCounts.moderate++;
      else if (t === 3) tierCounts.deep++;
      else tierCounts.severe++;
    }
  }
  // counting sort: deepest code first; scanning i ascending keeps ties in index order
  const order = new Int32Array(flooded);
  let pos = 0;
  for (let c = 255; c >= 1; c--) { const h = hist[c]; hist[c] = pos; pos += h; }
  for (let i = 0; i < n; i++) {
    const c = maxCode[i];
    if (c >= minCode && c > 0) order[hist[c]++] = i;
  }
  return { depthM, meanDepthM, speedMs, floodedCount: flooded, tierCounts, belowTierCount: below, order, minDepthM };
}

// ---------------------------------------------------------------- road index

function linesOf(geom) {
  if (!geom || typeof geom !== 'object' || !Array.isArray(geom.coordinates)) return null;
  if (geom.type === 'LineString') return [geom.coordinates];
  if (geom.type === 'MultiLineString') return geom.coordinates;
  return null;
}

// features: GeoJSON Feature[] (LineString / MultiLineString, WGS84). One entry per feature, same order:
//   { kind, count, width, height, spacingM, featOffsets: Int32Array(count+1) (sample range of feature f), pixel: Int32Array (-1 outside
//     the grid), weightM: Float64Array (ground metres each sample represents), lengthM: Float64Array(count), totalLengthM, inGridLengthM }
// Null, non-line, zero-length or one-point geometry gives an entry without samples (length 0).
export function buildRoadIndex(meta, features, opts = {}) {
  const G = gridOf(meta, 'buildRoadIndex');
  const spacing = opts.spacingM === undefined ? 10 : opts.spacingM;
  if (typeof spacing !== 'number' || !(spacing > 0) || !Number.isFinite(spacing)) throw new RangeError(`spacingM must be a positive number (got ${spacing})`);
  if (!Array.isArray(features)) throw new TypeError('buildRoadIndex: features must be an array');
  const n = features.length;
  const featOffsets = new Int32Array(n + 1);
  const lengthM = new Float64Array(n);
  let pixel = new Int32Array(1024), weight = new Float64Array(1024);
  let ns = 0;
  let lx = new Float64Array(64), ly = new Float64Array(64), cum = new Float64Array(64);
  let total = 0, inGridLen = 0;

  for (let f = 0; f < n; f++) {
    featOffsets[f] = ns;
    const feat = features[f];
    const lines = linesOf(feat && feat.geometry);
    if (!lines) continue;
    let featLen = 0;
    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      if (!Array.isArray(line)) continue;
      if (lx.length < line.length) { lx = new Float64Array(line.length * 2); ly = new Float64Array(line.length * 2); cum = new Float64Array(line.length * 2); }
      let m = 0;
      for (let j = 0; j < line.length; j++) {            // vertices with a non-finite coordinate are dropped
        const p = line[j];
        if (!Array.isArray(p) || typeof p[0] !== 'number' || typeof p[1] !== 'number' || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue;
        lngLatToMerc(p[0], p[1], _mc);
        lx[m] = _mc[0]; ly[m] = _mc[1]; m++;
      }
      if (m < 2) continue;
      cum[0] = 0;
      for (let j = 1; j < m; j++) {
        // cos(latitude) = 1/cosh(y/R) on the Mercator sphere, taken at the segment's mid latitude
        const cosLat = 1 / Math.cosh((ly[j] + ly[j - 1]) / 2 / EARTH_R);
        cum[j] = cum[j - 1] + Math.hypot(lx[j] - lx[j - 1], ly[j] - ly[j - 1]) * cosLat;
      }
      const L = cum[m - 1];
      if (!(L > 0)) continue;
      const cnt = Math.max(1, Math.ceil(L / spacing - 1e-9));
      const step = L / cnt;
      if (ns + cnt > pixel.length) {
        const sz = Math.max(pixel.length * 2, ns + cnt);
        const p2 = new Int32Array(sz); p2.set(pixel.subarray(0, ns)); pixel = p2;
        const w2 = new Float64Array(sz); w2.set(weight.subarray(0, ns)); weight = w2;
      }
      let k = 0;
      for (let s = 0; s < cnt; s++) {
        const pos = (s + 0.5) * step;
        while (k < m - 2 && cum[k + 1] < pos) k++;
        const d = cum[k + 1] - cum[k];
        let t = d > 0 ? (pos - cum[k]) / d : 0;
        if (t < 0) t = 0; else if (t > 1) t = 1;
        const gx = (lx[k] + (lx[k + 1] - lx[k]) * t - G.xmin) / G.cell;
        const gy = (G.ymax - (ly[k] + (ly[k + 1] - ly[k]) * t)) / G.cell;
        const col = Math.floor(gx), row = Math.floor(gy);
        const inside = col >= 0 && col < G.W && row >= 0 && row < G.H;
        pixel[ns] = inside ? row * G.W + col : -1;
        weight[ns] = step;
        if (inside) inGridLen += step;
        ns++;
      }
      featLen += step * cnt;
    }
    lengthM[f] = featLen;
    total += featLen;
  }
  featOffsets[n] = ns;
  return {
    kind: 'roads', count: n, width: G.W, height: G.H, cellM: G.cell, gridKey: G.key, spacingM: spacing,
    featOffsets, pixel: pixel.slice(0, ns), weightM: weight.slice(0, ns), lengthM, totalLengthM: total, inGridLengthM: inGridLen,
    enc: encodingOf(meta),
  };
}

// Returns { totalLengthM, floodedLengthM, pctFlooded (0..100), perFeaturePct: Float32Array(count) (0..100, 0 for zero-length),
//           floodedFeatures (features with any flooded sample), inGridLengthM, minDepthM }.
export function evaluateRoads(index, rgba, opts = {}) {
  const minDepthM = minDepthOf(opts);
  const n = index.count;
  const perFeaturePct = new Float32Array(n);
  checkGridBytes(index, rgba, 'evaluateRoads');
  const res = { totalLengthM: index.totalLengthM, floodedLengthM: 0, pctFlooded: 0, perFeaturePct, floodedFeatures: 0, inGridLengthM: index.inGridLengthM, minDepthM };
  if (rgba.isZeroGrid === true) return res;
  const minCode = minCodeFor(lutsFor(index.enc), minDepthM);
  const fo = index.featOffsets, px = index.pixel, w = index.weightM, len = index.lengthM, d = rgba;
  let flooded = 0, nf = 0;
  for (let f = 0; f < n; f++) {
    const e = fo[f + 1];
    let fl = 0;
    for (let s = fo[f]; s < e; s++) {
      const p = px[s];
      if (p < 0) continue;
      const q = p << 2;
      if (d[q + 3] !== 0 && d[q] >= minCode) fl += w[s];
    }
    if (fl > 0) {
      nf++;
      flooded += fl;
      perFeaturePct[f] = Math.min(100, 100 * fl / len[f]);
    }
  }
  res.floodedLengthM = flooded;
  res.floodedFeatures = nf;
  res.pctFlooded = index.totalLengthM > 0 ? Math.min(100, 100 * flooded / index.totalLengthM) : 0;
  return res;
}

// -------------------------------------------------------------- grid summary

const _sumHist = new Float64Array(256);

// { wetCells, wetAreaKm2, maxDepthM, meanDepthM (over wet cells), maxSpeedMs (over wet cells), volumeM3 } of one hour grid,
// with the definitions of meta.hourly; A = 0 pixels are ignored.
export function summarizeGrid(meta, rgba) {
  const G = gridOf(meta, 'summarizeGrid');
  const out = { wetCells: 0, wetAreaKm2: 0, maxDepthM: 0, meanDepthM: 0, maxSpeedMs: 0, volumeM3: 0 };
  if (rgba && rgba.isZeroGrid === true) return out;
  const need = G.W * G.H * 4;
  if (!rgba || rgba.length !== need) throw new FlowFormatError(`summarizeGrid: grid has ${rgba ? rgba.length : 'no'} bytes, meta says ${G.W} x ${G.H} x 4 = ${need}`);
  const enc = encodingOf(meta);
  const lut = lutsFor(enc);
  const wetCode = minCodeFor(lut, enc.wetThreshold);
  const vel = lut.vel;
  let cg = meta.cell_ground_m;
  if (!(cg > 0)) {                                 // older metas: ground size from the grid centre latitude
    const yc = (meta.bounds_merc[1] + meta.bounds_merc[3]) / 2;
    cg = meta.cell_m / Math.cosh(yc / EARTH_R);
  }
  const hist = _sumHist;
  hist.fill(0);
  let maxS2 = 0, maxR = 0;
  const d = rgba, total = need;
  for (let p = 0; p < total; p += 4) {
    const r = d[p];
    if (r < wetCode || d[p + 3] === 0) continue;
    hist[r]++;
    if (r > maxR) maxR = r;
    const u = vel[d[p + 1]], v = vel[d[p + 2]];
    const s2 = u * u + v * v;
    if (s2 > maxS2) maxS2 = s2;
  }
  let wet = 0, sum = 0;
  for (let c = 1; c < 256; c++) { wet += hist[c]; sum += hist[c] * lut.depth64[c]; }
  out.wetCells = wet;
  out.wetAreaKm2 = wet * cg * cg / 1e6;
  out.maxDepthM = maxR > 0 ? lut.depth64[maxR] : 0;
  out.meanDepthM = wet > 0 ? sum / wet : 0;
  out.maxSpeedMs = Math.sqrt(maxS2);
  out.volumeM3 = sum * cg * cg;
  return out;
}
