// stats-harness.js - tests for fluid-stats.js. Results: window.__results = [{name, pass, detail}], window.__done, window.__numbers.
// ?synthetic=1 skips the real-data tests, ?manual=1 does not run on load, ?runs=N sets the timing repetitions.
import * as ST from '../fluid-stats.js';
import { loadStorm, mercToLngLat, lngLatToMerc, decodeDepth, decodeVel, FlowFormatError } from '../fluid-data.js';

const params = new URLSearchParams(location.search);
window.__results = [];
window.__done = false;
window.__numbers = {};

const listEl = document.getElementById('list');
const summaryEl = document.getElementById('summary');
const statusEl = document.getElementById('status');

function record(name, pass, detail = '') {
  window.__results.push({ name, pass: !!pass, detail: String(detail) });
  const row = document.createElement('div');
  row.className = 'row ' + (pass ? 'pass' : 'fail');
  row.innerHTML = `<b>${pass ? 'PASS' : 'FAIL'}</b> ${esc(name)} <i>${esc(detail)}</i>`;
  listEl.appendChild(row);
}
function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
function check(name, cond, detail = '') { record(name, cond, detail); return !!cond; }
function near(a, b, tol) { return Math.abs(a - b) <= tol; }
function sameArr(a, b) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
function median(a) { const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; }
const nowMs = () => performance.now();

// ------------------------------------------------------------- hand-made grid

const HW = 24, CELL = 10, HX0 = -9625000, HY1 = 4435240;
const handMeta = (() => {
  const yc = HY1 - 120;
  const m = {
    format: 'bgflow', version: 1, storm: 'hand', width: HW, height: HW, cell_m: CELL,
    bounds_merc: [HX0, HY1 - HW * CELL, HX0 + HW * CELL, HY1], hours: [1],
    wet_threshold_m: 0.02, depth_encoding: { type: 'sqrt', max_m: 8 }, vel_encoding: { vmax_ms: 4, zero_code: 128 },
  };
  m.cell_ground_m = CELL * Math.cos(mercToLngLat(HX0, yc)[1] * Math.PI / 180);
  return m;
})();
const COSLAT = Math.cos(mercToLngLat(HX0, HY1 - 18.5 * CELL)[1] * Math.PI / 180);   // road rows are at constant latitude

const P = (gx, gy) => mercToLngLat(HX0 + gx * CELL, HY1 - gy * CELL);
const ringOf = (pts) => { const r = pts.map(([x, y]) => P(x, y)); r.push(r[0].slice()); return r; };
const rectPts = (x0, x1, y0, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
const polyG = (...rings) => ({ type: 'Polygon', coordinates: rings.map(ringOf) });
const feat = (geometry) => ({ type: 'Feature', properties: {}, geometry });
const rect = (c0, c1, r0, r1) => { const o = []; for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) o.push(r * HW + c); return o; };
const minus = (a, b) => { const s = new Set(b); return a.filter((v) => !s.has(v)); };

function handGrid() {
  const g = new Uint8Array(HW * HW * 4);
  for (let i = 0; i < HW * HW; i++) { g[i * 4 + 1] = 128; g[i * 4 + 2] = 128; g[i * 4 + 3] = 255; }
  const set = (c, r, R, G = 128, B = 128, A = 255) => { const p = (r * HW + c) * 4; g[p] = R; g[p + 1] = G; g[p + 2] = B; g[p + 3] = A; };
  for (let r = 4; r <= 9; r++) for (let c = 4; c <= 9; c++) set(c, r, 40 + 5 * (c - 4) + (r - 4), 128 + 3 * c, 128 - r);   // wet block A
  for (let r = 5; r <= 8; r++) for (let c = 0; c <= 2; c++) set(c, r, 60);                                              // wet patch on the west edge
  set(0, 6, 90);
  for (let r = 5; r <= 7; r++) for (let c = 15; c <= 17; c++) set(c, r, 120);                                          // water inside the courtyard hole
  for (let r = 20; r <= 21; r++) for (let c = 20; c <= 21; c++) set(c, r, 80, 100);                                    // multipolygon patch
  set(20, 20, 80, 140);
  for (let r = 20; r <= 22; r++) for (let c = 10; c <= 12; c++) set(c, r, 100, 128, 128, 0);                           // R set but A = 0 (outside the model domain)
  for (let r = 20; r <= 21; r++) set(13, r, 50);
  set(22, 2, 100); set(22, 4, 100); set(22, 6, 100);                                                                   // three equal depths
  set(0, 12, 20);                                                                                                      // 0.049 m, under the first tier
  for (let c = 0; c <= 11; c++) set(c, 18, 60);                                                                        // road row: wet west half
  return g;
}

const sub = (cx, cy) => polyG(rectPts(cx - 0.15, cx + 0.15, cy - 0.15, cy + 0.15));
const HAND = [
  { name: 'inside wet block', geom: polyG(rectPts(5.2, 7.8, 5.2, 7.8)), px: rect(5, 7, 5, 7), R: 58 },
  { name: 'inside wet block, clockwise ring', geom: { type: 'Polygon', coordinates: [ringOf(rectPts(5.2, 7.8, 5.2, 7.8)).reverse()] }, px: rect(5, 7, 5, 7), R: 58 },
  { name: 'corner pixel only wet', geom: polyG(rectPts(9.6, 12.4, 9.6, 12.4)), px: rect(9, 12, 9, 12), R: 70 },
  { name: 'fully dry', geom: polyG(rectPts(15.2, 17.8, 15.2, 17.8)), px: rect(15, 17, 15, 17), R: 0 },
  { name: 'outside east', geom: polyG(rectPts(30.2, 32.8, 5.2, 7.8)), px: [], R: 0 },
  { name: 'outside west', geom: polyG(rectPts(-9.8, -5.2, 5.2, 7.8)), px: [], R: 0 },
  { name: 'straddles west edge', geom: polyG(rectPts(-1.5, 1.5, 5.2, 7.8)), px: rect(0, 1, 5, 7), R: 90 },
  { name: 'courtyard hole holds the water', geom: polyG(rectPts(12.2, 20.8, 2.2, 10.8), rectPts(14.3, 18.7, 4.3, 8.7)), px: minus(rect(12, 20, 2, 10), rect(15, 17, 5, 7)), R: 0 },
  { name: 'same ring without hole', geom: polyG(rectPts(12.2, 20.8, 2.2, 10.8)), px: rect(12, 20, 2, 10), R: 120 },
  { name: 'MultiPolygon', geom: { type: 'MultiPolygon', coordinates: [[ringOf(rectPts(20.2, 21.8, 20.2, 21.8))], [ringOf(rectPts(2.2, 3.8, 15.2, 16.8))]] }, px: rect(2, 3, 15, 16).concat(rect(20, 21, 20, 21)), R: 80 },
  { name: 'smaller than a pixel', geom: sub(6.5, 6.5), px: [6 * HW + 6], R: 52 },
  { name: 'null geometry', geom: null, px: [], R: 0, empty: true },
  { name: '2-point ring', geom: { type: 'Polygon', coordinates: [[P(5.2, 5.2), P(7.8, 7.8), P(5.2, 5.2)]] }, px: [], R: 0, empty: true },
  { name: 'Point geometry', geom: { type: 'Point', coordinates: P(6.5, 6.5) }, px: [], R: 0, empty: true },
  { name: 'null feature', feature: null, px: [], R: 0, empty: true },
  { name: 'NaN coordinate', geom: { type: 'Polygon', coordinates: [[P(5.2, 5.2), [NaN, 36.9], P(7.8, 7.8), P(5.2, 7.8), P(5.2, 5.2)]] }, px: [], R: 0, empty: true },
  { name: 'collinear triangle', geom: polyG([[5.2, 5.2], [6.2, 6.2], [7.2, 7.2]]), px: [], R: 0, empty: true },
  { name: 'only A=0 pixels', geom: polyG(rectPts(10.2, 11.8, 20.2, 21.8)), px: rect(10, 11, 20, 21), R: 0 },
  { name: 'A=0 beside valid pixels', geom: polyG(rectPts(11.5, 13.5, 20.5, 21.5)), px: rect(11, 13, 20, 21), R: 50 },
  { name: 'tie 1', geom: sub(22.5, 2.5), px: [2 * HW + 22], R: 100 },
  { name: 'tie 2', geom: sub(22.5, 4.5), px: [4 * HW + 22], R: 100 },
  { name: 'tie 3', geom: sub(22.5, 6.5), px: [6 * HW + 22], R: 100 },
  { name: 'under first tier', geom: sub(0.5, 12.5), px: [12 * HW], R: 20 },
];
const handFeatures = HAND.map((b) => (b.feature !== undefined ? b.feature : feat(b.geom)));
const EXPECT_ORDER = [8, 19, 20, 21, 6, 9, 2, 0, 1, 10, 18];

function syntheticTier() {
  const cases = [[-1, null], [0, null], [NaN, null], [undefined, null], [0.05, null], [0.0999999, null], [0.1, 'shallow'], [0.2999999, 'shallow'], [0.3, 'moderate'],
    [0.5999999, 'moderate'], [0.6, 'deep'], [1.1999999, 'deep'], [1.2, 'severe'], [8, 'severe'], [Infinity, 'severe']];
  const bad = cases.filter(([d, t]) => ST.tierOfDepth(d) !== t).map(([d, t]) => `${d}->${ST.tierOfDepth(d)} (want ${t})`);
  check('1 tierOfDepth boundaries (lower edges inclusive)', bad.length === 0, bad.length ? bad.join('; ') : `${cases.length} cases, TIER_BINS_M=${JSON.stringify(ST.TIER_BINS_M)}`);
}

function handTests() {
  const grid = handGrid();
  const idx = ST.buildBuildingIndex(handMeta, handFeatures);
  check('2 index shape', idx.count === HAND.length && idx.offsets.length === HAND.length + 1 && idx.offsets instanceof Int32Array && idx.pixels instanceof Int32Array &&
    idx.centroids instanceof Float64Array && idx.centroids.length === 2 * HAND.length && idx.inGrid.length === HAND.length, `count ${idx.count}, pixels ${idx.pixels.length}`);
  const badPx = [];
  HAND.forEach((b, i) => {
    const got = Array.from(idx.pixels.subarray(idx.offsets[i], idx.offsets[i + 1]));
    if (!sameArr(got, b.px)) badPx.push(`#${i} ${b.name}: got [${got.join(',')}] want [${b.px.join(',')}]`);
  });
  check('2 exact pixel sets of all hand-made buildings (touch rule, hole, multipolygon, edge clip, degenerate)', badPx.length === 0, badPx.join(' | ') || `${HAND.length} buildings, ${idx.pixels.length} pixels`);
  const badIn = HAND.filter((b, i) => idx.inGrid[i] !== (b.px.length ? 1 : 0)).map((b) => b.name);
  check('2 inGrid flags', badIn.length === 0, badIn.join(', '));
  const emptyOk = HAND.every((b, i) => !b.empty || (Number.isNaN(idx.centroids[2 * i]) && Number.isNaN(idx.centroids[2 * i + 1]) && idx.offsets[i + 1] === idx.offsets[i]));
  check('2 degenerate geometry gives an empty entry with NaN centroid and does not throw', emptyOk);
  // centroids in grid coordinates
  const gc = (i) => { const m = lngLatToMerc(idx.centroids[2 * i], idx.centroids[2 * i + 1]); return [(m[0] - HX0) / CELL, (HY1 - m[1]) / CELL]; };
  const cexp = [[0, 6.5, 6.5], [7, 16.5, 6.5], [9, 12, 18.5], [10, 6.5, 6.5]];
  const badC = cexp.filter(([i, x, y]) => { const c = gc(i); return !near(c[0], x, 0.01) || !near(c[1], y, 0.01); }).map(([i]) => `#${i} ${gc(i)}`);
  check('2 centroids (simple, with hole, multipolygon area weighted, sub-pixel)', badC.length === 0, badC.join('; '));

  const res = ST.evaluateBuildings(idx, grid);
  const exp = HAND.map((b) => {
    let mx = 0, sum = 0, cnt = 0, first = -1;
    for (const p of b.px) {
      if (grid[p * 4 + 3] === 0) continue;
      const R = grid[p * 4]; cnt++; sum += decodeDepth(R);
      if (R > mx) { mx = R; first = p; }
    }
    const sp = mx > 0 ? Math.hypot(decodeVel(grid[first * 4 + 1]), decodeVel(grid[first * 4 + 2])) : 0;
    return { R: mx, depth: decodeDepth(mx), mean: cnt ? sum / cnt : 0, speed: sp };
  });
  check('2 hand-written max codes match the grid (sanity of the test data)', HAND.every((b, i) => exp[i].R === b.R), HAND.map((b, i) => exp[i].R === b.R ? '' : `#${i} ${exp[i].R}!=${b.R}`).join(''));
  const badD = [];
  HAND.forEach((b, i) => {
    if (!near(res.depthM[i], exp[i].depth, 1e-6)) badD.push(`#${i} depth ${res.depthM[i]} vs ${exp[i].depth}`);
    if (!near(res.meanDepthM[i], exp[i].mean, 1e-6)) badD.push(`#${i} mean ${res.meanDepthM[i]} vs ${exp[i].mean}`);
    if (!near(res.speedMs[i], exp[i].speed, 1e-6)) badD.push(`#${i} speed ${res.speedMs[i]} vs ${exp[i].speed}`);
  });
  check('2 depth / mean depth / speed at deepest pixel of every building', badD.length === 0, badD.join(' | ') || 'exact within 1e-6');
  check('2 hand-computed values: B0 depth 0.4139 m, B1 mean = depth/16 with only the corner pixel wet, MultiPolygon speed = first tied pixel (0.378 m/s)',
    near(res.depthM[0], decodeDepth(58), 1e-6) && near(res.depthM[0], 0.41386, 1e-4) && near(res.meanDepthM[2], decodeDepth(70) / 16, 1e-6) && near(res.speedMs[9], 12 / 127 * 4, 1e-6),
    `${res.depthM[0]}, ${res.meanDepthM[2]}, ${res.speedMs[9]}`);
  check('2 courtyard: water inside the hole is not counted, the same ring without hole sees it', res.depthM[7] === 0 && near(res.depthM[8], decodeDepth(120), 1e-6), `${res.depthM[7]} / ${res.depthM[8]}`);
  check('2 corner-only building is flooded with the corner depth', res.depthM[2] > 0.6 && res.depthM[2] === Math.fround(decodeDepth(70)), `${res.depthM[2]}`);
  check('2 floodedCount 11 and tierCounts {0,4,3,4}', res.floodedCount === 11 && res.tierCounts.shallow === 0 && res.tierCounts.moderate === 4 && res.tierCounts.deep === 3 && res.tierCounts.severe === 4 &&
    res.belowTierCount === 0, `${res.floodedCount} ${JSON.stringify(res.tierCounts)}`);
  check('2 order: depth descending, ties by index (hand-written)', res.order instanceof Int32Array && sameArr(res.order, EXPECT_ORDER), `[${Array.from(res.order)}] want [${EXPECT_ORDER}]`);
  const ref = Array.from(res.depthM.keys()).filter((i) => res.depthM[i] >= 0.1).sort((a, b) => res.depthM[b] - res.depthM[a] || a - b);
  check('2 order equals a comparator sort of the returned depths', sameArr(res.order, ref));
  check('2 result fields', res.minDepthM === 0.1 && res.depthM instanceof Float32Array && res.meanDepthM instanceof Float32Array && res.speedMs instanceof Float32Array && res.depthM.length === HAND.length);
  const lo = ST.evaluateBuildings(idx, grid, { minDepthM: 0.02 });
  check('2 minDepthM 0.02 adds the 0.049 m building, which has no tier', lo.floodedCount === 12 && lo.belowTierCount === 1 && lo.order[11] === 22 && lo.minDepthM === 0.02 &&
    lo.tierCounts.shallow + lo.tierCounts.moderate + lo.tierCounts.deep + lo.tierCounts.severe === 11, `${lo.floodedCount} below ${lo.belowTierCount} order ${Array.from(lo.order)}`);
  const hi = ST.evaluateBuildings(idx, grid, { minDepthM: 1.0 });
  check('2 minDepthM 1.0 keeps only depth >= 1.0 m', sameArr(hi.order, [8, 19, 20, 21]) && hi.floodedCount === 4, `[${Array.from(hi.order)}]`);

  // ringPx
  const idx1 = ST.buildBuildingIndex(handMeta, handFeatures, { ringPx: 1 });
  const px1 = (i) => Array.from(idx1.pixels.subarray(idx1.offsets[i], idx1.offsets[i + 1]));
  check('2 ringPx 1: sub-pixel building grows to its 3x3 block, edge building is clipped, outside buildings stay outside, courtyard hole shrinks',
    sameArr(px1(10), rect(5, 7, 5, 7)) && sameArr(px1(6), rect(0, 2, 4, 8)) && px1(4).length === 0 && px1(5).length === 0 && sameArr(px1(7), minus(rect(11, 21, 1, 11), rect(16, 16, 6, 6))),
    `#10 ${px1(10).length} px, #6 [${px1(6)}], #7 ${px1(7).length} px`);
  const near1 = ST.buildBuildingIndex(handMeta, [feat(polyG(rectPts(-1.8, -0.3, 16.3, 16.7)))]);
  const near2 = ST.buildBuildingIndex(handMeta, [feat(polyG(rectPts(-1.8, -0.3, 16.3, 16.7)))], { ringPx: 1 });
  check('2 ringPx 1 lets a building one pixel outside the grid reach in (3 pixels), ringPx 0 does not',
    near1.inGrid[0] === 0 && near2.inGrid[0] === 1 && sameArr(near2.pixels, rect(0, 0, 15, 17)), `[${Array.from(near2.pixels)}]`);
  let threw = false;
  try { ST.buildBuildingIndex(handMeta, [], { ringPx: -1 }); } catch (e) { threw = e instanceof RangeError; }
  check('2 invalid ringPx throws RangeError', threw);

  // only the grid geometry matters
  const m2 = { ...handMeta, hours: [1, 2, 3], storm: 'other', label: 'x' };
  const idx2 = ST.buildBuildingIndex(m2, handFeatures);
  check('2 the index depends only on the grid geometry', sameArr(idx2.pixels, idx.pixels) && sameArr(idx2.offsets, idx.offsets) && idx2.gridKey === idx.gridKey);
  let wrong = null;
  try { ST.evaluateBuildings(idx, new Uint8Array(100)); } catch (e) { wrong = e; }
  check('2 a grid of the wrong size throws FlowFormatError', wrong instanceof FlowFormatError, wrong ? wrong.message : 'no error');

  // every depth code: LUT tiers agree with tierOfDepth
  const m16 = { ...handMeta, width: 16, height: 16, bounds_merc: [HX0, HY1 - 160, HX0 + 160, HY1] };
  const f256 = [], g256 = new Uint8Array(16 * 16 * 4);
  for (let c = 0; c < 256; c++) {
    const cx = (c % 16) + 0.5, cy = Math.floor(c / 16) + 0.5, ll = (gx, gy) => mercToLngLat(HX0 + gx * CELL, HY1 - gy * CELL);
    f256.push(feat({ type: 'Polygon', coordinates: [[ll(cx - 0.1, cy - 0.1), ll(cx + 0.1, cy - 0.1), ll(cx + 0.1, cy + 0.1), ll(cx - 0.1, cy + 0.1), ll(cx - 0.1, cy - 0.1)]] }));
    g256[c * 4] = c; g256[c * 4 + 1] = 128; g256[c * 4 + 2] = 128; g256[c * 4 + 3] = 255;
  }
  const i256 = ST.buildBuildingIndex(m16, f256);
  const r256 = ST.evaluateBuildings(i256, g256, { minDepthM: 0 });
  const tc = { shallow: 0, moderate: 0, deep: 0, severe: 0 }; let fl = 0, below = 0, mism = 0;
  for (let c = 1; c < 256; c++) {
    const d = Math.fround(decodeDepth(c)), t = ST.tierOfDepth(d);
    fl++; if (t) tc[t]++; else below++;
    if (r256.depthM[c] !== d) mism++;
  }
  check('2 all 256 depth codes: tier counts and depths agree with tierOfDepth(decodeDepth)', mism === 0 && r256.floodedCount === fl && r256.belowTierCount === below &&
    ['shallow', 'moderate', 'deep', 'severe'].every((k) => tc[k] === r256.tierCounts[k]) && r256.order.length === 255 && r256.order[0] === 255, `${JSON.stringify(r256.tierCounts)} below ${r256.belowTierCount}`);
  return { grid, idx };
}

function zeroTests({ idx }) {
  const n = HW * HW * 4;
  const z = new Uint8Array(n);
  Object.defineProperty(z, 'isZeroGrid', { value: true });
  const trap = new Proxy(z, { get(t, k) { if (typeof k === 'string' && /^\d+$/.test(k)) throw new Error('pixel read ' + k); return Reflect.get(t, k); } });
  const r = ST.evaluateBuildings(idx, trap);
  check('3 zero grid: buildings answered without reading a pixel, all zero', r.floodedCount === 0 && r.order.length === 0 && r.depthM.length === HAND.length && r.depthM.every((v) => v === 0) &&
    r.speedMs.every((v) => v === 0) && r.meanDepthM.every((v) => v === 0) && r.tierCounts.severe === 0);
  const rIdx = ST.buildRoadIndex(handMeta, [feat({ type: 'LineString', coordinates: [P(0.5, 18.5), P(23.5, 18.5)] })]);
  const rr = ST.evaluateRoads(rIdx, trap);
  check('3 zero grid: roads and summary answered without reading a pixel', rr.floodedLengthM === 0 && rr.pctFlooded === 0 && rr.totalLengthM > 0 && rr.perFeaturePct[0] === 0 &&
    ST.summarizeGrid(handMeta, trap).wetCells === 0);
  // an untagged all-zero grid takes the normal path with the same answer
  const plain = ST.evaluateBuildings(idx, new Uint8Array(n));
  check('3 untagged all-zero grid gives the same result', plain.floodedCount === 0 && plain.order.length === 0);
  // R > 0 everywhere but A = 0 everywhere
  const a0 = new Uint8Array(n);
  for (let i = 0; i < n; i += 4) { a0[i] = 200; a0[i + 1] = 255; a0[i + 2] = 255; }
  const ra = ST.evaluateBuildings(idx, a0);
  const rra = ST.evaluateRoads(rIdx, a0);
  const sa = ST.summarizeGrid(handMeta, a0);
  check('3 A = 0 everywhere: no building, road sample or wet cell counts', ra.floodedCount === 0 && ra.depthM.every((v) => v === 0) && ra.meanDepthM.every((v) => v === 0) && rra.floodedLengthM === 0 &&
    sa.wetCells === 0 && sa.maxDepthM === 0 && sa.volumeM3 === 0, `${ra.floodedCount} ${rra.floodedLengthM} ${sa.wetCells}`);
  // the hand grid: building 17 sits on A=0 R=100 pixels only, building 18 mixes them with valid ones
  const g = handGrid();
  const rh = ST.evaluateBuildings(idx, g);
  check('3 A = 0 pixels with R set are ignored by max and mean (buildings 17, 18)', rh.depthM[17] === 0 && rh.meanDepthM[17] === 0 && near(rh.depthM[18], decodeDepth(50), 1e-6) && near(rh.meanDepthM[18], decodeDepth(50), 1e-6),
    `${rh.depthM[17]} ${rh.depthM[18]} mean ${rh.meanDepthM[18]}`);
}

function roadTests() {
  const g = handGrid();
  const L = (...pts) => ({ type: 'LineString', coordinates: pts.map(([x, y]) => P(x, y)) });
  const rows = [
    { name: 'wet half / dry half', geom: L([0.3, 18.5], [23.7, 18.5]), total: 23.4 * CELL * COSLAT, flooded: 11.7 * CELL * COSLAT },
    { name: 'MultiLineString wet + dry', geom: { type: 'MultiLineString', coordinates: [[P(2.5, 18.5), P(6.5, 18.5)], [P(2.5, 19.5), P(6.5, 19.5)]] }, total: 80 * COSLAT, flooded: 40 * COSLAT },
    { name: 'line leaving the grid', geom: L([-5, 18.5], [5, 18.5]), total: 100 * COSLAT, flooded: 50 * COSLAT },
    { name: 'line fully outside', geom: L([-9, 18.5], [-3, 18.5]), total: 60 * COSLAT, flooded: 0 },
    { name: 'zero-length line', geom: L([4.5, 18.5], [4.5, 18.5]), total: 0, flooded: 0 },
    { name: 'one-point line', geom: { type: 'LineString', coordinates: [P(4.5, 18.5)] }, total: 0, flooded: 0 },
    { name: 'null geometry', geom: null, total: 0, flooded: 0 },
    { name: 'Polygon geometry', geom: polyG(rectPts(1, 3, 1, 3)), total: 0, flooded: 0 },
    { name: 'line on A=0 pixels', geom: L([10.2, 21.5], [12.8, 21.5]), total: 2.6 * CELL * COSLAT, flooded: 0 },
    { name: 'short line (one sample)', geom: L([4.2, 18.5], [4.6, 18.5]), total: 0.4 * CELL * COSLAT, flooded: 0.4 * CELL * COSLAT },
    { name: 'polyline with a bend', geom: L([1.5, 18.2], [1.5, 18.8], [8.5, 18.8]), total: 7.6 * CELL * COSLAT, flooded: 7.6 * CELL * COSLAT },
  ];
  const feats = rows.map((r) => feat(r.geom));
  for (const spacing of [10, 5, 3]) {
    const idx = ST.buildRoadIndex(handMeta, feats, { spacingM: spacing });
    const res = ST.evaluateRoads(idx, g);
    const bad = [];
    rows.forEach((r, i) => {
      if (!near(idx.lengthM[i], r.total, 1e-4 * Math.max(1, r.total))) bad.push(`#${i} ${r.name} length ${idx.lengthM[i]} vs ${r.total}`);
      const fl = res.perFeaturePct[i] / 100 * idx.lengthM[i];
      if (Math.abs(fl - r.flooded) > spacing) bad.push(`#${i} ${r.name} flooded ${fl.toFixed(2)} vs ${r.flooded.toFixed(2)}`);
      if (r.total === 0 && (res.perFeaturePct[i] !== 0 || idx.featOffsets[i + 1] !== idx.featOffsets[i])) bad.push(`#${i} ${r.name} should be empty`);
    });
    const expTot = rows.reduce((s, r) => s + r.total, 0), expFl = rows.reduce((s, r) => s + r.flooded, 0);
    check(`4 roads, spacing ${spacing} m: per-feature lengths exact, flooded length within one spacing, empty entries`, bad.length === 0, bad.join(' | ') || `total ${res.totalLengthM.toFixed(2)} m`);
    check(`4 roads, spacing ${spacing} m: totals and pctFlooded within one spacing of the exact fraction`, near(res.totalLengthM, expTot, 1e-5 * expTot) && Math.abs(res.floodedLengthM - expFl) <= spacing * 3 &&
      near(res.pctFlooded, 100 * expFl / expTot, 100 * spacing * 3 / expTot), `flooded ${res.floodedLengthM.toFixed(1)} m vs exact ${expFl.toFixed(1)}, pct ${res.pctFlooded.toFixed(3)} vs ${(100 * expFl / expTot).toFixed(3)}`);
    if (spacing === 10) {
      const one = ST.evaluateRoads(ST.buildRoadIndex(handMeta, [feats[0]]), g);
      const exact = 100 * 11.7 / 23.4;
      check('4 the wet-half/dry-half line: pctFlooded within one sample spacing of 50.0 percent', Math.abs(one.pctFlooded - exact) <= 100 * 10 / (23.4 * CELL * COSLAT), `${one.pctFlooded.toFixed(3)} (exact ${exact}), one spacing = ${(100 * 10 / (23.4 * CELL * COSLAT)).toFixed(2)} pct points`);
      check('4 samples per feature: n = ceil(L/spacing), weights add up to the length', idx.featOffsets[1] - idx.featOffsets[0] === Math.ceil(23.4 * CELL * COSLAT / 10 - 1e-9) &&
        near(idx.weightM.reduce((s, w) => s + w, 0), idx.totalLengthM, 1e-6), `${idx.featOffsets[1]} samples for the first line`);
      check('4 inGridLengthM excludes the part outside; floodedFeatures counts affected features', near(idx.inGridLengthM, idx.totalLengthM - 110 * COSLAT, 10) && res.floodedFeatures === 5,
        `inGrid ${idx.inGridLengthM.toFixed(1)} of ${idx.totalLengthM.toFixed(1)}, floodedFeatures ${res.floodedFeatures}`);
    }
  }
  let threw = false;
  try { ST.buildRoadIndex(handMeta, [], { spacingM: 0 }); } catch (e) { threw = e instanceof RangeError; }
  check('4 spacingM <= 0 throws RangeError', threw);
}

function summaryTests() {
  const g = handGrid();
  const s = ST.summarizeGrid(handMeta, g);
  let wet = 0, sum = 0, mx = 0, ms = 0;
  for (let i = 0; i < HW * HW; i++) {
    const R = g[i * 4]; if (R === 0 || g[i * 4 + 3] === 0 || decodeDepth(R) < 0.02) continue;
    wet++; const d = decodeDepth(R); sum += d; mx = Math.max(mx, d);
    ms = Math.max(ms, Math.hypot(decodeVel(g[i * 4 + 1]), decodeVel(g[i * 4 + 2])));
  }
  const cg2 = handMeta.cell_ground_m ** 2;
  check('3 summarizeGrid on the hand grid equals a direct pixel loop (A = 0 excluded)', s.wetCells === wet && near(s.maxDepthM, mx, 1e-12) && near(s.meanDepthM, sum / wet, 1e-12) && near(s.volumeM3, sum * cg2, 1e-9) &&
    near(s.wetAreaKm2, wet * cg2 / 1e6, 1e-12) && near(s.maxSpeedMs, ms, 1e-12), JSON.stringify(s));
}

// Brute-force reference: a pixel is owned when its closed square meets the polygon (centre inside, or an edge crosses the square).
function segHitsRect(x1, y1, x2, y2, rx0, ry0, rx1, ry1) {
  let t0 = 0, t1 = 1; const dx = x2 - x1, dy = y2 - y1;
  for (const [p, q] of [[-dx, x1 - rx0], [dx, rx1 - x1], [-dy, y1 - ry0], [dy, ry1 - y1]]) {
    if (p === 0) { if (q < 0) return false; } else { const t = q / p; if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; } else { if (t < t0) return false; if (t < t1) t1 = t; } }
  }
  return true;
}
function insideRings(rings, x, y) {
  let c = false;
  for (const r of rings) for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
    if ((r[i][1] > y) !== (r[j][1] > y) && x < (r[j][0] - r[i][0]) * (y - r[i][1]) / (r[j][1] - r[i][1]) + r[i][0]) c = !c;
  }
  return c;
}
function exactTouch(rings, c, r) {
  if (insideRings(rings, c + 0.5, r + 0.5)) return true;
  for (const ring of rings) for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    if (segHitsRect(ring[j][0], ring[j][1], ring[i][0], ring[i][1], c, r, c + 1, r + 1)) return true;
  }
  return false;
}

function randomPolygonTest() {
  let seed = 12345;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const inside = insideRings;
  const star = (cx, cy, rad, n) => { const a0 = rnd() * 6.283, pts = []; for (let k = 0; k < n; k++) { const a = a0 + (k + rnd() * 0.5) / n * 6.283; const rr = rad * (0.4 + 0.6 * rnd()); pts.push([cx + rr * Math.cos(a), cy + rr * Math.sin(a)]); } return pts; };
  let bad = 0, total = 0, withHoles = 0, pxTotal = 0; const first = [];
  const polys = [], expect = [];
  for (let t = 0; t < 1500; t++) {
    const cx = -3 + rnd() * 30, cy = -3 + rnd() * 30, rad = Math.exp(Math.log(0.1) + rnd() * Math.log(150));
    let rings = [star(cx, cy, rad, 3 + Math.floor(rnd() * 8))];
    if (rad > 2 && rnd() < 0.5) { const hole = star(cx, cy, rad * 0.25, 3 + Math.floor(rnd() * 4)); if (hole.every((p) => inside([rings[0]], p[0], p[1]))) { rings.push(hole); withHoles++; } }
    if (rnd() < 0.5) rings = rings.map((r) => r.slice().reverse());
    polys.push(rings);
    const exp = [];
    for (let r = 0; r < HW; r++) for (let c = 0; c < HW; c++) {
      let hit = inside(rings, c + 0.5, r + 0.5);
      for (let q = 0; q < rings.length && !hit; q++) { const ring = rings[q]; for (let i = 0, j = ring.length - 1; i < ring.length && !hit; j = i++) hit = segHitsRect(ring[j][0], ring[j][1], ring[i][0], ring[i][1], c, r, c + 1, r + 1); }
      if (hit) exp.push(r * HW + c);
    }
    expect.push(exp);
  }
  const feats = polys.map((rings) => feat({ type: 'Polygon', coordinates: rings.map(ringOf) }));
  const idx = ST.buildBuildingIndex(handMeta, feats);
  polys.forEach((rings, i) => {
    total++;
    const got = Array.from(idx.pixels.subarray(idx.offsets[i], idx.offsets[i + 1]));
    pxTotal += got.length;
    if (!sameArr(got, expect[i])) { bad++; if (first.length < 3) first.push(`#${i} got ${got.length} px want ${expect[i].length}`); }
  });
  check('2 random polygons (stars, holes, both orientations, 0.1..150 px, crossing the grid edge) equal a brute-force closed-square intersection test', bad === 0,
    `${total} polygons (${withHoles} with holes), ${pxTotal} pixels, mismatches ${bad} ${first.join('; ')}`);
}

// ------------------------------------------------------------ real-data tests

const relDiff = (a, b) => (b === 0 ? (a === 0 ? 0 : Infinity) : Math.abs(a - b) / Math.abs(b));

function diffSets(a, b) {                       // both ascending
  const onlyA = [], onlyB = [];
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (j >= b.length || (i < a.length && a[i] < b[j])) onlyA.push(a[i++]);
    else if (i >= a.length || b[j] < a[i]) onlyB.push(b[j++]);
    else { i++; j++; }
  }
  return { onlyA, onlyB };
}

async function getJson(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

// buildings: JS index vs the GDAL pixel sets of the fixture; returns the mismatch records for make_stats_fixture.py explain
function compareSets(label, idx, goff, gpix, kind, out) {
  let nDiff = 0, jsOnlyPx = 0, gdalOnlyPx = 0;
  const diffIds = new Set();
  for (let i = 0; i < idx.count; i++) {
    const a = idx.pixels.subarray(idx.offsets[i], idx.offsets[i + 1]), b = gpix.subarray(goff[i], goff[i + 1]);
    if (a.length === b.length) { let same = true; for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) { same = false; break; } if (same) continue; }
    const d = diffSets(a, b);
    nDiff++; diffIds.add(i); jsOnlyPx += d.onlyA.length; gdalOnlyPx += d.onlyB.length;
    if (out.length < 600) out.push({ kind, id: i, jsOnly: d.onlyA, gdalOnly: d.onlyB });
  }
  return { nDiff, jsOnlyPx, gdalOnlyPx, diffIds, total: idx.pixels.length };
}

async function realAll() {
  const base = '/layers/flow';
  const num = window.__numbers;
  const runs = Number(params.get('runs')) || 15;
  let info;
  try { info = await getJson('./fixture/info.json'); } catch (e) {
    record('6 fixtures present (python make_stats_fixture.py build)', false, e.message);
    return;
  }
  statusEl.textContent = 'loading buildings and roads...';
  const bFeatures = (await getJson('/layers/buildings.geojson')).features;
  const rFeatures = (await getJson('/layers/roads.geojson')).features;
  num.nBuildings = bFeatures.length; num.nRoads = rFeatures.length;
  check('6 fixture was made from this many buildings and roads', info.nBuildings === bFeatures.length && info.nRoads === rFeatures.length, `${bFeatures.length} buildings, ${rFeatures.length} roads`);

  const storms = Object.keys(info.built);
  const meta0 = (await loadStorm(base, storms[0])).meta;
  const mismatches = [];

  // index builds (cold first, then median)
  let t = nowMs();
  let bIdx = ST.buildBuildingIndex(meta0, bFeatures);
  const coldBuild = nowMs() - t;
  const bTimes = [];
  for (let i = 0; i < 5; i++) { t = nowMs(); bIdx = ST.buildBuildingIndex(meta0, bFeatures); bTimes.push(nowMs() - t); }
  t = nowMs();
  let rIdx = ST.buildRoadIndex(meta0, rFeatures);
  const coldRoad = nowMs() - t;
  const rTimes = [];
  for (let i = 0; i < 5; i++) { t = nowMs(); rIdx = ST.buildRoadIndex(meta0, rFeatures); rTimes.push(nowMs() - t); }
  num.timings = { buildBuildingIndexColdMs: coldBuild, buildBuildingIndexMedianMs: median(bTimes), buildRoadIndexColdMs: coldRoad, buildRoadIndexMedianMs: median(rTimes) };
  const empties = Array.from({ length: bIdx.count }, (_, i) => bIdx.offsets[i + 1] === bIdx.offsets[i]).filter(Boolean).length;
  num.buildingPixels = bIdx.pixels.length; num.buildingsInGrid = bIdx.count - empties;
  check('7 buildBuildingIndex on the real buildings under 1.5 s', coldBuild < 1500 && median(bTimes) < 1500,
    `${bFeatures.length} buildings, ${bIdx.pixels.length} pixels, ${empties} outside the grid: cold ${coldBuild.toFixed(0)} ms, median of 5 ${median(bTimes).toFixed(0)} ms (excluding JSON parse)`);
  check('7 buildRoadIndex on the real roads', true, `${rFeatures.length} features, ${rIdx.pixel.length} samples, ${(rIdx.totalLengthM / 1000).toFixed(2)} km: cold ${coldRoad.toFixed(1)} ms, median ${median(rTimes).toFixed(1)} ms`);

  // set comparison with GDAL
  const bpx = new Int32Array(await (await fetch('./fixture/bpx.bin', { cache: 'no-cache' })).arrayBuffer());
  const nB = bpx[0], nnz = bpx[1];
  const goff = bpx.subarray(2, 3 + nB), gpix = bpx.subarray(3 + nB, 3 + nB + nnz);
  const sc = compareSets('real', bIdx, goff, gpix, 'building', mismatches);
  num.buildingSetMismatch = { buildings: sc.nDiff, jsOnlyPixels: sc.jsOnlyPx, gdalOnlyPixels: sc.gdalOnlyPx, totalPixels: sc.total };
  // every differing pixel is judged by an exact closed-square test on the same coordinates: the JS set must be the exact one
  let exactJs = 0, exactGdal = 0, wrongJs = 0;
  for (const m of mismatches) {
    const rings = bFeatures[m.id].geometry.coordinates.map((ring) => ring.map(([lng, lat]) => { const q = lngLatToMerc(lng, lat); return [(q[0] - meta0.bounds_merc[0]) / meta0.cell_m, (meta0.bounds_merc[3] - q[1]) / meta0.cell_m]; }));
    for (const p of m.jsOnly) { if (exactTouch(rings, p % meta0.width, Math.floor(p / meta0.width))) exactJs++; else wrongJs++; }
    for (const p of m.gdalOnly) { if (!exactTouch(rings, p % meta0.width, Math.floor(p / meta0.width))) exactGdal++; else wrongJs++; }
  }
  num.buildingSetMismatch.exactJsOnly = exactJs; num.buildingSetMismatch.exactGdalOnly = exactGdal; num.buildingSetMismatch.jsWrong = wrongJs;
  check('6 pixel sets of all real buildings vs rasterio all_touched=True: under 0.1 % of buildings differ and every difference is on the exact-geometry side of JS',
    sc.nDiff <= 0.001 * bFeatures.length && wrongJs === 0 && sc.jsOnlyPx + sc.gdalOnlyPx === exactJs + exactGdal,
    `${bFeatures.length} buildings, ${sc.total} JS pixels vs ${nnz} GDAL pixels; buildings with a different set ${sc.nDiff} (${(100 * sc.nDiff / bFeatures.length).toFixed(3)} %); JS-only pixels ${sc.jsOnlyPx} (exact test says touched: ${exactJs}), GDAL-only pixels ${sc.gdalOnlyPx} (exact test says not touched: ${exactGdal}); JS wrong ${wrongJs}. Offline: make_stats_fixture.py explain shows the JS-only pixels are corner clips of 1.6e-4 to 6.7e-4 px2 that GDAL's line walker misses, the GDAL-only ones lie 0.003 and 0.004 px (2 to 3 cm) away from the polygon`);

  // random polygons vs GDAL
  const rp = await getJson('./fixture/rand_polys.json');
  const gridsEqual = rp.grid.xmin === HX0 && rp.grid.ymax === HY1 && rp.grid.cell_m === CELL && rp.grid.width === HW;
  const rIdxP = ST.buildBuildingIndex(handMeta, rp.polys.map((rings) => feat({ type: 'Polygon', coordinates: rings })));
  const rpOff = Int32Array.from(rp.offsets), rpPix = Int32Array.from(rp.pixels);
  const rc = compareSets('rand', rIdxP, rpOff, rpPix, 'rand', mismatches);
  num.randSetMismatch = { polygons: rc.nDiff, jsOnlyPixels: rc.jsOnlyPx, gdalOnlyPixels: rc.gdalOnlyPx, totalPixels: rc.total, count: rp.polys.length };
  check('6 random polygons (stars, holes, slivers, tiny, huge) vs rasterio all_touched=True', gridsEqual && rc.nDiff === 0,
    `${rp.polys.length} polygons, ${rc.total} JS pixels vs ${rpPix.length} GDAL pixels; differing polygons ${rc.nDiff}, JS-only pixels ${rc.jsOnlyPx}, GDAL-only pixels ${rc.gdalOnlyPx}`);
  window.__mismatches = mismatches;

  // per storm and hour
  const sumWorst = { wetCells: 0, wetAreaKm2: 0, maxDepthM: 0, meanDepthM: 0, maxSpeedMs: 0, volumeM3: 0 };
  const sumWorstAt = {};
  let sumRows = 0, sumSpeedOver1pct = 0, sumDepthOver1pct = 0, sumSpeedUnexplained = 0, sumDepthUnexplained = 0;
  let sumMeanOver1pct = 0, sumMeanUnexplained = 0, sumMeanBoundMax = 0, sumVolOver1pct = 0;
  // |decoded sum - true sum| <= sum of half a code step (4(2R+1)/255^2 m per cell, rounding is to the nearest code)
  const quantBound = (rgba, meta) => {
    let dsum = 0, bsum = 0;
    for (let p = 0; p < rgba.length; p += 4) {
      const R = rgba[p]; if (R < 13 || rgba[p + 3] === 0) continue;
      dsum += decodeDepth(R, 8); bsum += 4 * (2 * R + 1) / (255 * 255);
    }
    return dsum > 0 ? bsum / dsum : 0;
  };
  const codeStep = (d) => { const R = Math.round(255 * Math.sqrt(Math.max(d, 0) / 8)); return 8 * (2 * R + 1) / (255 * 255); };
  const cross = [];
  for (const s of storms) {
    const storm = await loadStorm(base, s);
    const meta = storm.meta;
    if (meta.built !== info.built[s]) { check(`6 fixture matches the current build of ${s}`, false, `built ${meta.built} vs fixture ${info.built[s]}`); storm.dispose(); continue; }
    const hours = meta.hourly.map((h) => h.hour).filter((h) => meta.hours.includes(h));
    for (const h of hours) {
      statusEl.textContent = `${s} h${h}...`;
      const { rgba } = await storm.getHour(h);
      const hm = meta.hourly.find((x) => x.hour === h);
      const sm = ST.summarizeGrid(meta, rgba);
      const pairs = { wetCells: [sm.wetCells, hm.wet_cells], wetAreaKm2: [sm.wetAreaKm2, hm.wet_area_km2], maxDepthM: [sm.maxDepthM, hm.max_depth_m],
        meanDepthM: [sm.meanDepthM, hm.mean_depth_m], maxSpeedMs: [sm.maxSpeedMs, hm.max_speed_ms], volumeM3: [sm.volumeM3, hm.volume_m3] };
      sumRows++;
      for (const k of Object.keys(pairs)) {
        const [a, b] = pairs[k];
        // a difference over 1 % is explained when it is within one velocity code (4/127 m/s) or one depth code at that depth
        const rd = relDiff(a, b);
        if (k === 'maxSpeedMs' && rd > 0.01) { sumSpeedOver1pct++; if (Math.abs(a - b) > 4 / 127) sumSpeedUnexplained++; }
        if (k === 'maxDepthM' && rd > 0.01) { sumDepthOver1pct++; if (Math.abs(a - b) > codeStep(b)) sumDepthUnexplained++; }
        if ((k === 'meanDepthM' || k === 'volumeM3') && rd > 0.01) {
          const qb = quantBound(rgba, meta);          // worst case of 8-bit rounding for this very hour
          sumMeanOver1pct++;
          if (rd > qb) sumMeanUnexplained++;
          if (qb > sumMeanBoundMax) sumMeanBoundMax = qb;
          if (k === 'volumeM3') sumVolOver1pct++;
        }
        if (Number.isFinite(rd) && rd > sumWorst[k]) { sumWorst[k] = rd; sumWorstAt[k] = `${s} h${h} (${a} vs ${b})`; }
        if (!Number.isFinite(rd)) { sumWorst[k] = Infinity; sumWorstAt[k] = `${s} h${h} (${a} vs ${b})`; }
      }
      if (info.hours.includes(h)) cross.push(await crossCheck(s, h, meta, rgba, bIdx, rIdx, info, num, sc.diffIds));
    }
    // timings on the busiest hours of the last storm
    if (s === storms[storms.length - 1]) await timings(storm, bIdx, rIdx, num, runs);
    // an hour that is not in meta.hours comes back as the shared zero grid: no pixel is read
    const zg = (await storm.getHour(1)).rgba;
    t = nowMs();
    const zb = ST.evaluateBuildings(bIdx, zg), zr = ST.evaluateRoads(rIdx, zg), zs = ST.summarizeGrid(meta, zg);
    const zt = nowMs() - t;
    check(`7 ${s}: the loader's shared zero grid takes the fast path`, zg.isZeroGrid === true && zb.floodedCount === 0 && zr.floodedLengthM === 0 && zs.wetCells === 0 && zt < 5,
      `all three evaluations together ${zt.toFixed(2)} ms`);
    storm.dispose();
  }
  num.summaryWorst = sumWorst; num.summaryWorstAt = sumWorstAt;
  check('5 summarizeGrid vs meta.hourly: wet cell count and wet area agree', sumWorst.wetCells <= 0.01 && sumWorst.wetAreaKm2 <= 0.01,
    `${sumRows} storm hours; worst relative difference wetCells ${sumWorst.wetCells.toExponential(2)}${sumWorst.wetCells === 0 ? ' (identical in every hour)' : ' at ' + sumWorstAt.wetCells}, wetArea ${sumWorst.wetAreaKm2.toExponential(2)} at ${sumWorstAt.wetAreaKm2}`);
  check('5 summarizeGrid vs meta.hourly: mean depth and volume within 1 percent, or within the worst-case 8-bit rounding bound of that hour where over', sumMeanUnexplained === 0,
    `worst meanDepth ${(sumWorst.meanDepthM * 100).toFixed(3)} % at ${sumWorstAt.meanDepthM}; worst volume ${(sumWorst.volumeM3 * 100).toFixed(3)} % at ${sumWorstAt.volumeM3}; checks over 1 %: ${sumMeanOver1pct} (volume ${sumVolOver1pct}), unexplained by the bound: ${sumMeanUnexplained}, largest bound among them ${(sumMeanBoundMax * 100).toFixed(2)} % (hours with few wet cells hold shallow cells whose code is coarse: R = 13 is 8 % wide, and rounding does not average out over 42 cells)`);
  check('5 summarizeGrid vs meta.hourly: max depth within 1 percent, or within one depth code where over', sumDepthUnexplained === 0,
    `worst ${(sumWorst.maxDepthM * 100).toFixed(3)} % at ${sumWorstAt.maxDepthM}; hours over 1 %: ${sumDepthOver1pct}, unexplained by one code step: ${sumDepthUnexplained} (a code step is 8(2R+1)/255^2 m: 0.06 m near 6 m, a larger share of a shallow maximum)`);
  check('5 summarizeGrid vs meta.hourly: max speed within 1 percent, or within one velocity code (0.0315 m/s) where over', sumSpeedUnexplained === 0,
    `worst ${(sumWorst.maxSpeedMs * 100).toFixed(3)} % at ${sumWorstAt.maxSpeedMs}; hours over 1 %: ${sumSpeedOver1pct}, unexplained by one code: ${sumSpeedUnexplained} (u and v are 8-bit, step 4/127 m/s, and a maximum picks the cell whose rounding went upward)`);
  for (const rows of cross) for (const r of rows) record(r.name, r.pass, r.detail);
}

async function crossCheck(s, h, meta, rgba, bIdx, rIdx, info, num, diffIds) {
  const tag = `${s} h${String(h).padStart(2, '0')}`;
  const exp = await getJson(`./fixture/bld_${s}_h${String(h).padStart(2, '0')}.json`);
  const res = ST.evaluateBuildings(bIdx, rgba);
  const n = bIdx.count;
  let codeEq = 0, codeOne = 0, codeBig = 0, floodDisagree = 0;
  for (let i = 0; i < n; i++) {
    const code = res.depthM[i] > 0 ? Math.round(255 * Math.sqrt(res.depthM[i] / 8)) : 0;
    const dc = Math.abs(code - exp.maxR[i]);
    if (dc === 0) codeEq++; else if (dc === 1) codeOne++; else codeBig++;
    if ((res.depthM[i] >= 0.1) !== (decodeDepth(exp.maxR[i]) >= 0.1)) floodDisagree++;
  }
  const expFlooded = exp.tiers.flooded;
  const fdiff = res.floodedCount - expFlooded;
  const out = [];
  const tiersOk = ['shallow', 'moderate', 'deep', 'severe'].map((k) => res.tierCounts[k] - exp.tiers[k]);
  out.push({ name: `6 ${tag}: flooded building count (JS vs rasterio)`, pass: Math.abs(fdiff) <= Math.max(0, 0.003 * expFlooded),
    detail: `JS ${res.floodedCount} vs GDAL ${expFlooded} (diff ${fdiff}, ${expFlooded ? (100 * fdiff / expFlooded).toFixed(3) : 0} %); tier diffs shallow/moderate/deep/severe ${tiersOk.join('/')}` });
  out.push({ name: `6 ${tag}: per-building max depth within one depth code`, pass: codeBig === 0,
    detail: `${n} buildings: identical code ${codeEq}, off by one ${codeOne}, off by more ${codeBig}; flooded/not-flooded disagreements ${floodDisagree}` });
  num[`cross_${tag}`] = { jsFlooded: res.floodedCount, gdalFlooded: expFlooded, codeEq, codeOne, codeBig, floodDisagree, tierDiff: tiersOk };
  // mean depth and speed against the GDAL pixel sets (same pixels => same numbers)
  let mMax = 0, sMax = 0, mCnt = 0, skipped = 0;
  for (let i = 0; i < n; i++) {
    if (diffIds.has(i)) { skipped++; continue; }   // a different pixel set legitimately changes mean and speed
    const dm = Math.abs(res.meanDepthM[i] - exp.mean[i]); if (dm > mMax) mMax = dm; if (dm > 1e-5) mCnt++;
    const ds = Math.abs(res.speedMs[i] - exp.speed[i]); if (ds > sMax) sMax = ds;
  }
  out.push({ name: `6 ${tag}: mean depth and speed at the deepest pixel equal numpy on the same pixels (float32 rounding only)`, pass: mMax < 1e-5 && sMax < 1e-5,
    detail: `max |mean diff| ${mMax.toExponential(2)} m, max |speed diff| ${sMax.toExponential(2)} m/s over ${n - skipped} buildings with identical pixel sets (${skipped} with different sets skipped)` });
  num[`cross_${tag}`].meanMax = mMax; num[`cross_${tag}`].speedMax = sMax; num[`cross_${tag}`].meanOver = mCnt;

  const er = await getJson(`./fixture/roads_${s}_h${String(h).padStart(2, '0')}.json`);
  const rr = ST.evaluateRoads(rIdx, rgba);
  let pfMax = 0, pfSum = 0, pfBig = 0;
  for (let i = 0; i < rIdx.count; i++) { const d = Math.abs(rr.perFeaturePct[i] - er.per_feature_pct[i]); pfSum += d; if (d > pfMax) pfMax = d; if (d > 10) pfBig++; }
  out.push({ name: `6 ${tag}: road share under water within 1 percentage point of shapely/numpy (1 m samples)`, pass: Math.abs(rr.pctFlooded - er.pct) <= 1 && relDiff(rr.totalLengthM, er.total_length_m) < 1e-3,
    detail: `JS ${rr.pctFlooded.toFixed(3)} % vs ${er.pct.toFixed(3)} %; total length ${(rr.totalLengthM / 1000).toFixed(3)} km vs ${(er.total_length_m / 1000).toFixed(3)} km; per-feature |diff| mean ${(pfSum / rIdx.count).toFixed(2)} max ${pfMax.toFixed(1)} pct points (${pfBig} features over 10 pp); flooded features ${rr.floodedFeatures}` });
  num[`roads_${tag}`] = { js: rr.pctFlooded, ref: er.pct, perFeatureMean: pfSum / rIdx.count, perFeatureMax: pfMax, over10: pfBig };
  return out;
}

async function timings(storm, bIdx, rIdx, num, runs) {
  const meta = storm.meta;
  for (const h of [13, 24]) {
    if (!meta.hours.includes(h)) continue;
    const { rgba } = await storm.getHour(h);
    const time = (fn) => { fn(); fn(); const a = []; for (let i = 0; i < runs; i++) { const t = nowMs(); fn(); a.push(nowMs() - t); } return { median: median(a), min: Math.min(...a), max: Math.max(...a) }; };
    const tb = time(() => ST.evaluateBuildings(bIdx, rgba));
    const tr = time(() => ST.evaluateRoads(rIdx, rgba));
    const ts = time(() => ST.summarizeGrid(meta, rgba));
    num.timings[`${storm.storm}_h${h}`] = { evaluateBuildings: tb, evaluateRoads: tr, summarizeGrid: ts };
    check(`7 ${storm.storm} h${h}: evaluateBuildings and evaluateRoads each under 25 ms`, tb.median < 25 && tr.median < 25,
      `evaluateBuildings median ${tb.median.toFixed(2)} ms (min ${tb.min.toFixed(2)}, max ${tb.max.toFixed(2)}), evaluateRoads median ${tr.median.toFixed(2)} ms (min ${tr.min.toFixed(2)}, max ${tr.max.toFixed(2)}), summarizeGrid median ${ts.median.toFixed(1)} ms; ${runs} runs`);
  }
}

async function syntheticAll() {
  syntheticTier();
  const ctx = handTests();
  zeroTests(ctx);
  roadTests();
  summaryTests();
  randomPolygonTest();
}

let running = false;
async function runAll(syntheticOnly) {
  if (running) return;
  running = true;
  window.__results.length = 0;
  window.__done = false;
  listEl.textContent = '';
  statusEl.textContent = 'running...';
  const t0 = nowMs();
  try {
    await syntheticAll();
    if (!syntheticOnly && typeof realAll === 'function') await realAll();
  } catch (e) {
    record('harness crashed', false, e && e.stack ? e.stack : e);
  }
  const pass = window.__results.filter((r) => r.pass).length, total = window.__results.length;
  summaryEl.textContent = `${pass} / ${total} passed, ${total - pass} failed (${((nowMs() - t0) / 1000).toFixed(1)} s)`;
  statusEl.textContent = pass === total ? 'all passed' : 'FAILURES';
  window.__done = true;
  running = false;
}
document.getElementById('run').onclick = () => runAll(false);
document.getElementById('runSynthetic').onclick = () => runAll(true);
window.__run = runAll;
if (!params.has('manual')) runAll(params.has('synthetic'));
else statusEl.textContent = 'ready';
