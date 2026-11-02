// fluid-renderer.js - WebGL2 renderer for bgflow grids (see FLOW_FORMAT.md).
//
// Draws one REGION of the flow grid (a rectangle in Web Mercator metres) into its own
// transparent, premultiplied-alpha canvas, in this order:
//   1. depth colour (viewer water palette, soft shoreline)
//   2. animated flowing-water surface (flow-map advected procedural noise, sun glints, foam)
//   3. particle streaks (CPU advected, drawn as thick lines into a fading trail buffer)
//   4. flow arrows (instanced triangles on a screen-space grid)
//   5. flow-paths overlay (cumulative discharge channels)
// The canvas is meant to be draped over a map, so `opacity` only scales the water layer.
//
// Coordinates: the grid has row 0 at the NORTH edge, u points east and v points north.
// Shaders never see absolute Mercator metres (float32 would quantise them to ~1 m at
// x = -9.6e6); they work in metres relative to the grid's north-west corner.

const DEFAULTS = Object.freeze({
  showDepth: true,
  showFlow: true,
  showParticles: true,
  showArrows: true,
  showFlowpaths: false,
  opacity: 1,
  flowSpeed: 8,          // pattern drift = real velocity * flowSpeed
  particleCount: 60000,
  particleSpeed: 40,     // visual speed = real velocity * particleSpeed
  particleLife: 6,       // seconds
  trailFade: 0.04,       // fraction of the trail removed per 1/60 s
  arrowSpacingPx: 40,
  sunAzimuthDeg: 225,    // compass bearing of the sun, clockwise from north
  sunElevationDeg: 55,
  minDepthShown: 0.02,   // metres; the shoreline fades in over the next 0.08 m
  flowCycleSec: 3,       // flow-map cycle length
  rippleStrength: 1,
  rippleScalePx: 16,     // wavelength of the finest ripples on screen
  particleWidthPx: 1.6,
  particleAlpha: 0.9,
  quality: 2,            // 0 low, 1 medium, 2 high: ripple octaves 2/3/4 and foam taps 3/4/6
  particleBudgetMs: 6,   // CPU time one frame may spend advecting particles; the pass is sliced across frames to fit
  particleCoverage: 0.6, // target trail ink per screen pixel in the densest water; caps the ACTIVE particle count (0 = always use particleCount)
  arrowScale: 1,
  flowpathsOpacity: 0.9,
  viewPitchDeg: 0,       // 0 = looking straight down; raises the sky reflection
  viewBearingDeg: 0,     // heading of the camera for a pitched view
  seed: 1,
});

const BOOLEANS = ['showDepth', 'showFlow', 'showParticles', 'showArrows', 'showFlowpaths'];
const LIMITS = {
  opacity: [0, 1], flowSpeed: [0, 100], particleCount: [0, 200000], particleSpeed: [0, 1000],
  particleLife: [0.2, 120], trailFade: [0.002, 0.5], arrowSpacingPx: [8, 400],
  sunAzimuthDeg: [-720, 720], sunElevationDeg: [1, 89], minDepthShown: [0, 2],
  flowCycleSec: [0.5, 30], rippleStrength: [0, 4], rippleScalePx: [2, 64],
  particleWidthPx: [0.5, 8], particleAlpha: [0, 1], particleBudgetMs: [0.5, 1000], particleCoverage: [0, 8], quality: [0, 2], arrowScale: [0.25, 4],
  flowpathsOpacity: [0, 1], viewPitchDeg: [0, 85], viewBearingDeg: [-720, 720], seed: [0, 4294967295],
};

const MIN_DEPTH_PARTICLE = 0.03;   // m, below this a particle is on dry ground
const MIN_SPEED_STILL = 0.02;      // m/s
const STILL_SECONDS = 1.5;
const SPAWN_TRIES = 10;
const MAX_SLICES = 4;
const SPAWN_TRIAL_BUDGET = 30000;  // depth probes per frame, keeps the CPU pass bounded on sparse floods
const CAP_MIN = 256;               // fewest particles the coverage cap leaves active
const CAP_NX = 64, CAP_NY = 32;    // probe grid of the coverage estimate
const CAP_EVERY = 20;              // frames between two estimates
// Most canvas pixels setRegion will ask the browser for (4096 x 4096). A WebGL drawing buffer is 4 bytes
// per pixel, the trail buffer 8 (RGBA16F) and 4x multisampling multiplies the former; browsers also cut
// buffers above about 33 M pixels down without telling the page. Override with the maxCanvasPixels option.
const MAX_CANVAS_PIXELS = 1 << 24;
const EARTH_R = 6378137;
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

// ------------------------------------------------------------------ shaders

const VS_FULLSCREEN = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Shared by the water, arrow and flow-path programs: how a canvas position maps to the grid.
const GLSL_GRID = `
uniform vec2 uGridOff;     // metres from the grid NW corner to the region NW corner (east, south)
uniform vec2 uRegionSize;  // region extent in metres (east, north)
uniform vec2 uGridSize;    // grid extent in metres
uniform float uDepthMax;
uniform float uVmax;
`;

const GLSL_PALETTE = `
const vec3 PAL0 = vec3(0.5608, 0.8275, 0.9569); // #8fd3f4  0.1 m
const vec3 PAL1 = vec3(0.2275, 0.6275, 0.8784); // #3aa0e0  0.3 m
const vec3 PAL2 = vec3(0.0824, 0.3765, 0.7216); // #1560b8  0.6 m
const vec3 PAL3 = vec3(0.0392, 0.1843, 0.4784); // #0a2f7a  1.2 m and deeper
vec3 depthColor(float d) {
  vec3 c = mix(PAL0, PAL1, clamp((d - 0.1) / 0.2, 0.0, 1.0));
  c = mix(c, PAL2, clamp((d - 0.3) / 0.3, 0.0, 1.0));
  return mix(c, PAL3, clamp((d - 0.6) / 0.6, 0.0, 1.0));
}`;

const FS_WATER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTexA;
uniform sampler2D uTexB;
uniform float uMix;
uniform float uMercScale;
uniform float uPhase;
uniform float uCycle;
uniform float uTime;
uniform float uFlowSpeed;
uniform float uOpacity;
uniform float uMinDepth;
uniform vec3 uSunDir;
uniform vec3 uViewDir;
uniform float uShowDepth;
uniform float uShowFlow;
uniform float uRipple;
uniform float uOctF;
uniform float uOctBase;
uniform int uOctaves;
uniform int uFoamTaps;
${GLSL_GRID}
${GLSL_PALETTE}

uvec2 pcg2d(uvec2 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;
  v ^= v >> 16u;
  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;
  v ^= v >> 16u;
  return v;
}
vec2 grad2(vec2 c) {
  uvec2 h = pcg2d(uvec2(ivec2(c)));
  return vec2(h & 0xFFFFu) * (1.0 / 32767.5) - 1.0;
}
// Gradient noise with analytic derivatives: (value, d/dx, d/dy).
vec3 gnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = p - i;
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  vec2 ga = grad2(i);
  vec2 gb = grad2(i + vec2(1.0, 0.0));
  vec2 gc = grad2(i + vec2(0.0, 1.0));
  vec2 gd = grad2(i + vec2(1.0, 1.0));
  float va = dot(ga, f);
  float vb = dot(gb, f - vec2(1.0, 0.0));
  float vc = dot(gc, f - vec2(0.0, 1.0));
  float vd = dot(gd, f - vec2(1.0, 1.0));
  float k = va - vb - vc + vd;
  float val = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
  vec2 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
         + du * (u.yx * k + vec2(vb - va, vc - va));
  return vec3(val, d);
}

const float SLOPE0 = 0.18;      // slope amplitude of the finest ripple octave
const float LOG2_G = -0.86;     // log2 of the amplitude ratio between octaves (0.55)
const vec2 DRIFT[4] = vec2[4](vec2(0.055, 0.031), vec2(-0.037, 0.049), vec2(0.028, -0.052), vec2(-0.045, -0.023));
const vec2 OFFS[4] = vec2[4](vec2(0.0, 0.0), vec2(17.3, 5.9), vec2(-8.1, 23.7), vec2(41.9, -13.3));

// Two to four ripple octaves at fixed WORLD wavelengths (powers of two in metres); the finest one
// fades out and a coarser one fades in as the zoom changes, so the pattern stays anchored
// to the ground. Each octave also drifts slowly on its own (a fixed fraction of its
// wavelength per second) so that still water keeps a gentle shimmer. Returns the surface
// slope (east, north) in .yz; .x is the height in slope-metres (unused by the caller).
vec3 waterLayer(vec2 q, vec2 layerOff) {
  vec2 acc = vec2(0.0);
  float h = 0.0;
  float lam = uOctBase;
  for (int k = 0; k < 4; k++) {
    if (k >= uOctaves) break;
    float wk = (k == 0) ? (1.0 - uOctF) : ((k == uOctaves - 1) ? uOctF : 1.0);
    float sk = wk * SLOPE0 * exp2((float(k) - uOctF) * LOG2_G);
    vec3 n = gnoise(q / lam + DRIFT[k] * uTime + OFFS[k] + layerOff);
    acc += sk * n.yz;
    h += sk * lam * n.x;
    lam *= 2.0;
  }
  return vec3(h, acc);
}

// Foam texture stretched along the flow. The noise lattice is NOT rotated into the flow
// frame (|q| is thousands of metres, so any change of direction would twist the lattice
// wildly); instead 3 to 6 taps are averaged along the flow direction, spread over the same length.
float foamNoise(vec2 q, vec2 dir, float lf) {
  float s = 0.0;
  float n = float(uFoamTaps);
  float step = lf * 0.34 * 6.0 / n;
  for (int i = 0; i < 6; i++) {
    if (i >= uFoamTaps) break;
    s += gnoise((q + dir * ((float(i) - 0.5 * (n - 1.0)) * step)) / lf).x;
  }
  return s / n;
}

void main() {
  vec2 gm = uGridOff + vUv * uRegionSize;
  vec2 guv = gm / uGridSize;
  vec4 ta = texture(uTexA, guv);
  vec4 tb = texture(uTexB, guv);
  float inside = step(0.0, guv.x) * step(guv.x, 1.0) * step(0.0, guv.y) * step(guv.y, 1.0);

  float depth = mix(ta.r * ta.r, tb.r * tb.r, uMix) * uDepthMax;
  vec2 velA = (ta.gb - 128.0 / 255.0) * (255.0 / 127.0) * uVmax * step(0.0005, ta.r);
  vec2 velB = (tb.gb - 128.0 / 255.0) * (255.0 / 127.0) * uVmax * step(0.0005, tb.r);
  vec2 vel = mix(velA, velB, uMix);
  float valid = max(ta.a, tb.a) * inside;
  float shore = smoothstep(uMinDepth, uMinDepth + 0.08, depth);
  float wetA = shore * valid;
  if (wetA <= 0.0) { outColor = vec4(0.0); return; }

  vec3 baseC = depthColor(depth);
  float baseA = mix(0.42, 0.92, smoothstep(0.0, 1.0, depth)) * wetA * uOpacity;
  vec3 outC = baseC;
  float outA = uShowDepth > 0.5 ? baseA : 0.0;

  if (uShowFlow > 0.5) {
    float speed = length(vel);
    vec2 w = vec2(gm.x, -gm.y);                      // east, north
    vec2 vm = vel * (uMercScale * uFlowSpeed);       // pattern drift in Mercator m/s
    float ph0 = uPhase;
    float ph1 = fract(uPhase + 0.5);
    float wt0 = 1.0 - abs(2.0 * ph0 - 1.0);
    float wt1 = 1.0 - abs(2.0 * ph1 - 1.0);
    vec2 q0 = w - vm * ((ph0 - 0.5) * uCycle);
    vec2 q1 = w - vm * ((ph1 - 0.5) * uCycle);
    vec3 s0 = waterLayer(q0, vec2(0.0));
    vec3 s1 = waterLayer(q1, vec2(37.31, 11.77));
    float nrm = inversesqrt(wt0 * wt0 + wt1 * wt1);
    float rough = uRipple * mix(0.22, 1.0, smoothstep(0.0, 1.2, speed));
    vec2 slope = (wt0 * s0.yz + wt1 * s1.yz) * (nrm * rough);
    // streaky ripples: damp the slope ALONG the flow so highlights run in streamlines
    vec2 fdir = speed > 1e-3 ? vel / speed : vec2(1.0, 0.0);
    float stretch = 1.0 + 2.6 * smoothstep(0.08, 1.6, speed);
    float sAlong = dot(slope, fdir);
    slope += fdir * (sAlong / stretch - sAlong);

    vec3 N = normalize(vec3(-slope, 1.0));
    vec3 V = normalize(uViewDir);
    vec3 L = normalize(uSunDir);
    float shade = clamp((dot(N, L) - L.z) / max(L.z, 0.25), -0.7, 0.7);
    vec3 H = normalize(L + V);
    float nh = max(dot(N, H), 0.0);
    float spec = pow(nh, 420.0) * 1.9 + pow(nh, 26.0) * 0.07;
    vec3 R = reflect(-V, N);
    float fres = 0.04 + 0.96 * pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 5.0);
    vec3 sky = mix(vec3(0.80, 0.89, 0.97), vec3(0.34, 0.60, 0.92), clamp(R.z, 0.0, 1.0));

    // whitening and streaky foam above ~1 m/s
    float foamAmt = smoothstep(1.0, 3.0, speed);
    float foam = 0.0;
    if (foamAmt > 0.01) {
      float lf = uOctBase * 1.25;
      float f0 = foamNoise(q0, fdir, lf);
      float f1 = foamNoise(q1 + 91.0, fdir, lf);
      float fn = (wt0 * f0 + wt1 * f1) * nrm * 1.3 + 0.5;
      foam = smoothstep(0.56 - 0.25 * foamAmt, 0.98 - 0.25 * foamAmt, fn) * foamAmt;
    }

    if (uShowDepth > 0.5) {
      outC = baseC * (1.0 + shade * 1.1) + vec3(0.04, 0.06, 0.08) * max(shade, 0.0) * 2.0;
      outC = mix(outC, sky, 0.05 + fres * 0.9);
    } else {
      outC = shade > 0.0 ? vec3(0.86, 0.95, 1.0) : vec3(0.02, 0.10, 0.30);
      outA = min(abs(shade) * 1.8, 0.85) * wetA * uOpacity;
    }
    float fa = clamp(max(foam * 0.85, spec), 0.0, 1.0) * wetA * uOpacity;
    vec3 fc = mix(vec3(1.0, 0.97, 0.88), vec3(0.94, 0.98, 1.0), clamp(foam * 2.0, 0.0, 1.0));
    float a = fa + outA * (1.0 - fa);
    outC = (fc * fa + outC * outA * (1.0 - fa)) / max(a, 1e-4);
    outA = a;
  }

  float dn = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
  outC = clamp(outC + dn * (1.5 / 255.0), 0.0, 1.0);
  outA = clamp(outA + dn * (1.0 / 255.0), 0.0, 1.0);
  outColor = vec4(outC * outA, outA);
}`;

const FS_FADE = `#version 300 es
precision highp float;
uniform float uFade;
out vec4 outColor;
void main() { outColor = vec4(0.0, 0.0, 0.0, uFade); }`;

const FS_SOLID = `#version 300 es
precision highp float;
uniform float uValue;
out vec4 outColor;
void main() { outColor = vec4(uValue); }`;

const VS_PARTICLE = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aCorner;   // x along the segment 0..1, y across -1..1
layout(location = 1) in vec4 aSeg;      // previous and current position in canvas px
layout(location = 2) in vec2 aAttr;     // speed (m/s), alpha
uniform vec2 uCanvas;
uniform float uHalfW;
out float vAcross;
out float vSpeed;
out float vAlpha;
void main() {
  vec2 p0 = aSeg.xy;
  vec2 p1 = aSeg.zw;
  vec2 d = p1 - p0;
  float len = length(d);
  vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  float t = aCorner.x;
  vec2 p = mix(p0, p1, t) + dir * ((t * 2.0 - 1.0) * uHalfW) + nrm * (aCorner.y * uHalfW);
  gl_Position = vec4(p.x / uCanvas.x * 2.0 - 1.0, 1.0 - p.y / uCanvas.y * 2.0, 0.0, 1.0);
  vAcross = aCorner.y;
  vSpeed = aAttr.x;
  vAlpha = aAttr.y;
}`;

const FS_PARTICLE = `#version 300 es
precision highp float;
in float vAcross;
in float vSpeed;
in float vAlpha;
uniform float uBright;
out vec4 outColor;
void main() {
  float cover = 1.0 - smoothstep(0.45, 1.0, abs(vAcross));
  vec3 c = mix(vec3(0.58, 0.89, 1.0), vec3(1.0), smoothstep(0.1, 2.0, vSpeed));
  float a = vAlpha * cover * uBright;
  outColor = vec4(c * a, a);
}`;

const FS_TRAIL = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uTrail;
uniform vec2 uCanvas;
out vec4 outColor;
void main() {
  vec4 t = texture(uTrail, gl_FragCoord.xy / uCanvas);
  float a = min(t.a, 1.0);
  outColor = vec4(min(t.rgb, vec3(a)), a);
}`;

const VS_ARROW = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uTexA;
uniform sampler2D uTexB;
uniform float uMix;
uniform vec2 uCanvas;
uniform vec2 uMppXY;      // metres per pixel
uniform float uSpacing;
uniform int uNx;
uniform float uLod;
uniform float uInflate;
uniform float uScale;
uniform float uPass;      // 0 = outline, 1 = fill
${GLSL_GRID}
out vec4 vColor;

vec2 glyph(int id, float L, float ws, float hl, float hw, float e) {
  float tail = -0.5 * L - e;
  float xb = 0.5 * L - hl;
  float xs = xb + 0.5 * hl;
  float tip = 0.5 * L + e * 2.0;
  float wy = ws + e;
  float hy = hw + e * 1.5;
  float xhb = xb - e;
  if (id == 0) return vec2(tail, -wy);
  if (id == 1) return vec2(xs, -wy);
  if (id == 2) return vec2(xs, wy);
  if (id == 3) return vec2(tail, -wy);
  if (id == 4) return vec2(xs, wy);
  if (id == 5) return vec2(tail, wy);
  if (id == 6) return vec2(xhb, -hy);
  if (id == 7) return vec2(tip, 0.0);
  return vec2(xhb, hy);
}

void main() {
  int ix = gl_InstanceID % uNx;
  int iy = gl_InstanceID / uNx;
  vec2 c = (vec2(float(ix), float(iy)) + 0.5) * uSpacing;
  vec2 vUv = c / uCanvas;
  vec2 guv = (uGridOff + vUv * uRegionSize) / uGridSize;
  vec4 ta = textureLod(uTexA, guv, uLod);
  vec4 tb = textureLod(uTexB, guv, uLod);
  float inside = step(0.0, guv.x) * step(guv.x, 1.0) * step(0.0, guv.y) * step(guv.y, 1.0);
  float depth = mix(ta.r * ta.r, tb.r * tb.r, uMix) * uDepthMax;
  vec2 velA = (ta.gb - 128.0 / 255.0) * (255.0 / 127.0) * uVmax * step(0.0005, ta.r);
  vec2 velB = (tb.gb - 128.0 / 255.0) * (255.0 / 127.0) * uVmax * step(0.0005, tb.r);
  vec2 vel = mix(velA, velB, uMix);
  float speed = length(vel);
  float valid = max(ta.a, tb.a) * inside;
  if (c.x > uCanvas.x || c.y > uCanvas.y || valid < 0.5 || depth < 0.05 || speed < 0.03) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vColor = vec4(0.0);
    return;
  }
  vec2 dir = normalize(vec2(vel.x / uMppXY.x, -vel.y / uMppXY.y));
  float k = uSpacing * uScale;
  float L = k * clamp(0.20 + 0.52 * sqrt(speed), 0.30, 0.92);
  float ws = max(0.9, 0.032 * k);
  float hl = clamp(0.36 * L, 5.0, 0.5 * L + 2.0);
  float hw = 0.62 * hl;
  vec2 lp = glyph(gl_VertexID, L, ws, hl, hw, uInflate);
  vec2 p = c + dir * lp.x + vec2(-dir.y, dir.x) * lp.y;
  gl_Position = vec4(p.x / uCanvas.x * 2.0 - 1.0, 1.0 - p.y / uCanvas.y * 2.0, 0.0, 1.0);
  float t1 = smoothstep(0.0, 1.0, speed);
  float t2 = smoothstep(1.0, 3.0, speed);
  vec3 fill = mix(mix(vec3(0.72, 0.95, 1.0), vec3(1.0), t1), vec3(1.0, 0.78, 0.30), t2);
  vec4 col = uPass > 0.5 ? vec4(fill, 1.0) : vec4(0.02, 0.07, 0.18, 0.80);
  vColor = vec4(col.rgb * col.a, col.a);
}`;

const FS_ARROW = `#version 300 es
precision highp float;
in vec4 vColor;
out vec4 outColor;
void main() { outColor = vColor; }`;

const FS_FLOWPATH = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTexFP;
uniform float uOpacity;
${GLSL_GRID}
void main() {
  vec2 guv = (uGridOff + vUv * uRegionSize) / uGridSize;
  vec4 t = texture(uTexFP, guv, -0.5);
  float inside = step(0.0, guv.x) * step(guv.x, 1.0) * step(0.0, guv.y) * step(guv.y, 1.0);
  float q = t.g;                                   // sqrt(q / QMAX)
  float wet = smoothstep(0.03, 0.08, t.r);
  float a = smoothstep(0.09, 0.40, q) * wet * t.a * inside * uOpacity;
  vec3 c = mix(vec3(0.00, 0.20, 0.34), vec3(0.06, 0.86, 1.00), smoothstep(0.10, 0.60, q));
  c = mix(c, vec3(1.0), smoothstep(0.60, 1.00, q));
  outColor = vec4(c * a, a);
}`;

// ------------------------------------------------------------------ helpers

function compile(gl, type, src, label) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    const numbered = src.split('\n').map((l, i) => `${String(i + 1).padStart(3)}| ${l}`).join('\n');
    throw new Error(`FluidRenderer: ${label} failed to compile:\n${log}\n${numbered}`);
  }
  return sh;
}

function link(gl, vsSrc, fsSrc, label) {
  const vs = compile(gl, gl.VERTEX_SHADER, vsSrc, `${label} vertex shader`);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc, `${label} fragment shader`);
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(`FluidRenderer: ${label} failed to link: ${log}`);
  }
  return p;
}

function uniforms(gl, prog) {
  const out = {};
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(prog, i);
    out[info.name] = gl.getUniformLocation(prog, info.name);
  }
  return out;
}

function clampNum(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

function int32View(g) {
  return g && LITTLE_ENDIAN && (g.byteOffset & 3) === 0 ? new Int32Array(g.buffer, g.byteOffset, g.length >> 2) : null;
}

function unwrapGrid(g) {
  if (g === null || g === undefined) return null;
  if (g instanceof Uint8Array) return g;
  if (g.rgba instanceof Uint8Array) return g.rgba;
  throw new TypeError('FluidRenderer: expected a Uint8Array or {rgba: Uint8Array}');
}

// Slow water carries little information: spawning thins the particles out there and favours moving water.
function spawnWeight(speed) {
  return 0.12 + 0.88 * Math.min(1, speed * 2.5);
}

// Slow water leaves faint specks, fast water bright streaks.
function speedAlpha(speed) {
  const s = speed < 0.8 ? speed / 0.8 : 1;
  return 0.05 + 0.95 * s * s * (3 - 2 * s);
}

let supportedCache = null;

// ----------------------------------------------------------------- renderer

export class FluidRenderer {
  static supported() {
    if (supportedCache !== null) return supportedCache;
    let ok = false;
    try {
      const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(4, 4) : document.createElement('canvas');
      const gl = c.getContext('webgl2');
      if (gl) {
        ok = gl.getParameter(gl.MAX_TEXTURE_SIZE) >= 4096 && gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) >= 2;
        const lose = gl.getExtension('WEBGL_lose_context');
        if (lose) lose.loseContext();
      }
    } catch (e) {
      ok = false;
    }
    supportedCache = ok;
    return ok;
  }

  constructor(opts = {}) {
    const { canvas, antialias, maxCanvasPixels, ...options } = opts;
    this._canvas = canvas || document.createElement('canvas');
    this._antialias = antialias !== false;
    this._maxPx = Number.isFinite(maxCanvasPixels) && maxCanvasPixels >= 1 ? Math.floor(maxCanvasPixels) : MAX_CANVAS_PIXELS;
    this.options = { ...DEFAULTS };
    this.stats = {
      frames: 0, uploads: 0, shaderCompiles: 0, contextRestores: 0,
      particleMs: 0, drawMs: 0, particlesDrawn: 0, spawnFails: 0, trailFloat: false,
      particleSlices: 1, particleFullPassMs: 0, particleCap: 0,
    };
    this.gpu = { vendor: '', renderer: '' };

    this._meta = null;
    this._W = 0; this._H = 0; this._cell = 1; this._invCell = 1;
    this._gxmin = 0; this._gymax = 0; this._gw = 1; this._gh = 1;
    this._depthMax = 8; this._vmax = 4; this._mercScale = 1.25;

    this._region = null;      // [xmin, ymin, xmax, ymax]
    this._pxW = 0; this._pxH = 0;
    this._mppX = 1; this._mppY = 1;
    this._gridOffX = 0; this._gridOffY = 0;
    this._octBase = 2; this._octF = 0;

    this._A = null; this._B = null;       // arrays as passed (null = dry)
    this._cpuA = null; this._cpuB = null; // non-zero grids the particle pass samples
    this._i32A = null; this._i32B = null; // the same bytes as one int32 per texel (R | G<<8 | B<<16 | A<<24)
    this._hoursSynced = false;
    this._mix = 0;
    this._fpSrc = null;

    this._sun = new Float32Array(3);
    this._view = new Float32Array(3);

    this._time = 0;
    this._lastMs = null;
    this._lost = false;
    this._disposed = false;
    this._trailClear = true;
    this._warned = new Set();

    this._pn = 0;
    this._eff = 0; this._capTick = 0; this._capDirty = true; this._capEma = 0; this._capMix = 0; this._costPer = 0;
    this._slices = 1; this._chunk = 0; this._chunkDt = new Float64Array(MAX_SLICES); this._passEma = 0; this._passCount = 0;
    this._px = new Float64Array(0);
    this._py = new Float64Array(0);
    this._age = new Float32Array(0);
    this._life = new Float32Array(0);
    this._still = new Float32Array(0);
    this._inst = new Float32Array(0);
    this._rs = 1;
    this._spawnBudget = 0;
    this._sx0 = 0; this._sx1 = 0; this._sy0 = 0; this._sy1 = 0;
    this._sd = 0; this._su = 0; this._sv = 0; this._sok = false;

    this._onLost = (e) => { e.preventDefault(); this._lost = true; };
    this._onRestored = () => {
      if (this._disposed) return;
      this._initGL();
      this._restoreGL();
      this._lost = false;
      this.stats.contextRestores++;
    };
    this._canvas.addEventListener('webglcontextlost', this._onLost, false);
    this._canvas.addEventListener('webglcontextrestored', this._onRestored, false);

    this._initGL();
    this.setOptions(options);
  }

  get canvas() { return this._canvas; }
  get lost() { return this._lost; }
  get region() { return this._region ? this._region.slice() : null; }
  get meta() { return this._meta; }

  // ------------------------------------------------------------------ options

  setOptions(opts = {}) {
    const o = this.options;
    const prevParticles = o.showParticles;
    for (const key of Object.keys(opts)) {
      const v = opts[key];
      if (!(key in DEFAULTS)) {
        if (!this._warned.has(key)) { this._warned.add(key); console.warn(`[FluidRenderer] unknown option "${key}" ignored`); }
        continue;
      }
      if (BOOLEANS.includes(key)) { o[key] = Boolean(v); continue; }
      const n = Number(v);
      if (!Number.isFinite(n)) throw new TypeError(`FluidRenderer: option ${key} must be a finite number (got ${v})`);
      const lim = LIMITS[key];
      o[key] = lim ? clampNum(n, lim[0], lim[1]) : n;
    }
    o.particleCount = Math.round(o.particleCount);
    if ('seed' in opts) { o.seed = Math.floor(o.seed); this._rs = o.seed | 0; }
    const hadParticles = this._pn > 0;
    if (o.particleCount !== this._pn) this._resizeParticles(o.particleCount);
    this._capDirty = true;

    const az = o.sunAzimuthDeg * Math.PI / 180, el = o.sunElevationDeg * Math.PI / 180;
    this._sun[0] = Math.sin(az) * Math.cos(el);
    this._sun[1] = Math.cos(az) * Math.cos(el);
    this._sun[2] = Math.sin(el);
    const pitch = o.viewPitchDeg * Math.PI / 180, brg = o.viewBearingDeg * Math.PI / 180;
    this._view[0] = -Math.sin(brg) * Math.sin(pitch);
    this._view[1] = -Math.cos(brg) * Math.sin(pitch);
    this._view[2] = Math.cos(pitch);
    this._updateOctaves();
    if (prevParticles && !o.showParticles) this.clearTrails();
    if (o.particleCount === 0) {
      // frame() runs no trail pass without particles, so streaks left in the buffer would never fade
      if (hadParticles) this.clearTrails();
      this._pCount = 0;
      this.stats.particlesDrawn = 0;
      this.stats.particleCap = 0;
    }
    return this;
  }

  getOptions() { return { ...this.options }; }

  // ------------------------------------------------------------------ setup

  _initGL() {
    const gl = this._canvas.getContext('webgl2', {
      alpha: true, premultipliedAlpha: true, antialias: this._antialias,
      preserveDrawingBuffer: true, depth: false, stencil: false, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('FluidRenderer: WebGL2 is not available');
    this.gl = gl;
    this._maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this._maxRb = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE);
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    if (dbg) {
      this.gpu.vendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL);
      this.gpu.renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
    }
    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('EXT_color_buffer_half_float');

    this._pWater = link(gl, VS_FULLSCREEN, FS_WATER, 'water');
    this._pFade = link(gl, VS_FULLSCREEN, FS_FADE, 'trail fade');
    this._pSolid = link(gl, VS_FULLSCREEN, FS_SOLID, 'trail clear');
    this._pParticle = link(gl, VS_PARTICLE, FS_PARTICLE, 'particles');
    this._pTrail = link(gl, VS_FULLSCREEN, FS_TRAIL, 'trail composite');
    this._pArrow = link(gl, VS_ARROW, FS_ARROW, 'arrows');
    this._pFlowpath = link(gl, VS_FULLSCREEN, FS_FLOWPATH, 'flow paths');
    this.stats.shaderCompiles += 7;
    this._uWater = uniforms(gl, this._pWater);
    this._uFade = uniforms(gl, this._pFade);
    this._uSolid = uniforms(gl, this._pSolid);
    this._uParticle = uniforms(gl, this._pParticle);
    this._uTrail = uniforms(gl, this._pTrail);
    this._uArrow = uniforms(gl, this._pArrow);
    this._uFlowpath = uniforms(gl, this._pFlowpath);

    gl.useProgram(this._pWater);
    gl.uniform1i(this._uWater.uTexA, 0);
    gl.uniform1i(this._uWater.uTexB, 1);
    gl.useProgram(this._pArrow);
    gl.uniform1i(this._uArrow.uTexA, 0);
    gl.uniform1i(this._uArrow.uTexB, 1);
    gl.useProgram(this._pFlowpath);
    gl.uniform1i(this._uFlowpath.uTexFP, 2);
    gl.useProgram(this._pTrail);
    gl.uniform1i(this._uTrail.uTrail, 3);

    this._vaoEmpty = gl.createVertexArray();

    // particle quad (triangle strip) + per-instance segment buffer
    this._vaoParticle = gl.createVertexArray();
    gl.bindVertexArray(this._vaoParticle);
    this._bufCorner = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._bufCorner);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this._bufInst = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._bufInst);
    gl.bufferData(gl.ARRAY_BUFFER, Math.max(24, this._inst.byteLength), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 24, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 24, 16);
    gl.vertexAttribDivisor(2, 1);
    gl.bindVertexArray(null);

    // 1x1 all-zero texture standing in for dry hours
    this._zeroTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._zeroTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 1, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this._slots = [{ tex: null, src: null }, { tex: null, src: null }];
    this._hoursSynced = false;
    this._texA = this._zeroTex;
    this._texB = this._zeroTex;
    this._fpTex = null;
    this._trailTex = null;
    this._trailFbo = null;
    this._trailW = 0; this._trailH = 0;
    this._trailClear = true;
  }

  // After a context restore every GL object is new; re-create the ones that depend on state.
  _restoreGL() {
    if (this._meta) {
      this._syncHours();
      if (this._fpSrc) this._uploadFlowpaths(this._fpSrc);
    }
    if (this._region) this._ensureTrail();
  }

  _gridTexture(w, h) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const levels = Math.floor(Math.log2(Math.max(w, h))) + 1;
    gl.texStorage2D(gl.TEXTURE_2D, levels, gl.RGBA8, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  _uploadGrid(tex, arr) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this._W, this._H, gl.RGBA, gl.UNSIGNED_BYTE, arr);
    gl.generateMipmap(gl.TEXTURE_2D);
    const err = gl.getError();
    if (err !== gl.NO_ERROR) throw new Error(`FluidRenderer: texture upload failed (GL error 0x${err.toString(16)}); grid ${this._W}x${this._H}`);
    this.stats.uploads++;
  }

  _releaseGridTextures() {
    const gl = this.gl;
    for (const s of this._slots) {
      if (s.tex && !this._lost) gl.deleteTexture(s.tex);
      s.tex = null; s.src = null;
    }
    if (this._fpTex && !this._lost) gl.deleteTexture(this._fpTex);
    this._fpTex = null;
    this._texA = this._zeroTex;
    this._texB = this._zeroTex;
  }

  // ------------------------------------------------------------------ storm / data

  setStorm(meta) {
    if (!meta || !(meta.width > 0) || !(meta.height > 0) || !(meta.cell_m > 0) || !Array.isArray(meta.bounds_merc)) {
      throw new TypeError('FluidRenderer.setStorm: meta needs width, height, cell_m and bounds_merc');
    }
    if (meta.width > this._maxTex || meta.height > this._maxTex) {
      throw new Error(`FluidRenderer: grid ${meta.width}x${meta.height} exceeds this GPU's MAX_TEXTURE_SIZE ${this._maxTex}`);
    }
    if (!this._lost) this._releaseGridTextures();
    this._meta = meta;
    this._W = meta.width; this._H = meta.height;
    this._cell = meta.cell_m; this._invCell = 1 / meta.cell_m;
    this._gxmin = meta.bounds_merc[0];
    this._gymax = meta.bounds_merc[3];
    this._gw = meta.width * meta.cell_m;
    this._gh = meta.height * meta.cell_m;
    this._depthMax = meta.depth_encoding && meta.depth_encoding.max_m ? meta.depth_encoding.max_m : 8;
    this._vmax = meta.vel_encoding && meta.vel_encoding.vmax_ms ? meta.vel_encoding.vmax_ms : 4;
    const yMid = (meta.bounds_merc[1] + meta.bounds_merc[3]) / 2;
    const lat = 2 * Math.atan(Math.exp(yMid / EARTH_R)) - Math.PI / 2;
    this._mercScale = 1 / Math.cos(lat);   // Mercator metres per ground metre
    this._A = null; this._B = null; this._cpuA = null; this._cpuB = null; this._i32A = null; this._i32B = null; this._mix = 0;
    this._hoursSynced = false;
    this._fpSrc = null;
    this._killParticles();
    this._capDirty = true; this._capEma = 0;
    this.clearTrails();
    this._updateGridOffsets();
    this._updateSpawnRect();
    this._updateOctaves();
    return this;
  }

  setHours(rgbaA, rgbaB, mix = 0) {
    if (!this._meta) throw new Error('FluidRenderer.setHours: call setStorm first');
    const A = unwrapGrid(rgbaA), B = unwrapGrid(rgbaB);
    const m = Number(mix);
    this._mix = Number.isFinite(m) ? clampNum(m, 0, 1) : 0;
    if (A === this._A && B === this._B && this._hoursSynced) {   // only the blend factor changed
      if (Math.abs(this._mix - this._capMix) >= 0.1) this._capDirty = true;
      return this;
    }
    this._capDirty = true;
    const need = this._W * this._H * 4;
    if (A && A.length !== need) throw new Error(`FluidRenderer.setHours: array A has ${A.length} bytes, expected ${need}`);
    if (B && B.length !== need) throw new Error(`FluidRenderer.setHours: array B has ${B.length} bytes, expected ${need}`);
    this._A = A; this._B = B;
    this._cpuA = A && !A.isZeroGrid ? A : null;
    this._cpuB = B && !B.isZeroGrid ? B : null;
    this._i32A = int32View(this._cpuA);
    this._i32B = int32View(this._cpuB);
    this._hoursSynced = false;
    if (!this._lost) this._syncHours();
    return this;
  }

  // Make the two sampler slots hold A and B. Textures are only uploaded when an array
  // identity is new: advancing an hour (A <- B, B <- next) re-uses the existing texture
  // for B and uploads just one grid.
  _syncHours() {
    const gl = this.gl;
    const A = this._cpuA, B = this._cpuB;
    for (let i = 0; i < 2; i++) {
      const s = this._slots[i];
      if (!s.tex && (A || B)) s.tex = this._gridTexture(this._W, this._H);
    }
    let sa = A ? this._findSlot(A) : null;
    let sb = B ? this._findSlot(B) : null;
    if (A && !sa) sa = this._uploadToSlot(A, sb);
    if (B && !sb) sb = this._findSlot(B) || this._uploadToSlot(B, sa);
    this._texA = A ? sa.tex : this._zeroTex;
    this._texB = B ? sb.tex : this._zeroTex;
    this._hoursSynced = true;
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  _findSlot(arr) {
    for (let i = 0; i < 2; i++) if (this._slots[i].src === arr) return this._slots[i];
    return null;
  }

  // Upload `arr` into the slot that is not `protect`, preferring an empty one.
  _uploadToSlot(arr, protect) {
    let victim = null;
    for (let i = 0; i < 2 && !victim; i++) if (this._slots[i] !== protect && this._slots[i].src === null) victim = this._slots[i];
    for (let i = 0; i < 2 && !victim; i++) if (this._slots[i] !== protect) victim = this._slots[i];
    this._uploadGrid(victim.tex, arr);
    victim.src = arr;
    return victim;
  }

  setFlowpaths(rgba) {
    if (!this._meta) throw new Error('FluidRenderer.setFlowpaths: call setStorm first');
    const arr = unwrapGrid(rgba);
    if (arr && arr.length !== this._W * this._H * 4) {
      throw new Error(`FluidRenderer.setFlowpaths: array has ${arr.length} bytes, expected ${this._W * this._H * 4}`);
    }
    if (arr === this._fpSrc) return this;
    this._fpSrc = arr;
    if (!this._lost) this._uploadFlowpaths(arr);
    return this;
  }

  _uploadFlowpaths(arr) {
    const gl = this.gl;
    if (!arr) {
      if (this._fpTex) gl.deleteTexture(this._fpTex);
      this._fpTex = null;
      return;
    }
    if (!this._fpTex) this._fpTex = this._gridTexture(this._W, this._H);
    this._uploadGrid(this._fpTex, arr);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ------------------------------------------------------------------ region

  setRegion(region, pxW, pxH) {
    if (!Array.isArray(region) || region.length !== 4 || !region.every(Number.isFinite) || !(region[0] < region[2]) || !(region[1] < region[3])) {
      throw new TypeError('FluidRenderer.setRegion: region must be [xmin, ymin, xmax, ymax] with xmin < xmax and ymin < ymax');
    }
    let w = Math.round(pxW), h = Math.round(pxH);
    if (!(w >= 1 && h >= 1)) throw new TypeError(`FluidRenderer.setRegion: bad canvas size ${pxW}x${pxH}`);
    const lim = Math.min(this._maxTex, this._maxRb);
    if (w > lim || h > lim) {
      console.warn(`[FluidRenderer] canvas ${w}x${h} clamped to this GPU's limit ${lim}`);
      w = Math.min(w, lim); h = Math.min(h, lim);
    }
    if (w * h > this._maxPx) {
      // both sides shrink by the same factor, so the picture keeps its shape and still covers the whole region
      const s = Math.sqrt(this._maxPx / (w * h));
      const w2 = Math.max(1, Math.floor(w * s)), h2 = Math.max(1, Math.floor(h * s));
      console.warn(`[FluidRenderer] canvas ${w}x${h} (${(w * h / 1e6).toFixed(1)} M px) reduced to ${w2}x${h2}: the limit is ${(this._maxPx / 1e6).toFixed(1)} M px`);
      w = w2; h = h2;
    }
    this._region = region.slice();
    if (this._canvas.width !== w || this._canvas.height !== h) { this._canvas.width = w; this._canvas.height = h; }
    if (!this._lost) {
      // The browser may hand out a smaller drawing buffer than asked for (it never throws). Viewport,
      // trail texture and metres per pixel must follow the buffer that exists, or the picture is
      // cropped and drifts off the map without any error.
      const bw = this.gl.drawingBufferWidth, bh = this.gl.drawingBufferHeight;
      if (bw >= 1 && bh >= 1 && (bw !== w || bh !== h)) {
        console.warn(`[FluidRenderer] the browser allocated a ${bw}x${bh} drawing buffer for the requested ${w}x${h} canvas; using ${bw}x${bh}`);
        w = bw; h = bh;
        this._canvas.width = w; this._canvas.height = h;
      }
    }
    this._pxW = w; this._pxH = h;
    this._mppX = (region[2] - region[0]) / w;
    this._mppY = (region[3] - region[1]) / h;
    this._updateGridOffsets();
    this._updateSpawnRect();
    this._updateOctaves();
    this._capDirty = true; this._capEma = 0;
    if (!this._lost) this._ensureTrail();
    this.clearTrails();
    return this;
  }

  // Offset of the region's NW corner from the grid's NW corner (east, south), which the shaders use
  // to place the grid. It depends on both the storm bounds and the region, so whichever of
  // setStorm / setRegion comes last has to refresh it.
  _updateGridOffsets() {
    if (!this._region || !this._meta) return;
    this._gridOffX = this._region[0] - this._gxmin;
    this._gridOffY = this._gymax - this._region[3];
  }

  // canvas pixel (top-left origin) <-> Mercator metres
  pixelToWorld(px, py, out = [0, 0]) {
    const r = this._region;
    out[0] = r[0] + px * this._mppX;
    out[1] = r[3] - py * this._mppY;
    return out;
  }

  worldToPixel(x, y, out = [0, 0]) {
    const r = this._region;
    out[0] = (x - r[0]) / this._mppX;
    out[1] = (r[3] - y) / this._mppY;
    return out;
  }

  _updateOctaves() {
    const mpp = Math.max(this._mppX || 1, 1e-3);
    const L = Math.log2(mpp * this.options.rippleScalePx);
    const e0 = Math.floor(L);
    this._octF = L - e0;
    this._octBase = Math.pow(2, e0);
  }

  _updateSpawnRect() {
    if (!this._region || !this._meta) return;
    const r = this._region;
    this._sx0 = Math.max(r[0], this._gxmin);
    this._sx1 = Math.min(r[2], this._gxmin + this._gw);
    this._sy0 = Math.max(r[1], this._gymax - this._gh);
    this._sy1 = Math.min(r[3], this._gymax);
  }

  _ensureTrail() {
    const gl = this.gl;
    const w = this._pxW, h = this._pxH;
    if (this._trailTex && this._trailW === w && this._trailH === h) return;
    if (this._trailTex) { gl.deleteTexture(this._trailTex); gl.deleteFramebuffer(this._trailFbo); }
    const tryFormat = (internal, label) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, internal, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      if (!ok) { gl.deleteFramebuffer(fbo); gl.deleteTexture(tex); return null; }
      return { tex, fbo, label };
    };
    const t = tryFormat(gl.RGBA16F, 'RGBA16F') || tryFormat(gl.RGBA8, 'RGBA8');
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    if (!t) throw new Error('FluidRenderer: could not create a trail framebuffer');
    this._trailTex = t.tex; this._trailFbo = t.fbo;
    this._trailW = w; this._trailH = h;
    this._trailFloat = t.label === 'RGBA16F';
    this.stats.trailFloat = this._trailFloat;
    this._trailClear = true;
  }

  // Empty the trail buffer now (not at the next frame) so a following renderOnce() shows no stale streaks.
  clearTrails() {
    this._trailClear = true;
    if (this._trailFbo && !this._lost && !this._disposed) {
      const gl = this.gl;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._trailFbo);
      gl.disable(gl.BLEND);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this._trailClear = false;
    }
  }

  // ------------------------------------------------------------------ particles

  _resizeParticles(n) {
    if (n === this._pn && this._px.length === n) return;
    const keep = Math.min(n, this._px.length);
    const px = new Float64Array(n), py = new Float64Array(n);
    const age = new Float32Array(n).fill(-2), life = new Float32Array(n), still = new Float32Array(n);
    px.set(this._px.subarray(0, keep)); py.set(this._py.subarray(0, keep));
    age.set(this._age.subarray(0, keep)); life.set(this._life.subarray(0, keep)); still.set(this._still.subarray(0, keep));
    this._px = px; this._py = py; this._age = age; this._life = life; this._still = still;
    this._inst = new Float32Array(n * 6);
    this._pn = n;
    this._eff = this._eff > 0 ? Math.min(this._eff, n) : n;
    if (this.gl && !this._lost) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this._bufInst);
      gl.bufferData(gl.ARRAY_BUFFER, Math.max(24, this._inst.byteLength), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }
  }

  _killParticles() { this._age.fill(-2); }

  _rnd() {
    let t = (this._rs = (this._rs + 0x6D2B79F5) | 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Bilinear sample of the (A, B, mix) field at a Mercator position. Same maths as
  // fluid-data.js sample(), written into fields to avoid allocating in the hot loop.
  // Sets _sok (inside the domain and wet enough), _sd (m), _su/_sv (m/s).
  _samp(x, y) {
    this._sok = false;
    const W = this._W, H = this._H;
    const gx = (x - this._gxmin) * this._invCell;
    const gy = (this._gymax - y) * this._invCell;
    if (!(gx >= 0 && gx <= W && gy >= 0 && gy <= H)) return;
    const fx = gx - 0.5, fy = gy - 0.5;
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const tx = fx - x0, ty = fy - y0;
    const xa = x0 < 0 ? 0 : x0, xb = x0 + 1 > W - 1 ? W - 1 : x0 + 1;
    const ya = y0 < 0 ? 0 : y0, yb = y0 + 1 > H - 1 ? H - 1 : y0 + 1;
    const i00 = (ya * W + xa) << 2, i10 = (ya * W + xb) << 2, i01 = (yb * W + xa) << 2, i11 = (yb * W + xb) << 2;
    const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
    const mix = this._mix;
    const kd = this._depthMax / 65025, kv = this._vmax / 127;
    let depth = 0, u = 0, v = 0, av = 0;
    const A = this._cpuA, B = this._cpuB;
    if (A) {
      const r = A[i00] * w00 + A[i10] * w10 + A[i01] * w01 + A[i11] * w11;
      av = A[i00 + 3] * w00 + A[i10 + 3] * w10 + A[i01 + 3] * w01 + A[i11 + 3] * w11;
      if (r > 0) {
        const wa = 1 - mix;
        depth += wa * r * r * kd;
        u += wa * ((A[i00 + 1] * w00 + A[i10 + 1] * w10 + A[i01 + 1] * w01 + A[i11 + 1] * w11) - 128) * kv;
        v += wa * ((A[i00 + 2] * w00 + A[i10 + 2] * w10 + A[i01 + 2] * w01 + A[i11 + 2] * w11) - 128) * kv;
      }
    }
    if (B && mix > 0) {
      const r = B[i00] * w00 + B[i10] * w10 + B[i01] * w01 + B[i11] * w11;
      const a = B[i00 + 3] * w00 + B[i10 + 3] * w10 + B[i01 + 3] * w01 + B[i11 + 3] * w11;
      if (a > av) av = a;
      if (r > 0) {
        depth += mix * r * r * kd;
        u += mix * ((B[i00 + 1] * w00 + B[i10 + 1] * w10 + B[i01 + 1] * w01 + B[i11 + 1] * w11) - 128) * kv;
        v += mix * ((B[i00 + 2] * w00 + B[i10 + 2] * w10 + B[i01 + 2] * w01 + B[i11 + 2] * w11) - 128) * kv;
      }
    }
    this._sd = depth; this._su = u; this._sv = v;
    this._sok = av >= 127.5 && depth >= MIN_DEPTH_PARTICLE;
  }

  _spawn(i) {
    const sw = this._sx1 - this._sx0, sh = this._sy1 - this._sy0;
    if (!(sw > 0 && sh > 0)) return false;
    for (let t = 0; t < SPAWN_TRIES; t++) {
      if (--this._spawnBudget < 0) return false;
      const x = this._sx0 + this._rnd() * sw;
      const y = this._sy0 + this._rnd() * sh;
      this._samp(x, y);
      if (!this._sok) continue;
      const sp = Math.sqrt(this._su * this._su + this._sv * this._sv);
      if (sp < MIN_SPEED_STILL || this._rnd() > spawnWeight(sp)) continue;
      const life = this.options.particleLife * (0.7 + 0.6 * this._rnd());
      this._px[i] = x; this._py[i] = y;
      this._life[i] = life;
      this._age[i] = this._age[i] === -2 ? this._rnd() * life * 0.95 : 0;
      this._still[i] = 0;
      return true;
    }
    return false;
  }

  // Parks the particles above index `eff`; they respawn when the cap grows again. A change under 5 %
  // is ignored so that a noisy estimate does not park and respawn particles every few frames.
  _setEff(eff) {
    if (!(this._eff > 0 && eff !== this._pn && eff > this._eff * 0.95 && eff < this._eff * 1.05)) {
      if (eff < this._eff) this._age.fill(-2, eff, this._pn);
      this._eff = eff;
    }
    this.stats.particleCap = this._eff;
  }

  // Most particles one pass can advect within particleBudgetMs when it is split over MAX_SLICES frames,
  // from the measured cost per particle. Past that, shedding particles beats updating them below 15 Hz.
  _budgetN() {
    if (this._passCount < 60 || !(this._costPer > 0)) return Infinity;   // too early: the first passes also fill the array
    return Math.max(CAP_MIN, Math.floor(MAX_SLICES * this.options.particleBudgetMs / (1.25 * this._costPer)));
  }

  // How many particles can be active before the trail ink saturates (option particleCoverage).
  // Particles spawn in proportion to spawnWeight(speed), so the densest spot is where weight x ink
  // per particle is largest. A particle's ink is its line width times the length of its fading
  // tail (displacement per frame x the e-folding time of the fade), which grows with zoom and
  // speed: the same 60000 particles that look right at city scale turn a close-up channel white.
  // Estimated on a jittered probe grid over the region, every CAP_EVERY frames and after any
  // change of region, hours or options.
  _updateParticleCap() {
    const o = this.options;
    const n = this._pn;
    if (!this._capDirty && ++this._capTick < CAP_EVERY) return;
    this._capTick = 0; this._capDirty = false; this._capMix = this._mix;
    const lim = this._budgetN();
    if (!(o.particleCoverage > 0)) { this._setEff(Math.min(n, lim)); return; }
    const sw = this._sx1 - this._sx0, sh = this._sy1 - this._sy0;
    if (!(sw > 0 && sh > 0)) { this._setEff(Math.min(n, CAP_MIN)); return; }
    const mpp = Math.sqrt(this._mppX * this._mppY);
    const dispK = o.particleSpeed * this._mercScale / (60 * mpp);   // px per frame per m/s
    const width = o.particleWidthPx;
    const tau = -1 / Math.log(1 - o.trailFade);                     // frames
    let sumW = 0, maxInk = 0;
    for (let j = 0; j < CAP_NY; j++) {
      for (let i = 0; i < CAP_NX; i++) {
        this._samp(this._sx0 + (i + this._rnd()) / CAP_NX * sw, this._sy0 + (j + this._rnd()) / CAP_NY * sh);
        if (!this._sok) continue;
        const sp = Math.sqrt(this._su * this._su + this._sv * this._sv);
        if (sp < MIN_SPEED_STILL) continue;
        const w = spawnWeight(sp);
        const ink = w * width * (sp * dispK * tau + width) * o.particleAlpha * speedAlpha(sp);
        sumW += w;
        if (ink > maxInk) maxInk = ink;
      }
    }
    let eff;
    if (sumW === 0 || !(maxInk > 0)) {
      eff = Math.min(n, CAP_MIN);
    } else {
      const wetPx = sumW / (CAP_NX * CAP_NY) * (sw / this._mppX) * (sh / this._mppY);
      const target = o.particleCoverage * wetPx / maxInk;
      this._capEma = this._capEma > 0 ? 0.5 * this._capEma + 0.5 * target : target;
      eff = clampNum(Math.round(this._capEma), Math.min(n, CAP_MIN), n);
    }
    this._setEff(Math.min(eff, lim));
  }

  // Runs one slice of the particle pass. When a full pass over all particles would cost more
  // than particleBudgetMs, the particles are processed in 2..4 interleaved chunks, one chunk per
  // frame; a chunk then advances by the time since it was last processed, so the motion is the
  // same and only the update rate of an individual particle drops to 30/20/15 Hz.
  _runParticlePass(dt) {
    const n = this._eff;
    const S = this._slices;
    for (let k = 0; k < MAX_SLICES; k++) this._chunkDt[k] += dt;
    const c = this._chunk % S;
    this._chunk++;
    const dtc = this._chunkDt[c];
    this._chunkDt[c] = 0;
    const lo = Math.floor(c * n / S), hi = Math.floor((c + 1) * n / S);
    const t0 = performance.now();
    this._stepParticles(dtc, lo, hi);
    const ms = performance.now() - t0;
    if (hi > lo) {
      const per = ms / (hi - lo);
      this._costPer = this._passCount === 0 ? per : this._costPer * 0.9 + per * 0.1;
      this._passEma = this._costPer * n;
      if (++this._passCount % 30 === 0) {
        const want = clampNum(Math.ceil(this._passEma * 1.25 / this.options.particleBudgetMs), 1, MAX_SLICES);
        if (want !== this._slices) { this._slices = want; this._chunkDt.fill(0); this._chunk = 0; }
        this._capDirty = true;   // re-check the budget limit on the active count
      }
    }
    this.stats.particleSlices = this._slices;
    this.stats.particleFullPassMs = this._passEma;
  }

  // One CPU pass: advect every particle with the sampled velocity, respawn the ones that
  // left the region, went dry, aged out or stalled, and write the draw segments (canvas px).
  // The bilinear sample is inlined here (the same maths as _samp) because this loop runs
  // once per particle per frame and a call per particle costs about a third of the budget.
  _stepParticles(dt, lo, hi) {
    const px = this._px, py = this._py, age = this._age, life = this._life, still = this._still, inst = this._inst;
    const r = this._region;
    const rx0 = r[0], ry0 = r[1], rx1 = r[2], ry1 = r[3];
    const invX = 1 / this._mppX, invY = 1 / this._mppY;
    const k = this.options.particleSpeed * this._mercScale * dt;
    const alphaMax = this.options.particleAlpha;
    const W = this._W, H = this._H, invCell = this._invCell, gxmin = this._gxmin, gymax = this._gymax;
    const A = this._cpuA, B = this._cpuB, mix = this._mix, wa = 1 - mix;
    const useB = B !== null && mix > 0;
    const A32 = this._i32A, B32 = this._i32B;
    const fast = (A === null || A32 !== null) && (B === null || B32 !== null);
    const kd = this._depthMax / 65025, kv = this._vmax / 127;
    const still2 = MIN_SPEED_STILL * MIN_SPEED_STILL;
    this._spawnBudget = SPAWN_TRIAL_BUDGET;
    let cnt = 0, fails = 0;
    for (let i = lo; i < hi; i++) {
      if (age[i] < 0) {
        if (this._spawnBudget > 0 && this._spawn(i)) continue;
        fails++;
        continue;
      }
      const x = px[i], y = py[i];
      const gx = (x - gxmin) * invCell, gy = (gymax - y) * invCell;
      let ok = false, u = 0, v = 0;
      if (gx >= 0 && gx <= W && gy >= 0 && gy <= H) {
        if (fast) {
          const fx = gx - 0.5, fy = gy - 0.5;
          const x0 = fx < 0 ? -1 : fx | 0, y0 = fy < 0 ? -1 : fy | 0;
          const tx = fx - x0, ty = fy - y0;
          const xa = x0 < 0 ? 0 : x0, xb = x0 + 1 >= W ? W - 1 : x0 + 1;
          const ya = y0 < 0 ? 0 : y0, yb = y0 + 1 >= H ? H - 1 : y0 + 1;
          const j00 = ya * W + xa, j10 = ya * W + xb, j01 = yb * W + xa, j11 = yb * W + xb;
          const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
          let depth = 0, av = 0;
          if (A32 !== null) {
            const t00 = A32[j00], t10 = A32[j10], t01 = A32[j01], t11 = A32[j11];
            av = (t00 >>> 24) * w00 + (t10 >>> 24) * w10 + (t01 >>> 24) * w01 + (t11 >>> 24) * w11;
            const rc = (t00 & 255) * w00 + (t10 & 255) * w10 + (t01 & 255) * w01 + (t11 & 255) * w11;
            if (rc > 0) {
              depth = wa * rc * rc * kd;
              u = wa * ((((t00 >> 8) & 255) * w00 + ((t10 >> 8) & 255) * w10 + ((t01 >> 8) & 255) * w01 + ((t11 >> 8) & 255) * w11) - 128) * kv;
              v = wa * ((((t00 >> 16) & 255) * w00 + ((t10 >> 16) & 255) * w10 + ((t01 >> 16) & 255) * w01 + ((t11 >> 16) & 255) * w11) - 128) * kv;
            }
          }
          if (useB) {
            const t00 = B32[j00], t10 = B32[j10], t01 = B32[j01], t11 = B32[j11];
            const ab = (t00 >>> 24) * w00 + (t10 >>> 24) * w10 + (t01 >>> 24) * w01 + (t11 >>> 24) * w11;
            if (ab > av) av = ab;
            const rc = (t00 & 255) * w00 + (t10 & 255) * w10 + (t01 & 255) * w01 + (t11 & 255) * w11;
            if (rc > 0) {
              depth += mix * rc * rc * kd;
              u += mix * ((((t00 >> 8) & 255) * w00 + ((t10 >> 8) & 255) * w10 + ((t01 >> 8) & 255) * w01 + ((t11 >> 8) & 255) * w11) - 128) * kv;
              v += mix * ((((t00 >> 16) & 255) * w00 + ((t10 >> 16) & 255) * w10 + ((t01 >> 16) & 255) * w01 + ((t11 >> 16) & 255) * w11) - 128) * kv;
            }
          }
          ok = av >= 127.5 && depth >= MIN_DEPTH_PARTICLE;
        } else {
          this._samp(x, y);
          ok = this._sok; u = this._su; v = this._sv;
        }
      }
      if (!ok) {
        age[i] = -1;
        if (!this._spawn(i)) fails++;
        continue;
      }
      const sp2 = u * u + v * v;
      const st = sp2 < still2 ? still[i] + dt : 0;
      const nx = x + u * k, ny = y + v * k;
      const a = age[i] + dt;
      if (st > STILL_SECONDS || a > life[i] || nx < rx0 || nx > rx1 || ny < ry0 || ny > ry1) {
        age[i] = -1;
        if (!this._spawn(i)) fails++;
        continue;
      }
      still[i] = st;
      age[i] = a;
      px[i] = nx; py[i] = ny;
      const f = a / life[i];
      const fade = Math.min(f / 0.12, 1) * Math.min((1 - f) / 0.25, 1);
      const speed = Math.sqrt(sp2);
      const o = cnt * 6;
      inst[o] = (x - rx0) * invX;
      inst[o + 1] = (ry1 - y) * invY;
      inst[o + 2] = (nx - rx0) * invX;
      inst[o + 3] = (ry1 - ny) * invY;
      inst[o + 4] = speed;
      const sf = speed < 0.8 ? speed / 0.8 : 1;
      inst[o + 5] = alphaMax * fade * (0.05 + 0.95 * sf * sf * (3 - 2 * sf));   // speedAlpha(), inlined
      cnt++;
    }
    this._pCount = cnt;
    this.stats.particlesDrawn = cnt;
    this.stats.spawnFails = fails;
  }

  // Read-only views for tests and diagnostics. age < 0 marks a particle that is not alive.
  getParticles() {
    return { count: this._pn, active: this._eff, x: this._px, y: this._py, age: this._age, drawn: this._pCount || 0 };
  }

  // ------------------------------------------------------------------ drawing

  _ready() { return !this._lost && !this._disposed && this._meta && this._region && this._pxW > 0; }

  _advanceClock(nowMs) {
    let dt = 0;
    if (this._lastMs !== null && Number.isFinite(nowMs)) dt = clampNum((nowMs - this._lastMs) / 1000, 0, 0.1);
    if (Number.isFinite(nowMs)) this._lastMs = nowMs;
    this._time += dt;
    return dt;
  }

  frame(nowMs) {
    if (this._disposed) return;
    const dt = this._advanceClock(nowMs);
    if (!this._ready()) return;
    const t0 = performance.now();
    const o = this.options;
    if (o.showParticles && this._pn > 0) {
      this._updateParticleCap();
      this._runParticlePass(dt);
      this.stats.particleMs = performance.now() - t0;
      this._drawTrails(dt);
    } else {
      this.stats.particleMs = 0;
    }
    this._draw();
    this.stats.drawMs = performance.now() - t0 - this.stats.particleMs;
    this.stats.frames++;
  }

  // Draw the current state once without advancing time or the particle simulation.
  renderOnce() {
    if (!this._ready()) return;
    this._draw();
  }

  _drawTrails(dt) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._trailFbo);
    gl.viewport(0, 0, this._trailW, this._trailH);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this._vaoEmpty);
    if (this._trailClear) {
      gl.disable(gl.BLEND);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      this._trailClear = false;
    } else if (dt > 0) {
      const fade = 1 - Math.pow(1 - this.options.trailFade, dt * 60);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(this._pFade);
      gl.uniform1f(this._uFade.uFade, fade);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (!this._trailFloat) {
        // 8-bit targets cannot decay below the rounding step by multiplication alone
        gl.blendEquation(gl.FUNC_REVERSE_SUBTRACT);
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.useProgram(this._pSolid);
        gl.uniform1f(this._uSolid.uValue, 2 / 255);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.blendEquation(gl.FUNC_ADD);
      }
    }
    const cnt = this._pCount;
    if (cnt > 0) {
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this._vaoParticle);
      gl.bindBuffer(gl.ARRAY_BUFFER, this._bufInst);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this._inst, 0, cnt * 6);
      gl.useProgram(this._pParticle);
      gl.uniform2f(this._uParticle.uCanvas, this._trailW, this._trailH);
      gl.uniform1f(this._uParticle.uHalfW, this.options.particleWidthPx * 0.5);
      gl.uniform1f(this._uParticle.uBright, 1);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, cnt);
      gl.bindVertexArray(this._vaoEmpty);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _setGridUniforms(u) {
    const gl = this.gl;
    gl.uniform2f(u.uGridOff, this._gridOffX, this._gridOffY);
    gl.uniform2f(u.uRegionSize, this._region[2] - this._region[0], this._region[3] - this._region[1]);
    gl.uniform2f(u.uGridSize, this._gw, this._gh);
    gl.uniform1f(u.uDepthMax, this._depthMax);
    gl.uniform1f(u.uVmax, this._vmax);
  }

  _draw() {
    const gl = this.gl;
    const o = this.options;
    const w = this._pxW, h = this._pxH;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this._vaoEmpty);

    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this._texA);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this._texB);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this._fpTex || this._zeroTex);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this._trailTex);

    if (o.showDepth || o.showFlow) {
      const u = this._uWater;
      gl.useProgram(this._pWater);
      this._setGridUniforms(u);
      gl.uniform1f(u.uMix, this._mix);
      gl.uniform1f(u.uMercScale, this._mercScale);
      const cycle = o.flowCycleSec;
      gl.uniform1f(u.uCycle, cycle);
      gl.uniform1f(u.uPhase, (this._time / cycle) % 1);
      gl.uniform1f(u.uTime, this._time);
      gl.uniform1f(u.uFlowSpeed, o.flowSpeed);
      gl.uniform1f(u.uOpacity, o.opacity);
      gl.uniform1f(u.uMinDepth, o.minDepthShown);
      gl.uniform3f(u.uSunDir, this._sun[0], this._sun[1], this._sun[2]);
      gl.uniform3f(u.uViewDir, this._view[0], this._view[1], this._view[2]);
      gl.uniform1f(u.uShowDepth, o.showDepth ? 1 : 0);
      gl.uniform1f(u.uShowFlow, o.showFlow ? 1 : 0);
      gl.uniform1f(u.uRipple, o.rippleStrength);
      gl.uniform1f(u.uOctF, this._octF);
      gl.uniform1f(u.uOctBase, this._octBase);
      const qlt = Math.round(o.quality);
      gl.uniform1i(u.uOctaves, 2 + qlt);
      gl.uniform1i(u.uFoamTaps, qlt === 2 ? 6 : (qlt === 1 ? 4 : 3));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    if (o.showParticles && this._trailTex) {
      gl.useProgram(this._pTrail);
      gl.uniform2f(this._uTrail.uCanvas, this._trailW, this._trailH);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    if (o.showArrows) {
      const u = this._uArrow;
      const S = o.arrowSpacingPx;
      const nx = Math.ceil(w / S), ny = Math.ceil(h / S);
      gl.useProgram(this._pArrow);
      this._setGridUniforms(u);
      gl.uniform1f(u.uMix, this._mix);
      gl.uniform2f(u.uCanvas, w, h);
      gl.uniform2f(u.uMppXY, this._mppX, this._mppY);
      gl.uniform1f(u.uSpacing, S);
      gl.uniform1i(u.uNx, nx);
      // average over about 1/8 of the spacing: steadies the glyph without diluting narrow channels
      gl.uniform1f(u.uLod, clampNum(Math.log2(Math.max(S * this._mppX / this._cell, 1)) - 3, 0, 4));
      gl.uniform1f(u.uScale, o.arrowScale);
      gl.uniform1f(u.uInflate, 1.3);
      gl.uniform1f(u.uPass, 0);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 9, nx * ny);
      gl.uniform1f(u.uInflate, 0);
      gl.uniform1f(u.uPass, 1);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 9, nx * ny);
    }

    if (o.showFlowpaths && this._fpTex) {
      const u = this._uFlowpath;
      gl.useProgram(this._pFlowpath);
      this._setGridUniforms(u);
      gl.uniform1f(u.uOpacity, o.flowpathsOpacity);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    gl.activeTexture(gl.TEXTURE0);
  }

  // ------------------------------------------------------------------ readback

  // Whole canvas as premultiplied RGBA, first row = top of the picture. Allocates.
  readPixels() {
    if (this._lost || !this._pxW) return new Uint8Array(0);
    const gl = this.gl;
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const out = new Uint8Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
    const row = new Uint8Array(w * 4);
    for (let y = 0; y < (h >> 1); y++) {
      const a = y * w * 4, b = (h - 1 - y) * w * 4;
      row.set(out.subarray(a, a + w * 4));
      out.copyWithin(a, b, b + w * 4);
      out.set(row, b);
    }
    return out;
  }

  // One canvas pixel (x right, y down) as premultiplied [r, g, b, a]; also waits for the GPU.
  readPixel(x, y, out = new Uint8Array(4)) {
    if (this._lost) { out.fill(0); return out; }
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(x, gl.drawingBufferHeight - 1 - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
    return out;
  }

  // Block until the GPU has finished the queued work.
  sync() {
    if (this._lost || !this._pxW) return;
    this.readPixel(0, 0, this._syncPx || (this._syncPx = new Uint8Array(4)));
  }

  // Largest alpha in the particle trail buffer (diagnostics: 0 means the trails are clear).
  debugTrailMax() {
    if (this._lost || !this._trailFbo) return 0;
    const gl = this.gl;
    const w = this._trailW, h = this._trailH;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._trailFbo);
    let max = 0;
    if (this._trailFloat) {
      const buf = new Float32Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
      for (let i = 3; i < buf.length; i += 4) if (buf[i] > max) max = buf[i];
    } else {
      const buf = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      for (let i = 3; i < buf.length; i += 4) if (buf[i] > max) max = buf[i] / 255;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return max;
  }

  // ------------------------------------------------------------------ teardown

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this._canvas.removeEventListener('webglcontextlost', this._onLost, false);
    this._canvas.removeEventListener('webglcontextrestored', this._onRestored, false);
    const gl = this.gl;
    if (gl && !this._lost) {
      this._releaseGridTextures();
      for (const t of [this._zeroTex, this._trailTex]) if (t) gl.deleteTexture(t);
      if (this._trailFbo) gl.deleteFramebuffer(this._trailFbo);
      for (const b of [this._bufCorner, this._bufInst]) if (b) gl.deleteBuffer(b);
      for (const v of [this._vaoEmpty, this._vaoParticle]) if (v) gl.deleteVertexArray(v);
      for (const p of [this._pWater, this._pFade, this._pSolid, this._pParticle, this._pTrail, this._pArrow, this._pFlowpath]) if (p) gl.deleteProgram(p);
      const lose = gl.getExtension('WEBGL_lose_context');
      if (lose) lose.loseContext();
    }
    // a lost context keeps its drawing buffer until the canvas is collected: release it now
    this._canvas.width = 1; this._canvas.height = 1;
    this._A = this._B = this._cpuA = this._cpuB = this._fpSrc = null;
    this._px = new Float64Array(0); this._py = new Float64Array(0);
    this._age = new Float32Array(0); this._life = new Float32Array(0);
    this._still = new Float32Array(0); this._inst = new Float32Array(0);
    this._pn = 0; this._eff = 0;
  }
}
