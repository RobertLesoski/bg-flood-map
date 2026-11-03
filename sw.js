/* sw.js - offline shell for the Bowling Green Flood Map (PWA).
 *
 * Policy (see also the comments in viewer.html where it registers this file):
 *   shell   = viewer.html, manifest, icons, fluid/*.js, logo      -> cache-first with background refresh (stale-while-revalidate);
 *             precached at install. When the refresh brought a different file, the page is told ('shell-updated').
 *   cdn     = MapLibre JS/CSS from unpkg                           -> cache-first, stored at first use (opaque responses are fine for script/style)
 *   data    = small layers (*.geojson, meta.json, < 6 MB)          -> network-first, the last good copy is the offline fallback
 *   tiles   = OpenStreetMap + Terrarium DEM tiles                  -> stale-while-revalidate, at most TILE_MAX tiles kept
 *   NEVER cached, never touched (the browser handles them): flow grids (*.rgba.gz), hourly flood layers (layers/flood_hpc/),
 *             county building/road dumps (*.geojson.gz), anything with a Range header, anything that is not GET.
 * Bump VERSION when a shell file changes in a way the stale-while-revalidate refresh should not be trusted with.
 * Works on https:// and on http://localhost / 127.0.0.1 only (service workers need a secure context).
 */
const VERSION = 'v1';
const SHELL = 'bgflood-shell-' + VERSION;
const CDN = 'bgflood-cdn-' + VERSION;
const DATA = 'bgflood-data-' + VERSION;
const TILES = 'bgflood-tiles-' + VERSION;
const ALL = [SHELL, CDN, DATA, TILES];
const TILE_MAX = 400;
const DATA_MAX_BYTES = 6 * 1024 * 1024;

const SHELL_FILES = [
  'viewer.html', 'manifest.webmanifest', 'dsoc_logo.png',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-maskable-512.png', 'icons/apple-touch-icon.png', 'icons/favicon-32.png',
  'fluid/fluid-layer.js', 'fluid/fluid-renderer.js', 'fluid/fluid-data.js', 'fluid/fluid-stats.js',
];
const CDN_HOSTS = ['unpkg.com'];
const TILE_HOSTS = ['tile.openstreetmap.org', 's3.amazonaws.com'];

const scopePath = new URL(self.registration.scope).pathname;           // e.g. "/"
const shellPaths = new Set(SHELL_FILES.map((f) => new URL(f, self.registration.scope).pathname));

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL);
    // each file on its own: one missing optional file must not abort the install
    await Promise.all(SHELL_FILES.map(async (f) => {
      try { await cache.add(new Request(f, { cache: 'reload' })); } catch (e) { /* offline or missing: filled in at first use */ }
    }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith('bgflood-') && !ALL.includes(k)) await caches.delete(k);
    await self.clients.claim();
  })());
});

async function notify(msg) {
  for (const c of await self.clients.matchAll({ type: 'window' })) c.postMessage(msg);
}

function sameFile(a, b) {
  const ea = a.headers.get('etag'), eb = b.headers.get('etag');
  if (ea && eb) return ea === eb;
  const la = a.headers.get('last-modified'), lb = b.headers.get('last-modified');
  if (la && lb) return la === lb;
  return a.headers.get('content-length') === b.headers.get('content-length');
}

async function shellSWR(request, url) {
  const cache = await caches.open(SHELL);
  const key = url.origin + url.pathname;               // ?storm=...&hour=... and ?r=N must not create new entries
  const hit = await cache.match(key);
  const refresh = fetch(new Request(key, { cache: 'no-cache' })).then(async (res) => {
    if (res && res.ok) {
      if (hit && !sameFile(hit, res)) notify({ type: 'shell-updated', path: url.pathname });
      await cache.put(key, res.clone());
    }
    return res;
  }).catch(() => null);
  if (hit) return hit;                                  // cache-first
  const res = await refresh;                            // first visit: the network
  return res || new Response('Offline and not cached yet.', { status: 503, headers: { 'Content-Type': 'text/plain' } });
}

async function networkFirst(request, cacheName, maxBytes) {
  const cache = await caches.open(cacheName);
  try {
    const res = await fetch(request);
    if (res && res.ok) {
      const len = Number(res.headers.get('content-length'));
      if (!(len > maxBytes)) cache.put(request, res.clone()).catch(() => {});
    }
    return res;
  } catch (e) {
    const hit = await cache.match(request, { ignoreSearch: true });
    if (hit) return hit;
    throw e;
  }
}

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res && (res.ok || res.type === 'opaque')) cache.put(request, res.clone()).catch(() => {});
  return res;
}

async function tileSWR(request) {
  const cache = await caches.open(TILES);
  const hit = await cache.match(request);
  const net = fetch(request).then(async (res) => {
    if (res && res.ok) {
      cache.put(request, res.clone()).catch(() => {});
      const keys = await cache.keys();
      if (keys.length > TILE_MAX) for (const k of keys.slice(0, keys.length - TILE_MAX)) await cache.delete(k);
    }
    return res;
  }).catch(() => null);
  return hit || (await net) || new Response('', { status: 504 });
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || req.headers.has('range')) return;
  const url = new URL(req.url);

  if (CDN_HOSTS.includes(url.hostname)) { event.respondWith(cacheFirst(req, CDN)); return; }
  if (TILE_HOSTS.includes(url.hostname)) { if (/\.(png|jpg|jpeg|webp)$/i.test(url.pathname)) event.respondWith(tileSWR(req)); return; }
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith(scopePath)) return;

  const p = url.pathname;
  // never touch the big data: flow grids, hourly flood layers, gzipped county dumps, rasters
  if (/\.rgba\.gz$|\.geojson\.gz$|\.tif{1,2}$|\/layers\/flood_hpc\//i.test(p)) return;
  if (shellPaths.has(p)) { event.respondWith(shellSWR(req, url)); return; }
  if (/\/layers\/.*\.(geojson|json)$/i.test(p)) { event.respondWith(networkFirst(req, DATA, DATA_MAX_BYTES)); return; }
  // everything else (test harnesses, report, scripts, serve.py endpoints): untouched
});

self.addEventListener('message', (event) => {
  if (event.data === 'skip-waiting') self.skipWaiting();
  if (event.data === 'clear-caches') event.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k.startsWith('bgflood-')).map((k) => caches.delete(k)))));
});
