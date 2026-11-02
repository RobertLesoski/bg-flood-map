#!/usr/bin/env python3
"""
DSOC Flood Map - local web server (standard library only, Python 3.8+).

Serves the folder this file lives in (never the current directory) to the browser on this computer.

  python serve.py                    start on port 8000 (next free port if 8000 is taken by another program)
  python serve.py --port 8101        choose the port
  python serve.py --no-browser       do not open the browser
  python serve.py --find [--wait 10] print the URL of the already-running server for this folder (exit 1 if none)
  python serve.py --spawn            open the server in its own console window and return (what Launch Map.bat uses)

What it does beyond http.server:
  - threaded, HTTP/1.1 keep-alive, large listen backlog (many hour grids are requested at once)
  - correct MIME types from a fixed table (the Windows registry is never consulted); .gz is sent as
    application/gzip WITHOUT Content-Encoding, so the browser's DecompressionStream receives raw gzip bytes
  - Range requests (206 single range, suffix ranges, 416), HEAD, ETag / Last-Modified with 304
  - Cache-Control: no-cache on every file (ETag revalidation, 304 when unchanged), so a rebuilt flow grid is never stale
  - path traversal protection: ../ and encoded variants, backslashes, drive letters / NTFS streams, Windows
    device names, dot-files (.git ...), symlink and junction escapes; no directory listings
  - Host header check while bound to localhost (blocks DNS-rebinding pages from reading local files)
  - /__health: JSON status (used by the launcher and self-check)
  - logging: startup banner plus one short line per response that is not 2xx and not 304 (-v logs everything)
  - exclusive port binding on Windows (SO_REUSEADDR would let two programs share one port)
  - writes the chosen port (line 1) and the process id (line 2) to .serve_port, removed again on exit
"""

import argparse
import atexit
import datetime
import http.client
import json
import os
import platform
import re
import signal
import socket
import socketserver
import stat as statmod
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate, parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVER_ID = "dsoc-flood-map"
VERSION = "1.0.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
PORT_FILE = os.path.join(ROOT, ".serve_port")
DEFAULT_PORT = 8000
WINDOW_TITLE = "DSOC Flood Map Server"
PORT_TRIES = 50          # how many ports to try when the requested one is busy
FIND_SPAN = 12           # how many ports --find looks at (a refused connect costs ~2 s on Windows, so keep it small)
CHUNK = 256 * 1024
START_TIME = time.time()

MIME = {
    ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json", ".map": "application/json", ".webmanifest": "application/manifest+json",
    ".geojson": "application/geo+json",
    ".txt": "text/plain; charset=utf-8", ".md": "text/markdown; charset=utf-8", ".csv": "text/csv; charset=utf-8",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".svg": "image/svg+xml", ".ico": "image/x-icon",
    ".tif": "image/tiff", ".tiff": "image/tiff",
    ".gz": "application/gzip", ".zip": "application/zip",
    ".pdf": "application/pdf", ".wasm": "application/wasm", ".pbf": "application/x-protobuf",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".otf": "font/otf",
}

KEY_FILES = [
    "viewer.html", "flood_report.html", "dsoc_logo.png", "dem_bg.tif",
    "layers/fema.geojson", "layers/sinkholes.geojson", "layers/riverine.geojson", "layers/roads.geojson",
    "layers/buildings.geojson", "layers/all_parcels.geojson", "layers/parcels.geojson",
    "fluid/fluid-data.js", "fluid/fluid-renderer.js", "fluid/fluid-layer.js",
]

WINDOWS_DEVICE_NAMES = {"con", "prn", "aux", "nul", "conin$", "conout$", "clock$"} | \
    {"com%d" % i for i in range(1, 10)} | {"lpt%d" % i for i in range(1, 10)}
BUSY_ERRNOS = {98, 48, 10048, 10013, 13}   # EADDRINUSE (linux/mac/windows), EACCES / WSAEACCES (exclusive bind refused)
ABORT_WINERRORS = {64, 995, 10053, 10054, 10058}
ALLOWED_METHODS = "GET, HEAD, OPTIONS"


def now_str():
    return time.strftime("%H:%M:%S")


def say(msg=""):
    print(msg, flush=True)


def norm(path):
    return os.path.normcase(os.path.realpath(path))


def is_abort(exc):
    if isinstance(exc, (ConnectionError, socket.timeout)):
        return True
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) in ABORT_WINERRORS


# --------------------------------------------------------------------------------------------------
# Status for /__health (also printed in the banner)
# --------------------------------------------------------------------------------------------------

_meta_cache = {}


def _read_meta(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _meta_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    try:
        with open(path, "r", encoding="utf-8-sig") as f:        # tolerate a BOM; the browser's json() does too
            data = json.load(f)
        if not isinstance(data, dict):
            data = {"_error": "meta.json is not a JSON object"}
    except (OSError, ValueError) as e:
        data = {"_error": "meta.json unreadable: %s" % e}
    _meta_cache[path] = (key, data)
    return data


def storm_status(sdir):
    st = {"meta": False, "hours": 0, "hour_files": 0, "flowpaths": False, "bytes": 0, "complete": False}
    try:
        names = os.listdir(sdir)
    except OSError:
        return st
    hour_re = re.compile(r"^h(\d\d)\.rgba\.gz$")
    present = set()
    for n in names:
        try:
            size = os.path.getsize(os.path.join(sdir, n))
        except OSError:
            continue
        m = hour_re.match(n)
        if m:
            present.add(int(m.group(1)))
            st["hour_files"] += 1
            st["bytes"] += size
        elif n == "flowpaths.rgba.gz":
            st["flowpaths"] = size > 0
            st["bytes"] += size
        elif n == "meta.json":
            st["bytes"] += size
    meta = _read_meta(os.path.join(sdir, "meta.json"))
    if meta is not None:
        st["meta"] = "_error" not in meta
        if "_error" in meta:
            st["error"] = meta["_error"]
        else:
            hours = meta.get("hours")
            if isinstance(hours, list):
                st["hours"] = len(hours)
                st["complete"] = bool(st["flowpaths"]) and all(
                    isinstance(h, int) and h in present for h in hours)
            for k in ("source", "built", "width", "height", "cell_m"):
                if k in meta:
                    st[k] = meta[k]
    return st


def _storm_sort_key(name):
    m = re.match(r"^(\d+)", name)
    return (int(m.group(1)) if m else 10 ** 6, name)


def flow_status(root):
    base = os.path.join(root, "layers", "flow")
    out = {"present": False, "storms": {}}
    try:
        entries = [e for e in os.listdir(base) if os.path.isdir(os.path.join(base, e))]
    except OSError:
        return out
    out["present"] = True
    for name in sorted(entries, key=_storm_sort_key):
        out["storms"][name] = storm_status(os.path.join(base, name))
    return out


def file_status(root):
    files = {}
    for rel in KEY_FILES:
        try:
            files[rel] = {"exists": True, "bytes": os.path.getsize(os.path.join(root, *rel.split("/")))}
        except OSError:
            files[rel] = {"exists": False, "bytes": 0}
    return files


def hpc_status(root):
    base = os.path.join(root, "layers", "flood_hpc")
    out = {}
    pat = re.compile(r"^flood_h\d\d\.geojson$")
    try:
        for e in os.scandir(base):
            if e.is_dir():
                try:
                    out[e.name] = sum(1 for n in os.listdir(e.path) if pat.match(n))
                except OSError:
                    pass
            elif pat.match(e.name):
                out["(flat)"] = out.get("(flat)", 0) + 1
    except OSError:
        pass
    return out


def layer_names(root):
    try:
        return sorted(n[:-8] for n in os.listdir(os.path.join(root, "layers")) if n.endswith(".geojson"))
    except OSError:
        return []


# --------------------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------------------

class Reject(Exception):
    def __init__(self, code, why):
        Exception.__init__(self, why)
        self.code = code
        self.why = why


class FloodServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128            # the default of 5 refuses bursts of parallel hour-grid requests on Windows
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Without this a second program can bind the same port and steal requests (SO_REUSEADDR semantics on Windows).
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)      # not HTTPServer.server_bind: its getfqdn() can stall for seconds
        host, port = self.socket.getsockname()[:2]
        self.server_name = host
        self.server_port = port

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if is_abort(exc):
            self.count("aborted")
            return
        say("[%s] ERROR while handling a request from %s: %s: %s" % (now_str(), client_address[0], type(exc).__name__, exc))
        traceback.print_exc()
        sys.stderr.flush()

    def count(self, key):
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + 1


class FloodServer6(FloodServer):
    address_family = socket.AF_INET6


def make_server(host, port, verbose):
    cls = FloodServer6 if ":" in host else FloodServer
    httpd = cls((host, port), Handler)
    httpd.real_root = norm(ROOT)
    httpd.host = host
    httpd.verbose = verbose
    httpd.stats = {"served": 0, "aborted": 0, "errors": 0}
    httpd.stats_lock = threading.Lock()
    httpd.host_check = host in ("localhost", "::1") or host.startswith("127.")
    return httpd


class Handler(BaseHTTPRequestHandler):
    server_version = "DSOCFloodMap/" + VERSION
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 120

    def version_string(self):
        return self.server_version

    def setup(self):
        BaseHTTPRequestHandler.setup(self)
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    # ---- logging: only what a person needs to see ----
    def log_message(self, fmt, *args):
        pass

    def log_error(self, fmt, *args):
        pass

    def log_request(self, code="-", size="-"):
        try:
            c = int(code)
        except (TypeError, ValueError):
            c = 0
        self.server.count("served")
        if c >= 400:
            self.server.count("errors")
        if c == 304 or (200 <= c < 300):
            if not self.server.verbose:
                return
        # self.path / self.command do not exist yet when the request line itself was bad (414, 400)
        path = (getattr(self, "path", None) or getattr(self, "requestline", None) or "")[:160]
        path = path.encode("ascii", "backslashreplace").decode("ascii")
        path = re.sub(r"[\x00-\x1f\x7f]", "?", path)
        say("[%s] %s %s %s  (%s)" % (now_str(), c, getattr(self, "command", None) or "-", path, self.client_address[0]))

    # ---- methods ----
    def do_GET(self):
        self._guarded(False)

    def do_HEAD(self):
        self._guarded(True)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", ALLOWED_METHODS)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _not_allowed(self):
        self.send_response(405)
        self.send_header("Allow", ALLOWED_METHODS)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        body = b"405 Method Not Allowed: this server is read-only.\n"
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command != "HEAD":
            self.wfile.write(body)

    do_POST = do_PUT = do_DELETE = do_PATCH = _not_allowed

    def _guarded(self, head):
        try:
            self._serve(head)
        except Reject as r:
            self._reply_text(r.code, r.why, head)
        except Exception as e:
            if is_abort(e):
                self.close_connection = True
                self.server.count("aborted")
                return
            raise

    # ---- helpers for small replies ----
    def _reply_text(self, code, text, head=False, extra=None):
        body = (("%d %s\n" % (code, text))).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _reply_json(self, code, obj, head):
        body = json.dumps(obj, indent=1).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _redirect(self, location, code=302):
        self.send_response(code)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    # ---- request checks ----
    def _host_ok(self):
        if not self.server.host_check:
            return True
        h = self.headers.get("Host")
        if not h:
            return True                           # HTTP/1.0 client without a Host header
        h = h.strip()
        if h.startswith("["):
            name = h[1:h.find("]")] if "]" in h else h
        elif h.count(":") == 1:
            name = h.split(":", 1)[0]
        else:
            name = h
        name = name.lower().rstrip(".")
        return (name in ("localhost", "::1") or name.endswith(".localhost")
                or re.match(r"^127(\.\d{1,3}){3}$", name) is not None)

    def _raw_path(self):
        raw = self.path.split("#", 1)[0].split("?", 1)[0]
        if not raw.startswith("/"):
            if re.match(r"^https?://", raw, re.I):        # absolute-form request target
                raw = urllib.parse.urlsplit(raw).path or "/"
            else:
                raise Reject(400, "Bad request target")
        return raw

    def _query(self):
        return self.path.split("#", 1)[0].partition("?")[2]

    def resolve(self, raw):
        """Map a URL path to a path inside the folder, or raise Reject. Returns (fs_path, ends_with_slash)."""
        try:
            decoded = urllib.parse.unquote(raw, encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            raise Reject(400, "Bad URL encoding")
        if "\\" in decoded:
            raise Reject(403, "Forbidden: backslash in path")
        if re.search(r"[\x00-\x1f\x7f]", decoded):
            raise Reject(403, "Forbidden: control character in path")
        segments = []
        for seg in decoded.split("/"):
            if seg in ("", "."):
                continue
            if seg == "..":
                raise Reject(403, "Forbidden: path traversal")
            if ":" in seg:
                raise Reject(403, "Forbidden: drive letter or stream in path")
            if seg.startswith(".") or seg.endswith(".") or seg.endswith(" "):
                raise Reject(404, "Not found")                  # dot-files (.git, .serve_port) are never served
            if seg.split(".", 1)[0].strip().lower() in WINDOWS_DEVICE_NAMES:
                raise Reject(404, "Not found")                  # CON, NUL, COM1 ... would hang or leak the console
            segments.append(seg)
        root_real = self.server.real_root
        fs = os.path.join(ROOT, *segments) if segments else ROOT
        real = os.path.realpath(fs)
        real_n = os.path.normcase(real)
        try:
            inside = os.path.commonpath([root_real, real_n]) == root_real
        except ValueError:
            inside = False
        if not inside:
            raise Reject(403, "Forbidden: outside the map folder")        # includes symlink / junction escapes
        rel = os.path.relpath(real_n, root_real)
        if rel != "." and any(part.startswith(".") for part in rel.split(os.sep)):
            raise Reject(404, "Not found")                              # 8.3 short names (GIT~1) resolve to dot-folders
        return real, raw.endswith("/")

    # ---- main entry ----
    def _serve(self, head):
        if not self._host_ok():
            raise Reject(403, "Forbidden: unexpected Host header (this server only answers on localhost)")
        raw = self._raw_path()
        if raw == "/__health":
            return self._reply_json(200, self.server_health(), head)
        real, slash = self.resolve(raw)
        if raw == "/":
            if os.path.isfile(os.path.join(ROOT, "viewer.html")):
                return self._redirect("/viewer.html")
            raise Reject(404, "viewer.html is missing from this folder. Open /__health to see what is present.")
        try:
            st = os.stat(real)
        except OSError:
            if raw == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            raise Reject(404, "Not found")
        if statmod.S_ISDIR(st.st_mode):
            if not slash:
                q = self._query()
                # raw is already percent-encoded; strip leading slashes so //host/ can never become a protocol-relative redirect
                return self._redirect("/" + raw.lstrip("/") + "/" + ("?" + q if q else ""), 301)
            index = os.path.join(real, "index.html")
            if os.path.isfile(index):
                real = index
            else:
                raise Reject(404, "Not found (directory listing is disabled)")
        elif not statmod.S_ISREG(st.st_mode):
            raise Reject(404, "Not found")
        self._send_file(real, head)

    def server_health(self):
        root = self.server.real_root
        return {
            "ok": True,
            "server": SERVER_ID,
            "version": VERSION,
            "root": ROOT,
            "host": self.server.host,
            "port": self.server.server_port,
            "pid": os.getpid(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "started": datetime.datetime.fromtimestamp(START_TIME).isoformat(timespec="seconds"),
            "uptime_s": round(time.time() - START_TIME, 1),
            "requests": dict(self.server.stats),
            "flow": flow_status(ROOT),
            "files": file_status(ROOT),
            "layers": layer_names(ROOT),
            "flood_hpc": hpc_status(ROOT),
            "root_real": root,
        }

    def _send_file(self, path, head):
        ext = os.path.splitext(path)[1].lower()
        ctype = MIME.get(ext, "application/octet-stream")
        try:
            f = open(path, "rb")
        except OSError:
            raise Reject(404, "Not found")
        with f:
            st = os.fstat(f.fileno())
            size = st.st_size
            etag = '"%x-%x"' % (size, st.st_mtime_ns)
            lastmod = formatdate(int(st.st_mtime), usegmt=True)
            # Always revalidate (ETag, 304 costs a few ms here). The flow grids are rewritten under the same names by every
            # rebuild; max-age would keep the old water in the browser for its whole lifetime, whatever the URL query says.
            cache = "no-cache"

            # conditional GET
            inm = self.headers.get("If-None-Match")
            ims = self.headers.get("If-Modified-Since")
            not_modified = False
            if inm is not None:
                not_modified = etag_matches(inm, etag)
            elif ims:
                try:
                    not_modified = int(st.st_mtime) <= parsedate_to_datetime(ims).timestamp()
                except (TypeError, ValueError, IndexError, OverflowError):
                    not_modified = False
            if not_modified:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", lastmod)
                self.send_header("Cache-Control", cache)
                self.end_headers()
                return

            # range
            start, length, code = 0, size, 200
            rng = self.headers.get("Range")
            if rng:
                ifr = self.headers.get("If-Range")
                use = True
                if ifr:
                    ifr = ifr.strip()
                    use = (ifr == etag) if ifr.startswith(('"', "W/")) else (ifr == lastmod)
                if use:
                    r = parse_range(rng, size)
                    if r == "bad":
                        self.send_response(416)
                        self.send_header("Content-Range", "bytes */%d" % size)
                        self.send_header("Content-Length", "0")
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        return
                    if r is not None:
                        start, end = r
                        length = end - start + 1
                        code = 206

            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            if code == 206:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, start + length - 1, size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", lastmod)
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if head:
                return
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(CHUNK, remaining))
                if not chunk:                       # file shrank while it was being sent
                    self.close_connection = True
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def etag_matches(header, etag):
    """If-None-Match comparison (weak comparison: W/ prefixes are ignored)."""
    for tag in header.split(","):
        tag = tag.strip()
        if tag == "*":
            return True
        if tag.startswith("W/"):
            tag = tag[2:]
        if tag == etag:
            return True
    return False


def parse_range(header, size):
    """None = ignore the header (serve 200), "bad" = 416, else (first, last) inclusive.
    Unknown units and multi-range requests are ignored, which RFC 9110 allows."""
    m = re.match(r"^\s*bytes\s*=\s*(.*)$", header, re.I)
    if not m:
        return None
    spec = m.group(1).strip()
    if "," in spec:
        return None
    m = re.fullmatch(r"(\d*)\s*-\s*(\d*)", spec)
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return "bad"
    first, last = m.groups()
    if first == "":
        n = int(last)
        if n == 0 or size == 0:
            return "bad"
        return (max(0, size - n), size - 1)
    a = int(first)
    b = int(last) if last else size - 1
    if last and b < a:
        return "bad"
    if a >= size:
        return "bad"
    return (a, min(b, size - 1))


# --------------------------------------------------------------------------------------------------
# Finding a server that is already running
# --------------------------------------------------------------------------------------------------

def probe_health(host, port, connect_timeout=0.4, timeout=1.5):
    """Return the /__health dict of whatever answers on host:port, or None."""
    conn = http.client.HTTPConnection(host, port, timeout=connect_timeout)
    try:
        conn.connect()
        conn.sock.settimeout(timeout)
        conn.request("GET", "/__health")
        resp = conn.getresponse()
        body = resp.read(1 << 20)
        if resp.status != 200:
            return None
        data = json.loads(body.decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        try:
            conn.close()
        except OSError:
            pass


def is_this_server(info):
    if not info or info.get("server") != SERVER_ID:
        return False
    other = info.get("root_real") or info.get("root") or ""
    return os.path.normcase(os.path.realpath(other)) == norm(ROOT) if other else False


def read_port_hint():
    try:
        with open(PORT_FILE, "r", encoding="ascii") as f:
            return int(f.readline().strip())
    except (OSError, ValueError):
        return None


def find_running(host, port):
    candidates = []
    hint = read_port_hint()
    if hint:
        candidates.append(hint)
    candidates += [p for p in range(port, port + FIND_SPAN) if p not in candidates]
    return probe_many(host, [p for p in candidates if 0 < p < 65536])


def probe_many(host, ports):
    """Probe all ports at once; return (port, health) of the first one (in the given order) that is THIS server."""
    if not ports:
        return None
    with ThreadPoolExecutor(max_workers=min(len(ports), 32)) as pool:
        results = list(pool.map(lambda p: probe_health(host, p), ports))
    for p, info in zip(ports, results):
        if is_this_server(info):
            return p, info
    return None


def probe_host_for(host):
    return "127.0.0.1" if host in ("0.0.0.0", "", "localhost") else ("::1" if host == "::" else host)


def url_host(host):
    h = probe_host_for(host)
    return "[%s]" % h if ":" in h else h


# --------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------

def write_port_file(port, pid):
    try:
        with open(PORT_FILE, "w", encoding="ascii") as f:
            f.write("%d\n%d\n" % (port, pid))
        return True
    except OSError as e:
        say("  warning: could not write %s (%s)" % (PORT_FILE, e))
        return False


def remove_port_file(port, pid):
    try:
        with open(PORT_FILE, "r", encoding="ascii") as f:
            lines = f.read().split()
        if lines[:2] == [str(port), str(pid)]:
            os.remove(PORT_FILE)
    except (OSError, IndexError):
        pass


def open_browser(url):
    def go():
        try:
            import webbrowser
            webbrowser.open(url, new=2)
        except Exception:
            pass
    threading.Timer(0.4, go).start()


def flow_summary_line():
    fl = flow_status(ROOT)
    storms = fl["storms"]
    if not storms:
        return "Flow data   : not built yet: see RUN_LOCALLY.md (the map still works without it)"
    parts = []
    for name, s in storms.items():
        parts.append("%s(%dh%s)" % (name, s["hours"], "" if s["complete"] else ", INCOMPLETE"))
    return "Flow data   : " + " ".join(parts)


def cmd_find(args):
    host = probe_host_for(args.host)
    deadline = time.time() + max(0.0, args.wait)
    while True:
        hit = find_running(host, args.port)
        if hit:
            say("http://%s:%d" % (url_host(args.host), hit[0]))
            return 0
        if time.time() >= deadline:
            return 1
        time.sleep(0.25)


def cmd_spawn(args):
    """Open the server in its own console window and return at once.
    Unlike `start` in a batch file this leaks no inherited handles, so a caller that captures our output
    (a script, a CI job, an IDE) is not kept waiting until the server window is closed."""
    cmd = [sys.executable, os.path.abspath(__file__), "--no-browser", "--pause-on-error", "--title", WINDOW_TITLE,
           "--port", str(args.port), "--host", args.host]
    if args.verbose:
        cmd.append("-v")
    if os.name == "nt":
        kw = {"creationflags": subprocess.CREATE_NEW_CONSOLE, "close_fds": True}
    else:
        kw = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
              "start_new_session": True, "close_fds": True}
    try:
        subprocess.Popen(cmd, **kw)
    except OSError as e:
        say("Could not open the server window: %s" % e)
        return 1
    return 0


def cmd_serve(args):
    if args.title and os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(args.title)
        except Exception:
            pass
    host = args.host
    chosen, httpd = None, None
    skipped = []
    for port in range(args.port, min(args.port + PORT_TRIES, 65536)):
        try:
            httpd = make_server(host, port, args.verbose)
            chosen = port
            break
        except OSError as e:
            if e.errno not in BUSY_ERRNOS and getattr(e, "winerror", None) not in BUSY_ERRNOS:
                say("Could not start the server on %s:%d: %s" % (host, port, e))
                return 1
            skipped.append(port)
    # Is one of the busy ports our own server from an earlier launch? Ask them all at once, so programs that accept
    # connections but never answer HTTP cost one timeout in total, not one per port.
    hit = probe_many(probe_host_for(host), skipped) if skipped else None
    if hit:
        port, info = hit
        if httpd is not None:
            httpd.server_close()
        url = "http://%s:%d/" % (url_host(host), port)
        say("The DSOC Flood Map server for this folder is already running: %s" % url)
        write_port_file(port, int(info.get("pid") or 0))
        if not args.no_browser and args.open:
            open_browser(url + args.open.lstrip("/"))
        return 0
    if httpd is None:
        say("No free port found between %d and %d. Close other programs or choose another one with --port." %
            (args.port, min(args.port + PORT_TRIES - 1, 65535)))
        return 1

    pid = os.getpid()
    write_port_file(chosen, pid)
    atexit.register(remove_port_file, chosen, pid)

    def stop(signum, frame):
        raise KeyboardInterrupt
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), stop)
            except (OSError, ValueError):
                pass

    url = "http://%s:%d/" % (url_host(host), chosen)
    say("")
    say("DSOC Flood Map server %s" % VERSION)
    say("  Folder      : %s" % ROOT)
    say("  Address     : %s%s" % (url, "  (this computer only)" if httpd.host_check else ""))
    if skipped:
        say("  Note        : port %s is used by another program, so port %d is used instead" %
            (", ".join(str(p) for p in skipped), chosen))
    if not httpd.host_check:
        say("  WARNING     : bound to %s - other computers on your network can open the map (there is no password)" % host)
    say("  " + flow_summary_line())
    say("  Press Ctrl+C (or close this window) to stop.")
    say("")
    if not args.no_browser and args.open:
        target = args.open.lstrip("/")
        if os.path.isfile(os.path.join(ROOT, *target.split("/"))):
            open_browser(url + target)
        else:
            say("  (not opening the browser: %s is not in this folder)" % target)
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        say("")
        say("Stopping server...")
    finally:
        httpd.server_close()
        remove_port_file(chosen, pid)
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="DSOC Flood Map local web server (serves the folder this file is in).")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to listen on (default %d; the next free one is used if it is busy)" % DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1 = this computer only)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    ap.add_argument("--open", default="viewer.html", metavar="PAGE", help="page to open in the browser (default viewer.html)")
    ap.add_argument("--find", action="store_true", help="print the URL of the server already running for this folder and exit (1 if none)")
    ap.add_argument("--wait", type=float, default=0.0, metavar="SECONDS", help="with --find: keep looking for up to this long")
    ap.add_argument("--spawn", action="store_true", help="start the server in its own console window and return immediately (used by Launch Map.bat)")
    ap.add_argument("--title", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pause-on-error", action="store_true", help="when the server cannot start, wait for Enter so a double-clicked window stays readable")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every request, not only errors")
    ap.add_argument("--version", action="version", version="%s %s" % (SERVER_ID, VERSION))
    args = ap.parse_args(argv)
    if not (0 < args.port < 65536):
        ap.error("--port must be between 1 and 65535")
    if sys.version_info < (3, 8):
        say("Python 3.8 or newer is required (this is %s)." % platform.python_version())
        return 1
    try:
        code = cmd_find(args) if args.find else cmd_spawn(args) if args.spawn else cmd_serve(args)
    except KeyboardInterrupt:
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    if code and args.pause_on_error and not args.find and sys.stdin is not None and sys.stdin.isatty():
        say("")
        say("The server did not keep running (see above). Press Enter to close this window.")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
