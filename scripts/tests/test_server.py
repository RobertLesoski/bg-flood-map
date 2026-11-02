"""
Tests for the local web server (serve.py), the launcher (Launch Map.bat), the install self-check (selfcheck.py) and the
plain-language guide (RUN_LOCALLY.md) of the DSOC flood map.

  C:/Users/Minec/anaconda3/python.exe -m unittest discover -s C:/Users/Minec/Documents/BG_3D_Map/scripts/tests -p "test_server.py" -v

Every test talks to a REAL serve.py process over a real socket. serve.py is copied into a throw-away tree in the
scratch folder (the folder name has spaces and an ampersand on purpose); the real map folder is never modified.
Ports: 8101 and 8103 are preferred. When one of them is taken by a program this test did not start, the next free
port is used and the choice is printed, because the test must never touch somebody else's server.

The launcher tests (Windows only) run the real Launch Map.bat in copies of the tree; each one that starts a server opens it
in a console window of its own, as the launcher does, on the first free block of 14 ports from 8160. They stop only the
servers they started (pid from the copy's .serve_port, then a sweep for processes whose command line names this run's
private folder).

Environment overrides: BGMAP_APP (folder that holds serve.py and selfcheck.py), BGMAP_SCRATCH (where the tree is built),
BGMAP_REPO (code folder that holds RUN_LOCALLY.md).
"""
import atexit
import ctypes
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import gzip
from email.utils import formatdate
from pathlib import Path

APP = Path(os.environ.get("BGMAP_APP", r"C:/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map"))
_DEFAULT_SCRATCH = Path(r"C:/Users/Minec/AppData/Local/Temp/claude/C--Users-Minec-Documents-BG-3D-Map/"
                        r"6d20f384-9e95-4606-82dd-08b76f90f515/scratchpad")
SCRATCH = Path(os.environ.get("BGMAP_SCRATCH", str(_DEFAULT_SCRATCH)))
if not SCRATCH.is_dir():
    SCRATCH = Path(tempfile.gettempdir())

SECRET = b"TOP-SECRET-OUTSIDE-THE-ROOT-7f3a"
BIG_MB = 60
IS_WIN = os.name == "nt"
LOCAL = "127.0.0.1"

WORK = None          # temp dir for this run
ROOT = None          # the served tree (inside WORK)
PORT = None          # the shared server
BIG_SHA = None
LINKS = []           # symlinks / junctions created by tests, removed before the tree is deleted
_procs = []          # every process started here, so that only those can be stopped
_launched = []       # trees in which a launcher was started (it opens its server in a window of its own)
_used_ports = set()


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------

def port_is_free(p):
    s = socket.socket()
    try:
        if IS_WIN and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        s.bind((LOCAL, p))
        return True
    except OSError:
        return False
    finally:
        s.close()


def alloc_port(preferred=None):
    """A port nobody listens on (the preferred one when it is free). Never returns the same port twice."""
    cands = ([preferred] if preferred else []) + list(range(8101, 8300))
    for p in cands:
        if p not in _used_ports and port_is_free(p) and port_is_free(p + 1):
            _used_ports.add(p)
            if preferred and p != preferred:
                print("  [test] port %d is busy (not started by this test); using %d instead" % (preferred, p))
            return p
    raise RuntimeError("no free test port")


class Resp:
    def __init__(self, status, headers, body, line=""):
        self.status, self.headers, self.body, self.line = status, headers, body, line

    def h(self, name, default=None):
        return self.headers.get(name.lower(), default)


def req(port, method, path, headers=None, host=None, timeout=15):
    c = http.client.HTTPConnection(LOCAL, port, timeout=timeout)
    try:
        c.putrequest(method, path, skip_host=bool(host), skip_accept_encoding=True)
        if host:
            c.putheader("Host", host)
        for k, v in (headers or {}).items():
            c.putheader(k, v)
        c.endheaders()
        r = c.getresponse()
        body = r.read()
        return Resp(r.status, {k.lower(): v for k, v in r.getheaders()}, body, r.reason)
    finally:
        c.close()


def get(path, headers=None, port=None, **kw):
    return req(port or PORT, "GET", path, headers, **kw)


class HttpCache:
    """What a browser's HTTP cache does with these responses (RFC 9111, the part that matters here): a stored response is reused
    without asking the server while its Cache-Control says it is fresh; otherwise the server is asked with If-None-Match."""

    def __init__(self, port):
        self.port, self.stored = port, {}

    def get(self, path):
        hit = self.stored.get(path)
        if hit:
            cc = hit.h("cache-control", "")
            age = re.search(r"max-age=(\d+)", cc)
            if age and int(age.group(1)) > 0 and "no-cache" not in cc and "no-store" not in cc:
                return hit
        r = req(self.port, "GET", path, {"If-None-Match": hit.h("etag")} if hit and hit.h("etag") else None)
        if r.status == 304 and hit:
            return hit
        self.stored[path] = r
        return r


def raw(port, data, timeout=10, limit=1 << 20):
    """Send exact bytes, read until the server closes (or `limit`). Returns Resp (status 0 when nothing came back)."""
    s = socket.create_connection((LOCAL, port), timeout=timeout)
    chunks, total = [], 0
    try:
        s.sendall(data)
        while total < limit:
            try:
                c = s.recv(65536)
            except socket.timeout:
                break
            if not c:
                break
            chunks.append(c)
            total += len(c)
    except OSError:
        pass
    finally:
        s.close()
    buf = b"".join(chunks)
    head, _, body = buf.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1]) if lines and lines[0].startswith("HTTP/") and len(lines[0].split()) > 1 else 0
    hdrs = {}
    for ln in lines[1:]:
        k, _, v = ln.partition(":")
        hdrs[k.strip().lower()] = v.strip()
    return Resp(status, hdrs, body, lines[0])


def raw_get(port, target, extra="", host=None):
    host = host or "%s:%d" % (LOCAL, port)
    return raw(port, ("GET %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n%s\r\n" % (target, host, extra)).encode("latin-1"))


def gz(payload):
    return gzip.compress(payload, 1)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def make_tree(root, big=False):
    """A small map folder: real serve.py, a few key files, fixtures for MIME/flow tests, optionally a 60 MB gz."""
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(APP / "serve.py"), str(root / "serve.py"))
    write(root / "viewer.html", "<!doctype html><title>viewer</title>" + "<p>x</p>" * 400)
    write(root / "flood_report.html", "<!doctype html><title>report</title>")
    write(root / "dsoc_logo.png", b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8)
    write(root / "fluid" / "fluid-data.js", "export const a = 1;\n")
    write(root / "layers" / "fema.geojson", '{"type":"FeatureCollection","features":[]}')
    write(root / "layers" / "roads.geojson", '{"type":"FeatureCollection","features":[]}')
    for name in ("a.js", "a.mjs", "a.json", "a.geojson", "a.html", "a.css", "a.png", "a.tif", "a.gz", "a.bin", "a.svg", "a.txt"):
        write(root / "mime" / name, b"0123456789" * 3)
    write(root / "mime" / "noext", b"x")
    write(root / "dirindex" / "index.html", "<title>index</title>")
    (root / "emptydir").mkdir(exist_ok=True)
    write(root / ".hidden" / "x.txt", "dot folder")
    write(root / "range.bin", bytes(range(256)) * 4)                # 1024 bytes, byte i == i % 256
    write(root / "empty.bin", b"")
    # flow storms: 5yr complete, 10yr with a missing hour and no flowpaths, 25yr with a broken meta.json
    f = root / "layers" / "flow"
    for hour in (1, 2):
        write(f / "5yr" / ("h%02d.rgba.gz" % hour), gz(bytes(2 * 2 * 4)))
    write(f / "5yr" / "flowpaths.rgba.gz", gz(bytes(2 * 2 * 4)))
    write(f / "5yr" / "meta.json", json.dumps({"hours": [1, 2], "source": "tiers", "width": 2, "height": 2, "cell_m": 5.0}))
    write(f / "10yr" / "h01.rgba.gz", gz(bytes(16)))
    write(f / "10yr" / "h02.rgba.gz", gz(bytes(16)))
    write(f / "10yr" / "meta.json", json.dumps({"hours": [1, 2, 3], "source": "rasters", "width": 2, "height": 2}))
    write(f / "25yr" / "meta.json", "{ this is not json")
    if big:
        d = root / "bigdata"
        d.mkdir(exist_ok=True)
        h = hashlib.sha256()
        with open(str(d / "big.gz"), "wb") as out:
            for _ in range(BIG_MB):
                chunk = os.urandom(1 << 20)
                h.update(chunk)
                out.write(chunk)
        return h.hexdigest()
    return None


class Srv:
    """One serve.py process. Output goes to a log file next to the tree (never into a pipe that could fill up)."""

    def __init__(self, root, port, extra=(), cwd=None, python=None):
        self.root, self.port = Path(root), port
        self.log_path = self.root.parent / ("server_%d_%d.log" % (port, len(_procs)))
        self._log = open(str(self.log_path), "wb")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0
        self.proc = subprocess.Popen([python or sys.executable, str(self.root / "serve.py"), "--port", str(port), "--no-browser"] + list(extra),
                                     cwd=str(cwd or WORK), stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags)
        _procs.append(self.proc)

    def wait_ready(self, timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early (%s):\n%s" % (self.proc.returncode, self.text()))
            try:
                r = req(self.port, "GET", "/__health", timeout=2)
                if r.status == 200:
                    return self
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("server not ready after %ss:\n%s" % (timeout, self.text()))

    def text(self):
        try:
            self._log.flush()
        except (OSError, ValueError):
            pass
        return self.log_path.read_text(encoding="utf-8", errors="replace") if self.log_path.exists() else ""

    def alive(self):
        return self.proc.poll() is None

    def stop(self, graceful=True, timeout=10):
        """Ctrl+C equivalent first (CTRL_BREAK on Windows), kill only as a fallback. Returns the exit code."""
        if self.proc.poll() is None:
            try:
                if graceful:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT if IS_WIN else signal.SIGINT)
                else:
                    self.proc.terminate()
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
            except OSError:
                pass
        try:
            self._log.close()
        except OSError:
            pass
        return self.proc.returncode


def _kill_leftovers():
    for p in _procs:
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass


atexit.register(_kill_leftovers)


def health(port=None):
    return json.loads(req(port or PORT, "GET", "/__health").body.decode("utf-8"))


def setUpModule():
    global WORK, ROOT, PORT, BIG_SHA
    if not (APP / "serve.py").is_file():
        raise unittest.SkipTest("serve.py not found in %s" % APP)
    WORK = Path(tempfile.mkdtemp(prefix="srvtest_", dir=str(SCRATCH)))
    write(WORK / "secret.txt", SECRET)
    write(WORK / "outside_dir" / "secret.txt", SECRET)
    ROOT = WORK / "map site (a&b) x"
    BIG_SHA = make_tree(ROOT, big=True)
    PORT = alloc_port(8101)
    srv = Srv(ROOT, PORT, cwd=tempfile.gettempdir())     # cwd is NOT the tree: the server must root itself at serve.py
    setUpModule.srv = srv
    srv.wait_ready()


def tearDownModule():
    srv = getattr(setUpModule, "srv", None)
    if srv:
        srv.stop()
    for link in LINKS:                      # links first: rmtree must never follow them into the outside folder
        try:
            if os.path.isdir(link) and not os.path.islink(link):
                os.rmdir(link)              # NTFS junction
            else:
                os.unlink(link)
        except OSError:
            pass
    if WORK is not None:
        if _launched:
            kill_stragglers(WORK)
        shutil.rmtree(str(WORK), ignore_errors=True)


# --------------------------------------------------------------------------------------------------
# basics, MIME, cache headers
# --------------------------------------------------------------------------------------------------

class TestBasics(unittest.TestCase):
    def test_get_html(self):
        r = get("/viewer.html")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.h("content-type"), "text/html; charset=utf-8")
        self.assertEqual(r.body, (ROOT / "viewer.html").read_bytes())
        self.assertEqual(int(r.h("content-length")), len(r.body))
        self.assertEqual(r.h("accept-ranges"), "bytes")

    def test_root_redirects_to_viewer(self):
        r = get("/")
        self.assertIn(r.status, (301, 302))
        self.assertEqual(r.h("location"), "/viewer.html")

    def test_no_server_banner_leak(self):
        r = get("/viewer.html")
        self.assertNotIn("python", r.h("server", "").lower())

    def test_missing_file_is_404(self):
        self.assertEqual(get("/nothing-here.html").status, 404)
        self.assertEqual(get("/layers/nothing.geojson").status, 404)

    def test_no_directory_listing(self):
        for p in ("/layers/", "/layers", "/emptydir/", "/emptydir", "/bigdata/", "/mime/", "/layers/flow/"):
            r = get(p)
            self.assertIn(r.status, (301, 404), p)
            if r.status == 301:                               # slash added, then 404 (never a listing)
                r = get(r.h("location"))
                self.assertEqual(r.status, 404, p)
            self.assertNotIn(b"<li>", r.body)
            self.assertNotIn(b"fema.geojson", r.body)

    def test_directory_with_index_html(self):
        r = get("/dirindex/")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.body, b"<title>index</title>")
        r = get("/dirindex")
        self.assertEqual(r.status, 301)
        self.assertEqual(r.h("location"), "/dirindex/")

    def test_redirect_cannot_become_protocol_relative(self):
        r = raw_get(PORT, "//emptydir")
        self.assertNotEqual(r.h("location", "")[:2], "//", r.line)

    def test_query_string_ignored_for_file_lookup(self):
        r = get("/viewer.html?v=123&x=%2e%2e")
        self.assertEqual(r.status, 200)

    def test_percent_encoded_names_work(self):
        write(ROOT / "mime" / "sp ace.json", "{}")
        self.assertEqual(get("/mime/sp%20ace.json").status, 200)

    def test_favicon_without_file_is_quiet(self):
        self.assertEqual(get("/favicon.ico").status, 204)

    def test_write_methods_refused(self):
        for m in ("POST", "PUT", "DELETE", "PATCH"):
            r = raw(PORT, ("%s /viewer.html HTTP/1.1\r\nHost: %s:%d\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" % (m, LOCAL, PORT)).encode())
            self.assertEqual(r.status, 405, m)
            self.assertIn("GET", r.h("allow", ""))
        self.assertEqual(req(PORT, "OPTIONS", "/viewer.html").status, 204)

    def test_keep_alive_many_requests_one_connection(self):
        c = http.client.HTTPConnection(LOCAL, PORT, timeout=10)
        try:
            for i in range(60):
                c.request("GET", "/range.bin" if i % 3 else "/nothing-%d" % i)
                r = c.getresponse()
                r.read()
                self.assertIn(r.status, (200, 404))
        finally:
            c.close()

    def test_unicode_and_odd_but_legal_urls_do_not_crash(self):
        for t in ("/%E2%82%AC", "/a%20b", "/%7e", "/mime/%41.js", "/" + "a" * 3000):
            r = raw_get(PORT, t)
            self.assertIn(r.status, (200, 400, 404, 414), t)
        self.assertEqual(get("/mime/%61.js").status, 200)       # %61 is "a"
        self.assertTrue(setUpModule.srv.alive())

    def test_bad_request_lines_get_an_error_not_a_hang(self):
        for data in (b"\x00\x01\x02 garbage\r\n\r\n", b"GET\r\n\r\n", b"GET /x\r\n\r\n", b"GET /x HTTP/9.9\r\n\r\n",
                     b"GET C:\\Windows\\win.ini HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"):
            r = raw(PORT, data, timeout=5)
            self.assertIn(r.status, (0, 400, 403, 404, 505), data)
        self.assertEqual(get("/range.bin").status, 200)


class TestMime(unittest.TestCase):
    TYPES = {
        "a.js": "text/javascript", "a.mjs": "text/javascript", "a.json": "application/json",
        "a.geojson": "application/geo+json", "a.html": "text/html", "a.css": "text/css", "a.png": "image/png",
        "a.tif": "image/tiff", "a.gz": "application/gzip", "a.svg": "image/svg+xml", "a.txt": "text/plain",
        "a.bin": "application/octet-stream", "noext": "application/octet-stream",
    }

    def test_content_types(self):
        for name, ctype in self.TYPES.items():
            r = get("/mime/" + name)
            self.assertEqual(r.status, 200, name)
            self.assertEqual(r.h("content-type").split(";")[0], ctype, name)

    def test_text_types_declare_utf8(self):
        for name in ("a.js", "a.mjs", "a.html", "a.css"):
            self.assertIn("charset=utf-8", get("/mime/" + name).h("content-type"), name)

    def test_gz_has_no_content_encoding(self):
        # DecompressionStream in the browser needs the raw gzip bytes; Content-Encoding would make the browser inflate it first
        for path in ("/layers/flow/5yr/h01.rgba.gz", "/layers/flow/5yr/flowpaths.rgba.gz"):
            for extra in ({}, {"Accept-Encoding": "gzip, deflate, br"}):
                r = get(path, extra)
                self.assertEqual(r.status, 200)
                self.assertIsNone(r.h("content-encoding"), path)
                self.assertEqual(r.h("content-type"), "application/gzip")
                self.assertEqual(r.body[:2], b"\x1f\x8b")
                self.assertEqual(gzip.decompress(r.body), bytes(16))
        r = get("/mime/a.gz", {"Accept-Encoding": "gzip"})
        self.assertIsNone(r.h("content-encoding"))
        self.assertEqual(r.body, b"0123456789" * 3)             # untouched bytes, whatever they are

    def test_nosniff(self):
        self.assertEqual(get("/mime/a.json").h("x-content-type-options"), "nosniff")


class TestCache(unittest.TestCase):
    def test_no_cache_for_code_and_meta(self):
        for p in ("/viewer.html", "/fluid/fluid-data.js", "/layers/flow/5yr/meta.json", "/layers/fema.geojson", "/mime/a.gz"):
            self.assertEqual(get(p).h("cache-control"), "no-cache", p)

    GRIDS = ("/layers/flow/5yr/h01.rgba.gz", "/layers/flow/5yr/flowpaths.rgba.gz", "/layers/flow/10yr/h02.rgba.gz")

    def test_flow_grids_are_revalidated_not_cached(self):
        # A rebuild rewrites the grids under the same names. max-age would keep the old water in the browser for an hour
        # (and a changed grid size is a FlowFormatError in the loader) for every client that does not put a new ?v= on the URL.
        for p in self.GRIDS:
            for query in ("", "?v=2026-10-31T10%3A00%3A00Z"):
                cc = get(p + query).h("cache-control", "")
                self.assertEqual(cc, "no-cache", p + query)

    def test_rebuilt_grid_reaches_a_client_that_caches(self):
        folder = ROOT / "layers" / "flow" / "rebuild_probe"
        self.addCleanup(shutil.rmtree, str(folder), ignore_errors=True)
        grid = folder / "h03.rgba.gz"
        url = "/layers/flow/rebuild_probe/h03.rgba.gz"
        stored = lambda payload: gzip.compress(payload, 0)       # same payload length, same file length
        cases = {"another grid size": (gz(bytes(16)), gz(bytes(range(256)) * 5)),
                 "same grid size": (stored(bytes(16)), stored(bytes(range(16))))}
        self.assertEqual(len(cases["same grid size"][0]), len(cases["same grid size"][1]))
        for query in ("", "?v=2026-10-31T10%3A00%3A00Z"):         # build time unknown to the client, or unchanged
            for what, (old, new) in cases.items():
                write(grid, old)
                browser = HttpCache(PORT)
                self.assertEqual(browser.get(url + query).body, old)
                write(grid, new)
                later = os.stat(str(grid)).st_mtime_ns + 5 * 10 ** 9      # a real rebuild is seconds later, a test is not
                os.utime(str(grid), ns=(later, later))
                self.assertEqual(browser.get(url + query).body, new, "stale grid for %s after a rebuild with %s" % (url + query, what))

    def test_unchanged_grid_revalidates_with_304(self):
        r = get(self.GRIDS[0])
        self.assertEqual(r.status, 200)
        again = get(self.GRIDS[0], {"If-None-Match": r.h("etag")})
        self.assertEqual(again.status, 304)
        self.assertEqual(again.body, b"")
        self.assertEqual(again.h("cache-control"), "no-cache")

    def test_etag_and_last_modified_present(self):
        r = get("/viewer.html")
        self.assertTrue(r.h("etag", "").startswith('"'))
        self.assertIn("GMT", r.h("last-modified", ""))


# --------------------------------------------------------------------------------------------------
# Range requests
# --------------------------------------------------------------------------------------------------

class TestRange(unittest.TestCase):
    DATA = bytes(range(256)) * 4

    def rng(self, spec, path="/range.bin", extra=None):
        h = {"Range": spec}
        h.update(extra or {})
        return get(path, h)

    def test_simple_range(self):
        r = self.rng("bytes=0-9")
        self.assertEqual(r.status, 206)
        self.assertEqual(r.body, self.DATA[0:10])
        self.assertEqual(r.h("content-range"), "bytes 0-9/1024")
        self.assertEqual(r.h("content-length"), "10")
        self.assertEqual(r.h("content-type"), "application/octet-stream")

    def test_open_ended_range(self):
        r = self.rng("bytes=1000-")
        self.assertEqual((r.status, r.body), (206, self.DATA[1000:]))
        self.assertEqual(r.h("content-range"), "bytes 1000-1023/1024")

    def test_suffix_range(self):
        r = self.rng("bytes=-24")
        self.assertEqual((r.status, r.body), (206, self.DATA[-24:]))
        self.assertEqual(r.h("content-range"), "bytes 1000-1023/1024")

    def test_suffix_longer_than_file_gives_whole_file(self):
        r = self.rng("bytes=-99999")
        self.assertEqual((r.status, r.body), (206, self.DATA))
        self.assertEqual(r.h("content-range"), "bytes 0-1023/1024")

    def test_end_beyond_file_is_clamped(self):
        r = self.rng("bytes=1020-99999")
        self.assertEqual((r.status, r.body), (206, self.DATA[1020:]))

    def test_single_byte_and_last_byte(self):
        self.assertEqual(self.rng("bytes=5-5").body, self.DATA[5:6])
        self.assertEqual(self.rng("bytes=1023-1023").body, self.DATA[1023:])

    def test_unsatisfiable_is_416_with_size(self):
        for spec in ("bytes=1024-", "bytes=5000-6000", "bytes=-0"):
            r = self.rng(spec)
            self.assertEqual(r.status, 416, spec)
            self.assertEqual(r.h("content-range"), "bytes */1024", spec)
            self.assertEqual(r.body, b"")

    def test_malformed_is_416(self):
        for spec in ("bytes=9-3", "bytes=abc", "bytes=-", "bytes=--5", "bytes=1-2-3", "bytes=0x10-0x20", "bytes=a-b"):
            self.assertEqual(self.rng(spec).status, 416, spec)

    def test_ignored_forms_give_the_whole_file(self):
        for spec in ("items=0-5", "bytes=0-1,5-6", "garbage"):
            r = self.rng(spec)
            self.assertEqual((r.status, r.body), (200, self.DATA), spec)

    def test_empty_file_range_is_416(self):
        self.assertEqual(self.rng("bytes=0-", "/empty.bin").status, 416)
        self.assertEqual(get("/empty.bin").status, 200)

    def test_if_range(self):
        etag = get("/range.bin").h("etag")
        self.assertEqual(self.rng("bytes=0-3", extra={"If-Range": etag}).status, 206)
        r = self.rng("bytes=0-3", extra={"If-Range": '"stale"'})       # validator no longer matches: full body
        self.assertEqual((r.status, len(r.body)), (200, 1024))

    def test_head_with_range(self):
        r = req(PORT, "HEAD", "/range.bin", {"Range": "bytes=10-19"})
        self.assertEqual((r.status, r.body), (206, b""))
        self.assertEqual(r.h("content-range"), "bytes 10-19/1024")

    def test_range_on_the_big_file_is_exact(self):
        first = 37 * 1024 * 1024 + 123
        with open(str(ROOT / "bigdata" / "big.gz"), "rb") as f:
            f.seek(first)
            expect = f.read(70000)
        r = self.rng("bytes=%d-%d" % (first, first + 69999), "/bigdata/big.gz")
        self.assertEqual(r.status, 206)
        self.assertEqual(r.body, expect)


# --------------------------------------------------------------------------------------------------
# launcher (Launch Map.bat), self-check (selfcheck.py) and the plain-language guide (RUN_LOCALLY.md)
# --------------------------------------------------------------------------------------------------

REPO = Path(os.environ.get("BGMAP_REPO", str(Path(__file__).resolve().parents[2])))
PRESS = "Press any key to close this window"     # the launcher's own wait-for-a-key line (the text of `pause` depends on the language)


def load_selfcheck():
    import importlib.util
    spec = importlib.util.spec_from_file_location("selfcheck_under_test", str(APP / "selfcheck.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def alloc_span(preferred, n=14):
    """A base port whose next n-1 ports are free too (a server that finds its port busy walks upwards from there)."""
    for p in [preferred] + list(range(8120, 8300)):
        span = range(p, p + n)
        if not any(q in _used_ports for q in span) and all(port_is_free(q) for q in span):
            _used_ports.update(span)
            return p
    raise RuntimeError("no free port span")


def launch_env(**override):
    env = dict(os.environ)
    env.pop("DSOC_PORT", None)
    env["DSOC_PYTHON"] = sys.executable           # the launcher's own search is not what these tests are about
    env.update(override)
    return env


def hidden_python_env():
    """An environment in which the launcher cannot find any Python (no PATH entry, no py launcher, no Anaconda)."""
    empty = str(WORK / "no_python_home")
    os.makedirs(empty, exist_ok=True)
    env = launch_env(USERPROFILE=empty, ProgramData=empty, LOCALAPPDATA=empty,
                     PATH=os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"))
    env.pop("DSOC_PYTHON", None)
    return env


def run_bat(tree, args=(), env=None, stdin=subprocess.DEVNULL, name="Launch Map.bat", timeout=60):
    p = subprocess.run([str(Path(tree) / name)] + list(args), stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env=env or launch_env(), cwd=str(WORK), timeout=timeout)
    return p.returncode, p.stdout.decode("cp437", "replace")


def make_launch_tree(root, stub_server=False):
    """A map folder with the real launcher and self-check next to the real (or a stub) serve.py."""
    make_tree(root)
    for n in ("Launch Map.bat", "Run Checks.bat", "selfcheck.py"):
        shutil.copy2(str(APP / n), str(root / n))
    if stub_server:
        write(root / "serve.py", "import sys\nsys.exit(1)\n")


def kill_stragglers(folder):
    """Last resort at the end of the run: stop every process whose command line names this run's private folder.
    A server that a launcher spawned just before it gave up may not have written .serve_port yet, so stop_launched misses it."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($env:BGMAP_SWEEP) "
          "-and $_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], env=dict(os.environ, BGMAP_SWEEP=str(folder)),
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass


def stop_launched(root):
    """Stop the server that a launcher started in this tree (identified by the pid it wrote into .serve_port)."""
    _launched.append(str(root))
    try:
        pid = int((Path(root) / ".serve_port").read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@unittest.skipUnless(IS_WIN and (APP / "Launch Map.bat").is_file(), "needs Windows and Launch Map.bat")
class TestLauncher(unittest.TestCase):
    def tree(self, name="launch site (a&b) x", **kw):
        root = WORK / name                         # spaces, parentheses and an ampersand on purpose
        make_launch_tree(root, **kw)
        self.addCleanup(stop_launched, root)
        return root

    def test_simultaneous_launches_all_succeed(self):
        # two launchers that start within a second used to pick the same %RANDOM% temp file name; the loser failed with 13
        root = self.tree()
        port = alloc_span(8160)
        args = ["1", "/nobrowser", "/port", str(port)]
        for rnd in ("cold start", "server already running"):
            procs = [subprocess.Popen([str(root / "Launch Map.bat")] + args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, env=launch_env(), cwd=str(WORK)) for _ in range(3)]
            outs = [p.communicate(timeout=90)[0].decode("cp437", "replace") for p in procs]
            self.assertEqual([p.returncode for p in procs], [0, 0, 0], "%s:\n%s" % (rnd, "\n----\n".join(outs)))
            for o in outs:
                self.assertIn("Server running at http://127.0.0.1:%d\n" % port, o.replace("\r\n", "\n"))
        self.assertEqual(json.loads(req(port, "GET", "/__health").body.decode("utf-8"))["server"], "dsoc-flood-map")
        self.assertFalse(port_is_free(port))
        self.assertTrue(port_is_free(port + 1), "a second server was started")

    def test_launch_does_not_need_a_writable_temp_folder(self):
        root = self.tree()
        port = alloc_span(8160)
        gone = str(WORK / "no such temp folder")
        rc, out = run_bat(root, ["1", "/nobrowser", "/port", str(port)], env=launch_env(TEMP=gone, TMP=gone))
        self.assertEqual(rc, 0, out)
        self.assertIn("Server running at http://127.0.0.1:%d" % port, out)
        self.assertNotIn("cannot find the path", out.lower())

    def test_menu_runs_the_checks_and_gives_up_when_the_keyboard_is_gone(self):
        root = self.tree()
        port = alloc_span(8160)
        p = subprocess.run([str(root / "Launch Map.bat"), "/port", str(port)], input=b"4\r\n", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=launch_env(), cwd=str(WORK), timeout=90)
        out = p.stdout.decode("cp437", "replace")
        self.assertEqual(p.returncode, 0, out)
        self.assertIn("[4]  Check this installation", out)
        after_first_menu = out.split("[4]  Check this installation", 1)[1]
        self.assertIn("installation self-check", after_first_menu)                 # option 4 ran ...
        self.assertGreaterEqual(after_first_menu.count("[4]  Check this installation"), 1, out)   # ... and the menu came back
        self.assertIn("Please type 1, 2, 3 or 4.", out)
        self.assertIn("No choice made. The server keeps running at http://127.0.0.1:%d" % port, out)

    def test_python_in_a_folder_with_parentheses_and_ampersand(self):
        # "C:\\Program Files (x86)\\..." style paths: the address is read through cmd /c, which has its own quote rules
        root = self.tree()
        link = WORK / "py (x86)&1"
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(Path(sys.executable).parent)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if made != 0:
            self.skipTest("could not create a junction")
        LINKS.append(str(link))
        port = alloc_span(8160)
        rc, out = run_bat(root, ["1", "/nobrowser", "/port", str(port)], env=launch_env(DSOC_PYTHON=str(link / Path(sys.executable).name)))
        self.assertEqual(rc, 0, out)
        self.assertIn("Server running at http://127.0.0.1:%d" % port, out)

    def check_pause(self, tree, args, rc, env=None):
        """With a keypress source that is at EOF `pause` returns at once, so the text is what shows that it was asked for."""
        got, out = run_bat(tree, args, env=env)
        self.assertEqual(got, rc, "%s: %s" % (args, out))
        self.assertIn(PRESS, out, "%s should wait for a key" % (args,))
        got, out = run_bat(tree, list(args) + ["/nopause"], env=env)
        self.assertEqual(got, rc, "%s /nopause: %s" % (args, out))
        self.assertNotIn(PRESS, out, "%s /nopause must not wait" % (args,))
        got, out = run_bat(tree, ["/nopause"] + list(args), env=env)
        self.assertEqual(got, rc, "/nopause %s: %s" % (args, out))
        self.assertNotIn(PRESS, out, "/nopause %s must not wait" % (args,))

    def test_errors_wait_for_a_key_unless_nopause(self):
        root = WORK / "launch incomplete"
        root.mkdir()
        shutil.copy2(str(APP / "Launch Map.bat"), str(root / "Launch Map.bat"))
        self.check_pause(root, ["1"], 11)                       # folder incomplete, a shortcut with a choice
        self.check_pause(root, [], 11)                          # the menu start
        self.check_pause(root, ["9"], 10)                       # unknown option
        self.check_pause(root, ["/port", "99999"], 10)          # bad port
        rc, out = run_bat(root, ["/?"])                         # asking for help is not an error
        self.assertEqual(rc, 0, out)
        self.assertIn("Usage:", out)
        self.assertNotIn(PRESS, out)

    def test_python_not_found_waits_for_a_key(self):
        root = self.tree()
        rc, out = run_bat(root, ["1"], env=hidden_python_env())
        if rc != 12:
            self.skipTest("could not hide Python from the launcher on this computer (rc %s)" % rc)
        self.check_pause(root, ["1"], 12, env=hidden_python_env())
        self.check_pause(root, ["4"], 12, env=hidden_python_env())

    def test_server_that_does_not_start_waits_for_a_key(self):
        root = self.tree("launch stub", stub_server=True)
        self.check_pause(root, ["1", "/nobrowser"], 13)

    def test_option_4_waits_for_a_key_unless_nopause(self):
        root = self.tree()
        rc, out = run_bat(root, ["4"])
        self.assertIn(rc, (0, 14), out)
        self.assertIn("Summary:", out)
        self.check_pause(root, ["4"], rc)

    def test_it_really_blocks_until_a_key_arrives(self):
        root = WORK / "launch blocks"
        root.mkdir()
        shutil.copy2(str(APP / "Launch Map.bat"), str(root / "Launch Map.bat"))
        p = subprocess.Popen([str(root / "Launch Map.bat"), "1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=launch_env(), cwd=str(WORK))
        try:
            time.sleep(2.5)
            self.assertIsNone(p.poll(), "the window would have closed before the error could be read")
            p.stdin.write(b"\r\n")
            p.stdin.flush()
            self.assertEqual(p.wait(15), 11)
        finally:
            if p.poll() is None:
                p.kill()
            p.stdout.close()
            p.stdin.close()
        q = subprocess.Popen([str(root / "Launch Map.bat"), "1", "/nopause"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=launch_env(), cwd=str(WORK))
        try:
            self.assertEqual(q.wait(15), 11, "/nopause must return without a key")
        finally:
            if q.poll() is None:
                q.kill()
            q.stdout.close()
            q.stdin.close()


class TestSelfcheckHourlyLayers(unittest.TestCase):
    FEATURE = ('{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-86.5,37.0],[-86.5,37.1],[-86.4,37.1],[-86.5,37.0]]]},'
               '"properties":{"hour":1,"tier":"deep"}}')

    @classmethod
    def setUpClass(cls):
        cls.sc = load_selfcheck()

    def good(self, n=40):
        return '{"type":"FeatureCollection","features":[' + ",".join([self.FEATURE] * n) + ']}'

    def tree(self, name):
        root = WORK / name
        for scen, hours in self.sc.DEFAULT_HPC.items():
            for h in range(1, hours + 1):
                write(root / "layers" / "flood_hpc" / scen / ("flood_h%02d.geojson" % h), self.good())
        return root

    def hpc_row(self, root):
        rep = self.sc.Report()
        self.sc.check_hpc(rep, str(root))
        self.assertEqual(len(rep.rows), 1)
        return rep.rows[0]

    def hour(self, root, h, scen="2yr"):
        return root / "layers" / "flood_hpc" / scen / ("flood_h%02d.geojson" % h)

    def test_complete_set_is_ok_including_dry_and_pretty_printed_hours(self):
        root = self.tree("hpc ok")
        write(self.hour(root, 3), '{"type":"FeatureCollection","features":[]}')                     # a dry hour (42 bytes)
        write(self.hour(root, 4), json.dumps(json.loads(self.good(3)), indent=2) + "\n")            # tools that indent
        row = self.hpc_row(root)
        self.assertEqual((row.status, row.items), ("OK", []), row.detail)

    def test_truncated_hour_fails_and_is_named(self):
        root = self.tree("hpc cut")
        text = self.good()
        write(self.hour(root, 5), text[:len(text) // 2])
        row = self.hpc_row(root)
        self.assertEqual(row.status, "FAIL", row.detail)
        self.assertIn("flood_h05.geojson", row.detail)
        self.assertIn("2yr", row.detail)

    def test_error_page_instead_of_geojson_fails_and_is_named(self):
        root = self.tree("hpc html")
        write(self.hour(root, 6), "<html>502 Bad Gateway</html>")
        row = self.hpc_row(root)
        self.assertEqual(row.status, "FAIL", row.detail)
        self.assertIn("flood_h06.geojson", row.detail)

    def test_cut_that_happens_to_end_in_a_brace_fails(self):
        root = self.tree("hpc brace")
        text = self.good()
        cut = text.index("},{") + 1                           # right after the first feature's last brace
        self.assertTrue(text[:cut].endswith("}"))
        write(self.hour(root, 7, "v2_25yr"), text[:cut])
        row = self.hpc_row(root)
        self.assertEqual(row.status, "FAIL", row.detail)
        self.assertIn("v2_25yr", row.detail)

    def test_several_bad_hours_are_all_reported(self):
        root = self.tree("hpc many")
        write(self.hour(root, 2), "<html>x</html>")
        write(self.hour(root, 9), self.good()[:-5])
        row = self.hpc_row(root)
        self.assertEqual(row.status, "FAIL")
        entries = [m for m in row.items if m.startswith("2yr")]
        self.assertEqual(len(entries), 1, row.items)                      # one line per storm ...
        self.assertIn("flood_h02.geojson", entries[0])                    # ... naming every bad hour
        self.assertIn("flood_h09.geojson", entries[0])

    def test_zero_length_hour_still_fails(self):
        root = self.tree("hpc zero")
        write(self.hour(root, 8), b"")
        row = self.hpc_row(root)
        self.assertEqual(row.status, "FAIL", row.detail)
        self.assertIn("h08", row.detail)

    def complete_tree(self, name):
        """Everything the self-check treats as a FAIL when missing, in minimal size (the rest is WARN)."""
        root = self.tree(name)
        write(root / "viewer.html", "x" * 30000)
        write(root / "serve.py", "x" * 20000)
        for layer, size in (("buildings", 1024 * 1024), ("all_parcels", 2 * 1024 * 1024), ("fema", 100 * 1024),
                            ("sinkholes", 200 * 1024), ("riverine", 4 * 1024), ("roads", 50 * 1024)):
            write(root / "layers" / (layer + ".geojson"), '{"type":"FeatureCollection","features":[' + " " * (size + 1) + ']}')
        return root

    def selfcheck(self, root, *flags):
        return subprocess.run([sys.executable, str(APP / "selfcheck.py"), "--root", str(root)] + list(flags),
                              capture_output=True, text=True, timeout=60)

    def test_exit_code_and_summary_report_a_bad_hour(self):
        root = self.complete_tree("hpc cli bad")
        ok = self.selfcheck(root, "--json")
        self.assertEqual(ok.returncode, 0, ok.stdout)
        write(self.hour(root, 5), self.good()[:100])
        bad = self.selfcheck(root, "--json")
        self.assertEqual(bad.returncode, 1)
        row = [c for c in json.loads(bad.stdout)["checks"] if c["name"] == "Hourly flood layers"][0]
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("flood_h05.geojson", row["detail"])
        summ = self.selfcheck(root, "--summary")                  # what the launcher shows: the problem rows when something fails
        self.assertEqual(summ.returncode, 1)
        self.assertIn("flood_h05.geojson", summ.stdout)
        self.assertIn("Summary:", summ.stdout)

    def test_summary_is_one_line_when_nothing_fails(self):
        root = self.complete_tree("hpc cli ok")
        quiet = self.selfcheck(root, "--summary")
        self.assertEqual(quiet.returncode, 0, quiet.stdout)
        lines = [ln for ln in quiet.stdout.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1, quiet.stdout)
        self.assertTrue(lines[0].startswith("Summary: "), quiet.stdout)


class TestRunLocallyGuide(unittest.TestCase):
    GUIDE = REPO / "RUN_LOCALLY.md"

    def setUp(self):
        if not self.GUIDE.is_file():
            self.skipTest("RUN_LOCALLY.md not found in %s" % REPO)
        self.text = self.GUIDE.read_text(encoding="utf-8")

    def test_every_script_it_names_exists(self):
        names = set(re.findall(r"scripts[\\/]([A-Za-z0-9_.\-]+\.(?:py|sh))", self.text))
        self.assertTrue(names, "the guide should name the builder scripts")
        missing = sorted(n for n in names if not (REPO / "scripts" / n).is_file())
        self.assertEqual(missing, [], "RUN_LOCALLY.md sends the reader to scripts that do not exist")

    def test_files_of_the_app_folder_it_names_exist(self):
        for name in ("Launch Map.bat", "Run Checks.bat", "serve.py", "selfcheck.py", "viewer.html", "flood_report.html", "dem_bg.tif"):
            self.assertIn(name, self.text, name)
            self.assertTrue((APP / name).is_file(), "%s is named in the guide but not in the app folder" % name)

    def test_command_line_options_it_shows_exist(self):
        block = self.text.split("## For scripts and developers", 1)[1]
        for script in ("serve.py", "selfcheck.py"):
            helptext = subprocess.run([sys.executable, str(APP / script), "--help"], capture_output=True, text=True, timeout=60).stdout
            for line in block.splitlines():
                if line.startswith("python %s" % script):
                    for opt in re.findall(r"--[a-z][a-z-]*", line):
                        self.assertIn(opt, helptext, "%s: guide shows %s" % (script, opt))

    def test_launcher_options_it_shows_are_accepted(self):
        bat = (APP / "Launch Map.bat").read_text(encoding="ascii", errors="replace").lower()
        for opt in ("/nobrowser", "/port", "/nopause"):
            self.assertIn(opt, self.text.lower(), "the guide should mention %s" % opt)
            self.assertTrue(('"%s"' % opt) in bat, "the launcher does not know %s" % opt)

    def test_it_does_not_promise_the_old_grid_cache(self):
        # the server revalidates the grids on every request now; "cached for an hour" sent readers to Ctrl+F5 for nothing
        old = re.search(r"(?i)\b(?:an|one|1)[ -]hours?\b", self.text)
        self.assertIsNone(old, "RUN_LOCALLY.md still describes the old grid cache: %r" % (old and old.group(0)))


# --- more tests go above this line ---
if __name__ == "__main__":
    unittest.main(verbosity=2)
