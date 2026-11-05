"""Address index (ADDRESS_INDEX.md): loader, typeahead and per-storm flood lookup.  Pure CPU (numpy + json + gzip).

    idx = AddressIndex.load(path)                  # single file, or manifest + shards (merged); raises IndexError_ with a
                                                   # human readable message when the file is missing / unreadable
    idx.search("1500 scottsville r", limit=12)     # -> [Hit]  address rows (record index) or street rows
    rec = idx.record(i)                            # Record(address, lng, lat, use, value, parcel, bid, kind, ...)
    rows = idx.storm_rows(i)                       # [StormRow] one per storm of the index: depth cm, tier, first hour, hours
    idx.by_bid(bid) -> record index or -1          # footprint feature index (city) / OSM way id (county)

Flood arrays: `flood[storm]` is a flat int array, stride 5 `[index delta, depth cm, tier 0-4, first hour flooded (0 never),
hours flooded]`, indices accumulate from 0 (the first delta is relative to record 0).  A record missing from a storm is dry
(below the 0.02 m wet threshold, as the index builder drops it).  Depth = max over the footprint pixels (all-touched) of the
flow grid, flooded = depth >= 0.1 m (10 cm).  The normalisation / matching rules are those of scripts/build_address_index.py
(`ABBREV`, `normalize_street`, `search`); this module re-implements them (keep ABBREV in sync) and adds ranking.
"""
from __future__ import annotations

import bisect
import gzip
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ABBREV = {
    "STREET": "ST", "ROAD": "RD", "DRIVE": "DR", "AVENUE": "AVE", "BOULEVARD": "BLVD", "LANE": "LN", "COURT": "CT",
    "CIRCLE": "CIR", "PARKWAY": "PKWY", "HIGHWAY": "HWY", "TERRACE": "TER", "PLACE": "PL", "TRAIL": "TRL",
    "SQUARE": "SQ", "EXPRESSWAY": "EXPY", "ALLEY": "ALY", "BYPASS": "BYP", "CROSSING": "XING", "POINT": "PT",
    "HEIGHTS": "HTS", "RIDGE": "RDG", "VALLEY": "VLY", "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW",
}
TIER_EDGES_CM = (10, 30, 60, 120)
TIER_NAMES = ("dry", "shallow", "moderate", "deep", "severe")
FLOOD_MIN_CM = 10


class IndexError_(Exception):
    """The address index cannot be used; str(e) is the user-facing explanation."""


def clean_text(s) -> str:
    s = str(s or "").upper().replace("&", " AND ")
    s = re.sub(r"[.'`]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_street(street, partial: bool = False) -> str:
    """Upper case, punctuation out, suffixes / directions abbreviated; partial=True leaves the last token as typed."""
    toks = re.sub(r"[^A-Z0-9 ]", " ", clean_text(street)).split()
    out = []
    for i, t in enumerate(toks):
        out.append(t if (partial and i == len(toks) - 1) else ABBREV.get(t, t))
    return " ".join(out)


def tier_of_cm(cm) -> int:
    """0 below 10 cm, 1 shallow [10,30), 2 moderate [30,60), 3 deep [60,120), 4 severe >= 120 cm."""
    return int(np.searchsorted(TIER_EDGES_CM, cm, side="right"))


@dataclass
class Record:
    i: int
    address: str
    street: str
    num: str
    lng: float
    lat: float
    use: str
    value: int
    parcel: str
    bid: int
    kind: int          # 0 building / address, 1 lot (depth taken at the lot point)
    nb: int


@dataclass
class Hit:
    kind: str          # 'address' | 'street'
    idx: int           # record index (address) or street index (street)
    label: str
    sub: str = ""


@dataclass
class StormRow:
    key: str
    label: str
    source: str        # 'rasters' | 'tiers'
    depth_cm: int
    tier: int
    wet: bool          # present in the index (water >= 2 cm somewhere on the footprint)
    first_hour: int    # 0 = never flooded
    hours: int

    @property
    def flooded(self) -> bool:
        return self.depth_cm >= FLOOD_MIN_CM


class AddressIndex:
    def __init__(self, doc: dict, path: str | None = None):
        self.path = path
        self.doc = doc
        self.domain = doc.get("domain", "")
        self.label = doc.get("label", self.domain)
        self.built = doc.get("built", "")
        self.rules = doc.get("rules", "")
        self.storms = list(doc.get("storms") or [])
        self.storm_keys = [s["key"] for s in self.storms]
        self.use_legend = list(doc.get("use_legend") or [])
        self.n = int(doc.get("n") or len(doc.get("num", [])))
        ox, oy = doc["origin"]
        sc = float(doc.get("scale", 100000))
        self.lng = ox + np.asarray(doc["x"], np.int64) / sc
        self.lat = oy + np.asarray(doc["y"], np.int64) / sc
        self.num = list(doc["num"])
        self._num_arr = np.array(self.num, dtype="U12") if self.num else np.zeros(0, "U1")
        self.use = np.asarray(doc["use"], np.int16)
        self.val = np.asarray(doc["val"], np.int64)
        self.parcel = list(doc["parcel"])
        self.bid = np.asarray(doc["bid"], np.int64)
        self.kind = np.asarray(doc["kind"], np.int8)
        self.nb = np.asarray(doc.get("nb") or np.zeros(self.n), np.int32)
        st = doc["streets"]
        self.keys = list(st["key"])
        self.disp = list(st["disp"])
        self.start = list(st["start"])
        if not (len(self.num) == self.n == len(self.bid) == len(self.lng)):
            raise IndexError_(f"address index {path}: column lengths disagree (n={self.n})")
        self._street_of = np.repeat(np.arange(len(self.keys)), np.diff(np.asarray(self.start)))
        # flood arrays -> dense per storm
        self.depth_cm: dict[str, np.ndarray] = {}
        self.tier: dict[str, np.ndarray] = {}
        self.first: dict[str, np.ndarray] = {}
        self.hours: dict[str, np.ndarray] = {}
        self.wet: dict[str, np.ndarray] = {}
        for k in self.storm_keys:
            flat = np.asarray((doc.get("flood") or {}).get(k, []), np.int64)
            if flat.size % 5:
                raise IndexError_(f"address index {path}: flood array of {k} has a length that is not a multiple of 5")
            f = flat.reshape(-1, 5)
            ix = np.cumsum(f[:, 0]) if f.size else np.zeros(0, np.int64)
            d = np.zeros(self.n, np.int32)
            t = np.zeros(self.n, np.uint8)
            fh = np.zeros(self.n, np.uint8)
            hh = np.zeros(self.n, np.uint8)
            w = np.zeros(self.n, bool)
            if ix.size:
                if ix.min() < 0 or ix.max() >= self.n:
                    raise IndexError_(f"address index {path}: flood indices of {k} out of range")
                d[ix], t[ix], fh[ix], hh[ix], w[ix] = f[:, 1], f[:, 2], f[:, 3], f[:, 4], True
            self.depth_cm[k], self.tier[k], self.first[k], self.hours[k], self.wet[k] = d, t, fh, hh, w
        self._bid_map: dict[int, int] | None = None

    # ------------------------------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path) -> "AddressIndex":
        path = str(path)
        if not os.path.isfile(path):
            raise IndexError_(f"address index not found: {path} (build it with scripts/build_address_index.py)")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:                                                     # noqa: BLE001
            raise IndexError_(f"address index {path} cannot be read: {type(e).__name__}: {e}") from None
        if doc.get("format") != "bgaddr":
            raise IndexError_(f"{path} is not an address index (format {doc.get('format')!r})")
        if doc.get("sharded"):
            doc = _merge_shards(doc, os.path.dirname(path))
        try:
            return cls(doc, path)
        except KeyError as e:
            raise IndexError_(f"address index {path}: missing field {e}") from None

    # ------------------------------------------------------------------------------------------ records
    def street_of(self, i: int) -> int:
        return int(self._street_of[i])

    def address(self, i: int) -> str:
        return (self.num[i] + " " + self.disp[self.street_of(i)]).strip()

    def record(self, i: int) -> Record:
        s = self.street_of(i)
        return Record(int(i), self.address(i), self.disp[s], self.num[i], float(self.lng[i]), float(self.lat[i]),
                      self.use_legend[self.use[i]] if 0 <= self.use[i] < len(self.use_legend) else "", int(self.val[i]),
                      self.parcel[i], int(self.bid[i]), int(self.kind[i]), int(self.nb[i]))

    def by_bid(self, bid: int) -> int:
        """Record index of the building with this footprint id (city: feature index in buildings.geojson; county: OSM way id)."""
        if self._bid_map is None:
            m = {}
            for i in np.flatnonzero((self.bid >= 0) & (self.kind == 0)):
                m.setdefault(int(self.bid[i]), int(i))
            self._bid_map = m
        return self._bid_map.get(int(bid), -1)

    def nearest(self, lng: float, lat: float, max_m: float = 40.0, buildings_only: bool = True) -> int:
        """Record whose point is closest to (lng, lat) within max_m metres, else -1."""
        k = np.cos(np.radians(lat))
        dx = (self.lng - lng) * 111320.0 * k
        dy = (self.lat - lat) * 110540.0
        d2 = dx * dx + dy * dy
        if buildings_only:
            d2 = np.where(self.kind == 0, d2, np.inf)
        i = int(np.argmin(d2)) if d2.size else -1
        return i if i >= 0 and d2[i] <= max_m * max_m else -1

    def storm_rows(self, i: int) -> list[StormRow]:
        out = []
        for s in self.storms:
            k = s["key"]
            out.append(StormRow(k, s.get("label", k), s.get("source", ""), int(self.depth_cm[k][i]), int(self.tier[k][i]),
                                bool(self.wet[k][i]), int(self.first[k][i]), int(self.hours[k][i])))
        return out

    def summary(self, i: int, storm: str | None = None) -> tuple[int, int]:
        """(depth cm, tier) at record i for `storm` (default: the deepest over all storms)."""
        if storm in self.depth_cm:
            return int(self.depth_cm[storm][i]), int(self.tier[storm][i])
        if not self.storm_keys:
            return 0, 0
        d = max(int(self.depth_cm[k][i]) for k in self.storm_keys)
        return d, tier_of_cm(d)

    # ------------------------------------------------------------------------------------------ typeahead
    def search(self, query: str, limit: int = 12) -> list[Hit]:
        """Typeahead.  '1500 scottsville r' -> address rows; 'scottsville' -> one row per matching street (the addresses of the
        street when it is the only one); '1500' -> addresses with that number.  Exact house numbers rank first."""
        q = clean_text(str(query).split(",")[0])
        if not q or self.n == 0:
            return []
        m = re.match(r"^(\d+[A-Z]?)\s+(.*)$", q)
        if m:
            num_q, street_q = m.group(1), m.group(2)
        elif re.fullmatch(r"\d+[A-Z]?", q):
            num_q, street_q = q, ""
        else:
            num_q, street_q = "", q
        if street_q:
            trailing = str(query).rstrip(" ") != str(query) or str(query).endswith(" ")
            streets = self._match_streets(street_q, trailing)
            if num_q:
                return self._address_hits(streets, num_q, limit)
            if len(streets) == 1:
                return self._address_hits(streets, "", limit)
            return [Hit("street", si, self.disp[si], f"{self.start[si + 1] - self.start[si]} addresses") for si in streets[:limit]]
        # number only: linear scan (numpy)
        ok = np.char.startswith(self._num_arr, num_q) & (self.kind == 0)
        idx = np.flatnonzero(ok)
        exact = idx[self._num_arr[idx] == num_q]
        rest = idx[self._num_arr[idx] != num_q]
        out = []
        for i in list(exact[:limit]) + list(rest[:max(0, limit - len(exact))]):
            out.append(Hit("address", int(i), self.address(int(i)), self._sub(int(i))))
        return out[:limit]

    def _match_streets(self, street_q: str, trailing: bool) -> list[int]:
        keys = self.keys
        v1 = normalize_street(street_q, partial=not trailing)
        v2 = normalize_street(street_q, partial=False)
        found, seen = [], set()
        for pref in (v1, v2):
            if pref in seen:
                continue
            seen.add(pref)
            si = bisect.bisect_left(keys, pref)
            while si < len(keys) and keys[si].startswith(pref):
                if si not in found:
                    found.append(si)
                si += 1
        if not found and len(street_q) >= 3:                       # a word of the street starts with every typed word
            words = street_q.split()
            for si, k in enumerate(keys):
                kw = k.split()
                if all(any(w2.startswith(w) for w2 in kw) for w in words):
                    found.append(si)
        return found

    def _address_hits(self, streets: list[int], num_q: str, limit: int) -> list[Hit]:
        cand = []
        for si in streets:
            lo, hi = self.start[si], self.start[si + 1]
            for i in range(lo, hi):
                if self.kind[i] != 0 and num_q == "":
                    continue
                if self.num[i].startswith(num_q):
                    cand.append(i)
            if len(cand) > 4000:
                break
        if not cand:
            return []
        if num_q:
            cand.sort(key=lambda i: (self.num[i] != num_q, self.kind[i], _numkey(self.num[i])))
        else:
            cand.sort(key=lambda i: (self.kind[i], _numkey(self.num[i])))
        return [Hit("address", i, self.address(i), self._sub(i)) for i in cand[:limit]]

    def _sub(self, i: int) -> str:
        d, t = self.summary(i)
        bits = []
        if self.kind[i] == 1:
            bits.append("lot")
        if t >= 1:
            bits.append(f"up to {d} cm ({TIER_NAMES[t]})")
        return "  ".join(bits)


def _numkey(num: str):
    m = re.match(r"(\d+)(.*)", num or "")
    return (int(m.group(1)), m.group(2)) if m else (-1, num or "")


def _merge_shards(man: dict, base: str) -> dict:
    """Manifest + shards -> one document (the reference loader of build_address_index.load_index)."""
    parts = []
    for sh in man["shards"]:
        p = os.path.join(base, sh["file"])
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                parts.append(json.load(f))
        except Exception as e:                                                     # noqa: BLE001
            raise IndexError_(f"address index shard {p} cannot be read: {type(e).__name__}: {e}") from None
    doc = dict(man)
    for k in ("x", "y", "num", "use", "val", "parcel", "bid", "kind", "nb"):
        doc[k] = [v for p in parts for v in p[k]]
    doc["flood"] = {}
    for s in man["storms"]:
        nm = s["key"]
        flat, prev = [], 0
        for p in parts:
            f = p["flood"].get(nm, [])
            a = p["rec_lo"]
            for t in range(0, len(f), 5):
                a += f[t]
                flat += [a - prev, f[t + 1], f[t + 2], f[t + 3], f[t + 4]]
                prev = a
        doc["flood"][nm] = flat
    return doc


def index_path(root, domain_key: str) -> Path:
    """Where the index of a domain lives: <root>/layers/address_index.json.gz (city), layers/county/... (county)."""
    root = Path(root)
    return root / "layers" / ("county" if domain_key == "county" else "") / "address_index.json.gz"
