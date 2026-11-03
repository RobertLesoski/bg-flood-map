"""A tiny OpenStreetMap PBF WRITER for the tests (the viewer only reads PBF): enough protobuf to build files with dense nodes,
building / highway ways, tags, several blocks and either compression.  Not part of the app."""
from __future__ import annotations

import struct
import zlib


def varint(v: int) -> bytes:
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def zigzag(v: int) -> int:
    return (v << 1) ^ (v >> 63)


def key(field: int, wt: int) -> bytes:
    return varint((field << 3) | wt)


def f_varint(field: int, v: int) -> bytes:
    return key(field, 0) + varint(v)


def f_len(field: int, payload: bytes) -> bytes:
    return key(field, 2) + varint(len(payload)) + payload


def packed(vals, signed_delta=False, signed=False) -> bytes:
    out = b""
    prev = 0
    for v in vals:
        if signed_delta:
            out += varint(zigzag(v - prev))
            prev = v
        elif signed:
            out += varint(zigzag(v))
        else:
            out += varint(v)
    return out


class Block:
    """One PrimitiveBlock: add_nodes() or add_ways(); strings are interned per block."""

    def __init__(self, granularity: int = 100):
        self.strings = [b""]
        self.groups = []
        self.gran = granularity

    def s(self, text: str) -> int:
        b = text.encode("utf-8")
        if b not in self.strings:
            self.strings.append(b)
        return self.strings.index(b)

    def add_nodes(self, nodes):
        """nodes: [(id, lon_deg, lat_deg)] sorted by id (dense nodes, delta coded)."""
        ids = [n[0] for n in nodes]
        lat = [round(n[2] * 1e9 / self.gran) for n in nodes]
        lon = [round(n[1] * 1e9 / self.gran) for n in nodes]
        dense = (f_len(1, packed(ids, signed_delta=True)) + f_len(8, packed(lat, signed_delta=True))
                 + f_len(9, packed(lon, signed_delta=True)))
        self.groups.append(f_len(2, dense))

    def add_ways(self, ways):
        """ways: [(id, {tag: value}, [node ids])]."""
        body = b""
        for wid, tags, refs in ways:
            ks = [self.s(k) for k in tags]
            vs = [self.s(v) for v in tags.values()]
            w = (f_varint(1, wid) + f_len(2, packed(ks)) + f_len(3, packed(vs)) + f_len(8, packed(refs, signed_delta=True)))
            body += f_len(3, w)
        self.groups.append(body)

    def encode(self) -> bytes:
        st = b"".join(f_len(1, s) for s in self.strings)
        out = f_len(1, st)
        for g in self.groups:
            out += f_len(2, g)
        out += f_varint(17, self.gran)
        return out


def blob(btype: str, payload: bytes, compress: str = "zlib") -> bytes:
    if compress == "zlib":
        b = f_varint(2, len(payload)) + f_len(3, zlib.compress(payload))
    elif compress == "raw":
        b = f_len(1, payload)
    else:                                   # an lzma blob (field 4): the reader must refuse it
        b = f_varint(2, len(payload)) + f_len(4, payload)
    header = f_len(1, btype.encode()) + f_varint(3, len(b))
    return struct.pack(">I", len(header)) + header + b


def write_pbf(path, blocks, compress: str = "zlib"):
    data = blob("OSMHeader", f_len(4, b"OsmSchema-V0.6"), compress if compress != "lzma" else "zlib")
    for b in blocks:
        data += blob("OSMData", b.encode(), compress)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def ring_nodes(first_id: int, lon0: float, lat0: float, w_deg: float, h_deg: float):
    """Corner nodes of a rectangle (4) -> ([(id, lon, lat)], closed ring of node ids)."""
    pts = [(lon0, lat0), (lon0 + w_deg, lat0), (lon0 + w_deg, lat0 + h_deg), (lon0, lat0 + h_deg)]
    nodes = [(first_id + i, x, y) for i, (x, y) in enumerate(pts)]
    ids = [n[0] for n in nodes]
    return nodes, ids + [ids[0]]
