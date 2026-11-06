"""Colour-blind safety check for legends: simulate protanopia / deuteranopia / tritanopia (Machado et al. 2009, severity 1.0,
applied in linear RGB) and require every pair of legend colours to stay distinguishable (CIE76 delta-E >= MIN_DE) under each.
"""
from __future__ import annotations

import numpy as np

MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}
MIN_DE = 10.0


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(lin):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def min_delta_e(colours_srgb01) -> dict:
    """{'normal': dE, 'protan': dE, ...}: smallest pairwise delta-E of the colours (sRGB 0..1) under each vision type."""
    lin = _lin(np.asarray(colours_srgb01, float)[:, :3])
    out = {}
    for name, M in [("normal", np.eye(3))] + list(MACHADO.items()):
        lab = _lab(np.clip(lin @ M.T, 0, 1))
        d = np.linalg.norm(lab[:, None] - lab[None], axis=-1)
        d[np.eye(len(lab), dtype=bool)] = np.inf
        out[name] = float(d.min())
    return out


def safe(colours_srgb01, min_de: float = MIN_DE) -> bool:
    return all(v >= min_de for v in min_delta_e(colours_srgb01).values())
