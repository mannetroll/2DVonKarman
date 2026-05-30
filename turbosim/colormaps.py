"""256-entry RGB color tables built from a small set of anchor colors.

The tables are plain ``(256, 3)`` uint8 NumPy arrays; the GUI turns them into a
Qt color table.  ``Inferno`` is a faithful sampling of matplotlib's map;
``Ocean`` is a deep-sea black->blue->teal->white ramp; ``Gray`` is linear.
"""

from __future__ import annotations

import numpy as np

# Anchor colors (0-255), interpolated linearly to 256 entries.
_ANCHORS: dict[str, list[tuple[int, int, int]]] = {
    "Inferno": [
        (0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
        (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164),
    ],
    "Ocean": [
        (0, 0, 0), (0, 20, 80), (0, 60, 130), (0, 120, 160),
        (40, 180, 180), (160, 220, 210), (255, 255, 255),
    ],
    "Gray": [(0, 0, 0), (255, 255, 255)],
}

COLORMAP_NAMES = tuple(_ANCHORS.keys())


def _build(anchors: list[tuple[int, int, int]]) -> np.ndarray:
    a = np.asarray(anchors, dtype=np.float64)
    xp = np.linspace(0.0, 1.0, len(a))
    x = np.linspace(0.0, 1.0, 256)
    out = np.empty((256, 3), dtype=np.uint8)
    for c in range(3):
        out[:, c] = np.clip(np.interp(x, xp, a[:, c]), 0, 255).astype(np.uint8)
    return out


_TABLES: dict[str, np.ndarray] = {name: _build(a) for name, a in _ANCHORS.items()}


def colormap(name: str) -> np.ndarray:
    """Return the ``(256, 3)`` uint8 RGB table for ``name``."""
    return _TABLES[name]
