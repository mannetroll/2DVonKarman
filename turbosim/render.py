"""Field -> uint8 conversion (the PySide6-free half of fast rendering).

The Qt-specific part (wrapping the buffer in a ``QImage`` and building a
``QPixmap``) lives in :mod:`turbosim.gui`; this module only produces the
contiguous uint8 array and chooses sensible display limits, so it can be tested
without Qt.
"""

from __future__ import annotations

import numpy as np

# Fields that are signed and look best on a symmetric range about zero.
_SIGNED = {"Vorticity", "U-Velocity", "V-Velocity", "Stream function"}


def display_limits(field: np.ndarray, name: str) -> tuple[float, float]:
    """Robust (percentile-based) color limits for ``field``."""
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 0.0, 1.0
    if name in _SIGNED:
        a = float(np.percentile(np.abs(finite), 99.5))
        a = a if a > 0 else 1.0
        return -a, a
    vmax = float(np.percentile(finite, 99.5))
    return 0.0, vmax if vmax > 0 else 1.0


def to_uint8(field: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize ``field`` to ``[vmin, vmax]`` and return a contiguous uint8.

    The array is transposed so that the first axis (x) maps to image columns
    and the second axis (y) to image rows, matching screen orientation.
    """
    span = vmax - vmin
    if span <= 0:
        span = 1.0
    f = (field - vmin) * (1.0 / span)
    np.clip(f, 0.0, 1.0, out=f)
    u8 = (f * 255.0).astype(np.uint8)
    return np.ascontiguousarray(u8.T)
