"""Compute-backend selection: NumPy + ``scipy.fft`` (CPU) or CuPy +
``cupyx.scipy.fft`` (GPU).

This mirrors the CPU/GPU abstraction proven in ``fast/turbo_simulator.py`` so the
exact same solver code runs on an NVIDIA GPU (e.g. an RTX 3090) through CuPy, or
on the CPU through SciPy.  The solver holds an array module ``xp`` (``numpy`` or
``cupy``) and an FFT module, and never imports either backend directly.

Key ideas taken from the reference port:

* ``xp`` alias so array ops are backend-agnostic.
* FFT module is ``cupyx.scipy.fft`` on GPU (it reuses cuFFT plans via CuPy's
  internal plan cache) and ``scipy.fft`` on CPU (multithreaded via ``workers``).
* Everything runs in **float32 / complex64** — roughly half the memory traffic
  and, on consumer GPUs like the 3090, far higher throughput than float64.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as _scipy_fft

# CuPy and its SciPy-compatible FFT are both optional; absence => CPU only.
try:  # pragma: no cover - depends on the machine having CUDA + CuPy
    import cupy as _cp

    try:
        import cupyx.scipy.fft as _cupy_fft
    except Exception:
        _cupy_fft = None
except Exception:
    _cp = None
    _cupy_fft = None


def gpu_available() -> bool:
    """True if CuPy is importable and at least one CUDA device is present."""
    if _cp is None:
        return False
    try:  # pragma: no cover - needs a real GPU
        return _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_name() -> str | None:
    """Marketing name of the active CUDA device, e.g. ``NVIDIA GeForce RTX 3090``."""
    if not gpu_available():
        return None
    try:  # pragma: no cover - needs a real GPU
        props = _cp.cuda.runtime.getDeviceProperties(_cp.cuda.Device().id)
        name = props["name"]
        return name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
    except Exception:
        return None


def select_backend(backend: str = "auto"):
    """Resolve ``"cpu" | "gpu" | "auto"`` to ``(name, xp, fft_module)``.

    ``name`` is the concrete backend actually selected (``"cpu"`` or ``"gpu"``).
    ``backend="gpu"`` raises if no CUDA GPU is available; ``"auto"`` prefers the
    GPU and silently falls back to the CPU.
    """
    if backend == "gpu":
        if not gpu_available():
            raise RuntimeError(
                "backend='gpu' was requested but no CuPy/CUDA GPU is available"
            )
        return "gpu", _cp, (_cupy_fft if _cupy_fft is not None else _cp.fft)
    if backend == "auto" and gpu_available():
        return "gpu", _cp, (_cupy_fft if _cupy_fft is not None else _cp.fft)
    if backend not in ("cpu", "gpu", "auto"):
        raise ValueError(f"unknown backend {backend!r}")
    return "cpu", np, _scipy_fft


def to_host(arr) -> np.ndarray:
    """Bring an array to host NumPy: a no-op for NumPy, ``.get()`` for CuPy."""
    get = getattr(arr, "get", None)
    return get() if get is not None else np.asarray(arr)
