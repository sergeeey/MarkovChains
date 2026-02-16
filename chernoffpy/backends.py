"""Backend helpers for NumPy/CuPy interoperability.

CuPy is optional. The default backend is NumPy.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:  # pragma: no cover - depends on environment
    cp = None
    HAS_CUPY = False


def get_backend(name: str = "numpy"):
    """Return array backend module with NumPy-like API."""
    normalized = name.lower()
    if normalized == "cupy":
        if not HAS_CUPY:
            raise ImportError("CuPy not installed. Install with: pip install cupy-cuda12x")
        return cp
    if normalized != "numpy":
        raise ValueError(f"Unknown backend '{name}'. Expected 'numpy' or 'cupy'.")
    return np


def to_backend(arr, xp):
    """Convert array-like to target backend module."""
    if xp is np:
        return arr if isinstance(arr, np.ndarray) else np.asarray(arr)
    return xp.asarray(arr)


def to_numpy(arr):
    """Convert array-like to NumPy ndarray."""
    if isinstance(arr, np.ndarray):
        return arr
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return arr.get()
    return np.asarray(arr)
