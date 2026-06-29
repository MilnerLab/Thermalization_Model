"""Helpers to open HDF5 files without filesystem locking."""

from __future__ import annotations

import os
from typing import Any


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def open_h5(h5py_module: Any, *args: Any, **kwargs: Any):
    """Call ``h5py.File`` with locking disabled when supported."""
    try:
        return h5py_module.File(*args, locking=False, **kwargs)
    except TypeError:
        return h5py_module.File(*args, **kwargs)
