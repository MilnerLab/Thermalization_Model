#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""05b_free_rotor_drive_observables.py

Precompute angular observables in the antipodally symmetric |J,M> basis used by 05.
These matrices depend only on the basis and not on the driven eigensystem from 05a.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import multiprocessing as mp

import h5py
import numpy as np
from h5_locking import open_h5

params = importlib.reload(importlib.import_module("01_Parameters"))
_Y_BASIS_05B: np.ndarray | None = None
_WEIGHTS_05B: np.ndarray | None = None
_VALS_05B: dict[str, np.ndarray] = {}

OBS_KEYS = ["one", "x2", "y2", "sin2theta", "cos2phi_rot", "cos2theta2D"]


def get_defaults() -> dict:
    mod = importlib.reload(importlib.import_module("01_Parameters"))
    return dict(mod.get_defaults_for_case())


def get_paths(p: dict) -> tuple[Path, Path]:
    data_dir = Path(str(p.get("data_dir_03_free_rotor_drive", "data/03_free_rotor_drive")))
    return data_dir / "free_rotor_drive_diagonalization.h5", data_dir / "observable_projections.h5"


def get_default_ylm_path() -> Path:
    mod = importlib.reload(importlib.import_module("01_Parameters"))
    p_default = dict(mod.get_defaults_for_case("Default"))
    return Path(str(p_default.get("Ylm_h5_path", "data/01_spherical_harmonics/Ylm_blocks_JM.h5")))


def load_05a_basis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}. Run 05a_free_rotor_drive_compute.py first.")
    with open_h5(h5py, path, "r") as h5:
        if int(h5.attrs.get("antipodal_even_J_only", 0)) != 1:
            raise RuntimeError("05b expects 05a data built in the antipodally symmetric even-J subspace.")
        return h5["J"][...].astype(int), h5["M"][...].astype(int)


def phi_wrap(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def phi_antipodal_distance(phi: np.ndarray) -> np.ndarray:
    """Shortest azimuthal distance to 0 on the antipodally identified rotor."""
    phi_abs = np.abs(phi_wrap(phi))
    return np.minimum(phi_abs, np.pi - phi_abs)


def load_yjm_grid(path: Path, js: np.ndarray, ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = [path]
    fallback = get_default_ylm_path()
    if fallback not in candidates:
        candidates.append(fallback)

    last_error: Exception | None = None
    for cand in candidates:
        if not cand.exists():
            last_error = FileNotFoundError(f"Missing Ylm file: {cand}")
            continue
        try:
            with open_h5(h5py, cand, "r") as h5:
                if "YJM_grid" not in h5:
                    raise KeyError("Missing group 'YJM_grid'")
                grp = h5["YJM_grid"]
                if not all(key in h5 for key in ("J", "M")):
                    raise KeyError("Missing J/M basis datasets in Ylm file")
                if not all(key in grp for key in ("theta", "phi", "Y")):
                    raise KeyError("YJM_grid does not contain full-sphere theta/phi/Y datasets")
                js_full = h5["J"][...].astype(int)
                ms_full = h5["M"][...].astype(int)
                mask = (js_full <= int(np.max(js))) & ((js_full % 2) == 0)
                js_grid = js_full[mask]
                ms_grid = ms_full[mask]
                if not np.array_equal(js_grid, js) or not np.array_equal(ms_grid, ms):
                    raise RuntimeError("JM basis mismatch between 05a and 01b YJM_grid.")
                theta = grp["theta"][...].astype(float)
                phi = grp["phi"][...].astype(float)
                y_full = grp["Y"][...].astype(np.complex128)
                y_basis = y_full[mask]
            if cand != path:
                print(f"[05b] using fallback Ylm file: {cand}", flush=True)
            return theta, phi, y_basis
        except Exception as exc:
            last_error = exc
            continue

    assert last_error is not None
    raise OSError(
        f"Could not load a valid YJM_grid for 05b. Tried: {', '.join(str(c) for c in candidates)}. Last error: {last_error}"
    )


def project_function_to_jm(y_basis: np.ndarray, weights: np.ndarray, fvals: np.ndarray) -> np.ndarray:
    return np.einsum("aij,ij,ij,bij->ab", np.conjugate(y_basis), fvals, weights, y_basis, optimize=True)


def _init_05b_worker(y_basis: np.ndarray, weights: np.ndarray, vals: dict[str, np.ndarray]) -> None:
    global _Y_BASIS_05B, _WEIGHTS_05B, _VALS_05B
    _Y_BASIS_05B = y_basis
    _WEIGHTS_05B = weights
    _VALS_05B = vals


def _project_one_observable_05b(key: str) -> tuple[str, np.ndarray]:
    assert _Y_BASIS_05B is not None
    assert _WEIGHTS_05B is not None
    return key, project_function_to_jm(_Y_BASIS_05B, _WEIGHTS_05B, _VALS_05B[key]).astype(np.complex128)


def observable_matrices(js: np.ndarray, ms: np.ndarray, ylm_path: Path) -> dict[str, np.ndarray]:
    theta, phi, y_basis = load_yjm_grid(ylm_path, js, ms)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    dtheta = float(theta[1] - theta[0]) if theta.size > 1 else np.pi
    dphi = float(phi[1] - phi[0]) if phi.size > 1 else 2.0 * np.pi
    weights = np.sin(th) * dtheta * dphi
    x = th - 0.5 * np.pi
    y = phi_antipodal_distance(ph)
    sin2 = np.sin(th) ** 2
    cos2_th = np.cos(th) ** 2
    cos2phi_rot = np.cos(ph) ** 2
    cos2theta2d = (sin2 * cos2phi_rot) / np.maximum(sin2 * cos2phi_rot + cos2_th, 1e-15)

    vals = {
        "one": np.ones_like(th, dtype=float),
        "x2": x**2,
        "y2": y**2,
        "sin2theta": sin2,
        "cos2phi_rot": cos2phi_rot,
        "cos2theta2D": cos2theta2d,
    }
    nproc = max(1, int(get_defaults().get("nproc", 1)))
    if nproc <= 1 or len(OBS_KEYS) <= 1:
        _init_05b_worker(y_basis, weights, vals)
        pairs = [_project_one_observable_05b(key) for key in OBS_KEYS]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(nproc, len(OBS_KEYS)), initializer=_init_05b_worker, initargs=(y_basis, weights, vals)) as pool:
            pairs = pool.map(_project_one_observable_05b, OBS_KEYS, chunksize=1)
    return {key: mat for key, mat in pairs}


def main() -> None:
    p = get_defaults()
    data_05a, data_05b = get_paths(p)
    js, ms = load_05a_basis(data_05a)
    ylm_path = Path(str(p.get("Ylm_h5_path", "data/01_spherical_harmonics/Ylm_blocks_JM.h5")))
    theta, phi, _ = load_yjm_grid(ylm_path, js, ms)
    obs = observable_matrices(js, ms, ylm_path)
    l_matrix = np.diag(ms.astype(float)).astype(np.complex128)

    data_05b.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(h5py, data_05b, "w") as h5:
        h5.create_dataset("J", data=js.astype(np.int32))
        h5.create_dataset("M", data=ms.astype(np.int32))
        h5.create_dataset("L_re", data=np.real(l_matrix))
        h5.create_dataset("L_im", data=np.imag(l_matrix))
        grp = h5.create_group("observables_jm")
        for key in OBS_KEYS:
            grp.create_dataset(key, data=obs[key])
        h5.attrs["case_name"] = str(p.get("case_name", "Default"))
        h5.attrs["delta_phi_reference"] = 0.0
        h5.attrs["N_theta"] = int(theta.size)
        h5.attrs["N_phi"] = int(phi.size)
        h5.attrs["J_max"] = int(p.get("J_max", 20))
        h5.attrs["antipodal_even_J_only"] = 1
        h5.attrs["ylm_h5_path"] = str(ylm_path)
    print(f"Wrote: {data_05b}", flush=True)


if __name__ == "__main__":
    main()
