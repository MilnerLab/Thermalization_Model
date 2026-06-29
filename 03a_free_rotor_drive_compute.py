#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03a_free_rotor_drive_compute.py

Build the free-rotor Hamiltonian in the |J,M> basis, add the centrifuge drive
coupling Pi = -V0 cos^2(phi) sin^2(theta), diagonalize H0 + Pi along the drive,
and store the resulting eigensystem to HDF5.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import multiprocessing as mp

import h5py
import numpy as np

from h5_locking import open_h5

params = importlib.reload(importlib.import_module("01_Parameters"))
_Y_BLOCKS_03A: dict[tuple[int, int], np.ndarray] = {}
_COEFFS_03A: dict[tuple[int, int], complex] = {}
_BARE_BASE_03A: np.ndarray | None = None
_MS_03A: np.ndarray | None = None


def _init_03a_worker(
    y_blocks: dict[tuple[int, int], np.ndarray],
    coeffs: dict[tuple[int, int], complex],
    bare_base: np.ndarray,
    ms: np.ndarray,
) -> None:
    global _Y_BLOCKS_03A, _COEFFS_03A, _BARE_BASE_03A, _MS_03A
    _Y_BLOCKS_03A = y_blocks
    _COEFFS_03A = coeffs
    _BARE_BASE_03A = bare_base
    _MS_03A = ms


def get_defaults() -> dict:
    mod = importlib.reload(importlib.import_module("01_Parameters"))
    return dict(mod.get_defaults_for_case())


def get_data_h5(p: dict) -> Path:
    return Path(str(p.get("data_dir_03_free_rotor_drive", "data/03_free_rotor_drive"))) / "free_rotor_drive_diagonalization.h5"


def ensure_dirs(p: dict) -> None:
    get_data_h5(p).parent.mkdir(parents=True, exist_ok=True)


def get_default_ylm_path() -> Path:
    mod = importlib.reload(importlib.import_module("01_Parameters"))
    p_default = dict(mod.get_defaults_for_case("Default"))
    return Path(str(p_default.get("Ylm_h5_path", "data/01_spherical_harmonics/Ylm_blocks_JM.h5")))


def load_ylm_blocks(path: Path, j_max: int) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray]]:
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
                js = h5["J"][...].astype(int)
                ms = h5["M"][...].astype(int)
                mask = (js <= int(j_max)) & ((js % 2) == 0)
                js = js[mask]
                ms = ms[mask]
                if js.size == 0:
                    raise ValueError("Empty antipodally symmetric |J,M> basis after J_max truncation.")
                full_idx = np.flatnonzero(mask)
                grp = h5["Ylm"]
                blocks: dict[tuple[int, int], np.ndarray] = {}
                for lam_name in grp:
                    lam = int(lam_name.split("_")[1])
                    for mu_name in grp[lam_name]:
                        mu = int(mu_name.split("_")[1])
                        y_full = grp[lam_name][mu_name][...].astype(np.complex128)
                        blocks[(lam, mu)] = y_full[np.ix_(full_idx, full_idx)]
            if cand != path:
                print(f"[05a] using fallback Ylm file: {cand}", flush=True)
            return js, ms, blocks
        except Exception as exc:
            last_error = exc
            continue

    assert last_error is not None
    raise OSError(
        f"Could not load a valid Ylm file for 05a. Tried: {', '.join(str(c) for c in candidates)}. Last error: {last_error}"
    )


def compute_drive_coefficients(lam_max: int) -> dict[tuple[int, int], complex]:
    """Exact Y_lm decomposition of cos^2(phi) sin^2(theta).

    We use
      sin^2(theta) cos^2(phi)
        = 1/3
        - (1/3) sqrt(4 pi / 5) Y_20
        + sqrt(2 pi / 15) (Y_22 + Y_2,-2)

    so that
      Pi = -V0 * sum_{lm} a_lm Y_lm .
    """
    coeffs: dict[tuple[int, int], complex] = {}
    if int(lam_max) < 2:
        raise ValueError("Need Y_lm blocks up to lambda=2 to represent the drive exactly.")
    coeffs[(0, 0)] = complex(np.sqrt(4.0 * np.pi) / 3.0)
    coeffs[(2, 0)] = complex(-(1.0 / 3.0) * np.sqrt(4.0 * np.pi / 5.0))
    coeffs[(2, 2)] = complex(np.sqrt(2.0 * np.pi / 15.0))
    coeffs[(2, -2)] = complex(np.sqrt(2.0 * np.pi / 15.0))
    return coeffs


def build_drive_matrix(v0: float, coeffs: dict[tuple[int, int], complex], y_blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    keys = sorted(set(coeffs.keys()) & set(y_blocks.keys()))
    if not keys:
        raise RuntimeError("No common Ylm blocks available to build the drive matrix.")
    n = y_blocks[keys[0]].shape[0]
    out = np.zeros((n, n), dtype=np.complex128)
    for key in keys:
        out += coeffs[key] * y_blocks[key]
    return -float(v0) * out


def _compute_one_time_05a(it: int, v0: float, omega0: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert _BARE_BASE_03A is not None
    assert _MS_03A is not None
    bare_diag = _BARE_BASE_03A - float(omega0) * _MS_03A
    h0 = np.diag(bare_diag.astype(np.complex128))
    h_drive = build_drive_matrix(float(v0), _COEFFS_03A, _Y_BLOCKS_03A)
    h_total = h0 + h_drive
    evals, evecs = np.linalg.eigh(h_total)
    return it, h0, h_drive, np.real(evals), evecs


def _compute_one_task_05a(task: tuple[int, float, float]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_one_time_05a(*task)


def create_output_datasets_05a(
    h5: h5py.File,
    t_grid: np.ndarray,
    omega0_t: np.ndarray,
    v0_t: np.ndarray,
    js: np.ndarray,
    ms: np.ndarray,
    coeffs: dict[tuple[int, int], complex],
    p: dict,
    ylm_path: Path,
    j_max: int,
    b_rot: float,
    n_basis: int,
) -> dict[str, h5py.Dataset]:
    h5.create_dataset("t_grid", data=t_grid)
    h5.create_dataset("Omega0_t", data=omega0_t)
    h5.create_dataset("V0_t", data=v0_t)
    h5.create_dataset("J", data=js.astype(np.int32))
    h5.create_dataset("M", data=ms.astype(np.int32))
    ds = {
        "H0_re": h5.create_dataset("H0_re", shape=(t_grid.size, n_basis, n_basis), dtype=np.float64),
        "H0_im": h5.create_dataset("H0_im", shape=(t_grid.size, n_basis, n_basis), dtype=np.float64),
        "H_drive_re": h5.create_dataset("H_drive_re", shape=(t_grid.size, n_basis, n_basis), dtype=np.float64),
        "H_drive_im": h5.create_dataset("H_drive_im", shape=(t_grid.size, n_basis, n_basis), dtype=np.float64),
        "H_total_re": h5.create_dataset("H_total_re", shape=(t_grid.size, n_basis, n_basis), dtype=np.float64),
        "H_total_im": h5.create_dataset("H_total_im", shape=(t_grid.size, n_basis, n_basis), dtype=np.float64),
        "E_eval": h5.create_dataset("E_eval", shape=(t_grid.size, n_basis), dtype=np.float64),
        "U_evec_re": h5.create_dataset("U_evec_re", shape=(t_grid.size, n_basis, n_basis), dtype=np.float64),
        "U_evec_im": h5.create_dataset("U_evec_im", shape=(t_grid.size, n_basis, n_basis), dtype=np.float64),
    }
    coeff_grp = h5.create_group("drive_coefficients")
    for (lam, mu), val in coeffs.items():
        coeff_ds = coeff_grp.create_dataset(f"lam_{lam}_mu_{mu}", data=np.array([np.real(val), np.imag(val)], dtype=float))
        coeff_ds.attrs["lam"] = int(lam)
        coeff_ds.attrs["mu"] = int(mu)
    h5.attrs["case_name"] = str(p.get("case_name", "Default"))
    h5.attrs["J_max"] = int(j_max)
    h5.attrs["antipodal_even_J_only"] = 1
    h5.attrs["Nt_main"] = int(t_grid.size)
    h5.attrs["B"] = float(b_rot)
    h5.attrs["rotational_model"] = int(p.get("rotational_model", 1))
    if p.get("B_star", None) is not None:
        h5.attrs["B_star"] = float(p["B_star"])
    if p.get("D_star", None) is not None:
        h5.attrs["D_star"] = float(p["D_star"])
    h5.attrs["operator"] = "Pi = -V0*cos(phi)^2*sin(theta)^2"
    if int(p.get("rotational_model", 1)) == 1:
        h5.attrs["bare_frame_hamiltonian"] = "B*J*(J+1) - Omega0*M"
    else:
        h5.attrs["bare_frame_hamiltonian"] = "E_J(model_2) - Omega0*M"
    h5.attrs["ylm_h5_path"] = str(ylm_path)
    return ds


def main() -> None:
    p = get_defaults()
    ensure_dirs(p)
    grids = params.drive_grids_with_Nt(p, int(p.get("Nt_main", p.get("Nt_pendulon", p.get("rotor_Nt", 3)))))
    t_grid = np.asarray(grids["t"], dtype=float)
    omega0_t = np.asarray(grids["Omega0"], dtype=float)
    v0_t = np.asarray(grids["V0"], dtype=float)

    j_max = int(p.get("J_max", 20))
    ylm_path = Path(str(p.get("Ylm_h5_path", "data/01_spherical_harmonics/Ylm_blocks_JM.h5")))
    js, ms, y_blocks = load_ylm_blocks(ylm_path, j_max)
    n_basis = js.size

    lam_max_drive = max((lam for lam, _ in y_blocks.keys()), default=0)
    coeffs = compute_drive_coefficients(lam_max=lam_max_drive)

    b_rot = float(p["B"])
    bare_base = np.asarray(params.rotational_energy_levels(js, p), dtype=float)
    tasks = [(it, float(v0_t[it]), float(omega0_t[it])) for it in range(t_grid.size)]
    nproc = max(1, int(p.get("nproc", 1)))
    data_h5 = get_data_h5(p)
    data_h5.parent.mkdir(parents=True, exist_ok=True)
    bytes_needed = params.estimate_array_storage_bytes(
        ((t_grid.size, n_basis, n_basis), np.complex128),
        ((t_grid.size, n_basis, n_basis), np.complex128),
        ((t_grid.size, n_basis, n_basis), np.complex128),
        ((t_grid.size, n_basis), np.float64),
        ((t_grid.size, n_basis, n_basis), np.complex128),
    )
    use_streaming_output = params.exceeds_ram_threshold(p, bytes_needed)

    if use_streaming_output:
        with open_h5(h5py, data_h5, "w") as h5:
            ds = create_output_datasets_05a(
                h5,
                t_grid,
                omega0_t,
                v0_t,
                js,
                ms,
                coeffs,
                p,
                ylm_path,
                j_max,
                b_rot,
                n_basis,
            )
            _init_03a_worker(y_blocks, coeffs, bare_base, ms.astype(float))
            if nproc <= 1 or t_grid.size <= 1:
                result_iter = (_compute_one_time_05a(*task) for task in tasks)
            else:
                ctx = mp.get_context("fork")
                pool = ctx.Pool(
                    processes=nproc,
                    initializer=_init_03a_worker,
                    initargs=(y_blocks, coeffs, bare_base, ms.astype(float)),
                )
                result_iter = pool.imap(_compute_one_task_05a, tasks, chunksize=max(1, int(p.get("chunksize", 1))))
            try:
                for it, h0_i, h_drive_i, evals_i, evecs_i in result_iter:
                    h_total_i = h0_i + h_drive_i
                    ds["H0_re"][it] = np.real(h0_i)
                    ds["H0_im"][it] = np.imag(h0_i)
                    ds["H_drive_re"][it] = np.real(h_drive_i)
                    ds["H_drive_im"][it] = np.imag(h_drive_i)
                    ds["H_total_re"][it] = np.real(h_total_i)
                    ds["H_total_im"][it] = np.imag(h_total_i)
                    ds["E_eval"][it] = evals_i
                    ds["U_evec_re"][it] = np.real(evecs_i)
                    ds["U_evec_im"][it] = np.imag(evecs_i)
            finally:
                if nproc > 1 and t_grid.size > 1:
                    pool.close()
                    pool.join()
    else:
        h0 = np.zeros((t_grid.size, n_basis, n_basis), dtype=np.complex128)
        h_drive = np.zeros((t_grid.size, n_basis, n_basis), dtype=np.complex128)
        h_total = np.zeros_like(h_drive)
        evals = np.zeros((t_grid.size, n_basis), dtype=float)
        evecs = np.zeros((t_grid.size, n_basis, n_basis), dtype=np.complex128)
        if nproc <= 1 or t_grid.size <= 1:
            _init_03a_worker(y_blocks, coeffs, bare_base, ms.astype(float))
            results = [_compute_one_time_05a(*task) for task in tasks]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=nproc,
                initializer=_init_03a_worker,
                initargs=(y_blocks, coeffs, bare_base, ms.astype(float)),
            ) as pool:
                results = pool.starmap(_compute_one_time_05a, tasks, chunksize=max(1, int(p.get("chunksize", 1))))

        results.sort(key=lambda x: x[0])
        for it, h0_i, h_drive_i, evals_i, evecs_i in results:
            h0[it] = h0_i
            h_drive[it] = h_drive_i
            h_total[it] = h0_i + h_drive_i
            evals[it] = evals_i
            evecs[it] = evecs_i

        with open_h5(h5py, data_h5, "w") as h5:
            create_output_datasets_05a(
                h5,
                t_grid,
                omega0_t,
                v0_t,
                js,
                ms,
                coeffs,
                p,
                ylm_path,
                j_max,
                b_rot,
                n_basis,
            )
            h5["H0_re"][...] = np.real(h0)
            h5["H0_im"][...] = np.imag(h0)
            h5["H_drive_re"][...] = np.real(h_drive)
            h5["H_drive_im"][...] = np.imag(h_drive)
            h5["H_total_re"][...] = np.real(h_total)
            h5["H_total_im"][...] = np.imag(h_total)
            h5["E_eval"][...] = evals
            h5["U_evec_re"][...] = np.real(evecs)
            h5["U_evec_im"][...] = np.imag(evecs)
    print(f"Wrote: {data_h5}", flush=True)


if __name__ == "__main__":
    main()
