#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01c_precompute_steady_state.py

Central helpers and cache for the exact target Hamiltonian in the fixed JM basis

    H_target(t) = B J(J+1) + V_drive(t)

and its thermal target density matrix

    rho_target(t) = exp[-beta H_target(t)] / Z.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import multiprocessing as mp

import h5py
import numpy as np

from h5_locking import open_h5

params = importlib.reload(importlib.import_module("01_Parameters"))

_JS_01C: np.ndarray | None = None
_Y_BLOCKS_01C: dict[tuple[int, int], np.ndarray] = {}
_P_01C: dict[str, object] = {}
_BETA_01C: float = 0.0


def get_defaults() -> dict:
    mod = importlib.reload(importlib.import_module("01_Parameters"))
    return dict(mod.get_defaults_for_case())


def get_paths(p: dict) -> tuple[Path, Path, Path]:
    ylm_h5 = Path(str(p.get("Ylm_h5_path", "data/01_spherical_harmonics/Ylm_blocks_JM.h5")))
    out_h5 = Path(str(p.get("steady_state_target_h5_path", "data/01_spherical_harmonics/steady_state_target.h5")))
    return ylm_h5, out_h5


def k_matrix_from_eigensystem(evals: np.ndarray, evecs: np.ndarray, beta: float) -> np.ndarray:
    e = np.asarray(evals, dtype=float)
    u = np.asarray(evecs, dtype=np.complex128)
    if beta <= 0.0:
        d = np.zeros(e.size, dtype=np.complex128)
        d[int(np.argmin(e))] = 1.0
    else:
        d = np.exp(-beta * (e - np.min(e))).astype(np.complex128)
    return (u * d[None, :]) @ u.conj().T


def density_matrix_from_hamiltonian(ham: np.ndarray, beta: float) -> np.ndarray:
    h = np.asarray(ham, dtype=np.complex128)
    evals, evecs = np.linalg.eigh(h)
    return params.normalized_density_matrix(k_matrix_from_eigensystem(evals, evecs, beta))


def hermitian_normalized_density_matrix(mat: np.ndarray) -> np.ndarray:
    rho = np.asarray(mat, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    return params.normalized_density_matrix(rho)


def steady_state_density_from_target(
    rho_target: np.ndarray,
    evals_rot: np.ndarray,
    evecs_rot: np.ndarray,
    tau: float | None,
    degeneracy_tol: float,
) -> np.ndarray:
    u = np.asarray(evecs_rot, dtype=np.complex128)
    e = np.asarray(evals_rot, dtype=float)
    rho_t_eig = u.conj().T @ np.asarray(rho_target, dtype=np.complex128) @ u
    de = e[:, None] - e[None, :]
    if tau is None:
        mask = np.abs(de) <= float(max(degeneracy_tol, 0.0))
        rho_ss_eig = np.where(mask, rho_t_eig, 0.0)
    else:
        rho_ss_eig = rho_t_eig / (1.0 + 1j * float(tau) * de)
    rho_ss_eig = hermitian_normalized_density_matrix(rho_ss_eig)
    rho_ss = u @ rho_ss_eig @ u.conj().T
    return hermitian_normalized_density_matrix(rho_ss)


def load_ylm_blocks_for_basis(path: Path, js_target: np.ndarray, ms_target: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    with open_h5(h5py, path, "r") as h5:
        js_full = h5["J"][...].astype(int)
        ms_full = h5["M"][...].astype(int)
        idx_map = {(int(j), int(m)): i for i, (j, m) in enumerate(zip(js_full, ms_full))}
        cols = np.array([idx_map[(int(j), int(m))] for j, m in zip(js_target, ms_target)], dtype=int)
        grp = h5["Ylm"]
        out: dict[tuple[int, int], np.ndarray] = {}
        for lam_name in grp:
            lam = int(lam_name.split("_")[1])
            for mu_name in grp[lam_name]:
                mu = int(mu_name.split("_")[1])
                y_full = grp[lam_name][mu_name][...].astype(np.complex128)
                out[(lam, mu)] = y_full[np.ix_(cols, cols)]
    return out


def load_even_j_basis_from_ylm(path: Path, j_max: int) -> tuple[np.ndarray, np.ndarray]:
    with open_h5(h5py, path, "r") as h5:
        js_full = h5["J"][...].astype(int)
        ms_full = h5["M"][...].astype(int)
        mask = (js_full <= int(j_max)) & ((js_full % 2) == 0)
        js = js_full[mask]
        ms = ms_full[mask]
    if js.size == 0:
        raise RuntimeError("Empty even-J JM basis in 01c after J_max truncation.")
    return js, ms


def drive_coefficients() -> dict[tuple[int, int], complex]:
    return {
        (0, 0): complex(np.sqrt(4.0 * np.pi) / 3.0),
        (2, 0): complex(-(1.0 / 3.0) * np.sqrt(4.0 * np.pi / 5.0)),
        (2, 2): complex(np.sqrt(2.0 * np.pi / 15.0)),
        (2, -2): complex(np.sqrt(2.0 * np.pi / 15.0)),
    }


def build_drive_matrix(v0: float, y_blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    coeffs = drive_coefficients()
    keys = sorted(set(coeffs.keys()) & set(y_blocks.keys()))
    if not keys:
        raise RuntimeError("No common Ylm blocks available to build the target drive matrix.")
    out = np.zeros_like(np.asarray(y_blocks[keys[0]], dtype=np.complex128))
    for key in keys:
        out += coeffs[key] * np.asarray(y_blocks[key], dtype=np.complex128)
    return -float(v0) * out


def target_hamiltonian_jm(js: np.ndarray, y_blocks: dict[tuple[int, int], np.ndarray], v0: float, p: dict) -> np.ndarray:
    return params.rotational_energy_diagonal(js, p) + build_drive_matrix(v0, y_blocks)


def target_density_jm(js: np.ndarray, y_blocks: dict[tuple[int, int], np.ndarray], v0: float, beta: float, p: dict) -> np.ndarray:
    return density_matrix_from_hamiltonian(target_hamiltonian_jm(js, y_blocks, v0, p), beta)


def project_density_jm_to_basis(rho_jm: np.ndarray, basis_in_jm: np.ndarray) -> np.ndarray:
    u = np.asarray(basis_in_jm, dtype=np.complex128)
    rho = u.conj().T @ np.asarray(rho_jm, dtype=np.complex128) @ u
    return hermitian_normalized_density_matrix(rho)


def _init_01c_worker(js: np.ndarray, y_blocks: dict[tuple[int, int], np.ndarray], p: dict, beta: float) -> None:
    global _JS_01C, _Y_BLOCKS_01C, _P_01C, _BETA_01C
    _JS_01C = np.asarray(js, dtype=int)
    _Y_BLOCKS_01C = y_blocks
    _P_01C = dict(p)
    _BETA_01C = float(beta)


def _compute_one_time_01c(task: tuple[int, float]) -> tuple[int, np.ndarray, np.ndarray]:
    it, v0 = task
    assert _JS_01C is not None
    h_target_jm = target_hamiltonian_jm(_JS_01C, _Y_BLOCKS_01C, float(v0), _P_01C)
    rho_target_jm = density_matrix_from_hamiltonian(h_target_jm, _BETA_01C)
    return int(it), h_target_jm, rho_target_jm


def main() -> None:
    p = get_defaults()
    ylm_h5, out_h5 = get_paths(p)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    grids = params.drive_grids(p)
    t_grid = np.asarray(grids["t"], dtype=float)
    omega0_t = np.asarray(grids["Omega0"], dtype=float)
    v0_t = np.asarray(grids["V0"], dtype=float)
    js, ms = load_even_j_basis_from_ylm(ylm_h5, int(p.get("J_max", 20)))

    y_blocks = load_ylm_blocks_for_basis(ylm_h5, js, ms)
    beta = 0.0 if float(p.get("T_K", 0.0)) <= 0.0 else 1.0 / (float(p["kB_per_K"]) * float(p["T_K"]))
    nt = t_grid.size
    n_jm = js.size
    h_target_jm = np.zeros((nt, n_jm, n_jm), dtype=np.complex128)
    rho_target_jm = np.zeros((nt, n_jm, n_jm), dtype=np.complex128)

    tasks = [(int(it), float(v0_t[it])) for it in range(nt)]
    nproc = max(1, int(p.get("nproc", 1)))
    if nproc <= 1 or nt <= 1:
        _init_01c_worker(js, y_blocks, p, beta)
        results = [_compute_one_time_01c(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=nproc,
            initializer=_init_01c_worker,
            initargs=(js, y_blocks, p, beta),
        ) as pool:
            results = pool.map(_compute_one_time_01c, tasks, chunksize=max(1, int(p.get("chunksize", 1))))

    results.sort(key=lambda item: item[0])
    for it, h_jm_i, rho_jm_i in results:
        h_target_jm[it] = h_jm_i
        rho_target_jm[it] = rho_jm_i

    with open_h5(h5py, out_h5, "w") as h5:
        h5.create_dataset("t_grid", data=t_grid)
        h5.create_dataset("Omega0_t", data=omega0_t)
        h5.create_dataset("V0_t", data=v0_t)
        h5.create_dataset("J", data=js.astype(np.int32))
        h5.create_dataset("M", data=ms.astype(np.int32))
        h5.create_dataset("H_target_jm_re", data=np.real(h_target_jm))
        h5.create_dataset("H_target_jm_im", data=np.imag(h_target_jm))
        h5.create_dataset("rho_target_jm_re", data=np.real(rho_target_jm))
        h5.create_dataset("rho_target_jm_im", data=np.imag(rho_target_jm))
        h5.attrs["case_name"] = str(p.get("case_name", "Default"))
        h5.attrs["B"] = float(p["B"])
        h5.attrs["rotational_model"] = int(p.get("rotational_model", 1))
        if p.get("B_star", None) is not None:
            h5.attrs["B_star"] = float(p["B_star"])
        if p.get("D_star", None) is not None:
            h5.attrs["D_star"] = float(p["D_star"])
        h5.attrs["T_K"] = float(p.get("T_K", 0.0))
        h5.attrs["kB_per_K"] = float(p.get("kB_per_K", 1.0))
        h5.attrs["Nt_main"] = int(nt)
        h5.attrs["Ylm_h5_path"] = str(ylm_h5)
        h5.attrs["antipodal_even_J_only"] = int(np.all((js % 2) == 0))
    print(f"Wrote: {out_h5}", flush=True)


if __name__ == "__main__":
    main()
