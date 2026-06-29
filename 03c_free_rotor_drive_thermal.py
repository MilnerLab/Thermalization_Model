#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""05c_free_rotor_drive_thermal.py

Read the 05a eigensystem and 05b observable projections, then compute thermal
observables, Trotter diagnostics, J-cutoff diagnostics, and HO-basis comparison.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import multiprocessing as mp

import h5py
import numpy as np

from h5_locking import open_h5

params = importlib.reload(importlib.import_module("01_Parameters"))
stage03a = importlib.import_module("03a_free_rotor_drive_compute")
_OBS_KEYS_05C: list[str] = []
_BASE_OBS_05C: dict[str, np.ndarray] = {}
_MS_05C: np.ndarray | None = None
_EVALS_05C: np.ndarray | None = None
_EVECS_05C: np.ndarray | None = None
_OMEGA0_05C: np.ndarray | None = None
_DELTA_PHI_05C: np.ndarray | None = None
_V0_05C: np.ndarray | None = None
_JS_05C: np.ndarray | None = None
_RHO_TARGET_JM_05C: np.ndarray | None = None
_RHO_SS_05C: np.ndarray | None = None
_BETA_05C: float = 0.0
_N_STEPS_05C: int = 1
_TAU_05C: float | None = None
_DEGENERACY_TOL_05C: float = 1e-10
_COMPUTE_LAB_05C: bool = True
_COMPUTE_ROT_05C: bool = True
_B_ROT_05C: float = 1.0
_Y_TARGET_05C: dict[tuple[int, int], np.ndarray] = {}

OBS_KEYS = ["one", "x2", "y2", "sin2theta", "cos2phi_rot", "cos2phi_lab", "cos2theta2D"]


def get_defaults() -> dict:
    mod = importlib.reload(importlib.import_module("01_Parameters"))
    return dict(mod.get_defaults_for_case())


def get_paths(p: dict) -> tuple[Path, Path, Path, Path]:
    data_dir = Path(str(p.get("data_dir_03_free_rotor_drive", "data/03_free_rotor_drive")))
    return (
        data_dir / "free_rotor_drive_diagonalization.h5",
        data_dir / "observable_projections.h5",
        Path(str(p.get("steady_state_target_h5_path", "data/01_spherical_harmonics/steady_state_target.h5"))),
        data_dir / "observable_matrices.h5",
    )


def rotation_operator(ms: np.ndarray, delta_phi: float) -> np.ndarray:
    phase = np.exp((-1j * float(delta_phi)) * np.asarray(ms, dtype=float)).astype(np.complex128)
    return np.diag(phase)


def rotation_matrix_jm(ms: np.ndarray, beta: float, omega0: float) -> np.ndarray:
    phase = np.exp((-beta * float(omega0)) * np.asarray(ms, dtype=float)).astype(np.complex128)
    return np.diag(phase)


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


def target_hamiltonian_jm(js: np.ndarray, y_blocks: dict[tuple[int, int], np.ndarray], v0: float, p: dict) -> np.ndarray:
    h_bare = params.rotational_energy_diagonal(js, p)
    coeffs = stage03a.compute_drive_coefficients(lam_max=max((lam for lam, _ in y_blocks.keys()), default=2))
    h_drive = stage03a.build_drive_matrix(float(v0), coeffs, y_blocks)
    return h_bare + h_drive


def target_density_jm(js: np.ndarray, y_blocks: dict[tuple[int, int], np.ndarray], v0: float, beta: float, p: dict) -> np.ndarray:
    return density_matrix_from_hamiltonian(target_hamiltonian_jm(js, y_blocks, v0, p), beta)


def project_density_jm_to_basis(rho_jm: np.ndarray, basis_in_jm: np.ndarray) -> np.ndarray:
    u = np.asarray(basis_in_jm, dtype=np.complex128)
    rho = u.conj().T @ np.asarray(rho_jm, dtype=np.complex128) @ u
    return hermitian_normalized_density_matrix(rho)


def hermitian_normalized_density_matrix(mat: np.ndarray) -> np.ndarray:
    rho = np.asarray(mat, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    return params.normalized_density_matrix(rho)


def steady_state_density_from_target(
    rho_target_jm: np.ndarray,
    evals_rot: np.ndarray,
    evecs_rot: np.ndarray,
    tau_steady_state: float | None,
    degeneracy_tol: float,
) -> np.ndarray:
    u = np.asarray(evecs_rot, dtype=np.complex128)
    e = np.asarray(evals_rot, dtype=float)
    rho_t_eig = u.conj().T @ np.asarray(rho_target_jm, dtype=np.complex128) @ u
    de = e[:, None] - e[None, :]
    if tau_steady_state is None:
        mask = np.abs(de) <= float(max(degeneracy_tol, 0.0))
        rho_ss_eig = np.where(mask, rho_t_eig, 0.0)
    else:
        rho_ss_eig = rho_t_eig / (1.0 + 1j * float(tau_steady_state) * de)
    rho_ss_eig = hermitian_normalized_density_matrix(rho_ss_eig)
    rho_ss_jm = u @ rho_ss_eig @ u.conj().T
    return hermitian_normalized_density_matrix(rho_ss_jm)


def mode_weights_from_density(rho_jm: np.ndarray, evecs: np.ndarray) -> np.ndarray:
    rho_eig = np.asarray(evecs, dtype=np.complex128).conj().T @ np.asarray(rho_jm, dtype=np.complex128) @ np.asarray(evecs, dtype=np.complex128)
    w = np.real(np.diag(rho_eig)).astype(float)
    w = np.clip(w, 0.0, None)
    z = float(np.sum(w))
    if z <= 0.0:
        out = np.zeros_like(w)
        if out.size:
            out[0] = 1.0
        return out
    return w / z


def shell_weights_from_modes(js: np.ndarray, evecs: np.ndarray, mode_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j_vals = np.unique(np.asarray(js, dtype=int))
    prob_basis = np.abs(np.asarray(evecs, dtype=np.complex128)) ** 2
    shell_mode = np.zeros((mode_weights.size, j_vals.size), dtype=float)
    for ij, j in enumerate(j_vals):
        mask = np.asarray(js, dtype=int) == int(j)
        shell_mode[:, ij] = np.sum(prob_basis[mask, :], axis=0)
    shell_importance = np.sum(np.asarray(mode_weights, dtype=float)[:, None] * shell_mode, axis=0)
    shell_importance = np.clip(shell_importance, 0.0, None)
    z = float(np.sum(shell_importance))
    if z > 0.0:
        shell_importance = shell_importance / z
    return shell_importance, np.cumsum(shell_importance)


def thermal_observable_from_density(rho: np.ndarray, op: np.ndarray) -> float:
    return float(np.real(np.trace(np.asarray(rho, dtype=np.complex128) @ np.asarray(op, dtype=np.complex128))))


def build_steady_state_density_cache(
    evals: np.ndarray,
    evecs: np.ndarray,
    rho_target_jm: np.ndarray,
    tau_arr: np.ndarray | None,
    degeneracy_tol: float,
) -> np.ndarray:
    nt, n_basis = rho_target_jm.shape[:2]
    rho_ss = np.zeros((nt, n_basis, n_basis), dtype=np.complex128)
    report_every = max(1, nt // 10)
    for it in range(nt):
        tau_it = None if tau_arr is None else float(tau_arr[it])
        rho_ss[it] = steady_state_density_from_target(
            rho_target_jm[it],
            evals[it],
            evecs[it],
            tau_it,
            degeneracy_tol,
        )
        if (it + 1) % report_every == 0 or it == nt - 1:
            print(f"  03c: steady-state cache {it + 1}/{nt}", flush=True)
    return rho_ss


def _init_05c_worker(
    obs_keys: list[str],
    base_obs: dict[str, np.ndarray],
    ms: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    omega0: np.ndarray,
    delta_phi: np.ndarray,
    js: np.ndarray,
    rho_target_jm: np.ndarray,
    rho_ss: np.ndarray,
    beta: float,
    n_steps: int,
    tau_steady_state: float | None,
    degeneracy_tol: float,
    compute_lab: bool,
    compute_rot: bool,
) -> None:
    global _OBS_KEYS_05C, _BASE_OBS_05C, _MS_05C, _EVALS_05C, _EVECS_05C, _OMEGA0_05C, _DELTA_PHI_05C, _JS_05C, _RHO_TARGET_JM_05C, _RHO_SS_05C, _BETA_05C, _N_STEPS_05C, _TAU_05C, _DEGENERACY_TOL_05C, _COMPUTE_LAB_05C, _COMPUTE_ROT_05C
    _OBS_KEYS_05C = obs_keys
    _BASE_OBS_05C = base_obs
    _MS_05C = ms
    _EVALS_05C = evals
    _EVECS_05C = evecs
    _OMEGA0_05C = omega0
    _DELTA_PHI_05C = delta_phi
    _JS_05C = js
    _RHO_TARGET_JM_05C = rho_target_jm
    _RHO_SS_05C = rho_ss
    _BETA_05C = beta
    _N_STEPS_05C = n_steps
    _TAU_05C = tau_steady_state
    _DEGENERACY_TOL_05C = degeneracy_tol
    _COMPUTE_LAB_05C = bool(compute_lab)
    _COMPUTE_ROT_05C = bool(compute_rot)


def _compute_one_time_05c(it: int) -> dict[str, object]:
    assert _MS_05C is not None
    assert _EVALS_05C is not None
    assert _EVECS_05C is not None
    assert _OMEGA0_05C is not None
    assert _DELTA_PHI_05C is not None
    assert _JS_05C is not None
    assert _RHO_TARGET_JM_05C is not None
    assert _RHO_SS_05C is not None
    uz = rotation_operator(_MS_05C, float(_DELTA_PHI_05C[it]))
    obs_mats_i = {
        "one": _BASE_OBS_05C["one"],
        "x2": _BASE_OBS_05C["x2"],
        "y2": _BASE_OBS_05C["y2"],
        "sin2theta": _BASE_OBS_05C["sin2theta"],
        "cos2phi_rot": _BASE_OBS_05C["cos2phi_rot"],
        "cos2phi_lab": uz @ _BASE_OBS_05C["cos2phi_rot"] @ uz.conj().T,
        "cos2theta2D": uz @ _BASE_OBS_05C["cos2theta2D"] @ uz.conj().T,
    }
    evecs_i = np.asarray(_EVECS_05C[it], dtype=np.complex128)
    evals_i = np.asarray(_EVALS_05C[it], dtype=float)
    obs_per_mode_i: dict[str, np.ndarray] = {}
    for key in _OBS_KEYS_05C:
        eig_obs = evecs_i.conj().T @ obs_mats_i[key] @ evecs_i
        obs_per_mode_i[key] = np.real(np.diag(eig_obs))

    rho_rot = params.normalized_density_matrix(k_matrix_from_eigensystem(evals_i, evecs_i, _BETA_05C)) if _COMPUTE_ROT_05C else None
    rho_lab_target = np.asarray(_RHO_TARGET_JM_05C[it], dtype=np.complex128)
    rho_ss = np.asarray(_RHO_SS_05C[it], dtype=np.complex128)

    mode_weights_rot_i = mode_weights_from_density(rho_rot, evecs_i) if _COMPUTE_ROT_05C and rho_rot is not None else np.zeros(evals_i.size, dtype=float)
    mode_weights_lab_i = mode_weights_from_density(rho_lab_target, evecs_i) if _COMPUTE_LAB_05C else np.zeros(evals_i.size, dtype=float)
    mode_weights_ss_i = mode_weights_from_density(rho_ss, evecs_i)
    j_importance_lab_i, j_cumulative_lab_i = shell_weights_from_modes(_JS_05C, evecs_i, mode_weights_lab_i)
    j_importance_rot_i, j_cumulative_rot_i = shell_weights_from_modes(_JS_05C, evecs_i, mode_weights_rot_i)
    j_importance_ss_i, j_cumulative_ss_i = shell_weights_from_modes(_JS_05C, evecs_i, mode_weights_ss_i)
    thermal_rot_i = {key: (thermal_observable_from_density(rho_rot, obs_mats_i[key]) if _COMPUTE_ROT_05C and rho_rot is not None else float("nan")) for key in _OBS_KEYS_05C}
    thermal_lab_i = {key: (thermal_observable_from_density(rho_lab_target, obs_mats_i[key]) if _COMPUTE_LAB_05C else float("nan")) for key in _OBS_KEYS_05C}
    thermal_ss_i = {key: thermal_observable_from_density(rho_ss, obs_mats_i[key]) for key in _OBS_KEYS_05C}

    trotter_change_i = np.full(max(_N_STEPS_05C - 1, 0), np.nan, dtype=float)
    if _COMPUTE_LAB_05C and _N_STEPS_05C >= 2:
        for n in range(1, _N_STEPS_05C):
            beta_step = _BETA_05C / float(n) if _BETA_05C > 0.0 else 0.0
            k_step = k_matrix_from_eigensystem(evals_i, evecs_i, beta_step)
            r_step = rotation_matrix_jm(_MS_05C, 0.5 * beta_step, float(_OMEGA0_05C[it]))
            rho_trot = params.trotter_density_matrix(k_step, r_step, n)
            trotter_change_i[n - 1] = params.density_relative_change(rho_trot, rho_lab_target)

    return {
        "idx": it,
        "obs_per_mode": obs_per_mode_i,
        "thermal_lab": thermal_lab_i,
        "thermal_rot": thermal_rot_i,
        "thermal_ss": thermal_ss_i,
        "mode_weights_lab": mode_weights_lab_i,
        "mode_weights_rot": mode_weights_rot_i,
        "mode_weights_ss": mode_weights_ss_i,
        "j_importance_lab": j_importance_lab_i,
        "j_importance_rot": j_importance_rot_i,
        "j_importance_ss": j_importance_ss_i,
        "j_cumulative_lab": j_cumulative_lab_i,
        "j_cumulative_rot": j_cumulative_rot_i,
        "j_cumulative_ss": j_cumulative_ss_i,
        "trotter_change": trotter_change_i,
    }


def observable_matrices_at_time_05c(base_obs: dict[str, np.ndarray], ms: np.ndarray, delta_phi: float) -> dict[str, np.ndarray]:
    uz = rotation_operator(np.asarray(ms, dtype=float), float(delta_phi))
    return {
        "one": base_obs["one"],
        "x2": base_obs["x2"],
        "y2": base_obs["y2"],
        "sin2theta": base_obs["sin2theta"],
        "cos2phi_rot": base_obs["cos2phi_rot"],
        "cos2phi_lab": uz @ base_obs["cos2phi_rot"] @ uz.conj().T,
        "cos2theta2D": uz @ base_obs["cos2theta2D"] @ uz.conj().T,
    }


def main() -> None:
    p = get_defaults()
    data_05a, proj_05b, target_h5, out_05c = get_paths(p)

    with open_h5(h5py, data_05a, "r") as h5:
        if int(h5.attrs.get("antipodal_even_J_only", 0)) != 1:
            raise RuntimeError("05c expects 05a data built in the antipodally symmetric even-J subspace.")
        t = h5["t_grid"][...].astype(float)
        omega0 = h5["Omega0_t"][...].astype(float)
        v0_05a = h5["V0_t"][...].astype(float)
        js_05a = h5["J"][...].astype(int)
        ms_05a = h5["M"][...].astype(int)
        evals = h5["E_eval"][...].astype(float)
        evecs = h5["U_evec_re"][...].astype(float) + 1j * h5["U_evec_im"][...].astype(float)

    if not target_h5.exists():
        raise FileNotFoundError(f"Missing steady-state target file: {target_h5}. Run 01c_precompute_steady_state.py first.")
    with open_h5(h5py, target_h5, "r") as h5:
        t_target = h5["t_grid"][...].astype(float)
        omega_target = h5["Omega0_t"][...].astype(float)
        v0_target = h5["V0_t"][...].astype(float)
        js_target = h5["J"][...].astype(int)
        ms_target = h5["M"][...].astype(int)
        rho_target_jm = h5["rho_target_jm_re"][...].astype(float) + 1j * h5["rho_target_jm_im"][...].astype(float)
    if (
        not np.array_equal(js_target, js_05a)
        or not np.array_equal(ms_target, ms_05a)
        or t_target.shape != t.shape
        or not np.allclose(t_target, t, rtol=0.0, atol=1e-12)
        or omega_target.shape != omega0.shape
        or not np.allclose(omega_target, omega0, rtol=0.0, atol=1e-12)
        or v0_target.shape != v0_05a.shape
        or not np.allclose(v0_target, v0_05a, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("01c steady-state target grid/basis does not match 05a.")

    with open_h5(h5py, proj_05b, "r") as h5:
        js = h5["J"][...].astype(int)
        ms = h5["M"][...].astype(int)
        base_obs = {key: h5[f"observables_jm/{key}"][...].astype(np.complex128) for key in ["one", "x2", "y2", "sin2theta", "cos2phi_rot", "cos2theta2D"]}
        l_matrix = h5["L_re"][...].astype(float) + 1j * h5["L_im"][...].astype(float)

    nt = t.size
    n_basis = js.size
    grids = params.drive_grids_with_Nt(p, int(p.get("Nt_main", p.get("Nt_pendulon", p.get("rotor_Nt", nt)))))
    delta_phi = np.asarray(grids["Delta_phi"], dtype=float)
    if delta_phi.size != nt:
        delta_phi = np.interp(t, np.asarray(grids["t"], dtype=float), delta_phi)

    obs_per_mode = {key: np.zeros((nt, n_basis), dtype=float) for key in OBS_KEYS}
    thermal_lab = {key: np.zeros(nt, dtype=float) for key in OBS_KEYS}
    thermal_rot = {key: np.zeros(nt, dtype=float) for key in OBS_KEYS}
    thermal_ss = {key: np.zeros(nt, dtype=float) for key in OBS_KEYS}
    mode_weights_lab = np.zeros((nt, n_basis), dtype=float)
    mode_weights_rot = np.zeros((nt, n_basis), dtype=float)
    mode_weights_ss = np.zeros((nt, n_basis), dtype=float)
    j_values = np.unique(js.astype(int))
    j_importance_lab = np.zeros((nt, j_values.size), dtype=float)
    j_importance_rot = np.zeros((nt, j_values.size), dtype=float)
    j_importance_ss = np.zeros((nt, j_values.size), dtype=float)
    j_cumulative_lab = np.zeros((nt, j_values.size), dtype=float)
    j_cumulative_rot = np.zeros((nt, j_values.size), dtype=float)
    j_cumulative_ss = np.zeros((nt, j_values.size), dtype=float)

    n_steps = max(1, int(p.get("thermal_trotter_steps", 3)))
    trotter_change = np.full((max(n_steps - 1, 0), nt), np.nan, dtype=float)
    trotter_n_values = np.arange(1, n_steps, dtype=np.int32)
    beta = 0.0 if float(p.get("T_K", 0.0)) <= 0.0 else 1.0 / (float(p["kB_per_K"]) * float(p["T_K"]))
    tau_arr = params.tau_steady_state_grid(p, t)
    tau_smooth = p.get("tau_smooth", None)
    tau_smooth = None if tau_smooth is None else float(tau_smooth)
    degeneracy_tol = float(p.get("degeneracy_tol", 1e-10))
    compute_lab = bool(p.get("compute_thermal_lab_frame", True))
    compute_rot = bool(p.get("compute_thermal_rot_frame", True))
    compare_ho = bool(p.get("Compare_to_HO_basis", False))
    smooth_bytes_needed = params.estimate_array_storage_bytes(
        ((nt, n_basis), np.float64),
        ((nt, n_basis, n_basis), np.complex128),
        ((nt, n_basis, n_basis), np.complex128),
        ((nt, n_basis, n_basis), np.complex128),
        ((nt, n_basis, n_basis), np.complex128),
    )
    compute_ss_smooth = not params.exceeds_ram_threshold(p, smooth_bytes_needed)
    thermal_ss_smooth = {key: np.zeros(nt, dtype=float) for key in OBS_KEYS} if compute_ss_smooth else {}
    rho_ss_cache = build_steady_state_density_cache(evals, evecs, rho_target_jm, tau_arr, degeneracy_tol)
    rho_ss_smooth_cache = np.zeros_like(rho_ss_cache) if compute_ss_smooth else None

    nproc = max(1, int(p.get("nproc", 1)))
    worker_payload_bytes = params.estimate_array_storage_bytes(
        (evals.shape, evals.dtype),
        (evecs.shape, evecs.dtype),
        (rho_target_jm.shape, rho_target_jm.dtype),
        (rho_ss_cache.shape, rho_ss_cache.dtype),
        *[(arr.shape, arr.dtype) for arr in base_obs.values()],
    )
    parent_payload_specs = [
        (evals.shape, evals.dtype),
        (evecs.shape, evecs.dtype),
        (rho_target_jm.shape, rho_target_jm.dtype),
        (rho_ss_cache.shape, rho_ss_cache.dtype),
        *[(arr.shape, arr.dtype) for arr in base_obs.values()],
        *[(arr.shape, arr.dtype) for arr in obs_per_mode.values()],
        *[(arr.shape, arr.dtype) for arr in thermal_lab.values()],
        *[(arr.shape, arr.dtype) for arr in thermal_rot.values()],
        *[(arr.shape, arr.dtype) for arr in thermal_ss.values()],
        (mode_weights_lab.shape, mode_weights_lab.dtype),
        (mode_weights_rot.shape, mode_weights_rot.dtype),
        (mode_weights_ss.shape, mode_weights_ss.dtype),
        (j_importance_lab.shape, j_importance_lab.dtype),
        (j_importance_rot.shape, j_importance_rot.dtype),
        (j_importance_ss.shape, j_importance_ss.dtype),
        (j_cumulative_lab.shape, j_cumulative_lab.dtype),
        (j_cumulative_rot.shape, j_cumulative_rot.dtype),
        (j_cumulative_ss.shape, j_cumulative_ss.dtype),
        (trotter_change.shape, trotter_change.dtype),
    ]
    if rho_ss_smooth_cache is not None:
        parent_payload_specs.append((rho_ss_smooth_cache.shape, rho_ss_smooth_cache.dtype))
    parent_payload_bytes = params.estimate_array_storage_bytes(*parent_payload_specs)
    ram_guard_fraction = float(p.get("ram_guard_fraction", 0.85))
    threshold_bytes = float(p.get("ram_threshold_gb", 16.0)) * (1024.0 ** 3) * ram_guard_fraction

    def estimated_total_bytes(nproc_eff: int) -> int:
        return int(parent_payload_bytes + worker_payload_bytes * max(0, int(nproc_eff)))

    nproc_requested = nproc
    while nproc > 1 and estimated_total_bytes(nproc) > threshold_bytes:
        nproc = max(1, nproc // 2)
    if nproc != nproc_requested:
        print(
            "03c: reducing multiprocessing from "
            f"{nproc_requested} to {nproc} process(es); estimated parent payload is "
            f"{parent_payload_bytes / (1024.0 ** 3):.2f} GiB and worker payload is "
            f"{worker_payload_bytes / (1024.0 ** 3):.2f} GiB per process.",
            flush=True,
        )

    _init_05c_worker(OBS_KEYS, base_obs, ms.astype(float), evals, evecs, omega0, delta_phi, js, rho_target_jm, rho_ss_cache, beta, n_steps, tau_arr, degeneracy_tol, compute_lab, compute_rot)
    def _store_result(res: dict[str, object]) -> None:
        it = int(res["idx"])
        for key in OBS_KEYS:
            obs_per_mode[key][it] = res["obs_per_mode"][key]
            thermal_lab[key][it] = float(res["thermal_lab"][key])
            thermal_rot[key][it] = float(res["thermal_rot"][key])
            thermal_ss[key][it] = float(res["thermal_ss"][key])
        mode_weights_lab[it] = res["mode_weights_lab"]
        mode_weights_rot[it] = res["mode_weights_rot"]
        mode_weights_ss[it] = res["mode_weights_ss"]
        j_importance_lab[it] = res["j_importance_lab"]
        j_importance_rot[it] = res["j_importance_rot"]
        j_importance_ss[it] = res["j_importance_ss"]
        j_cumulative_lab[it] = res["j_cumulative_lab"]
        j_cumulative_rot[it] = res["j_cumulative_rot"]
        j_cumulative_ss[it] = res["j_cumulative_ss"]
        trotter_change[:, it] = res["trotter_change"]

    report_every = max(1, nt // 10)
    if nproc <= 1 or nt <= 1:
        for it in range(nt):
            _store_result(_compute_one_time_05c(it))
            if (it + 1) % report_every == 0 or it == nt - 1:
                print(f"  03c: observables {it + 1}/{nt}", flush=True)
    else:
        done = 0
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=nproc,
            initializer=_init_05c_worker,
            initargs=(OBS_KEYS, base_obs, ms.astype(float), evals, evecs, omega0, delta_phi, js, rho_target_jm, rho_ss_cache, beta, n_steps, tau_arr, degeneracy_tol, compute_lab, compute_rot),
        ) as pool:
            for res in pool.imap_unordered(_compute_one_time_05c, range(nt), chunksize=max(1, int(p.get("chunksize", 1)))):
                _store_result(res)
                done += 1
                if done % report_every == 0 or done == nt:
                    print(f"  03c: observables {done}/{nt}", flush=True)

    if compute_ss_smooth and rho_ss_smooth_cache is not None:
        print(f"  03c: smooth steady-state cache ({nt} steps)...", flush=True)
        for it in range(nt):
            rho_ss_smooth_cache[it] = params.causal_half_gaussian_average(
                float(t[it]),
                t[: it + 1],
                rho_ss_cache[: it + 1],
                tau_smooth,
            )
            obs_mats_i = observable_matrices_at_time_05c(base_obs, ms.astype(float), float(delta_phi[it]))
            for key in OBS_KEYS:
                thermal_ss_smooth[key][it] = thermal_observable_from_density(rho_ss_smooth_cache[it], obs_mats_i[key])

    grids_dense = params.drive_grids_with_Nt(p, int(p.get("Nt_plot_rotating_observables", nt)))
    t_dense = np.asarray(grids_dense["t"], dtype=float)
    omega0_dense = np.asarray(grids_dense["Omega0"], dtype=float)
    v0_dense = np.asarray(grids_dense["V0"], dtype=float)
    delta_phi_dense = np.asarray(grids_dense["Delta_phi"], dtype=float)
    left_idx, right_idx, w_left, w_right = params.interpolation_indices_and_weights(t_dense, t)
    dense_lab = {key: np.full(t_dense.size, np.nan, dtype=float) for key in ["cos2phi_lab", "cos2theta2D"]}
    dense_rot = {key: np.full(t_dense.size, np.nan, dtype=float) for key in ["cos2phi_lab", "cos2theta2D"]}
    dense_ss = {key: np.full(t_dense.size, np.nan, dtype=float) for key in ["cos2phi_lab", "cos2theta2D"]}
    dense_ss_smooth = {key: np.full(t_dense.size, np.nan, dtype=float) for key in ["cos2phi_lab", "cos2theta2D"]} if compute_ss_smooth else {}
    print(f"  03c: dense rotating observables ({t_dense.size} steps)...", flush=True)
    current_left = current_right = -1
    rho_lab_left = rho_lab_right = None
    rho_rot_left = rho_rot_right = None
    rho_ss_left = rho_ss_right = None
    rho_ss_smooth_left = rho_ss_smooth_right = None

    for it_out in range(t_dense.size):
        il = int(left_idx[it_out])
        ir = int(right_idx[it_out])
        if il != current_left:
            rho_lab_left = np.asarray(rho_target_jm[il], dtype=np.complex128)
            rho_rot_left = params.normalized_density_matrix(k_matrix_from_eigensystem(evals[il], evecs[il], beta)) if compute_rot else None
            rho_ss_left = np.asarray(rho_ss_cache[il], dtype=np.complex128)
            if compute_ss_smooth and rho_ss_smooth_cache is not None:
                rho_ss_smooth_left = np.asarray(rho_ss_smooth_cache[il], dtype=np.complex128)
            current_left = il
        if ir != current_right:
            rho_lab_right = np.asarray(rho_target_jm[ir], dtype=np.complex128)
            rho_rot_right = params.normalized_density_matrix(k_matrix_from_eigensystem(evals[ir], evecs[ir], beta)) if compute_rot else None
            rho_ss_right = np.asarray(rho_ss_cache[ir], dtype=np.complex128)
            if compute_ss_smooth and rho_ss_smooth_cache is not None:
                rho_ss_smooth_right = np.asarray(rho_ss_smooth_cache[ir], dtype=np.complex128)
            current_right = ir
        uz = rotation_operator(ms.astype(float), float(delta_phi_dense[it_out]))
        obs_dense = {
            "cos2phi_lab": uz @ base_obs["cos2phi_rot"] @ uz.conj().T,
            "cos2theta2D": uz @ base_obs["cos2theta2D"] @ uz.conj().T,
        }
        wl = float(w_left[it_out])
        wr = float(w_right[it_out])
        for key in ("cos2phi_lab", "cos2theta2D"):
            val_lab_left = thermal_observable_from_density(rho_lab_left, obs_dense[key]) if compute_lab else float("nan")
            val_lab_right = val_lab_left if il == ir else thermal_observable_from_density(rho_lab_right, obs_dense[key]) if compute_lab else float("nan")
            val_rot_left = thermal_observable_from_density(rho_rot_left, obs_dense[key]) if compute_rot and rho_rot_left is not None else float("nan")
            val_rot_right = val_rot_left if il == ir else thermal_observable_from_density(rho_rot_right, obs_dense[key]) if compute_rot and rho_rot_right is not None else float("nan")
            val_ss_left = thermal_observable_from_density(rho_ss_left, obs_dense[key])
            val_ss_right = val_ss_left if il == ir else thermal_observable_from_density(rho_ss_right, obs_dense[key])
            dense_lab[key][it_out] = wl * val_lab_left + wr * val_lab_right if compute_lab else np.nan
            dense_rot[key][it_out] = wl * val_rot_left + wr * val_rot_right if compute_rot else np.nan
            dense_ss[key][it_out] = wl * val_ss_left + wr * val_ss_right
            if compute_ss_smooth and rho_ss_smooth_left is not None and rho_ss_smooth_right is not None:
                val_ss_smooth_left = thermal_observable_from_density(rho_ss_smooth_left, obs_dense[key])
                val_ss_smooth_right = val_ss_smooth_left if il == ir else thermal_observable_from_density(rho_ss_smooth_right, obs_dense[key])
                dense_ss_smooth[key][it_out] = wl * val_ss_smooth_left + wr * val_ss_smooth_right

    out_05c.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(h5py, out_05c, "w") as h5:
        h5.create_dataset("t_grid", data=t)
        h5.create_dataset("Omega0_t", data=omega0)
        h5.create_dataset("Delta_phi_t", data=delta_phi)
        h5.create_dataset("J", data=js.astype(np.int32))
        h5.create_dataset("M", data=ms.astype(np.int32))
        h5.create_dataset("L_re", data=np.real(l_matrix))
        h5.create_dataset("L_im", data=np.imag(l_matrix))
        grp_obs = h5.create_group("observables_jm_base")
        for key, arr in base_obs.items():
            grp_obs.create_dataset(key, data=arr)
        grp_mode = h5.create_group("observables_per_mode")
        for key in OBS_KEYS:
            grp_mode.create_dataset(key, data=obs_per_mode[key])
        grp_th_lab = h5.create_group("thermal/lab_frame")
        grp_th_rot = h5.create_group("thermal/rotating_frame")
        grp_th_ss = h5.create_group("thermal/steady_state")
        for key in OBS_KEYS:
            grp_th_lab.create_dataset(key, data=thermal_lab[key])
            grp_th_rot.create_dataset(key, data=thermal_rot[key])
            grp_th_ss.create_dataset(key, data=thermal_ss[key])
        if compute_ss_smooth:
            grp_th_ss_smooth = h5.create_group("thermal/steady_state_smooth")
            for key in OBS_KEYS:
                grp_th_ss_smooth.create_dataset(key, data=thermal_ss_smooth[key])
        grp_dense = h5.create_group("dense_rotating_observables")
        grp_dense.create_dataset("t_grid", data=t_dense)
        grp_dense.create_dataset("Omega0_t", data=omega0_dense)
        grp_dense.create_dataset("V0_t", data=v0_dense)
        grp_dense.create_dataset("Delta_phi_t", data=delta_phi_dense)
        grp_dense_lab = grp_dense.create_group("lab_frame")
        grp_dense_rot = grp_dense.create_group("rotating_frame")
        grp_dense_ss = grp_dense.create_group("steady_state")
        for key in ["cos2phi_lab", "cos2theta2D"]:
            grp_dense_lab.create_dataset(key, data=dense_lab[key])
            grp_dense_rot.create_dataset(key, data=dense_rot[key])
            grp_dense_ss.create_dataset(key, data=dense_ss[key])
        if compute_ss_smooth:
            grp_dense_ss_smooth = grp_dense.create_group("steady_state_smooth")
            for key in ["cos2phi_lab", "cos2theta2D"]:
                grp_dense_ss_smooth.create_dataset(key, data=dense_ss_smooth[key])
        grp_w = h5.create_group("thermal/mode_weights")
        grp_w.create_dataset("lab_frame", data=mode_weights_lab)
        grp_w.create_dataset("rotating_frame", data=mode_weights_rot)
        grp_w.create_dataset("steady_state", data=mode_weights_ss)
        grp_j = h5.create_group("cutoff_J")
        grp_j.create_dataset("J_values", data=j_values.astype(np.int32))
        grp_j.create_dataset("importance_lab_frame", data=j_importance_lab)
        grp_j.create_dataset("importance_rotating_frame", data=j_importance_rot)
        grp_j.create_dataset("importance_steady_state", data=j_importance_ss)
        grp_j.create_dataset("cumulative_lab_frame", data=j_cumulative_lab)
        grp_j.create_dataset("cumulative_rotating_frame", data=j_cumulative_rot)
        grp_j.create_dataset("cumulative_steady_state", data=j_cumulative_ss)
        grp_trot = h5.create_group("thermal/trotter")
        grp_trot.create_dataset("n_values", data=trotter_n_values)
        grp_trot.create_dataset("change", data=trotter_change)
        h5.attrs["case_name"] = str(p.get("case_name", "Default"))
        h5.attrs["B"] = float(p["B"])
        h5.attrs["rotational_model"] = int(p.get("rotational_model", 1))
        if p.get("B_star", None) is not None:
            h5.attrs["B_star"] = float(p["B_star"])
        if p.get("D_star", None) is not None:
            h5.attrs["D_star"] = float(p["D_star"])
        h5.attrs["thermal_trotter_steps"] = int(n_steps)
        h5.attrs["lab_density_method"] = "exact_diagonalization"
        h5.attrs["compute_thermal_lab_frame"] = int(compute_lab)
        h5.attrs["compute_thermal_rot_frame"] = int(compute_rot)
        h5.attrs["Compare_to_HO_basis"] = int(compare_ho)
        h5.attrs["tau_steady_state"] = "None" if tau_arr is None else float(tau_arr[0])
        tau_f_p = p.get("tau_steady_state_final", None)
        h5.attrs["tau_steady_state_final"] = "None" if tau_f_p is None else float(tau_f_p)
        h5.attrs["tau_smooth"] = "None" if tau_smooth is None else float(tau_smooth)
        h5.attrs["Delta_phi_offset"] = float(p.get("Delta_phi_offset", 0.0))
        h5.attrs["compute_steady_state_smooth"] = int(compute_ss_smooth)
        h5.attrs["steady_state_smooth_kernel"] = "causal_half_gaussian_average_of_observables" if compute_ss_smooth else "disabled_due_to_ram_threshold"
        h5.attrs["degeneracy_tol"] = float(degeneracy_tol)
        h5.attrs["Nt_plot_rotating_observables"] = int(t_dense.size)
    print(f"Wrote: {out_05c}", flush=True)


if __name__ == "__main__":
    main()
