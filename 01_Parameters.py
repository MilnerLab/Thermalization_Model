#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central shared parameters and case management for the centrifuge project."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict
import zlib

_LOCAL_MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_MPLCONFIGDIR))
_LOCAL_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (6,6)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.use14corefonts"] = True
plt.rcParams["axes.unicode_minus"] = False
CASE_ENV_VAR = "PENDULON_CASE"
path_data = Path(__file__).resolve().parent / "data"


BASE_DEFAULTS: Dict[str, object] = dict(
    # Plotting and computating parameters
    Nt_plot_rotating_observables=100,
    N_plot_modes=6,
    nproc=1,
    chunksize=1,
    ram_threshold_gb=16.0,
    ram_guard_fraction=0.85,
    
    # Rotor and drive parameters
    B=3.0,
    rotational_model=1,    #1 = free rotor, 2 = with centrifugal distortion
    B_star=None,
    D_star=None,
    Delta_alpha=10,
    E0=5e10,
    rotor_t_min = -250e-3, # ns
    rotor_t_max = 250e-3, # ns
    Nt_main = 500,
    rotor_acceleration_ramp = 50.0, # dOmega0/dt in GHz / ns = MHz / ps
    rotor_frequency0 = 12.5, # GHz at t=0
    rotor_phi0 = 0.0, # rad at t=0
    rotor_sigma = 320 * 1e-3 / (2*np.sqrt(2*np.log(2))), #FWHM in ps converted to ns

    # Angular cutoffs
    lambda_max=2,
    lambda_max_angulon=8,
    spectral_weight_2lam1=1,
    J_max=25,
    N_theta=50,
    N_phi=200,

    # Temperature and width parameters
    T_K=0.37,
    kB_per_K=20.8366,
    thermal_nsigma_kBT=5.0,
    eta=0.2,
    
    # Parameters for 02: bath + angulon
    w_min_02=-100.0,
    w_max_02=400.0,
    Nw_02=2500,

    # Bath parameters
    bath_model=2,
    c_s=378,
    m_bath=6.3e-4,
    a_bb=3.3,
    gn=0.0,
    n=0.0218,
    do_density_sweep=1,
    Ln_min=-15,
    Ln_max=5,
    Nn=100,

    # Coupling to the bath
    r0=2.5,
    C06=0,
    C08=0,
    r1=2.5,
    C16=0,
    C18=0,
    r2=3.6 * 1.6,
    C26=1.38e3 * 242,
    C28=1080e3 * 242,
    r_other=1.0,
    C6_other=0,
    C8_other=0,
    
    # Space/wavevector grid for the bath
    Nr=250,
    r_cut_factor=5.0,
    k_min=1e-4,
    k_max=2.0,
    Nk=250,

    # Steady state thermalisation
    tau_steady_state=0.25,
    tau_steady_state_final=None,
    tau_smooth=0.001,
    degeneracy_tol=0,
    
    # Alternative thermalisations
    mode_weight_raw_relative_cutoff=1e-5,
    thermal_trotter_steps=10,
    compute_thermal_lab_frame=False,
    compute_thermal_rot_frame=False,
    compute_thermal_effective_modes=False,
    compute_thermal_quasiparticle_smooth=False,
    spectral_envelope_mode="gaussian_only",
    compute_thermal_quasiparticle_fit=False,
    
)


CASES: Dict[str, Dict[str, object]] = {
    "Default": {},
    "CS2": {"E0": 1e10, "tau_steady_state": 0.2, "tau_steady_state_final": 0.05, "tau_smooth": 0.02, "B": 3.27, "Delta_alpha": 7.77, "rotor_frequency0": 18.997041712122172, "rotor_acceleration_ramp": 92.08265680229961, "rotor_phi0": 0.010999796687703782, "C06": 18e3 * 242, "C08": 482e3 * 242, "r0": 3.6 * 1.6, "C26": 1.38e3 * 242, "C28": 1080e3 * 242, "r2": 3.6 * 1.6},
    "CS2_renormalised": {"E0": 3e10, "tau_steady_state": 0.2, "tau_steady_state_final": 0.05, "tau_smooth": 0.02, "B": 3.27, "rotational_model": 2, "B_star": 0.73, "D_star": 1.2e-3, "Delta_alpha": 7.77, "rotor_frequency0": 18.997041712122172, "rotor_acceleration_ramp": 92.08265680229961, "rotor_phi0": 0.010999796687703782, "C06": 18e3 * 242, "C08": 482e3 * 242, "r0": 3.6 * 1.6, "C26": 1.38e3 * 242, "C28": 1080e3 * 242, "r2": 3.6 * 1.6},
    "OCS": {"E0": 1e10, "tau_steady_state": 1.0, "tau_steady_state_final": 0.5, "tau_smooth": 0.05, "B": 6.08, "Delta_alpha": 4.67, "rotor_frequency0": 22.498737093913505, "rotor_acceleration_ramp": 97.59025698477419, "rotor_phi0": 0.5720343864739321, "C06": 18e3 * 242 / 2, "C08": 482e3 * 242 / 2, "r0": 3.6 * 1.6, "C26": 1.38e3 * 242 / 2, "C28": 1080e3 * 242 / 2, "r2": 3.6 * 1.6},
    "OCS_renormalised": {"E0": 3e10, "tau_steady_state": 1.0, "tau_steady_state_final": 0.5, "tau_smooth": 0.05, "B": 6.08, "rotational_model": 2, "B_star": 2.18, "D_star": 9.5e-3, "Delta_alpha": 4.67, "rotor_frequency0": 22.498737093913505, "rotor_acceleration_ramp": 97.59025698477419, "rotor_phi0": 0.5720343864739321, "C06": 18e3 * 242 / 2, "C08": 482e3 * 242 / 2, "r0": 3.6 * 1.6, "C26": 1.38e3 * 242 / 2, "C28": 1080e3 * 242 / 2, "r2": 3.6 * 1.6},
}


def _sanitize_case_name(case_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(case_name))
    return safe or "Default"


def get_active_case_name() -> str:
    if not CASES:
        raise KeyError("No cases defined in CASES.")
    fallback = "Default" if "Default" in CASES else next(iter(CASES))
    case_name = str(os.environ.get(CASE_ENV_VAR, fallback)).strip() or fallback
    if case_name not in CASES:
        return fallback
    return case_name


def get_all_case_names() -> list[str]:
    return list(CASES.keys())


def get_case_tag(case_name: str | None = None) -> str:
    active = get_active_case_name() if case_name is None else str(case_name)
    return _sanitize_case_name(active)


def get_case_root(case_name: str | None = None) -> Path:
    active = get_active_case_name() if case_name is None else str(case_name)
    if active == "Default":
        return Path(".")
    return Path(get_case_tag(active))


def get_case_stage_dir(stage_name: str, case_name: str | None = None) -> Path:
    return get_case_root(case_name) / stage_name


def get_case_fig_root(case_name: str | None = None) -> Path:
    active = get_active_case_name() if case_name is None else str(case_name)
    if active == "Default":
        return Path("figs")
    return Path("figs") / get_case_tag(active)


def get_case_data_root(case_name: str | None = None) -> Path:
    active = get_active_case_name() if case_name is None else str(case_name)
    if active == "Default":
        return path_data
    return path_data / get_case_tag(active)


def with_case_label(title: str, case_name: str | None = None) -> str:
    active = get_active_case_name() if case_name is None else str(case_name)
    return f"[{active}] {title}"


def centrifuge_V0(p: Dict[str, object]) -> float:
    dalpha = float(p["Delta_alpha"])
    E0 = float(p["E0"])
    return 0.5 * dalpha * E0 * 6.32e-10


def Omega0_grid(p: Dict[str, object], t_grid: np.ndarray | None = None) -> np.ndarray:
    ramp = float(p.get("rotor_acceleration_ramp", p.get("rotor_beta0", 50.0)))
    f0 = float(p.get("rotor_frequency0", 12.5))
    if t_grid is None:
        t_grid = time_grid(p)
    return f0 + ramp * np.asarray(t_grid, dtype=float)


def V0_grid(p: Dict[str, object], t_grid: np.ndarray | None = None) -> np.ndarray:
    if t_grid is None:
        t_grid = time_grid(p)
    t = np.asarray(t_grid, dtype=float)
    pref = centrifuge_V0(p)
    sigma_raw = p.get("rotor_sigma", 50 / 0.5)
    if sigma_raw is None:
        return pref * np.ones_like(t, dtype=float)
    t_min = float(p.get("rotor_t_min", 0.0))
    t_max = float(p.get("rotor_t_max", 0.5))
    t_center = 0.5 * (t_min + t_max)
    sigma = float(sigma_raw)
    return pref * (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)


def tau_steady_state_grid(p: Dict[str, object], t_grid: np.ndarray | None = None) -> np.ndarray | None:
    tau_0 = p.get("tau_steady_state", None)
    if tau_0 is None:
        return None
    tau_0 = float(tau_0)
    tau_f = p.get("tau_steady_state_final", None)
    if t_grid is None:
        t_grid = time_grid(p)
    t = np.asarray(t_grid, dtype=float)
    if tau_f is None or t.size <= 1:
        return np.full(t.size, tau_0)
    tau_f = float(tau_f)
    t0, t1 = float(t[0]), float(t[-1])
    if t1 == t0:
        return np.full(t.size, tau_0)
    alpha = (t - t0) / (t1 - t0)
    return tau_0 + alpha * (tau_f - tau_0)


def Delta_phi_grid(p: Dict[str, object], t_grid: np.ndarray | None = None, Omega0_t: np.ndarray | None = None) -> np.ndarray:
    if t_grid is None:
        t_grid = time_grid(p)
    t = np.asarray(t_grid, dtype=float)
    ramp = float(p.get("rotor_acceleration_ramp", p.get("rotor_beta0", 50.0)))
    f0 = float(p.get("rotor_frequency0", 12.5))
    phi0 = float(p.get("rotor_phi0", 0.0))
    return phi0 + 2.0 * np.pi * (f0 * t + 0.5 * ramp * t**2)


def time_grid_with_Nt(p: Dict[str, object], Nt: int) -> np.ndarray:
    t_min = float(p.get("rotor_t_min", 0.0))
    t_max = float(p.get("rotor_t_max", 0.5))
    return np.linspace(t_min, t_max, Nt)


def time_grid(p: Dict[str, object]) -> np.ndarray:
    Nt = int(p.get("Nt_main", p.get("Nt_pendulon", p.get("rotor_Nt", 1000))))
    return time_grid_with_Nt(p, Nt)


def drive_grids(p: Dict[str, object]) -> dict[str, np.ndarray]:
    t = time_grid(p)
    omega = Omega0_grid(p, t)
    v0 = V0_grid(p, t)
    dphi = Delta_phi_grid(p, t, omega)
    return {"t": t, "Omega0": omega, "V0": v0, "Delta_phi": dphi}


def drive_grids_with_Nt(p: Dict[str, object], Nt: int) -> dict[str, np.ndarray]:
    t = time_grid_with_Nt(p, int(Nt))
    omega = Omega0_grid(p, t)
    v0 = V0_grid(p, t)
    dphi = Delta_phi_grid(p, t, omega)
    return {"t": t, "Omega0": omega, "V0": v0, "Delta_phi": dphi}


def interpolation_indices_and_weights(
    t_out: np.ndarray,
    t_sel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_out_arr = np.asarray(t_out, dtype=float)
    t_sel_arr = np.asarray(t_sel, dtype=float)
    n_out = t_out_arr.size
    if t_sel_arr.size == 0:
        zeros = np.zeros(n_out, dtype=np.int32)
        ones = np.ones(n_out, dtype=float)
        return zeros, zeros, ones, np.zeros(n_out, dtype=float)

    right = np.searchsorted(t_sel_arr, t_out_arr, side="left").astype(np.int32)
    right = np.clip(right, 0, t_sel_arr.size - 1)
    left = np.maximum(right - 1, 0).astype(np.int32)

    hit = np.isclose(t_out_arr, t_sel_arr[right], rtol=0.0, atol=1e-15)
    left[hit] = right[hit]

    at_start = right == 0
    left[at_start] = 0
    right[at_start] = 0

    past_end = t_out_arr >= t_sel_arr[-1]
    left[past_end] = t_sel_arr.size - 1
    right[past_end] = t_sel_arr.size - 1

    w_left = np.ones(n_out, dtype=float)
    w_right = np.zeros(n_out, dtype=float)

    interp = left != right
    if np.any(interp):
        t_left = t_sel_arr[left[interp]]
        t_right = t_sel_arr[right[interp]]
        denom = np.maximum(t_right - t_left, 1e-30)
        w_right[interp] = (t_out_arr[interp] - t_left) / denom
        w_left[interp] = 1.0 - w_right[interp]

    return left, right, w_left, w_right


def causal_half_gaussian_average(
    t_now: float,
    t_history: np.ndarray,
    value_history: np.ndarray,
    tau: float | None,
) -> np.ndarray:
    """Return the normalized causal half-Gaussian average over the past history.

    The kernel is

        w(x) = exp(-x^2 / (2 tau^2)),   x >= 0

    and the discrete average uses

        avg(t_now) = sum_k w(t_now - t_k) value_k / sum_k w(t_now - t_k),

    with the understanding that only t_k <= t_now should be passed in.
    """
    values = np.asarray(value_history)
    if values.ndim == 0:
        return np.array(values, copy=True)
    if tau is None or tau <= 0.0 or values.shape[0] == 0:
        return np.array(values[-1], copy=True)
    times = np.asarray(t_history, dtype=float)
    dt = np.maximum(float(t_now) - times, 0.0)
    weights = np.exp(-0.5 * (dt / float(tau)) ** 2)
    norm = float(np.sum(weights))
    if norm <= np.finfo(float).tiny:
        return np.array(values[-1], copy=True)
    reshape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return np.sum(values * weights.reshape(reshape), axis=0) / norm


def estimate_array_storage_bytes(*specs: tuple[tuple[int, ...], object]) -> int:
    total = 0
    for shape, dtype in specs:
        total += int(np.prod(tuple(int(x) for x in shape), dtype=np.int64)) * np.dtype(dtype).itemsize
    return int(total)


def exceeds_ram_threshold(p: Dict[str, object], bytes_needed: int) -> bool:
    threshold_gb = float(p["ram_threshold_gb"])
    return int(bytes_needed) > threshold_gb * (1024.0 ** 3)


def select_time_indices(n_src: int, n_target: int) -> np.ndarray:
    """Return a monotone approximately-uniform subset of source indices."""
    n_src_i = int(n_src)
    n_target_i = max(1, int(n_target))
    if n_src_i <= 0:
        return np.zeros(0, dtype=np.int32)
    if n_target_i >= n_src_i:
        return np.arange(n_src_i, dtype=np.int32)
    idx = np.rint(np.linspace(0, n_src_i - 1, n_target_i)).astype(np.int32)
    return np.unique(idx)


def w_grid_rel(p: Dict[str, object]) -> np.ndarray:
    w_min = float(p.get("w_min", -200.0))
    w_max = float(p.get("w_max", 200.0))
    Nw = int(p.get("Nw", 400))
    return np.linspace(w_min, w_max, Nw)


def ho_frequency(B: float, V0: float) -> float:
    return 2.0 * np.sqrt(B * V0)


def ho_common_rotating_shift(Omega0: float, B: float) -> float:
    return -Omega0 * Omega0 / (4.0 * B)


def omega_window_shift(Omega0: float, B: float, V0: float) -> float:
    return -float(V0) + ho_frequency(B, V0) + ho_common_rotating_shift(Omega0, B)


def w_grid_abs_from_shift(w_rel: np.ndarray, shift: float | np.ndarray) -> np.ndarray:
    return np.asarray(w_rel, dtype=float) + np.asarray(shift, dtype=float)


def rotational_energy_levels(js: np.ndarray, p: Dict[str, object]) -> np.ndarray:
    j_arr = np.asarray(js, dtype=float)
    x = j_arr * (j_arr + 1.0)
    b0 = float(p["B"])
    model = int(p.get("rotational_model", 1))
    if model == 1:
        return b0 * x
    if model != 2:
        raise ValueError(f"Unsupported rotational_model={model}. Expected 1 or 2.")

    b_star_raw = p.get("B_star", None)
    d_star_raw = p.get("D_star", None)
    if b_star_raw is None or d_star_raw is None:
        raise ValueError("rotational_model=2 requires both B_star and D_star in parameters.")
    b_star = float(b_star_raw)
    d_star = float(d_star_raw)
    delta_b = b0 - b_star
    if delta_b <= 0.0:
        return b0 * x
    if d_star <= 0.0:
        raise ValueError("rotational_model=2 requires D_star > 0.")

    x0 = delta_b / d_star
    x0 = max(float(x0), np.finfo(float).tiny)
    correction = delta_b * x0 * x / (x0 + x)
    return b0 * x - correction


def rotational_energy_diagonal(js: np.ndarray, p: Dict[str, object]) -> np.ndarray:
    return np.diag(rotational_energy_levels(js, p).astype(np.complex128))


def save_pdf(path: Path, retries: int = 4, apply_tight_layout: bool = True) -> None:
    fig = plt.gcf()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if apply_tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            # Some figures with secondary axes or dense mathtext can make
            # tight_layout brittle or disproportionately slow. Fall back to the
            # current layout instead of aborting the whole pipeline.
            pass
    for attempt in range(max(1, int(retries))):
        tmp_path = path.with_name(f".{path.stem}.tmp{attempt}.pdf")
        try:
            fig.savefig(tmp_path, format="pdf")
            tmp_path.replace(path)
            break
        except Exception as e:
            msg = str(e).lower()
            is_timeout = isinstance(e, TimeoutError) or getattr(e, "errno", None) == 60
            is_stream_error = "stream state" in msg or "operation timed out" in msg
            is_pdf_font_error = isinstance(e, zlib.error) or "embedttf" in msg or "writefonts" in msg or "type42" in msg
            tmp_path.unlink(missing_ok=True)
            if is_pdf_font_error:
                # Fall back to simpler PDF settings when Type-42 font embedding
                # trips over backend_pdf state in long in-process pipeline runs.
                try:
                    with plt.rc_context({
                        "pdf.fonttype": 3,
                        "ps.fonttype": 3,
                        "pdf.use14corefonts": False,
                        "pdf.compression": 0,
                    }):
                        fig.savefig(tmp_path, format="pdf")
                    tmp_path.replace(path)
                    break
                except Exception as fallback_error:
                    msg = f"{msg} | fallback: {str(fallback_error).lower()}"
                    is_timeout = is_timeout or isinstance(fallback_error, TimeoutError) or getattr(fallback_error, "errno", None) == 60
                    is_stream_error = is_stream_error or "stream state" in str(fallback_error).lower() or "operation timed out" in str(fallback_error).lower()
                    tmp_path.unlink(missing_ok=True)
                    e = fallback_error
            if (not is_timeout and not is_stream_error and not is_pdf_font_error) or (attempt == retries - 1):
                plt.close(fig)
                raise
            time.sleep(0.3 * (attempt + 1))
    plt.close(fig)


def add_omega0_top_axis(ax: plt.Axes, t_ps: np.ndarray, Omega: np.ndarray) -> None:
    """Add a secondary top x-axis showing Omega0(t) for time-domain plots."""
    t = np.asarray(t_ps, dtype=float)
    om = np.asarray(Omega, dtype=float)
    if t.size < 2 or om.size != t.size:
        return

    if np.all(np.diff(t) > 0):
        tt, oo = t, om
    elif np.all(np.diff(t) < 0):
        tt, oo = t[::-1], om[::-1]
    else:
        order = np.argsort(t)
        tt, oo = t[order], om[order]

    if np.any(np.diff(oo) <= 0):
        return

    ax.set_xlim(float(np.min(tt)), float(np.max(tt)))

    def t_to_om(x: np.ndarray) -> np.ndarray:
        return np.interp(x, tt, oo, left=oo[0], right=oo[-1])

    def om_to_t(x: np.ndarray) -> np.ndarray:
        return np.interp(x, oo, tt, left=tt[0], right=tt[-1])

    sec = ax.secondary_xaxis("top", functions=(t_to_om, om_to_t))
    sec.set_xlabel(r"$\Omega_0(t)\;[\mathrm{GHz}]$")


def normalized_density_matrix(mat: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """Return mat / Tr(mat), with a safe fallback if the trace is too small."""
    rho = np.asarray(mat, dtype=np.complex128)
    tr = np.trace(rho)
    if abs(tr) <= eps:
        n = int(rho.shape[0]) if rho.ndim == 2 else 1
        return np.eye(n, dtype=np.complex128) / float(max(n, 1))
    return rho / tr


def trotter_density_matrix(K: np.ndarray, R: np.ndarray, n_steps: int, eps: float = 1e-14) -> np.ndarray:
    """Return rho_n from the symmetric Trotter factor [R K R]^n / Tr(...).

    Conventions:
      - K should already be K(beta / n)
      - R should already be R(beta / (2 n))
    so that this implements
      rho_n ~ [ R(beta/2n) K(beta/n) R(beta/2n) ]^n .
    """
    n_eff = max(1, int(n_steps))
    K_arr = np.asarray(K, dtype=np.complex128)
    R_arr = np.asarray(R, dtype=np.complex128)
    sym_factor = R_arr @ K_arr @ R_arr
    M = np.linalg.matrix_power(sym_factor, n_eff)
    return normalized_density_matrix(M, eps=eps)


def density_relative_change(rho_a: np.ndarray, rho_b: np.ndarray, eps: float = 1e-14) -> float:
    """Relative Frobenius change between two density matrices."""
    da = np.asarray(rho_a, dtype=np.complex128)
    db = np.asarray(rho_b, dtype=np.complex128)
    num = float(np.linalg.norm(db - da))
    den = max(float(np.linalg.norm(db)), eps)
    return num / den


def get_defaults_for_case(case_name: str | None = None) -> Dict[str, object]:
    active = get_active_case_name() if case_name is None else str(case_name)
    if active not in CASES:
        available = ", ".join(sorted(CASES))
        raise KeyError(f"Unknown case '{active}'. Available cases: {available}")

    out = dict(BASE_DEFAULTS)
    case_overrides = CASES[active]
    out.update(case_overrides)
    if "w_min_02" not in case_overrides and "w_min" in case_overrides:
        out["w_min_02"] = out["w_min"]
    if "w_max_02" not in case_overrides and "w_max" in case_overrides:
        out["w_max_02"] = out["w_max"]
    if "Nw_02" not in case_overrides and "Nw" in case_overrides:
        out["Nw_02"] = out["Nw"]

    out_root = get_case_root(active)
    fig_root = get_case_fig_root(active)
    data_root = get_case_data_root(active)
    out["case_name"] = active
    out["case_tag"] = get_case_tag(active)
    out["output_root"] = str(out_root)
    out["fig_root"] = str(fig_root)
    out["data_root"] = str(data_root)
    out["path_data"] = str(path_data)
    out["fig_dir_01"] = str(fig_root / "01_spherical_harmonics")
    shared_data_root = get_case_data_root("Default")
    out["data_dir_01"] = str(shared_data_root / "01_spherical_harmonics")
    out["steady_state_target_h5_path"] = str(data_root / "01_spherical_harmonics" / "steady_state_target.h5")
    out["fig_dir_02_bath"] = str(fig_root / "02_bath_and_angulon")
    out["data_dir_02_bath"] = str(data_root / "02_bath_and_angulon")
    out["spline_poly_path"] = str((data_root / "02_bath_and_angulon" / "spline_poly.npz"))
    out["fig_dir_03_localized"] = str(fig_root / "03_harmonic_oscillator")
    out["fig_tech_dir_03_localized"] = str(fig_root / "03_harmonic_oscillator" / "figs_technical")
    out["data_dir_03_localized"] = str(data_root / "03_harmonic_oscillator")
    out["fig_dir_04_renormalised_pendulon"] = str(fig_root / "04_renormalised_pendulon")
    out["fig_tech_dir_04_renormalised_pendulon"] = str(fig_root / "04_renormalised_pendulon" / "figs_technical")
    out["data_dir_04_renormalised_pendulon"] = str(data_root / "04_renormalised_pendulon")
    out["fig_dir_03_free_rotor_drive"] = str(fig_root / "03_free_rotor_drive")
    out["fig_tech_dir_03_free_rotor_drive"] = str(fig_root / "03_free_rotor_drive" / "figs_technical")
    out["data_dir_03_free_rotor_drive"] = str(data_root / "03_free_rotor_drive")
    out["fig_dir_06_renormalised_rotor_drive"] = str(fig_root / "06_renormalised_rotor_drive")
    out["fig_tech_dir_06_renormalised_rotor_drive"] = str(fig_root / "06_renormalised_rotor_drive" / "figs_technical")
    out["data_dir_06_renormalised_rotor_drive"] = str(data_root / "06_renormalised_rotor_drive")
    out["fig_dir_05_excited_subspace"] = out["fig_dir_04_renormalised_pendulon"]
    out["fig_tech_dir_05_excited_subspace"] = out["fig_tech_dir_04_renormalised_pendulon"]
    out["data_dir_05_excited_subspace"] = out["data_dir_04_renormalised_pendulon"]
    out["projection_h5_path"] = str(data_root / "03_harmonic_oscillator" / "projection_HO_to_JM_vs_Omega0.h5")
    out["observable_h5_path"] = str(data_root / "03_harmonic_oscillator" / "observable_matrices_HO.h5")
    out["Ylm_h5_path"] = str(shared_data_root / "01_spherical_harmonics" / "Ylm_blocks_JM.h5")
    env_overrides = {
        "PENDULON_NT_MAIN": ("Nt_main", int),
        "PENDULON_NT_PLOT_ROTATING_OBSERVABLES": ("Nt_plot_rotating_observables", int),
        "PENDULON_NPROC": ("nproc", int),
    }
    for env_name, (key, caster) in env_overrides.items():
        raw = os.environ.get(env_name)
        if raw is not None and str(raw).strip() != "":
            out[key] = caster(raw)
    return out


DEFAULTS: Dict[str, object] = get_defaults_for_case()
