#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_bath_angulon.py

Bath + one-loop (Fock) self-energy for the rotor.

We use the effective influence action result:
  χ(Ω,Ω';ω) = Σ_{λμ} Y_{λμ}(Ω) Y^*_{λμ}(Ω') χ_λ(ω),
  with
    χ_λ(ω) = Σ_k |U_λ(k)|^2 / (ω - ω_k + i0^+).                                  (Eq. 20 in notes) fileciteturn6file0

The one-loop self-energy is (after doing the μ-sums and ω' contour integral):
  Σ_λ(ω) = (1/4π) Σ_{λ1,λ2} Σ_k (2λ1+1)(2λ2+1) |U_{λ2}(k)|^2
                 * [ ( λ  λ2  λ1 ; 0 0 0 ) ]^2
                 / (ω - B λ1(λ1+1) - ω_k + i0^+).                                 (Eq. 24 in notes) fileciteturn6file0

Dyson (diagonal in λ):
  G_λ(ω) = 1 / [ ω - B λ(λ+1) - Σ_λ(ω) ].                                          (Eq. 25 in notes) fileciteturn6file0

When executed (__main__), this script:
  - Defines a bosonic bath ω_k and an interaction U_λ(k) (configurable)
  - Computes and plots χ_λ(ω) (diagnostic) for a few λ
  - Computes and plots Σ_λ(ω) and G_λ(ω): Re and Im (your "spectrum")
  - Extracts renormalised energies from peaks of A_λ(ω) = -(1/π)Im G_λ(ω)
  - Estimates B* from the first spacing
  - Shows a rotating-frame shift of the dressed levels: E^*_{λM} = E^*_λ - Ω0 M (pure frame shift)

All figures are saved as PDF under: figs/02_bath/
Optionally saves arrays under: data/02_bath/

Requirements:
  - 01_Parameters.py must be importable (same directory or on PYTHONPATH).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List

_LOCAL_MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_MPLCONFIGDIR))
_LOCAL_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import spherical_jn
from h5_locking import open_h5

# Optional HDF5
try:
    import h5py  # type: ignore
    _H5PY_OK = True
except Exception:
    _H5PY_OK = False

import importlib
params = importlib.import_module("01_Parameters")
ang0b = importlib.import_module("01b_precompute_Ylm_blocks")
_HE_MOD = None
FIGSIZE = params.FIGSIZE
plt.rcParams["figure.figsize"] = FIGSIZE
CASE_ENV_VAR = params.CASE_ENV_VAR
BASE_DEFAULTS = params.BASE_DEFAULTS
CASES = params.CASES
get_active_case_name = params.get_active_case_name
get_all_case_names = params.get_all_case_names
get_case_tag = params.get_case_tag
get_case_root = params.get_case_root
get_case_stage_dir = params.get_case_stage_dir
with_case_label = params.with_case_label
get_defaults_for_case = params.get_defaults_for_case
DEFAULTS: Dict[str, object] = params.DEFAULTS


# -----------------------------
# Output folders
# -----------------------------
FIG_DIR = Path(str(DEFAULTS.get("fig_dir_02_bath", "figs/02_bath")))
DATA_DIR = Path(str(DEFAULTS.get("data_dir_02_bath", "data/02_bath")))
RUN_METADATA_PATH = DATA_DIR / "02b_generalities_metadata.json"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_fig_dir(fig_dir: Path) -> int:
    removed = 0
    keep_names = {".gitkeep"}
    for child in fig_dir.iterdir():
        if not child.is_file():
            continue
        if child.name in keep_names or child.suffix.lower() != ".pdf":
            continue
        child.unlink(missing_ok=True)
        removed += 1
    return removed


def generalities_cache_signature(p: Dict[str, float | int | str]) -> Dict[str, float | int | str]:
    sig: Dict[str, float | int | str] = {
        "B": float(p["B"]),
        "bath_model": int(p["bath_model"]),
        "k_min": float(p["k_min"]),
        "k_max": float(p["k_max"]),
        "Nk": int(p["Nk"]),
        "Nr": int(p["Nr"]),
        "r_cut_factor": float(p["r_cut_factor"]),
        "w_min_02": float(p.get("w_min_02", p.get("w_min", -100.0))),
        "w_max_02": float(p.get("w_max_02", p.get("w_max", 400.0))),
        "Nw_02": int(p.get("Nw_02", p.get("N_w_02", p.get("Nw", 2000)))),
        "eta": float(p["eta"]),
        "lambda_max_angulon": int(p.get("lambda_max_angulon", p.get("lambda_max", 8))),
        "C06": float(p["C06"]),
        "C08": float(p["C08"]),
        "C16": float(p["C16"]),
        "C18": float(p["C18"]),
        "C26": float(p["C26"]),
        "C28": float(p["C28"]),
        "C6_other": float(p["C6_other"]),
        "C8_other": float(p["C8_other"]),
        "r0": float(p["r0"]),
        "r1": float(p["r1"]),
        "r2": float(p["r2"]),
        "r_other": float(p["r_other"]),
    }
    bath_model = int(p["bath_model"])
    if bath_model == 0:
        sig["c_s"] = float(p["c_s"])
    elif bath_model == 1:
        sig["m_bath"] = float(p["m_bath"])
        sig["a_bb"] = float(p["a_bb"])
        sig["n"] = float(p["n"])
        sig["gn"] = float(p["gn"])
    elif bath_model == 2:
        # Helium bath uses the imported tabulated dispersion; coupling still depends on the
        # generic interaction and grid parameters already included above.
        sig["helium_model"] = 1
    return sig


def save_generalities_run_metadata(p: Dict[str, float | int | str]) -> None:
    ensure_dirs()
    payload = generalities_cache_signature(p)
    RUN_METADATA_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def plot_complex_vs_w(w: np.ndarray, F: np.ndarray, title: str, out_prefix: Path) -> None:
    """Save Re and Im of a complex function vs ω on the same PDF."""
    fig, ax = plt.subplots()
    re_line, = ax.plot(w, np.real(F), color="C0", label=r"$\mathrm{Re}$")
    im_line, = ax.plot(w, np.imag(F), color="C1", label=r"$\mathrm{Im}$")
    ax.set_xlabel(r"$\omega$")
    # plt.title(title)
    try:
        ax.legend(handles=[re_line, im_line])
    except Exception:
        # In some long-lived runpy sessions, legend proxy construction can trip
        # over a corrupted Matplotlib state. Keep the plot usable and continue.
        ax.text(0.02, 0.98, r"$\mathrm{Re}$", color="C0", transform=ax.transAxes, ha="left", va="top")
        ax.text(0.02, 0.90, r"$\mathrm{Im}$", color="C1", transform=ax.transAxes, ha="left", va="top")
    params.save_pdf(out_prefix.with_name(out_prefix.name + ".pdf"), apply_tight_layout=False)


# ======================================================================================
# Bath ω_k
# ======================================================================================

def epsilon_k(k: np.ndarray, m: float) -> np.ndarray:
    return k**2 / (2.0 * m)


def omega_k(k: np.ndarray, p: Dict[str, float | int]) -> np.ndarray:
    global _HE_MOD
    model = int(p["bath_model"])
    if model == 0:
        c = float(p["c_s"])
        return c * k
    if model == 1:
        m = float(p["m_bath"])
        eps = epsilon_k(k, m)

        # Preferred: Bogoliubov with g_bb n, where g_bb = 4π a_bb / m
        if "a_bb" in p and "n" in p:
            a_bb = float(p["a_bb"])
            n = float(p["n"])
            g_bb = 4.0 * np.pi * a_bb / m
            return np.sqrt(eps * (eps + 2.0 * g_bb * n))

        # Backward-compatible fallback: user-supplied gn (same dimensions as g_bb n)
        gn = float(p["gn"])
        return np.sqrt(eps * (eps + 2.0 * gn))
    if model == 2:
        meV_to_GHz = 1.519267e3/(2*np.pi)  # convert meV to GHz
        if _HE_MOD is None:
            _HE_MOD = importlib.import_module("02a_superfluid_helium")
        return np.asarray(_HE_MOD.omega_k_he(k)*meV_to_GHz, dtype=float)

    raise ValueError("Unknown bath_model. Use 0, 1 or 2.")


# ======================================================================================
# Coupling U_λ(k)
# ======================================================================================

def _u_lam(lam: int, p: Dict[str, float | int]) -> Tuple[float, float]:
    if lam == 0:
        return float(p["C06"]), float(p["C08"])
    if lam == 1:
        return float(p["C16"]), float(p["C18"])
    if lam == 2:
        return float(p["C26"]), float(p["C28"])
    return float(p["C6_other"]), float(p["C8_other"])

def _r_lam(lam: int, p: Dict[str, float | int]) -> float:
    if lam == 0:
        return float(p["r0"])
    if lam == 1:
        return float(p["r1"])
    if lam == 2:
        return float(p["r2"])
    return float(p["r_other"])


def U_lam_r(lam: int, r: np.ndarray, p: Dict[str, float | int]) -> np.ndarray:
    """Radial interaction profile U_l(r) = -C6/r^6 - C8/r^8 for a given lambda."""
    C6, C8 = _u_lam(lam, p)
    rr = np.asarray(r, dtype=float)
    out = np.zeros_like(rr, dtype=float)
    mask = rr > 0.0
    out[mask] = -C6 / (rr[mask] ** 6) - C8 / (rr[mask] ** 8)
    return out


# Cache to avoid recomputing U_lambda(k) repeatedly inside chi_lam and Sigma_lam
_U_CACHE: Dict[Tuple[int, int, Tuple[Tuple[str, float | int], ...]], np.ndarray] = {}


def _radial_I_lam(lam: int, k: np.ndarray, p: Dict[str, float | int], C6, C8) -> np.ndarray:
    """Compute I_lambda(k) = ∫_0^∞ dr r^2 f_lambda(r) j_lambda(k r).

    f_lambda(r) = (2π)^(-3/2) exp[-r^2/(2 r_lambda^2)].

    We evaluate on an r-grid (vectorized) and truncate at
      r_max = r_cut_factor * r_lambda.
    """
    if C6 == 0 and C8 == 0:
        return np.zeros_like(k)
    rlam = _r_lam(lam, p)
    Nr = int(p.get("Nr", 1600))
    r_cut = float(p.get("r_cut_factor", 10.0))
    r_max = r_cut * rlam

    r = np.linspace(rlam, r_max, Nr)
    # f can be replaced here while keeping _radial_I_lam behavior unchanged.
    f = U_lam_r(lam, r, p)
    w = (r * r) * f  # r^2 f(r)

    kr = np.outer(k, r)          # (Nk, Nr)
    jl = spherical_jn(lam, kr)   # (Nk, Nr)

    return np.where(k<2*np.pi/rlam, np.trapezoid(jl * w[None, :], r, axis=1), 0)  # (Nk,)


def U_lam(lam: int, k: np.ndarray, p: Dict[str, float | int]) -> np.ndarray:
    """Microscopic U_lambda(k) as in the paper.

    U_lambda(k) = u_lambda * sqrt[ 8 n k^2 ε_k / ( ω_k (2λ+1) ) ] * I_lambda(k),
    ε_k = k^2/(2m), with m = m_bath here.

    Notes:
      - Requires p['n'] and (optionally) p['a_bb'] for Bogoliubov ω_k.
      - Uses caching because this function is called many times.
    """
    n = float(p["n"])
    C6, C8 = _u_lam(lam, p)

    m = float(p["m_bath"])
    ek = epsilon_k(k, m)
    wk = omega_k(k, p)

    p_key = tuple(sorted(p.items(), key=lambda kv: kv[0]))
    key = (lam, id(k), p_key)
    if key in _U_CACHE:
        return _U_CACHE[key]

    I = _radial_I_lam(lam, k, p, C6=C6, C8=C8)

    out = np.zeros_like(k, dtype=float)
    mask = (k > 0) & (wk > 0)
    out[mask] = np.sqrt((8.0 * n * (k[mask] ** 2) * ek[mask]) / (wk[mask] * (2.0 * lam + 1.0))) * I[mask]

    _U_CACHE[key] = out
    return out


def trapz_last_axis(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Stable trapezoidal integration along the last axis."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y)
    if xx.ndim != 1:
        raise ValueError("trapz_last_axis expects a 1D integration grid.")
    if yy.shape[-1] != xx.size:
        raise ValueError(
            f"trapz_last_axis shape mismatch: y.shape[-1]={yy.shape[-1]} but x.size={xx.size}."
        )
    dx = np.diff(xx)
    return np.sum(0.5 * (yy[..., 1:] + yy[..., :-1]) * dx, axis=-1)



# ======================================================================================
# χ_λ(ω)  (Eq. 20)
# ======================================================================================

def chi_lam(w: np.ndarray, lam: int, k: np.ndarray, wk: np.ndarray, p: Dict[str, float | int]) -> np.ndarray:
    eta = float(p["eta"])
    U = U_lam(lam, k, p)
    denom = (w[:, None] - wk[None, :] + 1j*eta)
    return trapz_last_axis((U**2)[None, :] / denom, k)


# ======================================================================================
# Σ_λ(ω)  (Eq. 24)
# ======================================================================================

def threej0_sq(lam: int, lam2: int, lam1: int) -> float:
    """Returns [ (lam lam2 lam1; 0 0 0) ]^2."""
    x = ang0b.w3j(lam, lam2, lam1, 0, 0, 0)
    return float(x * x)


def Sigma_lam(w: np.ndarray, lam: int, k: np.ndarray, wk: np.ndarray, p: Dict[str, float | int]) -> np.ndarray:
    """
    Implements Eq. (24) in notes:
      Σ_λ(ω) = (1/4π) Σ_{λ1,λ2} ∫ dk (2λ1+1)(2λ2+1) |U_{λ2}(k)|^2
               * (3j)^2 / (ω - B λ1(λ1+1) - ω_k + iη)
    """
    B = float(p["B"])
    eta = float(p["eta"])
    lamb_max = int(p.get("lambda_max_angulon", p.get("lambda_max", 8)))

    Sig = np.zeros_like(w, dtype=np.complex128)

    for lam2 in range(0, lamb_max + 1):
        U2 = U_lam(lam2, k, p)**2
        pref2 = (2.0*lam2 + 1.0)

        lam1_min = abs(lam - lam2)
        lam1_max_eff = min(lamb_max, lam + lam2)

        for lam1 in range(lam1_min, lam1_max_eff + 1):
            if (lam + lam2 + lam1) % 2 != 0:
                continue
            tj2 = threej0_sq(lam, lam2, lam1)
            if tj2 == 0.0:
                continue

            pref = (1.0/(4.0*np.pi)) * (2.0*lam1 + 1.0) * pref2 * tj2
            denom = (w[:, None] - B*lam1*(lam1+1) - wk[None, :] + 1j*eta)
            Sig += pref * trapz_last_axis(U2[None, :] / denom, k)

    return Sig


def G_lam(w: np.ndarray, lam: int, Sig: np.ndarray, p: Dict[str, float | int]) -> np.ndarray:
    B = float(p["B"])
    return 1.0 / (w - B*lam*(lam+1) - Sig)




def spectral_A_from_G(G: np.ndarray) -> np.ndarray:
    """A(ω) = -(1/π) Im G(ω)."""
    return -(1.0 / np.pi) * np.imag(G)


def full_spectral_function(w: np.ndarray, k: np.ndarray, wk: np.ndarray, p: Dict[str, float | int]) -> np.ndarray:
    """Compute the FULL spectral function A_full(ω) by summing over λ.

    A_full(ω) = Σ_{λ=0}^{lambda_max_angulon} w_λ A_λ(ω),
    A_λ(ω) = -(1/π) Im G_λ(ω).
    Weight w_λ = (2λ+1) if spectral_weight_2lam1=1, else w_λ=1.
    """
    lamb_max = int(p.get("lambda_max_angulon", p.get("lambda_max", 8)))
    use_weight = int(p.get("spectral_weight_2lam1", 1)) == 1

    A_full = np.zeros_like(w, dtype=float)
    for lam in range(0, lamb_max + 1):
        Sig = Sigma_lam(w, lam, k, wk, p)
        G = G_lam(w, lam, Sig, p)
        A = spectral_A_from_G(G)
        weight = (2.0 * lam + 1.0) if use_weight else 1.0
        A_full += weight * A
    return A_full

# ======================================================================================
# Peak extraction
# ======================================================================================

def peak_from_spectral(w: np.ndarray, G: np.ndarray) -> float:
    A = spectral_A_from_G(G)
    idx = int(np.argmax(A))
    return float(w[idx])


def effective_generalities_w_grid(p: Dict[str, float | int]) -> np.ndarray:
    """Frequency grid large enough to include the bare level up to lambda_max_angulon."""
    w_min = float(p.get("w_min_02", p.get("w_min", -100.0)))
    w_max = float(p.get("w_max_02", p.get("w_max", 400.0)))
    Nw = int(p.get("Nw_02", p.get("N_w_02", p.get("Nw", 2000))))
    B = float(p["B"])
    lamb_max = int(p.get("lambda_max_angulon", p.get("lambda_max", 8)))
    w_bare_max = B * lamb_max * (lamb_max + 1)
    margin = max(2.0, 0.05 * max(1.0, abs(w_bare_max)))
    w_hi_eff = max(w_max, w_bare_max + margin)
    return np.linspace(w_min, w_hi_eff, Nw)


# ======================================================================================
# HDF5
# ======================================================================================

def save_h5(path: Path, arrays: Dict[str, np.ndarray], attrs: Dict[str, float | int]) -> None:
    if not _H5PY_OK:
        print("h5py not available; skipping HDF5 save.")
        return
    import h5py  # type: ignore
    with open_h5(h5py, path, "w") as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v


# ======================================================================================
# __main__
# ======================================================================================

def plot_couplings_and_chi() -> None:
    ensure_dirs()
    p = DEFAULTS.copy()
    k = np.linspace(float(p["k_min"]), float(p["k_max"]), int(p["Nk"]))
    w = np.linspace(
        float(p.get("w_min_02", p.get("w_min", -100.0))),
        float(p.get("w_max_02", p.get("w_max", 400.0))),
        int(p.get("Nw_02", p.get("N_w_02", p.get("Nw", 2000)))),
    )
    wk = omega_k(k, p)

    # Plot bath dispersion
    plt.figure()
    plt.plot(k, wk)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\omega_k$")
    # plt.title(rf"Bath dispersion ($\mathrm{{model}}={int(p['bath_model'])}$)")
    params.save_pdf(FIG_DIR / "bath_dispersion.pdf", apply_tight_layout=False)

    # Plot couplings for a few λ
    for lam in range(0, 3):
        U = U_lam(lam, k, p)
        plt.figure()
        plt.plot(k, U)
        plt.xlabel(r"$k$ (A$^{-1}$)")
        plt.ylabel(rf"$U_{{\lambda={lam}}}(k) (GHz.A)$")
        # plt.title(rf"Coupling $U_\lambda(k)$")
        params.save_pdf(FIG_DIR / f"U_lambda_{lam}.pdf", apply_tight_layout=False)

        Nr = int(p.get("Nr", 1600))
        rlam = _r_lam(lam, p)
        r_cut = float(p.get("r_cut_factor", 10.0))
        r = np.linspace(rlam, r_cut * rlam, Nr)
        Ur = U_lam_r(lam, r, p)
        plt.figure()
        plt.plot(r, Ur)
        ax = plt.gca()
        plt.xlabel(r"$r\;(\mathrm{\AA})$")
        plt.ylabel(rf"$U_{{\lambda={lam}}}(r) (GHz)$")
        ghz_per_meV = 242.0
        secax = ax.secondary_yaxis(
            "right",
            functions=(lambda y: y / ghz_per_meV, lambda y: y * ghz_per_meV),
        )
        secax.set_ylabel(rf"$U_{{\lambda={lam}}}(r) (meV)$")
        # plt.title(rf"Radial coupling profile $U_\lambda(r)$")
        params.save_pdf(FIG_DIR / f"U_lambda_r_{lam}.pdf", apply_tight_layout=False)

    # χ_λ(ω) (diagnostic)
    for lam in range(0, 3):
        chi = chi_lam(w, lam, k, wk, p)
        plot_complex_vs_w(w, chi, title=rf"$\chi_\lambda(\omega)$, $\lambda={lam}$", out_prefix=FIG_DIR / f"chi_lam{lam}")


def plot_sigma_green_and_levels() -> None:
    ensure_dirs()
    p = DEFAULTS.copy()
    k = np.linspace(float(p["k_min"]), float(p["k_max"]), int(p["Nk"]))
    w = effective_generalities_w_grid(p)
    wk = omega_k(k, p)

    # Σ_λ(ω) and dressed G_λ(ω)
    lam_plot_max_default = int(p.get("lambda_max_angulon", p.get("lambda_max", 8)))
    Sigmas: List[np.ndarray] = []
    Gs: List[np.ndarray] = []

    B = float(p["B"])
    w_lo = float(np.min(w))
    w_hi = float(np.max(w))
    lam_plot_max = lam_plot_max_default
    w_bare_max = B * lam_plot_max * (lam_plot_max + 1)
    w_max_02 = float(p.get("w_max_02", p.get("w_max", 400.0)))
    if w_hi > w_max_02:
        print(
            f"Extended w-window upper bound from {w_max_02:.4g} to {w_hi:.4g} "
            f"to include the bare level B*lambda(lambda+1)={w_bare_max:.4g} for lambda_max_angulon={lam_plot_max}.",
            flush=True,
        )

    w_star = np.zeros(lam_plot_max + 1)
    Gamma = np.zeros(lam_plot_max + 1)

    for lam in range(0, lam_plot_max + 1):
        print(f"  02b: lambda={lam}/{lam_plot_max}", flush=True)
        Sig = Sigma_lam(w, lam, k, wk, p)
        G = G_lam(w, lam, Sig, p)

        Sigmas.append(Sig)
        Gs.append(G)

        plot_complex_vs_w(w, Sig, title=rf"$\Sigma_\lambda(\omega)$, $\lambda={lam}$", out_prefix=FIG_DIR / f"Sigma_lam{lam}")
        plot_complex_vs_w(w, G, title=rf"$G_\lambda(\omega)$, $\lambda={lam}$", out_prefix=FIG_DIR / f"G_lam{lam}")

        w_star[lam] = peak_from_spectral(w, G)
        Gamma[lam] = float(-2.0 * np.interp(w_star[lam], w, np.imag(Sig)))

    # Plot renormalised levels
    lams = np.arange(0, lam_plot_max + 1)
    valid_bstar = np.isfinite(w_star[:2]).all()
    plt.figure()
    plt.plot(lams, w_star, lw=2, label=r"$\omega_\lambda^\ast$")
    plt.plot(lams, B*lams*(lams+1), linestyle="--", lw=2, label=r"$B\lambda(\lambda+1)$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"Energy / frequency")
    # plt.title("Renormalised rotor levels from peaks of $-\\mathrm{Im}G$")
    if lam_plot_max >= 1 and valid_bstar:
        B_star = (w_star[1] - w_star[0]) / 2.0
        plt.text(
            0.04,
            0.96,
            rf"$B^* \approx {B_star:.4g}$",
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
        )
    plt.legend()
    params.save_pdf(FIG_DIR / "renormalised_levels.pdf", apply_tight_layout=False)

    # Estimate B*
    if lam_plot_max >= 1 and valid_bstar:
        lams_bstar = np.arange(0, lam_plot_max, dtype=float)
        B_star_local = (w_star[1:] - w_star[:-1]) / (2.0 * (lams_bstar + 1.0))
        plt.figure()
        plt.plot(lams_bstar, B_star_local, lw=2)
        plt.axhline(B, linestyle="--", color="k", alpha=0.6, label=r"$B$")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$B^*_\lambda$")
        plt.legend()
        params.save_pdf(FIG_DIR / "renormalised_Bstar_vs_lambda.pdf", apply_tight_layout=False)

        Delta = (w_star - w_star[0])/B - lams*(lams+1)
        plt.figure()
        plt.plot(lams, Delta, lw=2)
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\Delta_{\rm RLS}(\lambda)$")
        # plt.title("Differential rotational Lamb shift (from peaks)")
        params.save_pdf(FIG_DIR / "rotational_lamb_shift.pdf", apply_tight_layout=False)



def main_spectral_function() -> None:
    in_path = DATA_DIR / "A_full_vs_n.h5"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run compute_density_sweep.py first.")

    fig_dir = FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open_h5(h5py, in_path, "r") as f:
        n = f["n"][:]
        w = f["w"][:]
        A = f["A_full"][:]
    Amax = 10.0  # Cap large values for visualization   
    A[A>Amax] = Amax
    Ln = np.log(n)
    plt.figure()
    plt.pcolormesh(Ln, w, A.T, shading="auto")
    plt.xlabel(r"$Ln$")
    plt.ylabel(r"$\omega$")
    # plt.title(r"Full spectral function $A_{\rm full}(\omega;n)$")
    plt.colorbar(label=r"$A_{\rm full}$")
    plt.tight_layout()
    out_path = fig_dir / "Spectral_function_vs_n.pdf"
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"Saved: {out_path.resolve()}")


def main() -> None:
    ensure_dirs()
    removed = clean_fig_dir(FIG_DIR)
    print(f"Cleaned {removed} old file(s) in {FIG_DIR}", flush=True)
    p = DEFAULTS.copy()
    plot_couplings_and_chi()
    plot_sigma_green_and_levels()
    save_generalities_run_metadata(p)
    # main_spectral_function()


if __name__ == "__main__":
    main()
