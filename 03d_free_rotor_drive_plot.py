#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03d_free_rotor_drive_plot.py

Load the free-rotor + drive diagonalization data from 03a and generate plots:
  - bare and drive Hamiltonian matrices,
  - renormalized energies compared to bare rotor levels,
  - mixing diagnostics for the low-lying eigenvectors.
"""

from __future__ import annotations

from pathlib import Path
import importlib

import h5py
import numpy as np
params = importlib.reload(importlib.import_module("01_Parameters"))
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from h5_locking import open_h5

FIGSIZE = params.FIGSIZE
plt.rcParams["figure.figsize"] = FIGSIZE


def get_defaults() -> dict:
    mod = importlib.reload(importlib.import_module("01_Parameters"))
    return dict(mod.get_defaults_for_case())


def get_paths(p: dict) -> tuple[Path, Path, Path, Path, Path]:
    fig_dir = Path(str(p.get("fig_dir_03_free_rotor_drive", "figs/03_free_rotor_drive")))
    tech_fig_dir = Path(str(p.get("fig_tech_dir_03_free_rotor_drive", "figs/03_free_rotor_drive/figs_technical")))
    lab_fig_dir = fig_dir / "thermal_average_lab_frame"
    rot_fig_dir = fig_dir / "thermal_average_rot_frame"
    data_h5 = Path(str(p.get("data_dir_03_free_rotor_drive", "data/03_free_rotor_drive"))) / "free_rotor_drive_diagonalization.h5"
    return fig_dir, tech_fig_dir, lab_fig_dir, rot_fig_dir, data_h5


def get_obs_h5(p: dict) -> Path:
    return Path(str(p.get("data_dir_03_free_rotor_drive", "data/03_free_rotor_drive"))) / "observable_matrices.h5"


def ensure_dirs(fig_dir: Path, tech_fig_dir: Path, lab_fig_dir: Path, rot_fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    tech_fig_dir.mkdir(parents=True, exist_ok=True)
    lab_fig_dir.mkdir(parents=True, exist_ok=True)
    rot_fig_dir.mkdir(parents=True, exist_ok=True)


def clean_fig_dir(fig_dir: Path) -> int:
    removed = 0
    if not fig_dir.exists():
        return removed
    for child in fig_dir.iterdir():
        if not child.is_file():
            continue
        if child.suffix.lower() != ".pdf":
            continue
        child.unlink(missing_ok=True)
        removed += 1
    return removed


def plot_matrix_heatmap(mat: np.ndarray, outpath: Path, cbar_label: str, cmap: str = "RdBu_r") -> None:
    vmax = max(float(np.max(np.abs(mat))), 1e-14)
    plt.figure(figsize=FIGSIZE)
    plt.pcolormesh(
        np.arange(mat.shape[1] + 1) - 0.5,
        np.arange(mat.shape[0] + 1) - 0.5,
        np.real(mat),
        shading="flat",
        rasterized=True,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    plt.xlabel("basis index b")
    plt.ylabel("basis index a")
    plt.colorbar(label=cbar_label)
    params.save_pdf(outpath)


def add_half_reference_line(stem_or_key: str) -> None:
    token = str(stem_or_key)
    if any(name in token for name in ("cos2theta2d", "cos2theta2D", "cos2phi_rot", "sin2theta")):
        plt.axhline(0.5, color="k", ls=":", lw=1)


def plot_rotational_model_comparison(p: dict, outpath: Path) -> None:
    if int(p.get("rotational_model", 1)) != 2:
        return
    b0 = float(p["B"])
    b_star_raw = p.get("B_star", None)
    d_star_raw = p.get("D_star", None)
    if b_star_raw is None or d_star_raw is None:
        return
    b_star = float(b_star_raw)
    d_star = float(d_star_raw)
    j_max = max(0, int(p.get("J_max", 0)))
    js = np.arange(j_max + 1, dtype=float)
    x = js * (js + 1.0)
    e_model_1 = b0 * x
    e_model_2 = np.asarray(params.rotational_energy_levels(js, p), dtype=float)
    e_naive = b_star * x - d_star * x * x

    plt.figure(figsize=FIGSIZE)
    plt.plot(js, e_model_2, lw=2, label="Model 2")
    plt.plot(js, e_model_1, lw=2, ls="--", label="Model 1")
    plt.plot(js, e_naive, lw=2, ls=":", label=r"$B_* J(J+1) - D_* [J(J+1)]^2$")
    plt.xlabel(r"$J$")
    plt.ylabel(r"$E_J\;[\mathrm{GHz}]$")
    plt.legend(fontsize=8)
    params.save_pdf(outpath)


def curve_crossings_linear(x: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    crossings: list[tuple[float, float]] = []
    if x.size < 2 or y.size != x.size:
        return crossings
    for i in range(x.size - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == 0.0:
            crossings.append((float(x[i]), 0.0))
            continue
        if y0 * y1 > 0.0:
            continue
        frac = -y0 / (y1 - y0)
        x_cross = float(x[i] + frac * (x[i + 1] - x[i]))
        crossings.append((x_cross, 0.0))
    return crossings


def main() -> None:
    p = get_defaults()
    fig_dir, tech_fig_dir, lab_fig_dir, rot_fig_dir, data_h5 = get_paths(p)
    obs_h5 = get_obs_h5(p)
    compute_lab_requested = bool(p.get("compute_thermal_lab_frame", True))
    compute_rot_requested = bool(p.get("compute_thermal_rot_frame", True))
    ensure_dirs(fig_dir, tech_fig_dir, lab_fig_dir, rot_fig_dir)
    removed_main = clean_fig_dir(fig_dir)
    removed_tech = clean_fig_dir(tech_fig_dir)
    removed_lab = clean_fig_dir(lab_fig_dir) if compute_lab_requested else 0
    removed_rot = clean_fig_dir(rot_fig_dir) if compute_rot_requested else 0
    print(f"Cleaned {removed_main} old file(s) in {fig_dir}", flush=True)
    print(f"Cleaned {removed_tech} old file(s) in {tech_fig_dir}", flush=True)
    if compute_lab_requested:
        print(f"Cleaned {removed_lab} old file(s) in {lab_fig_dir}", flush=True)
    if compute_rot_requested:
        print(f"Cleaned {removed_rot} old file(s) in {rot_fig_dir}", flush=True)

    if not data_h5.exists():
        raise FileNotFoundError(f"Missing data file: {data_h5}. Run 03a_free_rotor_drive_compute.py first.")

    with open_h5(h5py, data_h5, "r") as h5:
        t_grid = h5["t_grid"][...].astype(float)
        omega0_t = h5["Omega0_t"][...].astype(float)
        v0_t = h5["V0_t"][...].astype(float)
        js = h5["J"][...].astype(int)
        ms = h5["M"][...].astype(int)
        h0 = h5["H0_re"][...].astype(float) + 1j * h5["H0_im"][...].astype(float)
        h_drive = h5["H_drive_re"][...].astype(float) + 1j * h5["H_drive_im"][...].astype(float)
        evals = h5["E_eval"][...].astype(float)
        evecs = h5["U_evec_re"][...].astype(float) + 1j * h5["U_evec_im"][...].astype(float)
        b_rot = float(h5.attrs["B"])
        even_j_only = int(h5.attrs.get("antipodal_even_J_only", 0))
    if even_j_only != 1:
        raise RuntimeError("03c expects 03a data built in the antipodally symmetric even-J subspace.")

    idx_list = list(dict.fromkeys([0, t_grid.size // 2, t_grid.size - 1]))
    if h0.ndim == 2:
        h0 = np.broadcast_to(h0[None, :, :], h_drive.shape).copy()
    for it in idx_list:
        tag = f"t_{t_grid[it]*1e3:.3g}ps_Omega0_{omega0_t[it]:.4g}".replace(".", "p")
        plot_matrix_heatmap(h0[it], tech_fig_dir / f"hamiltonian_bare_matrix_{tag}.pdf", r"$\Re H_{0,ab}(t)$")
    for it in idx_list:
        tag = f"t_{t_grid[it]*1e3:.3g}ps_Omega0_{omega0_t[it]:.4g}".replace(".", "p")
        plot_matrix_heatmap(h_drive[it], tech_fig_dir / f"hamiltonian_drive_matrix_{tag}.pdf", r"$\Re \Pi_{ab}(t)$")

    x = t_grid * 1e3
    x_label = r"$t\;[\mathrm{ps}]$"
    n_plot = min(int(p.get("N_plot_modes", 5)), evals.shape[1])
    bare_levels = np.sort(np.linalg.eigvalsh(h0), axis=1)
    bare_plot = bare_levels[:, :n_plot]

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax1.plot(x, v0_t, lw=2, color="tab:blue")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(r"$V_0\;[\mathrm{GHz}]$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    omega_h_t = params.ho_frequency(float(b_rot), np.maximum(v0_t, 0.0))
    ax2.plot(x, omega0_t, lw=2, color="tab:red", label=r"$\Omega_0(t)$")
    ax2.plot(x, omega_h_t, lw=2, color="tab:green", label=r"$\omega_h(t)=2\sqrt{BV_0(t)}$")
    crossings = curve_crossings_linear(x, omega_h_t - omega0_t)
    for i_cross, (t_cross, _) in enumerate(crossings):
        omega_cross = float(np.interp(t_cross, x, omega0_t))
        ax2.axvline(t_cross, color="0.25", ls="--", lw=1.2)
        ax2.annotate(
            rf"$t={t_cross:.1f}\,\mathrm{{ps}}$",
            xy=(t_cross, omega_cross),
            xytext=(6, 8 + 16 * i_cross),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
            color="0.15",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=2.0),
        )
    ax2.set_ylabel(r"frequency $[\mathrm{GHz}]$", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right", fontsize=8)
    params.save_pdf(fig_dir / "parameters_drive_vs_t.pdf")

    plot_rotational_model_comparison(p, fig_dir / "rotational_model_comparison_vs_J.pdf")

    plt.figure(figsize=FIGSIZE)
    for n in range(n_plot):
        plt.plot(x, evals[:, n], lw=2, label=rf"$E_{{{n}}}^{{\rm dr}}$")
        plt.plot(x, bare_plot[:, n], color="0.4", ls="--", lw=2)
    plt.xlabel(x_label)
    plt.ylabel("energy (GHz)")
    if n_plot <= 10:
        plt.legend(fontsize=8, ncol=2)
    params.add_omega0_top_axis(plt.gca(), x, omega0_t)
    params.save_pdf(fig_dir / "renormalised_energies_vs_t.pdf")

    evals_rel = evals - evals[:, [0]]
    bare_rel = bare_plot - bare_plot[:, [0]]
    plt.figure(figsize=FIGSIZE)
    for n in range(min(n_plot, evals_rel.shape[1])):
        plt.plot(x, evals_rel[:, n], lw=2, label=rf"$E_{{{n}}}^{{\rm dr}}-E_0^{{\rm dr}}$")
        plt.plot(x, bare_rel[:, n], color="0.4", ls="--", lw=2)
    plt.xlabel(x_label)
    plt.ylabel("energy relative to ground (GHz)")
    params.add_omega0_top_axis(plt.gca(), x, omega0_t)
    params.save_pdf(fig_dir / "renormalised_energies_shifted_vs_t.pdf")

    n_plot_evec = min(int(p.get("N_plot_modes", 5)), evecs.shape[2])
    mix_ipr = np.zeros((t_grid.size, n_plot_evec), dtype=float)
    for it in range(t_grid.size):
        for n in range(n_plot_evec):
            prob = np.abs(evecs[it, :, n]) ** 2
            mix_ipr[it, n] = 1.0 / max(float(np.sum(prob**2)), 1e-14)

    plt.figure(figsize=FIGSIZE)
    for n in range(n_plot_evec):
        plt.plot(x, mix_ipr[:, n], lw=2, label=rf"state {n}")
    plt.xlabel(x_label)
    plt.ylabel("participation ratio")
    if n_plot_evec <= 10:
        plt.legend(fontsize=8)
    params.add_omega0_top_axis(plt.gca(), x, omega0_t)
    params.save_pdf(fig_dir / "eigenvector_mixing_participation_ratio_vs_t.pdf")

    for it in idx_list:
        prob = np.abs(evecs[it, :, :n_plot_evec]) ** 2
        plt.figure(figsize=FIGSIZE)
        plt.pcolormesh(
            np.arange(n_plot_evec + 1) - 0.5,
            np.arange(prob.shape[0] + 1) - 0.5,
            prob,
            shading="flat",
            rasterized=True,
            cmap="magma",
            vmin=0.0,
            vmax=max(float(np.max(prob)), 1e-12),
        )
        plt.xlabel("eigenstate index n")
        plt.ylabel("basis index a")
        plt.colorbar(label=r"$|\langle JM|\psi_n\rangle|^2$")
        tag = f"t_{t_grid[it]*1e3:.3g}ps_Omega0_{omega0_t[it]:.4g}".replace(".", "p")
        params.save_pdf(fig_dir / f"eigenvector_mixing_heatmap_{tag}.pdf")

    if not obs_h5.exists():
        print(f"03b data not found, skipping thermal observable plots: {obs_h5}", flush=True)
        return

    with open_h5(h5py, obs_h5, "r") as h5:
        lab_density_method = str(h5.attrs.get("lab_density_method", "trotter"))
        compute_lab = bool(int(h5.attrs.get("compute_thermal_lab_frame", 1)))
        compute_rot = bool(int(h5.attrs.get("compute_thermal_rot_frame", 1)))
        th_lab = {key: h5[f"thermal/lab_frame/{key}"][...].astype(float) for key in ["one", "x2", "y2", "sin2theta", "cos2phi_rot", "cos2phi_lab", "cos2theta2D"]}
        th_rot = {key: h5[f"thermal/rotating_frame/{key}"][...].astype(float) for key in ["one", "x2", "y2", "sin2theta", "cos2phi_rot", "cos2phi_lab", "cos2theta2D"]}
        has_ss = "steady_state" in h5["thermal"]
        has_ss_smooth = "steady_state_smooth" in h5["thermal"]
        if has_ss:
            th_ss = {key: h5[f"thermal/steady_state/{key}"][...].astype(float) for key in ["one", "x2", "y2", "sin2theta", "cos2phi_rot", "cos2phi_lab", "cos2theta2D"]}
        else:
            th_ss = {key: np.full_like(th_lab[key], np.nan) for key in th_lab}
        if has_ss_smooth:
            th_ss_smooth = {key: h5[f"thermal/steady_state_smooth/{key}"][...].astype(float) for key in ["one", "x2", "y2", "sin2theta", "cos2phi_rot", "cos2phi_lab", "cos2theta2D"]}
        else:
            th_ss_smooth = {key: np.full_like(th_lab[key], np.nan) for key in th_lab}
        obs_per_mode = {key: h5[f"observables_per_mode/{key}"][...].astype(float) for key in ["one", "x2", "y2", "sin2theta", "cos2phi_rot", "cos2phi_lab", "cos2theta2D"]}
        w_lab = h5["thermal/mode_weights/lab_frame"][...].astype(float)
        w_rot = h5["thermal/mode_weights/rotating_frame"][...].astype(float)
        w_ss = h5["thermal/mode_weights/steady_state"][...].astype(float) if "steady_state" in h5["thermal/mode_weights"] else np.full_like(w_lab, np.nan)
        j_values = h5["cutoff_J/J_values"][...].astype(int)
        j_imp_lab = h5["cutoff_J/importance_lab_frame"][...].astype(float)
        j_imp_rot = h5["cutoff_J/importance_rotating_frame"][...].astype(float)
        j_imp_ss = h5["cutoff_J/importance_steady_state"][...].astype(float) if "importance_steady_state" in h5["cutoff_J"] else np.full_like(j_imp_lab, np.nan)
        j_cum_lab = h5["cutoff_J/cumulative_lab_frame"][...].astype(float)
        j_cum_rot = h5["cutoff_J/cumulative_rotating_frame"][...].astype(float)
        j_cum_ss = h5["cutoff_J/cumulative_steady_state"][...].astype(float) if "cumulative_steady_state" in h5["cutoff_J"] else np.full_like(j_cum_lab, np.nan)
        trotter_n = h5["thermal/trotter/n_values"][...].astype(float)
        trotter_change = h5["thermal/trotter/change"][...].astype(float)
        if "dense_rotating_observables" in h5:
            grp_dense = h5["dense_rotating_observables"]
            t_dense = grp_dense["t_grid"][...].astype(float)
            omega0_dense = grp_dense["Omega0_t"][...].astype(float)
            dense_lab = {key: grp_dense["lab_frame"][key][...].astype(float) for key in ["cos2phi_lab", "cos2theta2D"]}
            dense_rot = {key: grp_dense["rotating_frame"][key][...].astype(float) for key in ["cos2phi_lab", "cos2theta2D"]}
            dense_ss = {key: grp_dense["steady_state"][key][...].astype(float) for key in ["cos2phi_lab", "cos2theta2D"]}
            dense_ss_smooth = {key: grp_dense["steady_state_smooth"][key][...].astype(float) for key in ["cos2phi_lab", "cos2theta2D"]} if "steady_state_smooth" in grp_dense else {}
        else:
            t_dense = t_grid
            omega0_dense = omega0_t
            dense_lab = {}
            dense_rot = {}
            dense_ss = {}
            dense_ss_smooth = {}

    thermal_defs = [
        ("one", th_lab["one"], th_rot["one"], th_ss["one"], th_ss_smooth["one"], r"$\langle 1 \rangle_T$", "observable_one_thermal"),
        ("x2", np.sqrt(np.clip(th_lab["x2"], 0.0, None)), np.sqrt(np.clip(th_rot["x2"], 0.0, None)), np.sqrt(np.clip(th_ss["x2"], 0.0, None)), np.sqrt(np.clip(th_ss_smooth["x2"], 0.0, None)), r"$\sqrt{\langle (\theta-\pi/2)^2 \rangle_T}$", "observable_x2_thermal"),
        ("y2", np.sqrt(np.clip(th_lab["y2"], 0.0, None)), np.sqrt(np.clip(th_rot["y2"], 0.0, None)), np.sqrt(np.clip(th_ss["y2"], 0.0, None)), np.sqrt(np.clip(th_ss_smooth["y2"], 0.0, None)), r"$\sqrt{\langle \phi_{\rm rot}^2 \rangle_T}$", "observable_y2_thermal"),
        ("sin2theta", th_lab["sin2theta"], th_rot["sin2theta"], th_ss["sin2theta"], th_ss_smooth["sin2theta"], r"$\langle \sin^2(\theta) \rangle_T$", "observable_sin2theta_thermal"),
        ("cos2phi_rot", th_lab["cos2phi_rot"], th_rot["cos2phi_rot"], th_ss["cos2phi_rot"], th_ss_smooth["cos2phi_rot"], r"$\langle \cos^2(\phi_{\rm rot}) \rangle_T$", "observable_cos2phi_rot_thermal"),
        ("cos2phi_lab", th_lab["cos2phi_lab"], th_rot["cos2phi_lab"], th_ss["cos2phi_lab"], th_ss_smooth["cos2phi_lab"], r"$\langle \cos^2(\phi_{\rm lab}) \rangle_T$", "observable_cos2phi_lab_thermal"),
        ("cos2theta2D", th_lab["cos2theta2D"], th_rot["cos2theta2D"], th_ss["cos2theta2D"], th_ss_smooth["cos2theta2D"], r"$\langle \cos^2(\theta_{2D}) \rangle_T$", "observable_cos2theta2d_thermal"),
    ]
    for key, series_lab, series_rot, series_ss, series_ss_dyn, ylab, stem in thermal_defs:
        if key in {"cos2phi_lab", "cos2theta2D"} and key in dense_ss:
            x_use = t_dense * 1e3
            omega_use = omega0_dense
            series_lab_use = dense_lab.get(key, series_lab)
            series_rot_use = dense_rot.get(key, series_rot)
            series_ss_use = dense_ss.get(key, series_ss)
            series_ss_dyn_use = dense_ss_smooth.get(key, series_ss_dyn)
        else:
            x_use = x
            omega_use = omega0_t
            series_lab_use = series_lab
            series_rot_use = series_rot
            series_ss_use = series_ss
            series_ss_dyn_use = series_ss_dyn
        if compute_lab:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_lab_use, lw=2)
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(lab_fig_dir / f"{stem}_vs_t.pdf")

        if compute_rot:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_rot_use, lw=2)
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(rot_fig_dir / f"{stem}_vs_t.pdf")

        if has_ss:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_ss_use, lw=2)
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(fig_dir / f"{stem}_vs_t.pdf")

        if has_ss_smooth:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_ss_dyn_use, lw=2)
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(fig_dir / f"{stem}_steady_state_smooth_vs_t.pdf")

        if has_ss and compute_lab:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_lab_use, lw=2, label="lab frame")
            plt.plot(x_use, series_ss_use, lw=2, ls=":", label="steady state")
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            plt.legend(fontsize=8)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(lab_fig_dir / f"{stem}_compare_to_steady_state_vs_t.pdf")
        if has_ss and compute_rot:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_rot_use, lw=2, label="rotating frame")
            plt.plot(x_use, series_ss_use, lw=2, ls=":", label="steady state")
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            plt.legend(fontsize=8)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(rot_fig_dir / f"{stem}_compare_to_steady_state_vs_t.pdf")

        if has_ss_smooth and compute_lab:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_lab_use, lw=2, label="lab frame")
            plt.plot(x_use, series_ss_dyn_use, lw=2, ls="--", label="steady state smooth")
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            plt.legend(fontsize=8)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(lab_fig_dir / f"{stem}_compare_to_steady_state_smooth_vs_t.pdf")
        if has_ss_smooth and compute_rot:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_rot_use, lw=2, label="rotating frame")
            plt.plot(x_use, series_ss_dyn_use, lw=2, ls="--", label="steady state smooth")
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            plt.legend(fontsize=8)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(rot_fig_dir / f"{stem}_compare_to_steady_state_smooth_vs_t.pdf")
        if has_ss and has_ss_smooth:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x_use, series_ss_use, lw=2, ls=":", label="steady state")
            plt.plot(x_use, series_ss_dyn_use, lw=2, ls="--", label="steady state smooth")
            add_half_reference_line(stem)
            plt.xlabel(x_label)
            plt.ylabel(ylab)
            if stem == "observable_one_thermal":
                plt.ylim(0.9, 1.1)
            elif "cos2" in stem or "sin2" in stem:
                plt.ylim(0.0, 1.0)
            plt.legend(fontsize=8)
            params.add_omega0_top_axis(plt.gca(), x_use, omega_use)
            params.save_pdf(fig_dir / f"{stem}_steady_state_compare_to_smooth_vs_t.pdf")

        plt.figure(figsize=FIGSIZE)
        n_mode_plot = min(obs_per_mode[key].shape[1], int(p.get("N_plot_modes", 5)))
        for n in range(n_mode_plot):
            plt.plot(x, obs_per_mode[key][:, n], lw=2, label=rf"mode {n}")
        add_half_reference_line(f"observable_{key.lower()}_per_mode")
        plt.xlabel(x_label)
        plt.ylabel(ylab.replace(r"\langle ", r"\langle ").replace(r"\rangle_T", r"\rangle_n"))
        if key == "one":
            plt.ylim(0.9, 1.1)
        elif key.startswith("cos2") or key.startswith("sin2"):
            plt.ylim(0.0, 1.0)
        plt.legend(fontsize=8)
        params.add_omega0_top_axis(plt.gca(), x, omega0_t)
        params.save_pdf(fig_dir / f"observable_{key.lower()}_per_mode_vs_t.pdf")

    def _plot_weights(weights_arr: np.ndarray, out_dir: Path, stem: str) -> None:
        plt.figure(figsize=FIGSIZE)
        plt.pcolormesh(x, np.arange(weights_arr.shape[1]), weights_arr.T, shading="auto", rasterized=True, cmap="magma", vmin=0.0, vmax=1.0)
        plt.xlabel(x_label)
        plt.ylabel("mode index")
        plt.colorbar(label="thermal weight")
        params.add_omega0_top_axis(plt.gca(), x, omega0_t)
        params.save_pdf(out_dir / f"{stem}_vs_t.pdf")

    if compute_lab:
        _plot_weights(w_lab, lab_fig_dir, "thermal_weights")
    if compute_rot:
        _plot_weights(w_rot, rot_fig_dir, "thermal_weights")
    if has_ss:
        _plot_weights(w_ss, fig_dir, "thermal_weights")

    def _plot_j_heatmap(j_imp: np.ndarray, out_dir: Path, stem: str, label: str) -> None:
        plt.figure(figsize=FIGSIZE)
        Xc = np.broadcast_to(x[:, None], j_imp.shape)
        Yc = np.broadcast_to(j_values[None, :], j_imp.shape)
        plt.pcolormesh(Xc, Yc, j_imp, shading="auto", rasterized=True, cmap="magma")
        plt.xlabel(x_label)
        plt.ylabel(r"$J$")
        plt.colorbar(label=label)
        params.add_omega0_top_axis(plt.gca(), x, omega0_t)
        params.save_pdf(out_dir / f"{stem}.pdf")

    if compute_lab:
        _plot_j_heatmap(j_imp_lab, lab_fig_dir, "cutoff_J_importance_vs_t", "J-shell importance")
    if compute_rot:
        _plot_j_heatmap(j_imp_rot, rot_fig_dir, "cutoff_J_importance_vs_t", "J-shell importance")
    if has_ss:
        _plot_j_heatmap(j_imp_ss, fig_dir, "cutoff_J_importance_vs_t", "J-shell importance")

    def _plot_j_tail_single(j_cum: np.ndarray, out_dir: Path, stem: str) -> None:
        plt.figure(figsize=FIGSIZE)
        for jcut in [min(int(j_values[-1]), 2), min(int(j_values[-1]), 4), int(j_values[-1])]:
            idx = int(np.argmin(np.abs(j_values - jcut)))
            plt.plot(x, 1.0 - j_cum[:, idx], lw=2, label=rf"$J>{j_values[idx]}$")
        plt.xlabel(x_label)
        plt.ylabel(r"tail weight above $J_{\rm cut}$")
        plt.ylim(0.0, 1.0)
        plt.legend(fontsize=8)
        params.add_omega0_top_axis(plt.gca(), x, omega0_t)
        params.save_pdf(out_dir / stem)

    if compute_lab:
        _plot_j_tail_single(j_cum_lab, lab_fig_dir, "cutoff_J_tail_weight_vs_t.pdf")
    if compute_rot:
        _plot_j_tail_single(j_cum_rot, rot_fig_dir, "cutoff_J_tail_weight_vs_t.pdf")
    if has_ss:
        _plot_j_tail_single(j_cum_ss, fig_dir, "cutoff_J_tail_weight_vs_t.pdf")
        if compute_lab:
            plt.figure(figsize=FIGSIZE)
            for jcut in [min(int(j_values[-1]), 2), min(int(j_values[-1]), 4), int(j_values[-1])]:
                idx = int(np.argmin(np.abs(j_values - jcut)))
                plt.plot(x, 1.0 - j_cum_lab[:, idx], lw=2, label=rf"lab, $J>{j_values[idx]}$")
                plt.plot(x, 1.0 - j_cum_ss[:, idx], lw=2, ls=":", label=rf"steady, $J>{j_values[idx]}$")
            plt.xlabel(x_label)
            plt.ylabel(r"tail weight above $J_{\rm cut}$")
            plt.ylim(0.0, 1.0)
            plt.legend(fontsize=8)
            params.add_omega0_top_axis(plt.gca(), x, omega0_t)
            params.save_pdf(lab_fig_dir / "cutoff_J_tail_weight_compare_to_steady_state_vs_t.pdf")

        if compute_rot:
            plt.figure(figsize=FIGSIZE)
            for jcut in [min(int(j_values[-1]), 2), min(int(j_values[-1]), 4), int(j_values[-1])]:
                idx = int(np.argmin(np.abs(j_values - jcut)))
                plt.plot(x, 1.0 - j_cum_rot[:, idx], lw=2, label=rf"rotating, $J>{j_values[idx]}$")
                plt.plot(x, 1.0 - j_cum_ss[:, idx], lw=2, ls=":", label=rf"steady, $J>{j_values[idx]}$")
            plt.xlabel(x_label)
            plt.ylabel(r"tail weight above $J_{\rm cut}$")
            plt.ylim(0.0, 1.0)
            plt.legend(fontsize=8)
            params.add_omega0_top_axis(plt.gca(), x, omega0_t)
            params.save_pdf(rot_fig_dir / "cutoff_J_tail_weight_compare_to_steady_state_vs_t.pdf")

    idx_list = list(dict.fromkeys([0, t_grid.size // 2, t_grid.size - 1]))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(idx_list), 2)))
    if has_ss and compute_lab:
        plt.figure(figsize=FIGSIZE)
        for ic, ii in enumerate(idx_list):
            color = colors[ic % colors.shape[0]]
            label_base = f"Omega0={omega0_t[ii]:.4g} GHz"
            plt.plot(np.arange(w_lab.shape[1]), w_lab[ii], lw=2, color=color, label=label_base + ", lab")
            plt.plot(np.arange(w_ss.shape[1]), w_ss[ii], lw=2, ls=":", color=color, label=label_base + ", steady")
        plt.xlabel("mode index n")
        plt.ylabel("thermal weight")
        plt.ylim(0.0, 1.0)
        plt.legend(fontsize=8)
        params.save_pdf(lab_fig_dir / "thermal_weights_vs_n_compare_to_steady_state_vs_t.pdf")
    if has_ss and compute_rot:
        plt.figure(figsize=FIGSIZE)
        for ic, ii in enumerate(idx_list):
            color = colors[ic % colors.shape[0]]
            label_base = f"Omega0={omega0_t[ii]:.4g} GHz"
            plt.plot(np.arange(w_rot.shape[1]), w_rot[ii], lw=2, color=color, label=label_base + ", rotating")
            plt.plot(np.arange(w_ss.shape[1]), w_ss[ii], lw=2, ls=":", color=color, label=label_base + ", steady")
        plt.xlabel("mode index n")
        plt.ylabel("thermal weight")
        plt.ylim(0.0, 1.0)
        plt.legend(fontsize=8)
        params.save_pdf(rot_fig_dir / "thermal_weights_vs_n_compare_to_steady_state_vs_t.pdf")

    if compute_lab:
        plt.figure(figsize=FIGSIZE)
        Xc = np.broadcast_to(x[None, :], trotter_change.shape)
        Yc = np.broadcast_to(trotter_n[:, None], trotter_change.shape)
        plt.pcolormesh(Xc, Yc, trotter_change, shading="auto", rasterized=True, cmap="magma")
        plt.xlabel(x_label)
        plt.ylabel(r"Trotter step $n$")
        if lab_density_method == "exact_diagonalization":
            cbar = r"$\|\rho_n^{\rm Trotter}-\rho_{\rm lab}^{\rm exact}\|_F / \|\rho_{\rm lab}^{\rm exact}\|_F$"
        else:
            cbar = r"$\|\rho_n-\rho_{n+1}\|_F / \|\rho_{n+1}\|_F$"
        plt.colorbar(label=cbar)
        params.add_omega0_top_axis(plt.gca(), x, omega0_t)
        params.save_pdf(lab_fig_dir / "cutoff_trotter_importance_vs_t.pdf")
        if trotter_change.shape[0] > 0:
            plt.figure(figsize=FIGSIZE)
            plt.plot(x, trotter_change[-1], lw=2)
            plt.xlabel(x_label)
            if lab_density_method == "exact_diagonalization":
                plt.ylabel("selected Trotter vs exact change")
            else:
                plt.ylabel("selected Trotter change")
            params.add_omega0_top_axis(plt.gca(), x, omega0_t)
            params.save_pdf(lab_fig_dir / "cutoff_trotter_selected_vs_t.pdf")

    def _save_freq_diag(series: np.ndarray, t_series: np.ndarray, omega_series: np.ndarray, out_dir: Path, stem: str) -> None:
        if t_series.size < 2:
            return
        import pywt
        t_local = np.asarray(t_series, dtype=float)
        omega_local = np.asarray(omega_series, dtype=float)
        t_u = np.linspace(float(t_local[0]), float(t_local[-1]), t_local.size)
        omega_u = np.interp(t_u, t_local, omega_local)
        signal_u = np.interp(t_u, t_local, np.asarray(series, dtype=float) - float(np.mean(series)))
        dt = t_u[1] - t_u[0]
        freqs = np.linspace(5.0, 2.0 * float(np.max(omega_u)) + 1e-12, 300)
        wavelet = "cmor1.5-1.0"
        scales = pywt.central_frequency(wavelet) / (freqs * dt)
        valid = np.isfinite(scales) & (scales >= 1.0)
        if not np.any(valid):
            return
        freqs = freqs[valid]
        scales = scales[valid]
        coefs, _ = pywt.cwt(signal_u, scales, wavelet, sampling_period=dt)
        amp = np.abs(coefs)
        amp_vmax = max(1e-12, float(np.max(amp)))
        amp_vmin = 1e-2 * amp_vmax
        amp_plot = np.clip(amp, amp_vmin, amp_vmax)

        plt.figure(figsize=FIGSIZE)
        plt.pcolormesh(t_u * 1e3, freqs, amp, shading="auto", rasterized=True, cmap="magma")
        plt.xlabel("time (ps)")
        plt.ylabel("frequency (GHz)")
        plt.colorbar(label="amplitude")
        params.add_omega0_top_axis(plt.gca(), t_u * 1e3, omega_u)
        params.save_pdf(out_dir / f"{stem}_linscale.pdf")

        plt.figure(figsize=FIGSIZE)
        plt.pcolormesh(t_u * 1e3, freqs, amp_plot, shading="auto", rasterized=True, cmap="magma", norm=LogNorm(vmin=amp_vmin, vmax=amp_vmax))
        plt.xlabel("time (ps)")
        plt.ylabel("frequency (GHz)")
        plt.colorbar(label="amplitude (log scale)")
        params.add_omega0_top_axis(plt.gca(), t_u * 1e3, omega_u)
        params.save_pdf(out_dir / f"{stem}_logscale.pdf")

    if compute_lab:
        _save_freq_diag(dense_lab.get("cos2theta2D", th_lab["cos2theta2D"]), t_dense if "cos2theta2D" in dense_lab else t_grid, omega0_dense if "cos2theta2D" in dense_lab else omega0_t, lab_fig_dir, "observable_cos2theta2d_thermal_frequency_diagram")
    if compute_rot:
        _save_freq_diag(dense_rot.get("cos2theta2D", th_rot["cos2theta2D"]), t_dense if "cos2theta2D" in dense_rot else t_grid, omega0_dense if "cos2theta2D" in dense_rot else omega0_t, rot_fig_dir, "observable_cos2theta2d_thermal_frequency_diagram")
    if has_ss:
        _save_freq_diag(dense_ss.get("cos2theta2D", th_ss["cos2theta2D"]), t_dense if "cos2theta2D" in dense_ss else t_grid, omega0_dense if "cos2theta2D" in dense_ss else omega0_t, fig_dir, "observable_cos2theta2d_thermal_frequency_diagram")
    if has_ss_smooth:
        _save_freq_diag(dense_ss_smooth.get("cos2theta2D", th_ss_smooth["cos2theta2D"]), t_dense if "cos2theta2D" in dense_ss_smooth else t_grid, omega0_dense if "cos2theta2D" in dense_ss_smooth else omega0_t, fig_dir, "observable_cos2theta2d_steady_state_smooth_frequency_diagram")


if __name__ == "__main__":
    main()
