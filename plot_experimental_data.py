#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot experimental cos^2(theta_2D) traces and local frequency maps from CSV files."""

from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np
import csv


params = importlib.import_module("01_Parameters")
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
FIGSIZE = params.FIGSIZE
plt.rcParams["figure.figsize"] = FIGSIZE
import pywt


DATA_DIR = Path("Experimental_data")
PREDICTION_DATA_H5 = DATA_DIR / "prediction_data.h5"
FREQ_MIN = 5.0
FREQ_MAX = 100.0
N_FREQ = 400
WAVELET = "cmor1.5-1.0"
FREQ_MAP_VMAX = 0.04
FREQ_MAP_VMIN = FREQ_MAP_VMAX / 10.0
PLOT_T_MIN_PS = -250.0
PLOT_T_MAX_PS = 250.0
ALPHA_FIT_T_MIN_PS = -150.0
ALPHA_FIT_T_MAX_PS = 150.0
FITTED_ROTATIONAL_CHIRP_BY_CASE = {
    "CS2": {
        "frequency0_ghz": 18.997041712122172,
        "ramp_mhz_per_ps": 92.08265680229961,
    },
    "OCS": {
        "frequency0_ghz": 22.498737093913505,
        "ramp_mhz_per_ps": 97.59025698477419,
    },
}
PREDICTION_TIME_SHIFT_PS_ACCELERATING = 0.0
PREDICTION_TIME_SHIFT_PS_DECELERATING = 0.0
EXPERIMENTAL_DATA_PATTERNS = (
    "*_accelerating_droplets.csv",
    "*_decelerating_droplets.csv",
)

PREDICTION_SIGNAL_ALPHA_BY_DATASET = {
    "CS2_accelerating_droplets": {
        "CS2": 0.072697915*1.5,
        "CS2_renormalised": 0.0449806687*1.5,
    },
    "CS2_decelerating_droplets": {
        "CS2": 0.067970648*1.5,
        "CS2_renormalised": 0.0434283417*1.5,
    },
    "OCS_accelerating_droplets": {
        "OCS": 0.554924647,
        "OCS_renormalised": 0.268849693,
    },
    "OCS_decelerating_droplets": {
        "OCS": 0.658379079,
        "OCS_renormalised": 0.321609377,
    },
}

CASE_BY_TOKEN = {
    "OCS": "OCS",
    "CS2": "CS2",
}

PREDICTION_COLORS = ["#c03a2b", "#2769a8"]

def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}")

    order = np.argsort(data[:, 0])
    t_raw = data[order, 0]
    t_ps = t_raw * 1e12 if np.nanmax(np.abs(t_raw)) < 1e-6 else t_raw
    mask = t_ps >= -250.0
    t = t_ps[mask] * 1e-3  # input ps stored as ns
    cos2theta2d = data[order, 1]
    cos2theta2d = cos2theta2d[mask]
    err = data[order, 2]
    err = err[mask]
    return t, cos2theta2d, err


def prediction_case_from_path(path: Path) -> str | None:
    name = path.stem.upper()
    if "DROPLET" not in name and "DROPLETS" not in name:
        return None
    for token, case_name in CASE_BY_TOKEN.items():
        if token in name:
            return case_name
    return None


def is_decelerating_path(path: Path) -> bool:
    return "DECELERATING" in path.stem.upper()


def fitted_cos2_signal_frequency_ghz(t_ps: np.ndarray, path: Path) -> np.ndarray:
    case_name = prediction_case_from_path(path)
    if case_name not in FITTED_ROTATIONAL_CHIRP_BY_CASE:
        raise ValueError(f"Could not determine fitted chirp parameters for {path}")
    fit = FITTED_ROTATIONAL_CHIRP_BY_CASE[case_name]
    sign = -1.0 if is_decelerating_path(path) else 1.0
    f_rot = float(fit["frequency0_ghz"]) + sign * float(fit["ramp_mhz_per_ps"]) * np.asarray(t_ps, dtype=float) / 1000.0
    return 2.0 * f_rot


def load_prediction_trace(case_name: str) -> tuple[np.ndarray, np.ndarray]:
    if not PREDICTION_DATA_H5.exists():
        raise FileNotFoundError(f"Missing compact prediction file: {PREDICTION_DATA_H5}. Run generate_prediction_data.py first.")
    with h5py.File(PREDICTION_DATA_H5, "r") as h5:
        if case_name not in h5:
            raise FileNotFoundError(f"Missing case '{case_name}' in compact prediction file: {PREDICTION_DATA_H5}")
        pred = h5[f"{case_name}/prediction"]
        t = pred["t"][...].astype(float)
        y = pred["cos2theta2D"][...].astype(float)
    return t, y


def prediction_variants_for_case(case_name: str) -> list[tuple[str, str]]:
    return [
        (case_name, "model"),
        (f"{case_name}_renormalised", "model renormalised"),
    ]


def maybe_reverse_prediction_time(t: np.ndarray, signal: np.ndarray, reverse_time: bool) -> tuple[np.ndarray, np.ndarray]:
    if not reverse_time:
        return t, signal
    t_reversed = -np.asarray(t, dtype=float)
    order = np.argsort(t_reversed)
    return t_reversed[order], np.asarray(signal)[order]


def prediction_time_shift_ps(reverse_prediction_time: bool) -> float:
    if reverse_prediction_time:
        return float(PREDICTION_TIME_SHIFT_PS_DECELERATING)
    return float(PREDICTION_TIME_SHIFT_PS_ACCELERATING)


def prediction_signal_alpha(case_name: str, path: Path | None = None) -> float:
    if path is not None:
        dataset_alphas = PREDICTION_SIGNAL_ALPHA_BY_DATASET.get(Path(path).stem, {})
        if case_name in dataset_alphas:
            return float(dataset_alphas[case_name])
    return float(PREDICTION_SIGNAL_ALPHA_BY_CASE[str(case_name)])


def rescale_prediction_signal(signal: np.ndarray, alpha: float) -> np.ndarray:
    return float(alpha) * (np.asarray(signal, dtype=float) - 0.5) + 0.5


def fit_prediction_signal_alpha(
    t: np.ndarray,
    signal: np.ndarray,
    err: np.ndarray,
    t_pred: np.ndarray,
    signal_pred: np.ndarray,
) -> float:
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    err = np.asarray(err, dtype=float)
    t_pred = np.asarray(t_pred, dtype=float)
    signal_pred = np.asarray(signal_pred, dtype=float)
    t_ps = t * 1e3
    overlap = (
        (t >= float(np.min(t_pred)))
        & (t <= float(np.max(t_pred)))
        & (t_ps >= ALPHA_FIT_T_MIN_PS)
        & (t_ps <= ALPHA_FIT_T_MAX_PS)
    )
    if np.count_nonzero(overlap) < 2:
        return float("nan")
    pred_interp = np.interp(t[overlap], t_pred, signal_pred)
    basis = pred_interp - 0.5
    target = signal[overlap] - 0.5
    weights = 1.0 / np.maximum(err[overlap], 1e-12) ** 2
    denom = float(np.sum(weights * basis**2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(weights * basis * target) / denom)


def plot_trace(path: Path, t: np.ndarray, signal: np.ndarray, err: np.ndarray) -> None:
    plt.figure(figsize=FIGSIZE)
    plt.fill_between(t*1e3, signal - err, signal + err, color="0.85", alpha=1.0, label="uncertainty")
    plt.plot(t*1e3, signal, color="black", lw=1.2, label=r"$\langle \cos^2(\theta_{2D}) \rangle$")
    plt.axhline(0.5, color="black", ls=":", lw=1.0)
    plt.xlabel(r"$t$ (ps)")
    plt.ylabel(r"$\cos^2(\theta_{2D})$")
    plt.xlim(PLOT_T_MIN_PS, PLOT_T_MAX_PS)
    plt.ylim(0.45, 0.65)
    # plt.title(path.stem.replace("_", " "))
    plt.legend()
    params.save_pdf(path.with_name(f"{path.stem}_cos2theta2D_vs_t.pdf"))


def transform_prediction_trace(
    t_pred: np.ndarray,
    signal_pred: np.ndarray,
    reverse_prediction_time: bool,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_out, signal_out = maybe_reverse_prediction_time(t_pred, signal_pred, reverse_prediction_time)
    t_out = np.asarray(t_out, dtype=float) + prediction_time_shift_ps(reverse_prediction_time) * 1e-3
    signal_out = rescale_prediction_signal(signal_out, alpha)
    return t_out, signal_out


def plot_trace_with_predictions(
    path: Path,
    t: np.ndarray,
    signal: np.ndarray,
    err: np.ndarray,
    predictions: list[tuple[str, str, np.ndarray, np.ndarray]],
    reverse_prediction_time: bool = False,
) -> None:
    transformed = []
    for case_name, label, t_pred, signal_pred in predictions:
        t_unscaled, signal_unscaled = maybe_reverse_prediction_time(t_pred, signal_pred, reverse_prediction_time)
        t_unscaled = np.asarray(t_unscaled, dtype=float) + prediction_time_shift_ps(reverse_prediction_time) * 1e-3
        alpha = prediction_signal_alpha(case_name, path)
        print(f"  alpha for {path.name} / {label}: {alpha:.6g}", flush=True)
        t_scaled, signal_scaled = transform_prediction_trace(t_pred, signal_pred, reverse_prediction_time, alpha)
        transformed.append((case_name, label, alpha, t_scaled, signal_scaled))
    y_all = np.concatenate([signal - err, signal + err] + [signal_pred for _, _, _, _, signal_pred in transformed])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    pad = max(0.01, 0.08 * (y_max - y_min))
    plt.figure(figsize=FIGSIZE)
    plt.fill_between(t*1e3, signal - err, signal + err, color="0.85", alpha=1.0)
    plt.plot(t*1e3, signal, color="black", lw=1.2, label="experiment")
    colors = PREDICTION_COLORS
    for i, (case_name, label, alpha, t_pred, signal_pred) in enumerate(transformed):
        plt.plot(
            t_pred * 1e3,
            signal_pred,
            color=colors[i % len(colors)],
            ls="-",
            alpha=0.5,
            lw=1.6,
            label=rf"{label}, $\alpha={alpha:g}$",
        )
    plt.axhline(0.5, color="black", ls=":", lw=1.0)
    plt.xlabel(r"$t$ (ps)")
    plt.ylabel(r"$\cos^2(\theta_{2D})$")
    plt.xlim(PLOT_T_MIN_PS, PLOT_T_MAX_PS)
    plt.ylim(y_min - pad, y_max + pad)
    plt.legend(loc="upper left")
    params.save_pdf(path.with_name(f"{path.stem}_cos2theta2D_vs_t_with_predictions.pdf"))


def _prediction_file_suffix(label: str) -> str:
    return "_renormalised" if "renormalised" in label.lower() else ""


def plot_prediction_only(
    path: Path,
    case_name: str,
    label: str,
    t_pred: np.ndarray,
    signal_pred: np.ndarray,
    color: str,
    reverse_prediction_time: bool = False,
) -> None:
    alpha = prediction_signal_alpha(case_name, path)
    t_out, signal_out = transform_prediction_trace(t_pred, signal_pred, reverse_prediction_time, alpha)
    y_min = float(np.min(signal_out))
    y_max = float(np.max(signal_out))
    pad = max(0.01, 0.08 * (y_max - y_min))
    plt.figure(figsize=FIGSIZE)
    plt.plot(t_out * 1e3, signal_out, color=color, ls="-", alpha=0.7, lw=1.6, label=rf"{label}, $\alpha={alpha:g}$")
    plt.axhline(0.5, color="black", ls=":", lw=1.0)
    plt.xlabel(r"$t$ (ps)")
    plt.ylabel(r"$\cos^2(\theta_{2D})$")
    plt.xlim(PLOT_T_MIN_PS, PLOT_T_MAX_PS)
    plt.ylim(y_min - pad, y_max + pad)
    plt.legend(loc="upper left")
    suffix = _prediction_file_suffix(label)
    params.save_pdf(path.with_name(f"{path.stem}_cos2theta2D_vs_t_prediction{suffix}.pdf"))


def plot_trace_with_single_prediction(
    path: Path,
    t: np.ndarray,
    signal: np.ndarray,
    err: np.ndarray,
    case_name: str,
    label: str,
    t_pred: np.ndarray,
    signal_pred: np.ndarray,
    color: str,
    reverse_prediction_time: bool = False,
) -> None:
    alpha = prediction_signal_alpha(case_name, path)
    t_out, signal_out = transform_prediction_trace(t_pred, signal_pred, reverse_prediction_time, alpha)
    y_all = np.concatenate([signal - err, signal + err, signal_out])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    pad = max(0.01, 0.08 * (y_max - y_min))
    plt.figure(figsize=FIGSIZE)
    plt.fill_between(t * 1e3, signal - err, signal + err, color="0.85", alpha=1.0)
    plt.plot(t * 1e3, signal, color="black", lw=1.2, label="experiment")
    plt.plot(t_out * 1e3, signal_out, color=color, ls="-", alpha=0.5, lw=1.6, label=rf"{label}, $\alpha={alpha:g}$")
    plt.axhline(0.5, color="black", ls=":", lw=1.0)
    plt.xlabel(r"$t$ (ps)")
    plt.ylabel(r"$\cos^2(\theta_{2D})$")
    plt.xlim(PLOT_T_MIN_PS, PLOT_T_MAX_PS)
    plt.ylim(y_min - pad, y_max + pad)
    plt.legend(loc="upper left")
    suffix = _prediction_file_suffix(label)
    params.save_pdf(path.with_name(f"{path.stem}_cos2theta2D_vs_t_with_model{suffix}.pdf"))


def compute_frequency_map(path: Path, t: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t.size < 2:
        raise ValueError(f"Need at least 2 time samples for wavelet map: {path}")

    dt = float(np.median(np.diff(t)))
    if dt <= 0.0:
        raise ValueError(f"Non-positive time spacing in {path}")

    signal_centered = signal - np.mean(signal)

    central_freq = pywt.central_frequency(WAVELET)
    freq_max_allowed = 0.95 * central_freq / dt
    freq_max = min(FREQ_MAX, freq_max_allowed)
    if freq_max <= FREQ_MIN:
        raise ValueError(
            f"Time step too large for requested wavelet range in {path}: "
            f"dt={dt:.6g}, max allowed frequency={freq_max_allowed:.6g}"
        )

    freqs = np.linspace(FREQ_MIN, freq_max, N_FREQ)
    scales = central_freq / (freqs * dt)
    coefs, _ = pywt.cwt(signal_centered, scales, WAVELET, sampling_period=dt)
    amp = np.abs(coefs)
    return t * 1e3, freqs, amp

def save_data_csv(
    path: Path,
    case_name: str,
    label: str,
    t_pred: np.ndarray,
    signal_pred: np.ndarray,    
    reverse_prediction_time: bool = False,
) -> None:
    alpha = prediction_signal_alpha(case_name, path)
    t_out, signal_out = transform_prediction_trace(t_pred, signal_pred, reverse_prediction_time, alpha)
 #   params.save_pdf(path.with_name(f"{path.stem}_cos2theta2D_vs_t_with_model{suffix}.pdf"))
    suffix = _prediction_file_suffix(label) 
    with Path(path.with_name(f"{path.stem}_cos2theta2D_vs_t_with_model{suffix}.csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for d, m,v in zip(t_pred*1e3, signal_pred,signal_out):
                w.writerow([d, m,v])
        print("Data saved to:",path)

def plot_frequency_map(
    path: Path,
    t: np.ndarray,
    signal: np.ndarray,
    frequency_map_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frequency_map_data is None:
        t_ps, freqs, amp = compute_frequency_map(path, t, signal)
    else:
        t_ps, freqs, amp = frequency_map_data
    plt.figure(figsize=FIGSIZE)
    plt.pcolormesh(
        t_ps,
        freqs,
        amp,
        shading="auto",
        rasterized=True,
        norm=LogNorm(vmin=FREQ_MAP_VMIN, vmax=FREQ_MAP_VMAX),
        cmap="magma",
    )
    try:
        fitted_t_ps = np.linspace(PLOT_T_MIN_PS, PLOT_T_MAX_PS, 400)
        plt.plot(
            fitted_t_ps,
            fitted_cos2_signal_frequency_ghz(fitted_t_ps, path),
            color="white",
            lw=1.6,
            ls="-",
            label=r"fit $2\Omega_\mathrm{rot}(t)$",
        )
    except ValueError:
        pass
    plt.xlabel("Time (ps)")
    plt.ylabel("Frequency (GHz)")
    plt.xlim(PLOT_T_MIN_PS, PLOT_T_MAX_PS)
    plt.ylim(0.0, float(np.max(freqs)))
    # plt.title(f"{path.stem.replace('_', ' ')}: local Fourier amplitude")
    plt.colorbar(label="amplitude")
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(loc="upper left", fontsize=8)
    params.save_pdf(path.with_name(f"{path.stem}_frequency_map_vs_t.pdf"))
    return t_ps, freqs, amp


def main() -> None:
    csv_paths = sorted(
        {
            path
            for pattern in EXPERIMENTAL_DATA_PATTERNS
            for path in DATA_DIR.glob(pattern)
        }
    )
    if len(csv_paths) != 4:
        raise FileNotFoundError(f"Expected 4 new accelerating/decelerating CSV files in {DATA_DIR}, found {len(csv_paths)}")

    for path in csv_paths:
        t, signal, err = load_dataset(path)
        print(f"Processing {path.name}: {t.size} points", flush=True)
        plot_trace(path, t, signal, err)
        frequency_map_data = compute_frequency_map(path, t, signal)
        plot_frequency_map(path, t, signal, frequency_map_data=frequency_map_data)
        case_name = prediction_case_from_path(path)
        if case_name is not None:
            predictions = []
            for pred_case_name, label in prediction_variants_for_case(case_name):
                try:
                    t_pred, signal_pred = load_prediction_trace(pred_case_name)
                except FileNotFoundError as exc:
                    print(f"Skipping {label} overlay for {path.name}: {exc}", flush=True)
                else:
                    predictions.append((pred_case_name, label, t_pred, signal_pred))
            if predictions:
                reverse = is_decelerating_path(path)
                plot_trace_with_predictions(
                    path,
                    t,
                    signal,
                    err,
                    predictions,
                    reverse_prediction_time=reverse,
                )
                for i, (pred_case_name, label, t_pred, signal_pred) in enumerate(predictions):
                    color = PREDICTION_COLORS[i % len(PREDICTION_COLORS)]
                    plot_prediction_only(
                        path, pred_case_name, label, t_pred, signal_pred, color,
                        reverse_prediction_time=reverse,
                    )
                    plot_trace_with_single_prediction(
                        path, t, signal, err, pred_case_name, label, t_pred, signal_pred, color,
                        reverse_prediction_time=reverse,
                    )
                    save_data_csv(    
                        path = path,
                        case_name = pred_case_name,
                        label = label,
                        t_pred = t_pred,
                        signal_pred= signal_pred,    
                                  )


if __name__ == "__main__":
    main()
