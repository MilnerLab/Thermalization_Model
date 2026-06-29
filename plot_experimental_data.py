#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot experimental cos^2(theta_2D) traces against model predictions."""

from __future__ import annotations

from pathlib import Path
import importlib
import os

import numpy as np
import csv


params = importlib.import_module("01_Parameters")
import h5py
import matplotlib.pyplot as plt
FIGSIZE = params.FIGSIZE
plt.rcParams["figure.figsize"] = FIGSIZE
    

DATA_DIR = Path("Experimental_data")
PREDICTION_DATA_H5 = DATA_DIR / "prediction_data.h5"
PLOT_T_MIN_PS = -250.0
PLOT_T_MAX_PS = 250.0
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


def _label_from_case_name(case_name: str) -> str:
    parts = case_name.split("_")
    for i, part in enumerate(parts):
        if part.lower() in ("accel", "decel"):
            return " ".join(parts[i + 1:])
    return case_name


def prediction_variants_for_case(
    mol: str,
    direction: str,
    enabled_case_names: list[str] | None = None,
) -> list[tuple[str, str]]:
    if enabled_case_names is not None:
        matching = [
            cn for cn in enabled_case_names
            if _mol_dir_from_case_name(cn) == (mol, direction)
        ]
        if matching:
            return [(cn, _label_from_case_name(cn)) for cn in matching]
    return [(mol, "model")]


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


def _mol_and_renorm_from_pred_case(case_name: str) -> tuple[str | None, bool]:
    is_renorm = case_name.endswith("_renormalised")
    base = case_name[:-len("_renormalised")] if is_renorm else case_name
    if base in CASE_BY_TOKEN.values():
        return base, is_renorm
    base_upper = base.upper()
    for token, mol in CASE_BY_TOKEN.items():
        if base_upper.startswith(token):
            return mol, is_renorm
    return None, is_renorm


def prediction_signal_alpha(case_name: str, path: Path | None = None) -> float:
    if path is not None:
        dataset_alphas = PREDICTION_SIGNAL_ALPHA_BY_DATASET.get(Path(path).stem, {})
        if case_name in dataset_alphas:
            return float(dataset_alphas[case_name])
        mol, is_renorm = _mol_and_renorm_from_pred_case(case_name)
        if mol is not None:
            mol_key = f"{mol}_renormalised" if is_renorm else mol
            if mol_key in dataset_alphas:
                return float(dataset_alphas[mol_key])
    return float("nan")


def rescale_prediction_signal(signal: np.ndarray, alpha: float) -> np.ndarray:
    return float(alpha) * (np.asarray(signal, dtype=float) - 0.5) + 0.5


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
    params.save_png(path.with_name(f"{path.stem}_{case_name}_cos2theta2D_vs_t_with_model.png"))


def save_data_csv(
    path: Path,
    case_name: str,
    t_pred: np.ndarray,
    signal_pred: np.ndarray,
    reverse_prediction_time: bool = False,
) -> None:
    alpha = prediction_signal_alpha(case_name, path)
    t_out, signal_out = transform_prediction_trace(t_pred, signal_pred, reverse_prediction_time, alpha)
    with Path(path.with_name(f"{path.stem}_{case_name}_cos2theta2D_vs_t_model_only.csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for d, m, v in zip(t_pred * 1e3, signal_pred, signal_out):
            w.writerow([d, m, v])
        print("Data saved to:", path)


def _csv_mol_dir(path: Path) -> tuple[str, str] | None:
    mol = prediction_case_from_path(path)
    if mol is None:
        return None
    direction = "decel" if is_decelerating_path(path) else "accel"
    return (mol, direction)


def _mol_dir_from_case_name(case_name: str) -> tuple[str, str] | None:
    name_upper = case_name.upper()
    mol = next((CASE_BY_TOKEN[tok] for tok in CASE_BY_TOKEN if name_upper.startswith(tok)), None)
    if mol is None:
        return None
    if "_ACCEL_" in name_upper or name_upper.endswith("_ACCEL"):
        return (mol, "accel")
    if "_DECEL_" in name_upper or name_upper.endswith("_DECEL"):
        return (mol, "decel")
    return None


def load_enabled_case_names() -> list[str] | None:
    raw = os.environ.get("PENDULON_ENABLED_CASES", "")
    if raw:
        return [cn.strip() for cn in raw.split(",") if cn.strip()]
    try:
        pipeline = importlib.import_module("01_Pipeline")
        return [name for name, enabled in pipeline.CASE_ENABLED.items() if enabled]
    except Exception:
        return None  # can't determine — no filter, try all


def main() -> None:
    enabled_case_names = load_enabled_case_names()
    enabled_mol_dirs: set[tuple[str, str]] | None = None
    if enabled_case_names is not None:
        mds = {_mol_dir_from_case_name(cn) for cn in enabled_case_names} - {None}
        enabled_mol_dirs = mds if mds else None  # type: ignore[assignment]

    csv_paths = sorted({
        path
        for pattern in EXPERIMENTAL_DATA_PATTERNS
        for path in DATA_DIR.glob(pattern)
    })
    if enabled_mol_dirs is not None:
        csv_paths = [p for p in csv_paths if _csv_mol_dir(p) in enabled_mol_dirs]
    if not csv_paths:
        print("plot_experimental_data: no matching CSV files for enabled cases.", flush=True)
        return

    for path in csv_paths:
        t, signal, err = load_dataset(path)
        print(f"Processing {path.name}: {t.size} points", flush=True)
        mol = prediction_case_from_path(path)
        if mol is not None:
            reverse = is_decelerating_path(path)
            direction = "decel" if reverse else "accel"
            for i, (pred_case_name, label) in enumerate(
                prediction_variants_for_case(mol, direction, enabled_case_names)
            ):
                try:
                    t_pred, signal_pred = load_prediction_trace(pred_case_name)
                except FileNotFoundError as exc:
                    print(f"Skipping {label} for {path.name}: {exc}", flush=True)
                    continue
                color = PREDICTION_COLORS[i % len(PREDICTION_COLORS)]
                plot_trace_with_single_prediction(
                    path, t, signal, err, pred_case_name, label, t_pred, signal_pred, color,
                    reverse_prediction_time=reverse,
                )
                save_data_csv(
                    path=path,
                    case_name=pred_case_name,
                    t_pred=t_pred,
                    signal_pred=signal_pred,
                )


if __name__ == "__main__":
    main()
