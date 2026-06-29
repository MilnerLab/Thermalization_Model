#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export compact prediction traces for comparison with experimental data."""

from __future__ import annotations

from pathlib import Path
import importlib

import h5py
import numpy as np

from h5_locking import open_h5

params = importlib.import_module("01_Parameters")

CASE_NAMES = ("CS2", "CS2_renormalised", "OCS", "OCS_renormalised")
OUT_H5 = Path("Experimental_data") / "prediction_data.h5"
METADATA_KEYS = (
    "case_name",
    "case_tag",
    "B",
    "B_star",
    "D_star",
    "rotational_model",
    "Delta_alpha",
    "E0",
    "rotor_t_min",
    "rotor_t_max",
    "Nt_main",
    "rotor_acceleration_ramp",
    "rotor_frequency0",
    "rotor_phi0",
    "rotor_sigma",
    "T_K",
    "tau_steady_state",
    "tau_smooth",
    "degeneracy_tol",
    "thermal_trotter_steps",
    "compute_thermal_lab_frame",
    "compute_thermal_rot_frame",
    "Nt_plot_rotating_observables",
)


def prediction_h5_path(case_name: str) -> Path:
    defaults = params.get_defaults_for_case(case_name)
    return Path(str(defaults["data_dir_03_free_rotor_drive"])) / "observable_matrices.h5"


def write_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    group.create_dataset(
        name,
        data=np.asarray(data),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def write_metadata(group: h5py.Group, case_name: str, src_h5: h5py.File, src_path: Path) -> None:
    defaults = params.get_defaults_for_case(case_name)
    meta = group.create_group("metadata")
    meta.attrs["source_h5"] = str(src_path)
    for key in METADATA_KEYS:
        if key in defaults:
            value = defaults[key]
        elif key in src_h5.attrs:
            value = src_h5.attrs[key]
        else:
            continue
        if value is None:
            meta.attrs[key] = "None"
        else:
            meta.attrs[key] = value


def export_case(out: h5py.File, case_name: str) -> None:
    src_path = prediction_h5_path(case_name)
    if not src_path.exists():
        raise FileNotFoundError(f"Missing prediction file for {case_name}: {src_path}")

    with open_h5(h5py, src_path, "r") as src:
        grp = out.create_group(case_name)
        write_metadata(grp, case_name, src, src_path)

        model = grp.create_group("model_grid")
        write_dataset(model, "t", src["t_grid"][...])
        write_dataset(model, "Omega0", src["Omega0_t"][...])
        if "dense_rotating_observables/V0_t" in src:
            v0_model = np.interp(
                src["t_grid"][...].astype(float),
                src["dense_rotating_observables/t_grid"][...].astype(float),
                src["dense_rotating_observables/V0_t"][...].astype(float),
            )
            write_dataset(model, "V0", v0_model)
        elif "V0_t" in src:
            write_dataset(model, "V0", src["V0_t"][...])

        dense = src["dense_rotating_observables"]
        phase = grp.create_group("phase_grid")
        write_dataset(phase, "t", dense["t_grid"][...])
        write_dataset(phase, "Omega0", dense["Omega0_t"][...])
        write_dataset(phase, "V0", dense["V0_t"][...])
        write_dataset(phase, "Delta_phi", dense["Delta_phi_t"][...])

        pred = grp.create_group("prediction")
        write_dataset(pred, "t", src["t_grid"][...])
        write_dataset(pred, "cos2theta2D", src["thermal/steady_state/cos2theta2D"][...])


def main() -> None:
    OUT_H5.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(h5py, OUT_H5, "w") as out:
        out.attrs["description"] = "Compact steady-state cos2theta2D prediction data exported from pipeline HDF5 files."
        out.attrs["time_unit"] = "ns"
        out.attrs["Omega0_unit"] = "GHz"
        out.attrs["V0_unit"] = "GHz"
        out.attrs["Delta_phi_unit"] = "rad"
        for case_name in CASE_NAMES:
            export_case(out, case_name)
            print(f"Exported {case_name}", flush=True)
    print(f"Wrote: {OUT_H5}", flush=True)


if __name__ == "__main__":
    main()
