from __future__ import annotations

import importlib
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import h5py  # type: ignore
    _H5PY_OK = True
except Exception:
    _H5PY_OK = False

if _H5PY_OK:
    from h5_locking import open_h5

STAGE_ORDER = [
    ("02B", "02b_bath_angulon.py"),
    ("01B", "01b_precompute_Ylm_blocks.py"),
    ("01C", "01c_precompute_steady_state.py"),
    ("03A", "03a_free_rotor_drive_compute.py"),
    ("03B", "03b_free_rotor_drive_observables.py"),
    ("03C", "03c_free_rotor_drive_thermal.py"),
    ("03D", "03d_free_rotor_drive_plot.py"),
]

# Non-case-dependent stages that run once after all cases
POST_STAGE_ORDER = [
    ("GEN_PRED", "generate_prediction_data.py"),
    ("PLOT_EXP", "plot_experimental_data.py"),
]

ALL_STAGE_CODES = [code for code, _ in STAGE_ORDER]
ALL_STAGE_SET = set(ALL_STAGE_CODES)
POST_STAGE_CODES = [code for code, _ in POST_STAGE_ORDER]
ALL_POST_STAGE_SET = set(POST_STAGE_CODES)
PLOT_STAGE_SET = {"03D", "GEN_PRED", "PLOT_EXP"}

ACTIVE_RUN_PLAN = "run_plan_full"
CASE_ENABLED: dict[str, bool] = {
    "Default": False,
    "CS2": False,
    "CS2_renormalised": False,
    "OCS": True,
    "OCS_renormalised": False,
}
CASE_STAGE_OVERRIDES: dict[str, dict[str, bool] | str] = {
    # Example:
    #"OCS_renormalised": {"01B": True, "01C": True},
}


def _plan_dict_from_enabled(enabled_codes: set[str], case_names: list[str]) -> dict[str, dict[str, bool]]:
    enabled = set(enabled_codes) & ALL_STAGE_SET
    return {case_name: {code: (code in enabled) for code in ALL_STAGE_CODES} for case_name in case_names}


def _post_plan_dict_from_enabled(enabled_codes: set[str]) -> dict[str, bool]:
    enabled = set(enabled_codes) & ALL_POST_STAGE_SET
    return {code: (code in enabled) for code in POST_STAGE_CODES}


def _build_run_plans(case_names: list[str]) -> dict[str, dict[str, dict[str, bool]]]:
    run_plan_custom = _plan_dict_from_enabled({"03C", "03D"}, case_names)
    run_plan_full = _plan_dict_from_enabled(set(ALL_STAGE_CODES), case_names)
    run_plan_02 = _plan_dict_from_enabled({"02B"}, case_names)
    run_plan_03 = _plan_dict_from_enabled({"01B", "01C", "03A", "03B", "03C", "03D"}, case_names)
    run_plan_plot = _plan_dict_from_enabled({"03B", "03C", "03D"}, case_names)
    return {
        "run_plan_custom": run_plan_custom,
        "run_plan_full": run_plan_full,
        "run_plan_02": run_plan_02,
        "run_plan_03": run_plan_03,
        "run_plan_plot": run_plan_plot,
    }


def _build_post_plans() -> dict[str, dict[str, bool]]:
    """Build post-processing plans (non-case-dependent stages) for each run plan."""
    post_plan_custom = _post_plan_dict_from_enabled(set())
    post_plan_full = _post_plan_dict_from_enabled({"GEN_PRED", "PLOT_EXP"})
    post_plan_02 = _post_plan_dict_from_enabled(set())
    post_plan_03 = _post_plan_dict_from_enabled(set())
    post_plan_plot = _post_plan_dict_from_enabled({"GEN_PRED", "PLOT_EXP"})
    return {
        "run_plan_custom": post_plan_custom,
        "run_plan_full": post_plan_full,
        "run_plan_02": post_plan_02,
        "run_plan_03": post_plan_03,
        "run_plan_plot": post_plan_plot,
    }


def _apply_case_controls(
    base_plan: dict[str, dict[str, bool]],
    case_names: list[str],
) -> dict[str, dict[str, bool]]:
    run_plans = _build_run_plans(case_names)
    out: dict[str, dict[str, bool]] = {}
    for case_name in case_names:
        case_plan = dict(base_plan[case_name])
        if not bool(CASE_ENABLED.get(case_name, False)):
            case_plan = {code: False for code in ALL_STAGE_CODES}
        override = CASE_STAGE_OVERRIDES.get(case_name, {})
        if isinstance(override, str):
            if override not in run_plans:
                available = ", ".join(sorted(run_plans.keys()))
                raise KeyError(f"Unknown per-case run plan '{override}' for case '{case_name}'. Available plans: {available}")
            case_plan = dict(run_plans[override][case_name])
        else:
            for code, flag in override.items():
                if code in ALL_STAGE_SET:
                    case_plan[code] = bool(flag)
        out[case_name] = case_plan
    return out


PARAMS_03B = (
    "B",
    "J_max",
    "n_theta_max",
    "n_phi_max",
    "n_theta_low",
    "n_phi_low",
    "N_theta",
    "N_phi",
    "rotor_t_min",
    "rotor_t_max",
    "Nt_main",
    "rotor_acceleration_ramp",
    "rotor_frequency0",
    "rotor_phi0",
    "rotor_sigma",
    "Delta_alpha",
    "E0",
)


def load_defaults() -> dict[str, object]:
    par = importlib.import_module("01_Parameters")
    defaults = getattr(par, "DEFAULTS", None)
    if not isinstance(defaults, dict):
        raise RuntimeError("Cannot load DEFAULTS from 01_Parameters.py")
    return dict(defaults)


PROJECT_MODULES = [
    "01_Parameters",
    "01b_precompute_Ylm_blocks",
    "01c_precompute_steady_state",
    "02a_superfluid_helium",
    "02b_bath_angulon",
    "02c_bath_angulon_data_density",
    "03a_free_rotor_drive_compute",
    "03b_free_rotor_drive_observables",
    "03c_free_rotor_drive_thermal",
    "03d_free_rotor_drive_plot",
]


def reset_project_modules() -> None:
    for name in PROJECT_MODULES:
        sys.modules.pop(name, None)


def load_defaults_for_case(case_name: str) -> dict[str, object]:
    os.environ["PENDULON_CASE"] = case_name
    reset_project_modules()
    return load_defaults()


def load_case_names() -> list[str]:
    os.environ.pop("PENDULON_CASE", None)
    reset_project_modules()
    par = importlib.import_module("01_Parameters")
    get_all = getattr(par, "get_all_case_names", None)
    if callable(get_all):
        return list(get_all())
    return []


def _coerce_scalar(x: object) -> object:
    if isinstance(x, np.generic):
        return x.item()
    return x


def _same_value(a: object, b: object) -> bool:
    a = _coerce_scalar(a)
    b = _coerce_scalar(b)

    if a is None or b is None:
        return a is b

    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-12))

    return a == b


def _data_root(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("path_data", Path(__file__).resolve().parent / "data")))


def _projection_path(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("projection_h5_path", _data_root(defaults) / "03_harmonic_oscillator" / "projection_HO_to_JM_vs_Omega0.h5")))


def _ylm_path(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("Ylm_h5_path", _data_root(defaults) / "01_spherical_harmonics" / "Ylm_blocks_JM.h5")))


def _observable_path(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("observable_h5_path", _data_root(defaults) / "03_harmonic_oscillator" / "observable_matrices_HO.h5")))


def _01c_path(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("steady_state_target_h5_path", _data_root(defaults) / "01_spherical_harmonics" / "steady_state_target.h5")))


def _04a_path(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("data_dir_04_renormalised_pendulon", _data_root(defaults) / "04_renormalised_pendulon"))) / "excited_subspace_dressed_modes.h5"


def _03a_path(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("data_dir_03_free_rotor_drive", _data_root(defaults) / "03_free_rotor_drive"))) / "free_rotor_drive_diagonalization.h5"


def _03b_path(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("data_dir_03_free_rotor_drive", _data_root(defaults) / "03_free_rotor_drive"))) / "observable_projections.h5"


def _03c_path(defaults: dict[str, object]) -> Path:
    return Path(str(defaults.get("data_dir_03_free_rotor_drive", _data_root(defaults) / "03_free_rotor_drive"))) / "observable_matrices.h5"


def _check_rotational_model_attrs(h5, defaults: dict[str, object], stage_code: str) -> tuple[bool, str | None]:
    rotational_model_expected = int(defaults.get("rotational_model", 1))
    rotational_model_old = int(h5.attrs.get("rotational_model", 1))
    if rotational_model_old != rotational_model_expected:
        return False, (
            f"rotational_model changed for {stage_code} "
            f"(stored={rotational_model_old}, current={rotational_model_expected})"
        )
    if rotational_model_expected == 2:
        b_star_expected = defaults.get("B_star", None)
        d_star_expected = defaults.get("D_star", None)
        if not _same_value(h5.attrs.get("B_star", None), b_star_expected):
            return False, (
                f"B_star changed for {stage_code} "
                f"(stored={h5.attrs.get('B_star', None)}, current={b_star_expected})"
            )
        if not _same_value(h5.attrs.get("D_star", None), d_star_expected):
            return False, (
                f"D_star changed for {stage_code} "
                f"(stored={h5.attrs.get('D_star', None)}, current={d_star_expected})"
            )
    return True, None


def check_03a_cache(defaults: dict[str, object]) -> tuple[bool, str]:
    if not _H5PY_OK:
        return False, "h5py unavailable, cannot validate cache"

    p = _projection_path(defaults)
    if not p.exists():
        return False, f"missing file: {p}"

    required_dsets = ("t_grid", "Omega0_t", "V0_t", "J", "M", "nx", "ny", "C_alpha_JM")
    try:
        par = importlib.import_module("01_Parameters")
        mod03a = importlib.import_module("03a_precompute_ho_projection_JM")
        traj = par.drive_grids(defaults)
        t_expected = np.asarray(traj["t"], dtype=float)
        omega_expected = np.asarray(traj["Omega0"], dtype=float)
        v0_expected = np.asarray(traj["V0"], dtype=float)
        states_expected = mod03a.build_state_list(int(defaults.get("J_max", 20)))
        js_expected = np.array([j for j, _ in states_expected], dtype=int)
        ms_expected = np.array([m for _, m in states_expected], dtype=int)
        modes_expected = mod03a.build_projection_mode_list(defaults)
        nx_expected = np.array([m.nx for m in modes_expected], dtype=int)
        ny_expected = np.array([m.ny for m in modes_expected], dtype=int)

        with open_h5(h5py, p, "r") as h5:
            for name in required_dsets:
                if name not in h5:
                    return False, f"missing dataset '{name}' in {p}"
            if int(h5.attrs.get("antipodal_even_J_only", 0)) != 1:
                return False, "03a data not marked as antipodally symmetric even-J basis"

            t_old = h5["t_grid"][...].astype(float)
            omega_old = h5["Omega0_t"][...].astype(float)
            v0_old = h5["V0_t"][...].astype(float)
            js_old = h5["J"][...].astype(int)
            ms_old = h5["M"][...].astype(int)
            nx_old = h5["nx"][...].astype(int)
            ny_old = h5["ny"][...].astype(int)

            if t_old.shape != t_expected.shape or not np.allclose(t_old, t_expected, rtol=0.0, atol=1e-12):
                return False, "time trajectory changed for 03b"
            if omega_old.shape != omega_expected.shape or not np.allclose(omega_old, omega_expected, rtol=0.0, atol=1e-12):
                return False, "Omega0(t) trajectory changed for 03b"
            if v0_old.shape != v0_expected.shape or not np.allclose(v0_old, v0_expected, rtol=0.0, atol=1e-12):
                return False, "V0(t) trajectory changed for 03b"
            if js_old.shape != js_expected.shape or ms_old.shape != ms_expected.shape:
                return False, "JM basis shape changed for 03a"
            if not np.array_equal(js_old, js_expected) or not np.array_equal(ms_old, ms_expected):
                return False, "JM basis content changed for 03a"
            if nx_old.shape != nx_expected.shape or ny_old.shape != ny_expected.shape:
                return False, "HO mode list shape changed for 03a"
            if not np.array_equal(nx_old, nx_expected) or not np.array_equal(ny_old, ny_expected):
                return False, "HO mode list content changed for 03a"

            for key in PARAMS_03B:
                cur = defaults.get(key, None)
                if cur is None:
                    # None-valued defaults may not be serialized into HDF5 attrs.
                    continue
                if key not in h5.attrs:
                    return False, f"missing attr '{key}' in {p}"
                old = h5.attrs[key]
                if key in ("N_theta", "N_phi"):
                    old_i = int(_coerce_scalar(old))
                    cur_i = int(_coerce_scalar(cur))
                    if old_i < cur_i:
                        return False, (
                            f"parameter changed for 03a: {key} "
                            f"(stored={old_i}, current={cur_i}; stored grid too coarse)"
                        )
                    continue
                if not _same_value(old, cur):
                    return False, f"parameter changed for 03a: {key} (stored={old}, current={cur})"
    except Exception as e:
        return False, f"cannot validate {p}: {e}"

    return True, f"cached data valid: {p}"


def check_03b_cache(defaults: dict[str, object]) -> tuple[bool, str]:
    if not _H5PY_OK:
        return False, "h5py unavailable, cannot validate cache"
    path = _03b_path(defaults)
    if not path.exists():
        return False, f"missing file: {path}"
    try:
        mod03a = importlib.import_module("03a_free_rotor_drive_compute")
        mod03b = importlib.import_module("03b_free_rotor_drive_observables")
        j_max = int(defaults.get("J_max", 20))
        js_expected, ms_expected, _ = mod03a.load_ylm_blocks(_ylm_path(defaults), j_max)
        obs_keys_expected = sorted(list(getattr(mod03b, "OBS_KEYS", ())))
        with open_h5(h5py, path, "r") as h5:
            if int(h5.attrs.get("antipodal_even_J_only", 1)) != 1:
                return False, "03b data not marked as antipodally symmetric even-J basis"
            js_old = h5["J"][...].astype(int)
            ms_old = h5["M"][...].astype(int)
            if js_old.shape != js_expected.shape or ms_old.shape != ms_expected.shape:
                return False, "JM basis shape changed for 03b"
            if not np.array_equal(js_old, js_expected) or not np.array_equal(ms_old, ms_expected):
                return False, "JM basis content changed for 03b"
            if "observables_jm" not in h5:
                return False, "missing group 'observables_jm' in 03b output"
            obs_keys_old = sorted(list(h5["observables_jm"].keys()))
            if obs_keys_old != obs_keys_expected:
                return False, f"observable list changed for 03b (stored={obs_keys_old}, current={obs_keys_expected})"
            n_theta_old = h5.attrs.get("N_theta", h5.attrs.get("N_theta_05_decomp", None))
            n_phi_old = h5.attrs.get("N_phi", h5.attrs.get("N_phi_05_decomp", None))
            if n_theta_old is not None and int(n_theta_old) != int(defaults.get("N_theta", 50)):
                return False, f"N_theta changed for 03b (stored={int(n_theta_old)}, current={int(defaults.get('N_theta', 50))})"
            if n_phi_old is not None and int(n_phi_old) != int(defaults.get("N_phi", 50)):
                return False, f"N_phi changed for 03b (stored={int(n_phi_old)}, current={int(defaults.get('N_phi', 50))})"
    except Exception as e:
        return False, f"cannot validate {path}: {e}"
    return True, f"cached data valid: {path}"


def check_03a_cache(defaults: dict[str, object]) -> tuple[bool, str]:
    if not _H5PY_OK:
        return False, "h5py unavailable, cannot validate cache"
    path = _03a_path(defaults)
    if not path.exists():
        return False, f"missing file: {path}"
    try:
        par = importlib.import_module("01_Parameters")
        mod03a = importlib.import_module("03a_free_rotor_drive_compute")
        nt_main = int(defaults.get("Nt_main", defaults.get("Nt_pendulon", defaults.get("rotor_Nt", 3))))
        grids = par.drive_grids_with_Nt(defaults, nt_main)
        t_expected = np.asarray(grids["t"], dtype=float)
        omega_expected = np.asarray(grids["Omega0"], dtype=float)
        v0_expected = np.asarray(grids["V0"], dtype=float)
        j_max = int(defaults.get("J_max", 20))
        js_expected, ms_expected, _ = mod03a.load_ylm_blocks(_ylm_path(defaults), j_max)
        with open_h5(h5py, path, "r") as h5:
            if int(h5.attrs.get("antipodal_even_J_only", 0)) != 1:
                return False, "03a data not marked as antipodally symmetric even-J basis"
            if int(h5.attrs.get("J_max", -1)) != j_max:
                return False, f"J_max changed for 03a (stored={int(h5.attrs.get('J_max', -1))}, current={j_max})"
            nt_old = int(h5.attrs.get("Nt_main", h5.attrs.get("Nt_03", -1)))
            if nt_old != int(t_expected.size):
                return False, f"Nt_main changed for 03a (stored={nt_old}, current={int(t_expected.size)})"
            if not _same_value(h5.attrs.get("B", None), defaults.get("B", None)):
                return False, f"B changed for 03a (stored={h5.attrs.get('B', None)}, current={defaults.get('B', None)})"
            ok_rot, why_rot = _check_rotational_model_attrs(h5, defaults, "03a")
            if not ok_rot:
                return False, str(why_rot)
            t_old = h5["t_grid"][...].astype(float)
            omega_old = h5["Omega0_t"][...].astype(float)
            v0_old = h5["V0_t"][...].astype(float)
            js_old = h5["J"][...].astype(int)
            ms_old = h5["M"][...].astype(int)
            if t_old.shape != t_expected.shape or not np.allclose(t_old, t_expected, rtol=0.0, atol=1e-12):
                return False, "time trajectory changed for 03a"
            if omega_old.shape != omega_expected.shape or not np.allclose(omega_old, omega_expected, rtol=0.0, atol=1e-12):
                return False, "Omega0(t) trajectory changed for 03a"
            if v0_old.shape != v0_expected.shape or not np.allclose(v0_old, v0_expected, rtol=0.0, atol=1e-12):
                return False, "V0(t) trajectory changed for 03a"
            if js_old.shape != js_expected.shape or ms_old.shape != ms_expected.shape:
                return False, "JM basis shape changed for 03a"
            if not np.array_equal(js_old, js_expected) or not np.array_equal(ms_old, ms_expected):
                return False, "JM basis content changed for 03a"
    except Exception as e:
        return False, f"cannot validate {path}: {e}"
    return True, f"cached data valid: {path}"


def check_05b_cache(defaults: dict[str, object]) -> tuple[bool, str]:
    if not _H5PY_OK:
        return False, "h5py unavailable, cannot validate cache"
    path = _03b_path(defaults)
    if not path.exists():
        return False, f"missing file: {path}"
    try:
        mod03a = importlib.import_module("03a_free_rotor_drive_compute")
        mod03b = importlib.import_module("03b_free_rotor_drive_observables")
        j_max = int(defaults.get("J_max", 20))
        js_expected, ms_expected, _ = mod03a.load_ylm_blocks(_ylm_path(defaults), j_max)
        obs_keys_expected = sorted(list(getattr(mod03b, "OBS_KEYS", ())))
        with open_h5(h5py, path, "r") as h5:
            if int(h5.attrs.get("antipodal_even_J_only", 1)) != 1:
                return False, "03b data not marked as antipodally symmetric even-J basis"
            js_old = h5["J"][...].astype(int)
            ms_old = h5["M"][...].astype(int)
            if js_old.shape != js_expected.shape or ms_old.shape != ms_expected.shape:
                return False, "JM basis shape changed for 03b"
            if not np.array_equal(js_old, js_expected) or not np.array_equal(ms_old, ms_expected):
                return False, "JM basis content changed for 03b"
            if "observables_jm" not in h5:
                return False, "missing group 'observables_jm' in 03b output"
            obs_keys_old = sorted(list(h5["observables_jm"].keys()))
            if obs_keys_old != obs_keys_expected:
                return False, f"observable list changed for 03b (stored={obs_keys_old}, current={obs_keys_expected})"
            n_theta_old = h5.attrs.get("N_theta", h5.attrs.get("N_theta_05_decomp", None))
            n_phi_old = h5.attrs.get("N_phi", h5.attrs.get("N_phi_05_decomp", None))
            if n_theta_old is not None and int(n_theta_old) != int(defaults.get("N_theta", 50)):
                return False, f"N_theta changed for 03b (stored={int(n_theta_old)}, current={int(defaults.get('N_theta', 50))})"
            if n_phi_old is not None and int(n_phi_old) != int(defaults.get("N_phi", 50)):
                return False, f"N_phi changed for 03b (stored={int(n_phi_old)}, current={int(defaults.get('N_phi', 50))})"
    except Exception as e:
        return False, f"cannot validate {path}: {e}"
    return True, f"cached data valid: {path}"


def check_05c_cache(defaults: dict[str, object]) -> tuple[bool, str]:
    if not _H5PY_OK:
        return False, "h5py unavailable, cannot validate cache"
    path = _03c_path(defaults)
    path_03a = _03a_path(defaults)
    path_03b = _03b_path(defaults)
    path_01c = _01c_path(defaults)
    if not path.exists():
        return False, f"missing file: {path}"
    if not path_03a.exists():
        return False, f"missing dependency: {path_03a}"
    if not path_03b.exists():
        return False, f"missing dependency: {path_03b}"
    if not path_01c.exists():
        return False, f"missing dependency: {path_01c}"
    try:
        par = importlib.import_module("01_Parameters")
        grids = par.drive_grids(defaults)
        t_expected = np.asarray(grids["t"], dtype=float)
        omega_expected = np.asarray(grids["Omega0"], dtype=float)
        dphi_expected = np.asarray(grids["Delta_phi"], dtype=float)
        nt_dense = int(defaults.get("Nt_plot_rotating_observables", t_expected.size))
        dense_grids = par.drive_grids_with_Nt(defaults, nt_dense)
        t_dense_expected = np.asarray(dense_grids["t"], dtype=float)
        omega_dense_expected = np.asarray(dense_grids["Omega0"], dtype=float)
        dphi_dense_expected = np.asarray(dense_grids["Delta_phi"], dtype=float)
        tau_steady_state_expected = defaults.get("tau_steady_state", None)
        tau_steady_state_expected = "None" if tau_steady_state_expected is None else float(tau_steady_state_expected)
        tau_steady_state_final_expected = defaults.get("tau_steady_state_final", None)
        tau_steady_state_final_expected = "None" if tau_steady_state_final_expected is None else float(tau_steady_state_final_expected)
        tau_smooth_expected = defaults.get("tau_smooth", None)
        tau_smooth_expected = "None" if tau_smooth_expected is None else float(tau_smooth_expected)
        degeneracy_tol_expected = float(defaults.get("degeneracy_tol", 1e-10))
        compute_lab_expected = int(bool(defaults.get("compute_thermal_lab_frame", True)))
        compute_rot_expected = int(bool(defaults.get("compute_thermal_rot_frame", True)))
        compare_ho_expected = int(bool(defaults.get("Compare_to_HO_basis", False)))
        trotter_expected = int(defaults.get("thermal_trotter_steps", 3))
        with open_h5(h5py, path, "r") as h5:
            required = (
                "t_grid",
                "Omega0_t",
                "Delta_phi_t",
                "J",
                "M",
                "observables_per_mode",
                "thermal",
                "thermal/steady_state",
                "thermal/mode_weights",
                "dense_rotating_observables",
            )
            for key in required:
                if key not in h5:
                    return False, f"missing dataset/group '{key}' in {path}"
            if not _same_value(h5.attrs.get("tau_steady_state", None), tau_steady_state_expected):
                return False, (
                    "tau_steady_state changed for 03c "
                    f"(stored={h5.attrs.get('tau_steady_state', None)}, current={tau_steady_state_expected})"
                )
            if not _same_value(h5.attrs.get("tau_steady_state_final", "None"), tau_steady_state_final_expected):
                return False, (
                    "tau_steady_state_final changed for 03c "
                    f"(stored={h5.attrs.get('tau_steady_state_final', 'None')}, current={tau_steady_state_final_expected})"
                )
            if not _same_value(h5.attrs.get("tau_smooth", None), tau_smooth_expected):
                return False, (
                    f"tau_smooth changed for 03c "
                    f"(stored={h5.attrs.get('tau_smooth', None)}, current={tau_smooth_expected})"
                )
            if not _same_value(h5.attrs.get("degeneracy_tol", None), degeneracy_tol_expected):
                return False, f"degeneracy_tol changed for 03c (stored={h5.attrs.get('degeneracy_tol', None)}, current={degeneracy_tol_expected})"
            ok_rot, why_rot = _check_rotational_model_attrs(h5, defaults, "03c")
            if not ok_rot:
                return False, str(why_rot)
            n_basis = int(h5["J"].shape[0])
            smooth_bytes_needed = par.estimate_array_storage_bytes(
                ((t_expected.size, n_basis), np.float64),
                ((t_expected.size, n_basis, n_basis), np.complex128),
                ((t_expected.size, n_basis, n_basis), np.complex128),
                ((t_expected.size, n_basis, n_basis), np.complex128),
                ((t_expected.size, n_basis, n_basis), np.complex128),
            )
            compute_ss_smooth_expected = int(not par.exceeds_ram_threshold(defaults, smooth_bytes_needed))
            if int(h5.attrs.get("compute_steady_state_smooth", 1)) != compute_ss_smooth_expected:
                return False, (
                    "compute_steady_state_smooth changed for 03c "
                    f"(stored={int(h5.attrs.get('compute_steady_state_smooth', 1))}, current={compute_ss_smooth_expected})"
                )
            if int(h5.attrs.get("compute_thermal_lab_frame", -1)) != compute_lab_expected:
                return False, f"compute_thermal_lab_frame changed for 03c (stored={int(h5.attrs.get('compute_thermal_lab_frame', -1))}, current={compute_lab_expected})"
            if int(h5.attrs.get("compute_thermal_rot_frame", -1)) != compute_rot_expected:
                return False, f"compute_thermal_rot_frame changed for 03c (stored={int(h5.attrs.get('compute_thermal_rot_frame', -1))}, current={compute_rot_expected})"
            if int(h5.attrs.get("Compare_to_HO_basis", -1)) != compare_ho_expected:
                return False, f"Compare_to_HO_basis changed for 03c (stored={int(h5.attrs.get('Compare_to_HO_basis', -1))}, current={compare_ho_expected})"
            if int(h5.attrs.get("thermal_trotter_steps", -1)) != trotter_expected:
                return False, f"thermal_trotter_steps changed for 03c (stored={int(h5.attrs.get('thermal_trotter_steps', -1))}, current={trotter_expected})"
            if int(h5.attrs.get("Nt_plot_rotating_observables", -1)) != int(t_dense_expected.size):
                return False, f"Nt_plot_rotating_observables changed for 03c (stored={int(h5.attrs.get('Nt_plot_rotating_observables', -1))}, current={int(t_dense_expected.size)})"

            t_old = h5["t_grid"][...].astype(float)
            omega_old = h5["Omega0_t"][...].astype(float)
            dphi_old = h5["Delta_phi_t"][...].astype(float)
            if t_old.shape != t_expected.shape or not np.allclose(t_old, t_expected, rtol=0.0, atol=1e-12):
                return False, "time trajectory changed for 03c"
            if omega_old.shape != omega_expected.shape or not np.allclose(omega_old, omega_expected, rtol=0.0, atol=1e-12):
                return False, "Omega0(t) trajectory changed for 03c"
            if dphi_old.shape != dphi_expected.shape or not np.allclose(dphi_old, dphi_expected, rtol=0.0, atol=1e-12):
                return False, "Delta_phi(t) trajectory changed for 03c"

            grp_dense = h5["dense_rotating_observables"]
            for key in ("t_grid", "Omega0_t", "Delta_phi_t", "steady_state"):
                if key not in grp_dense:
                    return False, f"missing dataset/group 'dense_rotating_observables/{key}' in {path}"
            t_dense_old = grp_dense["t_grid"][...].astype(float)
            omega_dense_old = grp_dense["Omega0_t"][...].astype(float)
            dphi_dense_old = grp_dense["Delta_phi_t"][...].astype(float)
            if t_dense_old.shape != t_dense_expected.shape or not np.allclose(t_dense_old, t_dense_expected, rtol=0.0, atol=1e-12):
                return False, "dense time trajectory changed for 03c"
            if omega_dense_old.shape != omega_dense_expected.shape or not np.allclose(omega_dense_old, omega_dense_expected, rtol=0.0, atol=1e-12):
                return False, "dense Omega0(t) trajectory changed for 03c"
            if dphi_dense_old.shape != dphi_dense_expected.shape or not np.allclose(dphi_dense_old, dphi_dense_expected, rtol=0.0, atol=1e-12):
                return False, "dense Delta_phi(t) trajectory changed for 03c"
            has_ss_smooth = "steady_state_smooth" in h5["thermal"]
            has_dense_ss_smooth = "steady_state_smooth" in grp_dense
            if bool(compute_ss_smooth_expected) != has_ss_smooth:
                return False, (
                    "thermal/steady_state_smooth presence changed for 03c "
                    f"(stored={has_ss_smooth}, current={bool(compute_ss_smooth_expected)})"
                )
            if bool(compute_ss_smooth_expected) != has_dense_ss_smooth:
                return False, (
                    "dense steady_state_smooth presence changed for 03c "
                    f"(stored={has_dense_ss_smooth}, current={bool(compute_ss_smooth_expected)})"
                )

            if compare_ho_expected:
                if "ho_comparison" not in h5:
                    return False, "missing group 'ho_comparison' in 05c output"
                proj_old = h5.attrs.get("projection_h5_path", None)
                proj_cur = str(_projection_path(defaults))
                if not _same_value(proj_old, proj_cur):
                    return False, f"projection_h5_path changed for 03c (stored={proj_old}, current={proj_cur})"
            else:
                if "ho_comparison" in h5:
                    return False, "ho_comparison present although Compare_to_HO_basis=False for 05c"
    except Exception as e:
        return False, f"cannot validate {path}: {e}"
    return True, f"cached data valid: {path}"


def check_01b_cache(defaults: dict[str, object]) -> tuple[bool, str]:
    if not _H5PY_OK:
        return False, "h5py unavailable, cannot validate cache"

    path = _ylm_path(defaults)
    if not path.exists():
        return False, f"missing file: {path}"

    lam_max_req = int(defaults.get("lambda_max", 2))
    j_max_req = int(defaults.get("J_max", 0))
    n_theta_req = int(defaults.get("N_theta", 0))
    n_phi_req = int(defaults.get("N_phi", 0))

    try:
        with open_h5(h5py, path, "r") as h5:
            if "lam_max" not in h5.attrs:
                return False, f"missing attr 'lam_max' in {path}"
            lam_old = int(h5.attrs["lam_max"])
            if lam_old < lam_max_req:
                return False, (
                    f"lambda_max increased for 01b "
                    f"(stored={lam_old}, current={lam_max_req})"
                )

            if "J" not in h5 or "M" not in h5:
                return False, f"missing J/M datasets in {path}"
            js_old = h5["J"][...].astype(int)
            ms_old = h5["M"][...].astype(int)
            states_req = [(J, M) for J in range(j_max_req + 1) for M in range(-J, J + 1)]
            js_req = np.array([J for J, _ in states_req], dtype=int)
            ms_req = np.array([M for _, M in states_req], dtype=int)
            idx_map = {(int(j), int(m)) for j, m in zip(js_old, ms_old)}
            for j, m in zip(js_req, ms_req):
                if (int(j), int(m)) not in idx_map:
                    return False, (
                        f"J_max increased for 01b "
                        f"(stored max J={int(np.max(js_old)) if js_old.size else -1}, current={j_max_req})"
                    )

            if "YJM_grid" not in h5:
                return False, f"missing group 'YJM_grid' in {path}"
            grp = h5["YJM_grid"]
            theta_ds = "theta" if "theta" in grp else "theta0" if "theta0" in grp else None
            phi_ds = "phi" if "phi" in grp else "phi0" if "phi0" in grp else None
            y_ds = "Y" if "Y" in grp else "Y0" if "Y0" in grp else None
            if theta_ds is None or phi_ds is None or y_ds is None:
                return False, f"incomplete YJM_grid in {path}"
            theta_old = grp[theta_ds][...].astype(float)
            phi_old = grp[phi_ds][...].astype(float)
            if int(theta_old.size) != n_theta_req:
                return False, f"N_theta changed for 01b (stored={int(theta_old.size)}, current={n_theta_req})"
            if int(phi_old.size) != n_phi_req:
                return False, f"N_phi changed for 01b (stored={int(phi_old.size)}, current={n_phi_req})"
    except Exception as e:
        return False, f"cannot validate {path}: {e}"

    return True, f"cached data valid: {path}"


def check_01c_cache(defaults: dict[str, object]) -> tuple[bool, str]:
    if not _H5PY_OK:
        return False, "h5py unavailable, cannot validate cache"
    path = _01c_path(defaults)
    ylm = _ylm_path(defaults)
    if not path.exists():
        return False, f"missing file: {path}"
    if not ylm.exists():
        return False, f"missing dependency: {ylm}"
    try:
        par = importlib.import_module("01_Parameters")
        stage01c = importlib.import_module("01c_precompute_steady_state")
        traj = par.drive_grids(defaults)
        t_expected = np.asarray(traj["t"], dtype=float)
        omega_expected = np.asarray(traj["Omega0"], dtype=float)
        v0_expected = np.asarray(traj["V0"], dtype=float)
        js_expected, ms_expected = stage01c.load_even_j_basis_from_ylm(ylm, int(defaults.get("J_max", 20)))
        with open_h5(h5py, path, "r") as h5:
            for key in ("t_grid", "Omega0_t", "V0_t", "J", "M", "H_target_jm_re", "H_target_jm_im", "rho_target_jm_re", "rho_target_jm_im"):
                if key not in h5:
                    return False, f"missing dataset '{key}' in {path}"
            if not _same_value(h5.attrs.get("B", None), defaults.get("B", None)):
                return False, f"B changed for 01c (stored={h5.attrs.get('B', None)}, current={defaults.get('B', None)})"
            ok_rot, why_rot = _check_rotational_model_attrs(h5, defaults, "01c")
            if not ok_rot:
                return False, str(why_rot)
            if not _same_value(h5.attrs.get("T_K", None), defaults.get("T_K", None)):
                return False, f"T_K changed for 01c (stored={h5.attrs.get('T_K', None)}, current={defaults.get('T_K', None)})"
            t_old = h5["t_grid"][...].astype(float)
            omega_old = h5["Omega0_t"][...].astype(float)
            v0_old = h5["V0_t"][...].astype(float)
            if t_old.shape != t_expected.shape or not np.allclose(t_old, t_expected, rtol=0.0, atol=1e-12):
                return False, "time trajectory changed for 01c"
            if omega_old.shape != omega_expected.shape or not np.allclose(omega_old, omega_expected, rtol=0.0, atol=1e-12):
                return False, "Omega0(t) trajectory changed for 01c"
            if v0_old.shape != v0_expected.shape or not np.allclose(v0_old, v0_expected, rtol=0.0, atol=1e-12):
                return False, "V0(t) trajectory changed for 01c"
            if not np.array_equal(h5["J"][...].astype(int), js_expected):
                return False, "JM J basis changed for 01c"
            if not np.array_equal(h5["M"][...].astype(int), ms_expected):
                return False, "JM M basis changed for 01c"
    except Exception as e:
        return False, f"cannot validate {path}: {e}"
    return True, f"cached data valid: {path}"


def check_03c_cache(defaults: dict[str, object]) -> tuple[bool, str]:
    return check_05c_cache(defaults)


def format_seconds(dt: float) -> str:
    if dt < 60.0:
        return f"{dt:.1f} s"
    minutes, seconds = divmod(dt, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {seconds:.1f} s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)} h {int(minutes)} min {seconds:.1f} s"


def run_stage_script(script_name: str, case_name: str) -> None:
    env = os.environ.copy()
    env["PENDULON_CASE"] = case_name
    subprocess.run(
        [sys.executable, script_name],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parent,
    )


def run_post_stage_script(script_name: str) -> None:
    """Run a post-processing stage script (not case-dependent)."""
    env = os.environ.copy()
    env.pop("PENDULON_CASE", None)
    subprocess.run(
        [sys.executable, script_name],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parent,
    )


def main() -> None:
    pipeline_start = time.perf_counter()
    timings: list[tuple[str, str, float]] = []
    old_case_env = os.environ.get("PENDULON_CASE")

    try:
        case_names = load_case_names()
        run_plans = _build_run_plans(case_names)
        if ACTIVE_RUN_PLAN not in run_plans:
            available = ", ".join(sorted(run_plans))
            raise KeyError(f"Unknown run plan '{ACTIVE_RUN_PLAN}'. Available plans: {available}")
        active_plan = _apply_case_controls(run_plans[ACTIVE_RUN_PLAN], case_names)
        active_case_names = [case_name for case_name in case_names if any(active_plan[case_name].values())]
        if not active_case_names:
            print(f"Pendulon pipeline: {ACTIVE_RUN_PLAN} selects no enabled case.", flush=True)
            return
        print(f"Pendulon pipeline: using {ACTIVE_RUN_PLAN} for cases: {', '.join(active_case_names)}", flush=True)
        post_plans = _build_post_plans()
        active_post_plan = post_plans[ACTIVE_RUN_PLAN]
        for case_name in active_case_names:
            defaults = load_defaults_for_case(case_name)
            print(f"[case] {case_name}", flush=True)
            case_plan = active_plan[case_name]
            for stage_code, script_name in STAGE_ORDER:
                enabled = bool(case_plan.get(stage_code, False))
                if not enabled:
                    print(f"[skip][{case_name}] {script_name}", flush=True)
                    continue

                if stage_code == "01B":
                    ok, why = check_01b_cache(defaults)
                    if ok:
                        print(f"[cache][{case_name}] {script_name}: {why}", flush=True)
                        continue
                    print(f"[stale][{case_name}] {script_name}: {why}", flush=True)

                if stage_code == "01C":
                    ok, why = check_01c_cache(defaults)
                    if ok:
                        print(f"[cache][{case_name}] {script_name}: {why}", flush=True)
                        continue
                    print(f"[stale][{case_name}] {script_name}: {why}", flush=True)

                if stage_code == "03A":
                    ok, why = check_03a_cache(defaults)
                    if ok:
                        print(f"[cache][{case_name}] {script_name}: {why}", flush=True)
                        continue
                    print(f"[stale][{case_name}] {script_name}: {why}", flush=True)

                if stage_code == "03B":
                    ok, why = check_03b_cache(defaults)
                    if ok:
                        print(f"[cache][{case_name}] {script_name}: {why}", flush=True)
                        continue
                    print(f"[stale][{case_name}] {script_name}: {why}", flush=True)

                if stage_code == "03C":
                    ok, why = check_03c_cache(defaults)
                    if ok:
                        print(f"[cache][{case_name}] {script_name}: {why}", flush=True)
                        continue
                    print(f"[stale][{case_name}] {script_name}: {why}", flush=True)

                print(f"[run ][{case_name}] {script_name}", flush=True)
                reset_project_modules()
                os.environ["PENDULON_CASE"] = case_name
                t0 = time.perf_counter()
                run_stage_script(script_name, case_name)
                dt = time.perf_counter() - t0
                timings.append((case_name, script_name, dt))
                print(f"[done][{case_name}] {script_name} in {format_seconds(dt)}", flush=True)

        # Run post-processing stages (not case-dependent)
        for stage_code, script_name in POST_STAGE_ORDER:
            enabled = bool(active_post_plan.get(stage_code, False))
            if not enabled:
                print(f"[skip][post] {script_name}", flush=True)
                continue
            print(f"[run ][post] {script_name}", flush=True)
            reset_project_modules()
            t0 = time.perf_counter()
            run_post_stage_script(script_name)
            dt = time.perf_counter() - t0
            timings.append(("[post]", script_name, dt))
            print(f"[done][post] {script_name} in {format_seconds(dt)}", flush=True)

        total_dt = time.perf_counter() - pipeline_start
        print("Pendulon pipeline: completed.", flush=True)
        if timings:
            print("Executed stages:", flush=True)
            for case_name, script_name, dt in timings:
                print(f"  [{case_name}] {script_name}: {format_seconds(dt)}", flush=True)
        print(f"Total elapsed time: {format_seconds(total_dt)}", flush=True)
    finally:
        reset_project_modules()
        if old_case_env is None:
            os.environ.pop("PENDULON_CASE", None)
        else:
            os.environ["PENDULON_CASE"] = old_case_env


if __name__ == "__main__":
    main()
