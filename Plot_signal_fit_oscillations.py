# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:29:24 2026

@author: Ian
"""
from pathlib import Path

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FNAMES = [
    r"Experimental_data/CS2_accelerating_droplets.csv",
    r"Experimental_data/OCS_accelerating_droplets.csv",
]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def signal_model(t_arr, baseline, peak, t_max, signal_width, t_start, decay_time, phase, fc, ramp):
    s_env = peak*np.exp(-(t_arr-t_max)**2 / (2*(((signal_width/2.35))**2)))
    s_profile = s_env*np.exp(-(t_arr- t_start) / ((decay_time)))
    phi = phase + 2*np.pi*(fc*t_arr/1000 + 0.5*ramp*t_arr**2/1E6)
    return baseline + s_env + s_profile*np.cos(phi)**2


def initial_parameters(fname):
    fit_start = -200
    fit_end = 180

    # This set for CS2_usCFG_gas_20260507, overwritten if another matched file name is used
    params = {
        "baseline": 0.5,       # asymptotic value
        "peak": 0.06,          # maximum amplitude of the signal above baseline
        "t_max": 0,            # time of the maximum of the signal
        "signal_width": 300,   # approximate FWHM of the overall profile
        "t_start": 150,        # shifts the maximum of oscillations away from t=0
        "decay_time": -5000,   # approximate decay time of the oscillations
        "phase": 0.0,          # rotational phase at t=0
        "fc": 19,              # GHz, rotational frequency at t=0
        "ramp": 88,            # MHz/ps, rotational acceleration rate
    }

    if "OCS_accelerating_droplets" in fname:
        params.update({
            "baseline": 0.5,
            "peak": 0.075,
            "t_max": 50,
            "signal_width": 420,
            "t_start": -200,
            "decay_time": 160,
            "phase": np.pi/2,
            "fc": 22.25,
            "ramp": 87.5,
        })
        fit_start = -200
        fit_end = 200
    elif "OCS_decelerating_droplets" in fname:
        params.update({
            "baseline": 0.5,
            "peak": 0.1,
            "t_max": 0,
            "signal_width": 420,
            "t_start": 300,
            "decay_time": -160,
            "phase": np.pi/2,
            "fc": 22.25,
            "ramp": -87.5,
        })
        fit_start = -120
        fit_end = 180
    elif "CS2_accelerating_droplets" in fname:
        params.update({
            "baseline": 0.5,
            "peak": 0.025,
            "t_max": 0,
            "signal_width": 300,
            "t_start": -120,
            "decay_time": 160,
            "phase": 0.0,
            "fc": 19,
            "ramp": 88,
        })
        fit_start = -200
        fit_end = 200
    elif "CS2_decelerating_droplets" in fname:
        params.update({
            "baseline": 0.5,
            "peak": 0.025,
            "t_max": 0,
            "signal_width": 300,
            "t_start": 150,
            "decay_time": -160,
            "phase": 0.0,
            "fc": 19,
            "ramp": -88,
        })
        fit_start = -120
        fit_end = 180

    return params, fit_start, fit_end


def fit_and_plot(fname):
    data = pd.read_csv(
        fname,
        header=None,
        names=["delay", "c2t", "c2t_err"],
        sep=",",
        lineterminator="\n",
        dtype=float,
    )

    params, fit_start, fit_end = initial_parameters(fname)

    # Select only the portion of the signal with good quality oscillations.
    x_all = data.delay*1E12
    y_all = data.c2t
    ind_start = find_nearest(x_all, fit_start)
    ind_end = find_nearest(x_all, fit_end)
    x = x_all[ind_start:ind_end]
    y = y_all[ind_start:ind_end]

    # Not used for the fit, just to demonstrate the initial-guess pieces.
    t_arr = x
    s_env = params["peak"]*np.exp(
        -(t_arr-params["t_max"])**2 / (2*(((params["signal_width"]/2.35))**2))
    )
    s_profile = s_env*np.exp(-(t_arr- params["t_start"]) / ((params["decay_time"])))
    phi = params["phase"] + 2*np.pi*(
        params["fc"]*t_arr/1000 + 0.5*params["ramp"]*t_arr**2/1E6
    )
    s_osc = s_profile*np.cos(phi)**2
    s_total = params["baseline"] + s_env + s_osc

    sig_model = lmfit.Model(signal_model)
    sig_fitted = sig_model.fit(y, t_arr=x, max_nfev=2000, **params)

    print("\nFit to:", fname)
    print("Best fit:")
    print("Rotational phase at t=0 :", sig_fitted.best_values["phase"], "rad")
    print("Rotational frequency at t=0 :", sig_fitted.best_values["fc"], "GHz")
    print("Rotational acceleration :", sig_fitted.best_values["ramp"], "MHz/ps")
    print("cos^2 signal frequency at t=0 :", 2*sig_fitted.best_values["fc"], "GHz")
    print("cos^2 signal acceleration :", 2*sig_fitted.best_values["ramp"], "MHz/ps")

    myfig, axs = plt.subplots(1)
    a = axs
    a.plot(data.delay*1E12, data.c2t, label="Actual Signal")
    a.plot(t_arr, s_env+params["baseline"], label="Baseline + Main envelope (initial guess)")
    a.plot(t_arr, s_total, label="Initial Guess")
    a.plot(x, sig_fitted.best_fit, "k--", label="Best Fit")

    a.grid()
    a.legend(fontsize=7)
    a.set_xlim([-300, 250])
    a.set_xlabel("Probe Delay (ps)")
    a.set_ylabel("$\langle \cos^2 \\theta_{\mathrm{2D}} \\rangle$", horizontalalignment="center")
    a.set_title("Fit to "+fname)

    out_path = Path(fname).with_name(Path(fname).stem + "_cos2_signal_fit_oscillations.pdf")
    myfig.tight_layout()
    myfig.savefig(out_path)
    plt.close(myfig)
    print("Saved plot:", out_path)


for fname in FNAMES:
    fit_and_plot(fname)
