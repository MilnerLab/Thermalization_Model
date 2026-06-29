#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01b_precompute_Ylm_blocks.py

Precompute Y_{lambda,mu} matrices in a truncated |J,M> basis and store to HDF5.

Defaults:
  - JM basis source: J_max from 01_Parameters.py
  - Output file:      data/01/Ylm_blocks_JM.h5

Usage examples:
  python 01b_precompute_Ylm_blocks.py
  python 01b_precompute_Ylm_blocks.py --lam-max 6 --J-max 12
"""

from __future__ import annotations

import argparse
import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Tuple, Iterable

import h5py  # type: ignore
import numpy as np

import importlib
from h5_locking import open_h5
from scipy.special import sph_harm_y

params = importlib.import_module("01_Parameters")

try:
    from scipy.special import wigner_3j as _sp_wigner_3j
    _HAS_SCIPY_W3J = True
except Exception:
    _HAS_SCIPY_W3J = False
    _sp_wigner_3j = None  # type: ignore[assignment]

from sympy import S
from sympy.physics.wigner import wigner_3j as _wigner_3j

_JS: np.ndarray | None = None
_MS: np.ndarray | None = None
_IDX_OF: dict[tuple[int, int], int] | None = None
_J_MAX: int | None = None
_W3J_CACHE: Dict[Tuple[int, int, int, int, int, int], float] = {}


def triangle_ok(j1: int, j2: int, j3: int) -> bool:
    return abs(j1 - j2) <= j3 <= (j1 + j2)


def w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    key = (j1, j2, j3, m1, m2, m3)
    if key in _W3J_CACHE:
        return _W3J_CACHE[key]

    if (m1 + m2 + m3) != 0 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3 or not triangle_ok(j1, j2, j3):
        _W3J_CACHE[key] = 0.0
        return 0.0

    if _HAS_SCIPY_W3J:
        fval = float(_sp_wigner_3j(j1, j2, j3, m1, m2, m3))
    else:
        val = _wigner_3j(S(j1), S(j2), S(j3), S(m1), S(m2), S(m3))
        try:
            fval = float(val)
        except TypeError:
            fval = float(val.evalf())

    _W3J_CACHE[key] = fval
    return fval


def Ylm_me(J: int, M: int, lam: int, mu: int, Jp: int, Mp: int) -> complex:
    """Matrix element <J M | Y_{lam,mu} | J' M'>."""
    pref = ((2 * J + 1) * (2 * lam + 1) * (2 * Jp + 1) / (4.0 * np.pi)) ** 0.5
    val = ((-1.0) ** M) * pref
    val *= w3j(J, lam, Jp, 0, 0, 0)
    val *= w3j(J, lam, Jp, -M, mu, Mp)
    return complex(val)


def precompute_Ylm_blocks(Js: np.ndarray, Ms: np.ndarray, lam_max: int) -> Dict[Tuple[int, int], np.ndarray]:
    """Precompute Y_{lam,mu} matrices in the truncated |J,M> basis."""
    _init_worker(Js, Ms)
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for task in _tasks(lam_max):
        lam, mu, Y = _compute_one(task)
        out[(lam, mu)] = Y
    return out


def parse_args() -> argparse.Namespace:
    default_lam_max = int(params.DEFAULTS.get("lambda_max", 2))
    default_j_max = int(params.DEFAULTS.get("J_max", 12))
    default_nproc = int(params.DEFAULTS.get("nproc", max(1, os.cpu_count() or 1)))
    default_chunksize = int(params.DEFAULTS.get("chunksize", 1))
    p = argparse.ArgumentParser(description="Precompute and store Y_{lambda,mu} blocks.")
    p.add_argument(
        "--out-h5",
        type=Path,
        default=Path(str(params.DEFAULTS.get("Ylm_h5_path", "data/01/Ylm_blocks_JM.h5"))),
        help="Output HDF5 path for precomputed Y blocks.",
    )
    p.add_argument(
        "--lam-max",
        type=int,
        default=default_lam_max,
        help=f"Maximum lambda to precompute (default: 01_Parameters.lambda_max={default_lam_max}).",
    )
    p.add_argument(
        "--J-max",
        "--J-self-max",
        dest="J_max",
        type=int,
        default=default_j_max,
        help="Truncate basis to J <= J-max before precomputing.",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=default_nproc,
        help="Number of worker processes.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=default_chunksize,
        help="Task chunksize for worker dispatch.",
    )
    return p.parse_args()


def build_jm_basis(J_max: int) -> Tuple[np.ndarray, np.ndarray]:
    states = [(J, M) for J in range(J_max + 1) for M in range(-J, J + 1)]
    Js = np.array([J for J, _ in states], dtype=int)
    Ms = np.array([M for _, M in states], dtype=int)
    return Js, Ms


def build_patch_grids() -> tuple[np.ndarray, np.ndarray]:
    n_theta = int(params.DEFAULTS["N_theta"])
    n_phi = int(params.DEFAULTS["N_phi"])
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    return theta, phi


def precompute_Yjm_grid(Js: np.ndarray, Ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, phi = build_patch_grids()
    njm = Js.size
    n_theta = theta.size
    n_phi = phi.size
    y = np.empty((njm, n_theta, n_phi), dtype=np.complex128)
    for i, (J, M) in enumerate(zip(Js.astype(int), Ms.astype(int))):
        y[i, :, :] = sph_harm_y(int(J), int(M), theta[:, None], phi[None, :])
    return theta, phi, y


def save_blocks(path: Path, Js: np.ndarray, Ms: np.ndarray, lam_max: int, blocks: Dict[Tuple[int, int], np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(h5py, path, "w") as h5:
        h5.attrs["lam_max"] = int(lam_max)
        h5.attrs["case_name"] = str(params.DEFAULTS.get("case_name", "Default"))
        h5.create_dataset("J", data=Js.astype(np.int32))
        h5.create_dataset("M", data=Ms.astype(np.int32))
        grp = h5.create_group("Ylm")
        for lam in range(lam_max + 1):
            g_lam = grp.create_group(f"lam_{lam}")
            for mu in range(-lam, lam + 1):
                g_lam.create_dataset(f"mu_{mu}", data=blocks[(lam, mu)], compression="gzip", compression_opts=4)
        theta, phi, y = precompute_Yjm_grid(Js, Ms)
        grp_grid = h5.create_group("YJM_grid")
        grp_grid.attrs["N_theta"] = int(params.DEFAULTS["N_theta"])
        grp_grid.attrs["N_phi"] = int(params.DEFAULTS["N_phi"])
        grp_grid.create_dataset("theta", data=theta)
        grp_grid.create_dataset("phi", data=phi)
        grp_grid.create_dataset("Y", data=y, compression="gzip", compression_opts=4)

def _init_worker(Js: np.ndarray, Ms: np.ndarray) -> None:
    global _JS, _MS, _IDX_OF, _J_MAX
    _JS = Js
    _MS = Ms
    _IDX_OF = {(int(Js[i]), int(Ms[i])): i for i in range(Js.size)}
    _J_MAX = int(np.max(Js))


def _compute_one(task: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    lam, mu = task
    assert _JS is not None and _MS is not None and _IDX_OF is not None and _J_MAX is not None
    NJM = _JS.size
    Y = np.zeros((NJM, NJM), dtype=np.complex128)
    for a in range(NJM):
        J = int(_JS[a])
        M = int(_MS[a])
        Mp = M - mu
        Jp_min = max(abs(Mp), abs(J - lam))
        Jp_max = min(_J_MAX, J + lam)
        if Jp_min > Jp_max:
            continue
        for Jp in range(Jp_min, Jp_max + 1):
            b = _IDX_OF.get((Jp, Mp))
            if b is None:
                continue
            Y[a, b] = Ylm_me(J, M, lam, mu, Jp, Mp)
    return lam, mu, Y


def _tasks(lam_max: int) -> Iterable[tuple[int, int]]:
    for lam in range(lam_max + 1):
        for mu in range(-lam, lam + 1):
            yield (lam, mu)


def precompute_blocks_parallel(Js: np.ndarray, Ms: np.ndarray, lam_max: int, nproc: int, chunksize: int) -> Dict[Tuple[int, int], np.ndarray]:
    n_tasks = (lam_max + 1) ** 2
    if nproc <= 1:
        _init_worker(Js, Ms)
        out: Dict[Tuple[int, int], np.ndarray] = {}
        cur_lam = -1
        for t in _tasks(lam_max):
            lam, mu, Y = _compute_one(t)
            out[(lam, mu)] = Y
            if lam != cur_lam:
                cur_lam = lam
                print(f"  01b: lambda={lam}/{lam_max}", flush=True)
        return out

    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    out: Dict[Tuple[int, int], np.ndarray] = {}
    report_every = max(1, n_tasks // 5)
    done = 0
    with ctx.Pool(processes=nproc, initializer=_init_worker, initargs=(Js, Ms)) as pool:
        for lam, mu, Y in pool.imap_unordered(_compute_one, _tasks(lam_max), chunksize=max(1, chunksize)):
            out[(lam, mu)] = Y
            done += 1
            if done % report_every == 0 or done == n_tasks:
                print(f"  01b: {done}/{n_tasks} blocks done", flush=True)
    return out


def main() -> None:
    args = parse_args()
    Js, Ms = build_jm_basis(int(args.J_max))
    print(f"Precomputing Y blocks: lam_max={args.lam_max}, NJM={Js.size}, nproc={args.nproc}")
    blocks = precompute_blocks_parallel(
        Js=Js,
        Ms=Ms,
        lam_max=int(args.lam_max),
        nproc=max(1, int(args.nproc)),
        chunksize=max(1, int(args.chunksize)),
    )
    print("  01b: saving to HDF5...", flush=True)
    save_blocks(args.out_h5, Js, Ms, int(args.lam_max), blocks)
    print(f"Wrote: {args.out_h5}")


if __name__ == "__main__":
    main()
