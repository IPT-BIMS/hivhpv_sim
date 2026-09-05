"""Shared 17-parameter calibration runner.

This module is the single calibration code path intended for both the Jupyter
notebooks and the Shiny interface.  Keeping the objective and optimizer here
prevents the UI from maintaining a second, slightly different calibration
implementation.

The numerical formulation matches the historical notebook setup:
- 17 calibrated values (15 age-function parameters + 2 tunnel durations)
- L-BFGS-B, one deterministic start
- incidence/mortality/HIV weights 1/1/5
- regularisation weight 0.05
- incidence-RR penalty weight 0.05
- no explicit ``eps`` option
"""
from __future__ import annotations

import platform
import sys

import numpy as np
import pandas as pd
import scipy
import scipy.optimize as sopt

from .constants import ages, CC_AGE_BINS
from .model import run_model, var_output, agg_ageclasses_from_yearly
from .calibration import (
    params_fixed as DEFAULT_PARAMS_FIXED,
    params as DEFAULT_PARAMS,
    EXPECTED_CALIBRATION_VECTOR_SIZE,
    probability_to_rate,
    build_calibration_vector,
    build_params_fixed_from_durations,
    calculate_incidence_rr_table,
    incidence_rr_penalty,
    validate_candidate_vector,
    CALIBRATION_WEIGHTS,
    REGULARIZATION_WEIGHT,
    RR_REGULARIZATION_WEIGHT,
)
from .vaccination import vax_params as NOVAX


def normalized_mse(observed, simulated):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    scale = max(float(np.mean(np.abs(observed))), 1e-9)
    return float(np.mean(((simulated - observed) / scale) ** 2))


def notebook_calibration_setup():
    """Return the exact historical initial vector and 17 L-BFGS-B bounds."""
    initial = build_calibration_vector(
        age_parameters=DEFAULT_PARAMS,
        duration_neg_total=24.0,
        duration_pos_total=15.0,
    )

    bounds = [
        (probability_to_rate(0.03), probability_to_rate(0.85)),
        (17.0, 45.0), (3.0, 25.0),
        (probability_to_rate(0.05), probability_to_rate(0.60)),
        (18.0, 35.0), (4.0, 20.0),
        (probability_to_rate(0.01), probability_to_rate(0.85)),
        (45.0, 90.0), (2.5, 30.0),
        (probability_to_rate(0.01), probability_to_rate(0.90)),
        (35.0, 90.0), (3.0, 25.0),
        (1e-6, 0.04), (18.0, 45.0), (3.0, 20.0),
        (18.0, 27.0),
        (12.0, 25.0),
    ]
    if len(bounds) != EXPECTED_CALIBRATION_VECTOR_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_CALIBRATION_VECTOR_SIZE} bounds, got {len(bounds)}."
        )
    lo = np.array([x[0] for x in bounds], dtype=float)
    hi = np.array([x[1] for x in bounds], dtype=float)
    initial = np.clip(initial, lo + 1e-10, hi - 1e-10)
    return initial, bounds, lo, hi


def vector_to_components(vector, fixed_base=None):
    vector = np.asarray(vector, dtype=float)
    if vector.size != EXPECTED_CALIBRATION_VECTOR_SIZE:
        raise ValueError(
            f"Calibration vector must have {EXPECTED_CALIBRATION_VECTOR_SIZE} elements."
        )

    age_params = {
        "lambda_HPV": vector[0:3].tolist(),
        "lambda_HPV_pos": vector[3:6].tolist(),
        "mu_c": vector[6:9].tolist(),
        "mu_c_pos": vector[9:12].tolist(),
        "lambda_HIV": vector[12:15].tolist(),
    }
    age_vector = vector[:15].copy()
    fixed_base = DEFAULT_PARAMS_FIXED if fixed_base is None else fixed_base
    pf = build_params_fixed_from_durations(
        duration_neg_total=float(vector[15]),
        duration_pos_total=float(vector[16]),
    )

    # Preserve all fixed parameters supplied by the caller except the six
    # progression rates derived from the two calibrated tunnel durations.
    tmp = dict(fixed_base)
    for key in [
        "lambda_P1P2", "lambda_P2P3", "lambda_P3C",
        "lambda_P1P2_pos", "lambda_P2P3_pos", "lambda_P3C_pos",
    ]:
        tmp[key] = pf[key]
    pf = tmp

    return pf, age_params, age_vector, float(vector[15]), float(vector[16])


def calculate_calibration_error_components(
    candidate_vector,
    field_data,
    fixed_base=None,
    vax_params=NOVAX,
):
    pf, age_params, age_vector, dur_neg, dur_pos = vector_to_components(
        candidate_vector, fixed_base=fixed_base
    )
    sn, sp = run_model(
        age_vector,
        params_fixed=pf,
        vax_params=vax_params,
        run_tests=False,
    )
    model_ages = np.asarray(ages[:sn.shape[0]], dtype=float)
    df_yearly = var_output(sn, sp, model_ages)
    df_agg = agg_ageclasses_from_yearly(df_yearly, CC_AGE_BINS)

    obs_i = field_data["cc_incidence_per100k"].to_numpy(float)
    obs_m = field_data["cc_mortality_per100k"].to_numpy(float)
    obs_h = field_data["hiv_prevalence"].to_numpy(float)

    mod_i = df_agg["inc_cc_ageclass_per100k"].to_numpy(float)
    mod_m = df_agg["mort_cc_ageclass_per100k"].to_numpy(float)
    mod_h = df_agg["hiv_prev_ageclass_pct"].to_numpy(float)

    lengths = {
        len(obs_i), len(obs_m), len(obs_h),
        len(mod_i), len(mod_m), len(mod_h),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Observed and modelled incidence, mortality and HIV prevalence "
            "must have the same number of age classes."
        )

    e_i = normalized_mse(obs_i, mod_i)
    e_m = normalized_mse(obs_m, mod_m)
    e_h = normalized_mse(obs_h, mod_h)

    initial, _, lo, hi = notebook_calibration_setup()
    reg_scale = np.maximum(hi - lo, 1e-9)
    reg = float(
        np.mean(((np.asarray(candidate_vector, dtype=float) - initial) / reg_scale) ** 2)
    )

    rr = float(incidence_rr_penalty(sn, sp, model_ages))
    data_error = (
        CALIBRATION_WEIGHTS["inc"] * e_i
        + CALIBRATION_WEIGHTS["mort"] * e_m
        + CALIBRATION_WEIGHTS["hiv"] * e_h
    )
    total = (
        data_error
        + REGULARIZATION_WEIGHT * reg
        + RR_REGULARIZATION_WEIGHT * rr
    )

    return {
        "incidence": float(e_i),
        "mortality": float(e_m),
        "hiv": float(e_h),
        "data_error": float(data_error),
        "regularization": float(reg),
        "rr_penalty": float(rr),
        "total_error": float(total),
        "duration_neg_total": dur_neg,
        "duration_pos_total": dur_pos,
        "params_fixed": pf,
        "params_dict": age_params,
        "params_vector": age_vector,
        "state_neg": sn,
        "state_pos": sp,
        "df_yearly": df_yearly,
        "df_agg": df_agg,
        "df_rr": calculate_incidence_rr_table(sn, sp, model_ages),
    }


def calibration_objective(candidate_vector, field_data, fixed_base=None):
    candidate_vector = np.asarray(candidate_vector, dtype=float)
    _, bounds, _, _ = notebook_calibration_setup()

    if not validate_candidate_vector(candidate_vector):
        return 1e20

    for x, (lo, hi) in zip(candidate_vector, bounds):
        if x < lo or x > hi:
            return 1e20

    try:
        return calculate_calibration_error_components(
            candidate_vector,
            field_data,
            fixed_base=fixed_base,
        )["total_error"]
    except Exception:
        return 1e20


def run_notebook_calibration(field_data, fixed_base=None, maxiter=10000):
    """Run the same deterministic single-start L-BFGS-B calibration as notebook."""
    initial, bounds, _, _ = notebook_calibration_setup()

    initial_components = calculate_calibration_error_components(
        initial, field_data, fixed_base=fixed_base
    )
    initial_objective = calibration_objective(
        initial, field_data, fixed_base=fixed_base
    )

    result = sopt.minimize(
        calibration_objective,
        x0=initial,
        args=(field_data, fixed_base),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(maxiter),
            "maxfun": 150000,
            "ftol": 1e-10,
            "gtol": 1e-7,
            "maxls": 50,
        },
    )

    final_components = calculate_calibration_error_components(
        result.x, field_data, fixed_base=fixed_base
    )
    return {
        "result": result,
        "initial_vector": initial,
        "initial_components": initial_components,
        "initial_objective": initial_objective,
        "final_components": final_components,
        "optimized_vector": result.x.copy(),
        "optimized_fixed": final_components["params_fixed"],
        "environment": calibration_environment(),
    }


def calibration_environment():
    """Return software versions useful for recording calibration provenance."""
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }
