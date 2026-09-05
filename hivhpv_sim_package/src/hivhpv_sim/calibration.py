"""Parameter calibration — Cameroon Updated Framework.

All 17 calibration functions extracted verbatim. Application-only code
(calibration_objective, calibration_vector_to_parameters, normalized_mse,
regularization_penalty, calculate_error_components) stays in the notebook.

New vs previous Cameroon version:
  - RR_AGE_GROUPS (16 age groups vs 6 before)
  - CALIBRATION_WEIGHTS / REGULARIZATION_WEIGHT / RR_REGULARIZATION_WEIGHT as
    module-level constants (previously only inside objective function)
  - maximum_valid_tunnel_duration (→ sensitivity.py — see below)
"""
import numpy as np
import pandas as pd

from .model import prob_to_rate, build_params_vector, create_age_functions
from .constants import NEG_NAMES, POS_NAMES

# ============================================================
# 1. FIXED NATURAL-HISTORY PARAMETERS
# ============================================================
params_fixed_proba = {
    # HPV infection -> first precancer stage
    "lambda_IP1": 0.040,
    "lambda_IP1_pos": 0.080,

    # Precancer progression
    "lambda_P1P2": 0.125,
    "lambda_P2P3": 0.125,
    "lambda_P3C": 0.125,

    "lambda_P1P2_pos": 0.300,
    "lambda_P2P3_pos": 0.300,
    "lambda_P3C_pos": 0.300,

    # Precancer clearance -> susceptible
    "c_P1": 0.10,
    "c_P2": 0.08,
    "c_P3": 0.05,

    "c_P1_pos": 0.07,
    "c_P2_pos": 0.06,
    "c_P3_pos": 0.04,

    # Cancer -> remission
    "r_c": 0.151,
    "r_c_pos": 0.068,

    # HPV infection clearance
    "c_HPV": 0.300,
    "c_HPV_pos": 0.212,
}
# Convert annual probabilities to continuous-time rates
params_fixed = {
    name: prob_to_rate(probability)
    for name, probability in params_fixed_proba.items()
}
# ============================================================
# 2. AGE-DEPENDENT PARAMETERS
# ============================================================
params = {
    "lambda_HPV": [prob_to_rate(0.256), 21.0, 5.0],
    "lambda_HPV_pos": [prob_to_rate(0.380), 19.0, 5.0],
    "mu_c": [prob_to_rate(0.35), 52.0, 10.0],
    "mu_c_pos": [prob_to_rate(0.55), 45.0, 10.0],
    "lambda_HIV": [0.004, 33.0, 16.0],
}
params_vector = build_params_vector(params)
# ============================================================
# 3. NO-VACCINATION PARAMETERS
# ============================================================


def extract_calibration_targets(df_field_data):
    """Pull calibration target arrays from the field-data sheet."""
    cc_incidence_data    = df_field_data["cc_incidence_per100k"].to_numpy(dtype=float)
    cc_mortality_data    = df_field_data["cc_mortality_per100k"].to_numpy(dtype=float)
    cc_hivprevalence_data = df_field_data["hiv_prevalence"].to_numpy(dtype=float)
    if not (len(cc_incidence_data) == len(cc_mortality_data) == len(cc_hivprevalence_data)):
        raise ValueError("Incidence, mortality and HIV prevalence must have the same number of age classes.")
    return cc_incidence_data, cc_mortality_data, cc_hivprevalence_data


AGE_PARAMETER_NAMES = [
    "lambda_HPV",
    "lambda_HPV_pos",
    "mu_c",
    "mu_c_pos",
    "lambda_HIV",
]

EXPECTED_AGE_VECTOR_SIZE = 15
EXPECTED_CALIBRATION_VECTOR_SIZE = 17

def probability_to_rate(probability):
    probability = float(probability)

    if not 0.0 <= probability < 1.0:
        raise ValueError(
            "Probability must satisfy 0 <= p < 1. "
            f"Received {probability}."
        )

    return -np.log1p(-probability)

def rate_to_probability(rate):
    rate = float(rate)

    if rate < 0.0:
        raise ValueError(
            "Rate must be non-negative. "
            f"Received {rate}."
        )

    return 1.0 - np.exp(-rate)

def build_calibration_vector(
    age_parameters,
    duration_neg_total,
    duration_pos_total,
):
    values = []

    for name in AGE_PARAMETER_NAMES:
        group = age_parameters[name]

        if len(group) != 3:
            raise ValueError(
                f"{name} must contain exactly three values: "
                "amplitude, mean and sigma."
            )

        values.extend([
            float(group[0]),
            float(group[1]),
            float(group[2]),
        ])

    values.extend([
        float(duration_neg_total),
        float(duration_pos_total),
    ])

    return np.asarray(
        values,
        dtype=float,
    )

def derive_progression_rate(
    total_duration,
    clearance_rate,
    number_of_stages=3,
):
    total_duration = float(total_duration)
    clearance_rate = float(clearance_rate)
    number_of_stages = int(number_of_stages)

    if total_duration <= 0:
        raise ValueError(
            "Total tunnel duration must be positive."
        )

    if number_of_stages <= 0:
        raise ValueError(
            "Number of stages must be positive."
        )

    stage_duration = (
        total_duration
        / number_of_stages
    )

    progression_rate = (
        1.0 / stage_duration
        - clearance_rate
    )

    if progression_rate <= 0:
        raise ValueError(
            "The selected tunnel duration is incompatible "
            "with the fixed clearance rate. "
            f"Total duration={total_duration:.6f}, "
            f"clearance rate={clearance_rate:.6f}, "
            f"derived progression rate={progression_rate:.6f}."
        )

    return float(progression_rate)

def build_params_fixed_from_durations(
    duration_neg_total,
    duration_pos_total,
):
    candidate_params_fixed = params_fixed.copy()

    candidate_params_fixed["lambda_P1P2"] = (
        derive_progression_rate(
            total_duration=duration_neg_total,
            clearance_rate=candidate_params_fixed["c_P1"],
        )
    )

    candidate_params_fixed["lambda_P2P3"] = (
        derive_progression_rate(
            total_duration=duration_neg_total,
            clearance_rate=candidate_params_fixed["c_P2"],
        )
    )

    candidate_params_fixed["lambda_P3C"] = (
        derive_progression_rate(
            total_duration=duration_neg_total,
            clearance_rate=candidate_params_fixed["c_P3"],
        )
    )

    candidate_params_fixed["lambda_P1P2_pos"] = (
        derive_progression_rate(
            total_duration=duration_pos_total,
            clearance_rate=candidate_params_fixed["c_P1_pos"],
        )
    )

    candidate_params_fixed["lambda_P2P3_pos"] = (
        derive_progression_rate(
            total_duration=duration_pos_total,
            clearance_rate=candidate_params_fixed["c_P2_pos"],
        )
    )

    candidate_params_fixed["lambda_P3C_pos"] = (
        derive_progression_rate(
            total_duration=duration_pos_total,
            clearance_rate=candidate_params_fixed["c_P3_pos"],
        )
    )

    return candidate_params_fixed

RR_AGE_GROUPS = [
    ("0-14", 0, 14),
    ("15-19", 15, 19),
    ("20-24", 20, 24),
    ("25-29", 25, 29),
    ("30-34", 30, 34),
    ("35-39", 35, 39),
    ("40-44", 40, 44),
    ("45-49", 45, 49),
    ("50-54", 50, 54),
    ("55-59", 55, 59),
    ("60-64", 60, 64),
    ("65-69", 65, 69),
    ("70-74", 70, 74),
    ("75-79", 75, 79),
    ("80-84", 80, 84),
    ("85-100", 85, 100),
]

def calculate_incidence_rr_table(
    state_neg,
    state_pos,
    model_ages,
):
    state_neg = np.asarray(
        state_neg,
        dtype=float,
    )

    state_pos = np.asarray(
        state_pos,
        dtype=float,
    )

    model_ages = np.asarray(
        model_ages,
        dtype=float,
    )

    alive_neg_indices = [
        NEG_NAMES.index("S"),
        NEG_NAMES.index("V"),
        NEG_NAMES.index("I"),
        NEG_NAMES.index("P1"),
        NEG_NAMES.index("P2"),
        NEG_NAMES.index("P3"),
        NEG_NAMES.index("C"),
        NEG_NAMES.index("R"),
    ]

    alive_pos_indices = [
        POS_NAMES.index("Sp"),
        POS_NAMES.index("Vp"),
        POS_NAMES.index("Vp_b"),
        POS_NAMES.index("Ip"),
        POS_NAMES.index("Pp1"),
        POS_NAMES.index("Pp2"),
        POS_NAMES.index("Pp3"),
        POS_NAMES.index("Cp"),
        POS_NAMES.index("Rp"),
    ]

    alive_neg = state_neg[
        :,
        alive_neg_indices,
    ].sum(axis=1)

    alive_pos = state_pos[
        :,
        alive_pos_indices,
    ].sum(axis=1)

    cases_neg = state_neg[
        :,
        NEG_NAMES.index("new_cases"),
    ]

    cases_pos = state_pos[
        :,
        POS_NAMES.index("new_cases_p"),
    ]

    rows = []

    for age_group, age_min, age_max in RR_AGE_GROUPS:
        mask = (
            (model_ages >= age_min)
            & (model_ages <= age_max)
        )

        population_neg = alive_neg[mask].sum()
        population_pos = alive_pos[mask].sum()

        cancer_cases_neg = cases_neg[mask].sum()
        cancer_cases_pos = cases_pos[mask].sum()

        incidence_neg = (
            cancer_cases_neg
            / population_neg
            * 100000
            if population_neg > 0
            else np.nan
        )

        incidence_pos = (
            cancer_cases_pos
            / population_pos
            * 100000
            if population_pos > 0
            else np.nan
        )

        incidence_rr = (
            incidence_pos
            / incidence_neg
            if (
                np.isfinite(incidence_neg)
                and incidence_neg > 0
            )
            else np.nan
        )

        rows.append({
            "age_group": age_group,
            "age_min": age_min,
            "age_max": age_max,
            "hiv_negative_population": population_neg,
            "hiv_positive_population": population_pos,
            "hiv_negative_cases": cancer_cases_neg,
            "hiv_positive_cases": cancer_cases_pos,
            "cc_incidence_hiv_negative_per100k": incidence_neg,
            "cc_incidence_hiv_positive_per100k": incidence_pos,
            "incidence_rate_ratio": incidence_rr,
        })

    return pd.DataFrame(rows)

RR_TARGET = 6.0
RR_MIN = 1.0
RR_MAX = 10.0
RR_AGE_MIN = 20
RR_AGE_MAX = 69

def incidence_rr_penalty(
    state_neg,
    state_pos,
    model_ages,
    rr_target=RR_TARGET,
    rr_min=RR_MIN,
    rr_max=RR_MAX,
):
    df_rr = calculate_incidence_rr_table(
        state_neg,
        state_pos,
        model_ages,
    )

    valid = (
        df_rr["incidence_rate_ratio"].notna()
        & np.isfinite(df_rr["incidence_rate_ratio"])
        & (df_rr["age_min"] >= RR_AGE_MIN)
        & (df_rr["age_max"] <= RR_AGE_MAX)
        & (
            (
                df_rr["hiv_negative_cases"]
                + df_rr["hiv_positive_cases"]
            ) > 1e-8
        )
    )

    rr_values = df_rr.loc[
        valid,
        "incidence_rate_ratio",
    ].to_numpy(dtype=float)

    if rr_values.size == 0:
        return 0.0

    rr_values = np.maximum(
        rr_values,
        1e-9,
    )

    target_component = (
        np.log(
            rr_values
            / rr_target
        )
    ) ** 2

    lower_component = (
        np.maximum(
            rr_min - rr_values,
            0.0,
        )
        / rr_target
    ) ** 2

    upper_component = (
        np.maximum(
            rr_values - rr_max,
            0.0,
        )
        / rr_target
    ) ** 2

    return float(
        np.mean(
            target_component
            + lower_component
            + upper_component
        )
    )

def validate_candidate_vector(candidate_vector):
    candidate_vector = np.asarray(
        candidate_vector,
        dtype=float,
    )

    if candidate_vector.size != EXPECTED_CALIBRATION_VECTOR_SIZE:
        return False

    if not np.all(
        np.isfinite(candidate_vector)
    ):
        return False

    amplitude_indices = [
        0,
        3,
        6,
        9,
        12,
    ]

    sigma_indices = [
        2,
        5,
        8,
        11,
        14,
    ]

    if np.any(
        candidate_vector[amplitude_indices] < 0
    ):
        return False

    if np.any(
        candidate_vector[sigma_indices] <= 0
    ):
        return False

    if candidate_vector[15] <= 0:
        return False

    if candidate_vector[16] <= 0:
        return False

    return True

CALIBRATION_WEIGHTS = {
    "inc": 1.0,
    "mort": 1.0,
    "hiv": 5.0,
}

REGULARIZATION_WEIGHT = 0.05
RR_REGULARIZATION_WEIGHT = 0.05

failed_evaluations = {
    "count": 0,
}

