"""Vaccination effect vectors + baseline scenarios — Cameroon Updated Framework.

Key differences vs previous version:
  - DURATION_BOOST = 30 (not 65)
  - DURATION_POS = 10
  - OPT_COVERAGE = 0.98 (cell 147)
  - S3-S6 catch-up uses 3-dose HIV-positive campaign (CATCHUP_DOSE_NUMBERS_HIV_POS=3)
  - build_vax_vectors_by_age_old (cell 140) dropped — dead code.
"""
import numpy as np
from typing import Dict, List

from .model import prob_to_rate
from .constants import N_YEARS

def build_vax_vectors_by_age(
    ages: np.ndarray,
    campaigns: List[Dict],
) -> Dict[str, np.ndarray]:

    ages = np.asarray(ages, dtype=float)
    N_YEARS = len(ages)

    gamma = np.zeros(N_YEARS)      # vaccination HIV-
    gamma_h = np.zeros(N_YEARS)    # vaccination HIV+
    rho = np.zeros(N_YEARS)        # booster HIV+

    VE = 0.0
    VE_p = 0.0
    VE_boost = 0.0

    delta = np.ones(N_YEARS)
    delta_h = np.ones(N_YEARS)
    delta_boost = np.ones(N_YEARS)

    has_delta = np.zeros(N_YEARS, dtype=bool)
    has_delta_h = np.zeros(N_YEARS, dtype=bool)
    has_delta_boost = np.zeros(N_YEARS, dtype=bool)

    def _apply_waning(waning_array, has_array, idx, duration, VE_x):
        for i in idx:
            age_vax = ages[i]

            if duration is None:
                end_idx = N_YEARS
            else:
                end_age = age_vax + duration
                end_idx = np.searchsorted(ages, end_age, side="right")

            t = ages[i:end_idx] - age_vax

            if duration is None:
                # Permanent protection:
                # residual risk remains equal to 1 - VE_x
                w = np.full_like(t, 1.0 - VE_x, dtype=float)
            else:
                # Residual risk increases from 1 - VE_x to 1
                # as vaccine protection wanes.
                w = (1.0 - VE_x) + VE_x * (t / duration)
                w = np.clip(w, 1.0 - VE_x, 1.0)

            waning_array[i:end_idx] = np.where(
                has_array[i:end_idx],
                np.minimum(waning_array[i:end_idx], w),
                w
            )

            has_array[i:end_idx] = True

    for camp in campaigns:
        hiv_grp = camp["targets"].get("hiv", "both")
        age_list = np.array(camp["targets"].get("age_ranges", []), dtype=int)
        action = camp.get("action", "primary")
        cov = float(camp.get("coverage", 0.0))
        eff = camp.get("efficacy", {})
        dur_p = camp.get("duration_primary", None)
        dur_b = camp.get("duration_boost", None)

        mask = np.isin(ages.astype(int), age_list)

        if not np.any(mask):
            continue

        idx = np.where(mask)[0]

        if action == "primary":
            if hiv_grp in ("neg", "both"):
                gamma[idx] = np.maximum(gamma[idx], cov)
                ve_val = float(eff.get("VE", 0.0))
                VE = max(VE, ve_val)
                _apply_waning(delta, has_delta, idx, dur_p, ve_val)

            if hiv_grp in ("pos", "both"):
                gamma_h[idx] = np.maximum(gamma_h[idx], cov)
                ve_val = float(eff.get("VE_p", 0.0))
                VE_p = max(VE_p, ve_val)
                _apply_waning(delta_h, has_delta_h, idx, dur_p, ve_val)

        elif action == "booster":
            idx_boost = idx[idx < N_YEARS]

            rho[idx_boost] = np.maximum(rho[idx_boost], cov)
            ve_val = float(eff.get("VE_boost", 0.0))
            VE_boost = max(VE_boost, ve_val)

            _apply_waning(
                delta_boost,
                has_delta_boost,
                idx_boost,
                dur_b,
                ve_val
            )

        else:
            raise ValueError(f"Unknown action '{action}'")

    return {
        "gamma": prob_to_rate(gamma),
        "gamma_h": prob_to_rate(gamma_h),
        "rho": prob_to_rate(rho),
        "delta": delta,
        "delta_h": delta_h,
        "delta_boost": delta_boost,
        "VE": VE,
        "VE_p": VE_p,
        "VE_b": VE_boost,
    }

vax_params = {
    "gamma": np.zeros(N_YEARS),
    "gamma_h": np.zeros(N_YEARS),
    "rho": np.zeros(N_YEARS),
    # Residual infection risk: 1 = no protection
    "delta": np.ones(N_YEARS),
    "delta_h": np.ones(N_YEARS),
    "delta_boost": np.ones(N_YEARS),
    "VE": np.zeros(N_YEARS),
    "VE_p": np.zeros(N_YEARS),
    "VE_b": np.zeros(N_YEARS),
}



def build_baseline_scenarios(
    ages,
    *,
    age_vacc=14,
    base_coverage=0.50,
    opt_coverage=0.98,
    delta_neg=0.96,
    duration_neg=20,
    delta_pos=0.87,
    duration_pos=10,
    delta_boost=0.96,
    duration_boost=30,
):
    """Build S-1..S6 vaccination-scenario dicts (cell-147 values as defaults).

    NOTE on duration_boost: cell 147 sets DURATION_BOOST=30 (not 65 as in the
    previous Cameroon version). All defaults come from cell 147.

    NOTE on S3-S6: these scenarios include HIV-positive catch-up with 3 physical
    doses per recipient, tracked via apply_hiv_pos_catchup_dose_adjustment in
    multicohort.run_scenarios_to_summary_and_plots.
    """
    s0 = [
        {"targets": {"hiv": "neg", "age_ranges": [age_vacc]}, "action": "primary",
         "coverage": base_coverage, "efficacy": {"VE": delta_neg}, "duration_primary": duration_neg},
        {"targets": {"hiv": "pos", "age_ranges": [age_vacc]}, "action": "primary",
         "coverage": base_coverage, "efficacy": {"VE_p": delta_pos}, "duration_primary": duration_pos},
    ]
    vec_s0 = build_vax_vectors_by_age(ages, s0)

    s_no_vax = [{"targets": {"hiv": "both", "age_ranges": [age_vacc]}, "action": "primary",
                  "coverage": 0.0, "efficacy": {"VE": 0.0, "VE_p": 0.0}, "duration_primary": None}]
    vec_no_vax = build_vax_vectors_by_age(ages, s_no_vax)

    def _boost(age): return {"targets": {"hiv": "pos", "age_ranges": [age]}, "action": "booster",
        "coverage": base_coverage, "efficacy": {"VE_boost": delta_boost}, "duration_boost": duration_boost}
    def _cu(age, cov=None): return {"targets": {"hiv": "pos", "age_ranges": [age]}, "action": "primary",
        "coverage": cov or base_coverage, "efficacy": {"VE_p": delta_pos}, "duration_primary": duration_pos}
    def _cu_boost(age, cov=None): return {"targets": {"hiv": "pos", "age_ranges": [age]}, "action": "booster",
        "coverage": cov or base_coverage, "efficacy": {"VE_boost": delta_boost}, "duration_boost": duration_boost}
    def _cu_opt(age): return _cu(age, opt_coverage)
    def _cu_boost_opt(age): return _cu_boost(age, opt_coverage)

    vec_s1 = build_vax_vectors_by_age(ages, s0 + [_boost(18)])
    vec_s2 = build_vax_vectors_by_age(ages, s0 + [_boost(24)])
    vec_s3 = build_vax_vectors_by_age(ages, s0 + [_cu(18), _boost(18)])
    vec_s4 = build_vax_vectors_by_age(ages, s0 + [_cu(24), _boost(24)])
    vec_s5 = build_vax_vectors_by_age(ages, s0 + [_cu_opt(18), _cu_boost_opt(18)])
    vec_s6 = build_vax_vectors_by_age(ages, s0 + [_cu_opt(24), _cu_boost_opt(24)])

    strategy_descriptions = {
        "S0": "Routine primary HPV vaccination at age 14 (HIV-/HIV+, 2 doses, 50% coverage)",
        "S1": "S0 + 1-dose booster HIV+ age 18 (50% coverage)",
        "S2": "S0 + 1-dose booster HIV+ age 24 (50% coverage)",
        "S3": "S0 + 3-dose catch-up HIV+ age 18 + booster (50% coverage)",
        "S4": "S0 + 3-dose catch-up HIV+ age 24 + booster (50% coverage)",
        "S5": "S3 at optimised coverage (98%)",
        "S6": "S4 at optimised coverage (98%)",
    }

    scenarios_s0 = {k: {"vec": v, "description": strategy_descriptions[k]}
                    for k, v in zip(["S0","S1","S2","S3","S4","S5","S6"],
                                    [vec_s0,vec_s1,vec_s2,vec_s3,vec_s4,vec_s5,vec_s6])}
    s_novax_spec = {"S-1": {"vec": vec_no_vax, "description": "No vaccination (reference)"}}
    scenario_novax = {**s_novax_spec, **scenarios_s0}

    # Scenario specs for apply_hiv_pos_catchup_dose_adjustment
    scenario_specs = {
        "S0": {"vax_type": "primary"},
        "S1": {"vax_type": "booster"},
        "S2": {"vax_type": "booster"},
        "S3": {"vax_type": "catchup", "cu_age": 18},
        "S4": {"vax_type": "catchup", "cu_age": 24},
        "S5": {"vax_type": "catchup", "cu_age": 18, "cu_cov": opt_coverage},
        "S6": {"vax_type": "catchup", "cu_age": 24, "cu_cov": opt_coverage},
    }

    return {
        "SCENARIOS_SO":       scenarios_s0,
        "SCENARIO_novax":     scenario_novax,
        "SCENARIOS_SO_Only":  {"S0": scenarios_s0["S0"]},
        "scenario_specs":     scenario_specs,
        "strategy_descriptions": strategy_descriptions,
    }
