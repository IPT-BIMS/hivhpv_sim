"""Core simulation engine — Cameroon Updated Framework.

Key additions vs previous version:
  apply_hiv_pos_catchup_dose_adjustment (cell 53): corrects dose counts for
  S3-S6 HIV-positive catch-up recipients (3 physical doses vs 2 for routine).

prob_to_rate: cell-141 vectorised version used (cell-39 scalar dropped, confirmed).
validate_params_detailed: still a no-op (pre-existing bug from source notebook).
"""
import numpy as np
import pandas as pd

from .competing_risks import split_competing_risks, _mass_conservation_check
from .demography import remainlife_expectancy, discounted_years
from .constants import (
    N_YEARS, SIMULATION_YEARS, NEG_NAMES, POS_NAMES,
    n_HIV_neg, n_HIV_pos, DISABILITY_WEIGHT, DISCOUNT_RATE_HEALTH,
    CANCER_COST, DOSE_PRICE, DOSE_PRICE_BOOST, COST_PER_PHYSICAL_DOSE,
    CATCHUP_DOSE_NUMBERS_HIV_POS, DOSE_PRICE_CATCHUP_HIV_POS,
    CC_AGE_BINS, discountVectCost, ages, DOSE_NUMBERS,
)

get_mortality_rate = None
get_mortality_HIV  = None


def configure_mortality(get_mortality_rate_fn, get_mortality_HIV_fn):
    """Bind mortality functions before calling run_model."""
    global get_mortality_rate, get_mortality_HIV
    get_mortality_rate = get_mortality_rate_fn
    get_mortality_HIV  = get_mortality_HIV_fn

def gamma(x, alpha, beta, A):
    """Scaled gamma curve fit (x must be > 0)."""
    return A * x**(alpha - 1) * np.exp(-x / beta)

def gaussian(x, A, mu, sigma):
    """Gaussian.
    
    Parameters
    ----------
    x : Any
        Parameter `x`.
    A : Any
        Parameter `A`.
    mu : Any
        Parameter `mu`.
    sigma : Any
        Parameter `sigma`.
    
    Returns
    -------
    Any
        Function output.
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def prob_to_rate(p):
    """ Convert annual probability/probabilities into continuous-time rate/rates. rate = -log(1 - p) """
    p = np.asarray(p, dtype=float)
    rate = -np.log1p(-p)
    return float(rate) if rate.ndim == 0 else rate

def build_params_vector(params_dict):
    """
    Construit le vecteur ordonné depuis le dictionnaire params.
    Ordre garanti : HPV, HPV_pos, mu_c, mu_c_pos, HIV
    """
    return (
        params_dict['lambda_HPV']
        + params_dict['lambda_HPV_pos']
        + params_dict['mu_c']
        + params_dict['mu_c_pos']
        + params_dict['lambda_HIV']
    )

def create_age_functions(params): 
    params = np.asarray(params, dtype=float)

    if len(params) != 15:
        raise ValueError("Expected 15 parameters")

    if np.any(params[[2, 5, 8, 11, 14]] <= 0):
        raise ValueError("All Gaussian standard deviations must be positive")
    
# age-dependent Gaussian functions.
    """
    Crée les fonctions age-dépendantes à partir du vecteur params.

    Ordre attendu dans params :
        [0,1,2]   → lambda_HPV     (amp, mean_age, std)
        [3,4,5]   → lambda_HPV_pos (amp, mean_age, std)
        [6,7,8]   → mu_c           (amp, mean_age, std)
        [9,10,11] → mu_c_pos       (amp, mean_age, std)
        [12,13,14]→ lambda_HIV     (amp, mean_age, std)
    """

    # Extraction explicite — plus robuste que params[-3]
    amp_hpv,     mu_hpv,     sig_hpv     = params[0],  params[1],  params[2]
    amp_hpv_pos, mu_hpv_pos, sig_hpv_pos = params[3],  params[4],  params[5]
    amp_mc,      mu_mc,      sig_mc      = params[6],  params[7],  params[8]
    amp_mc_pos,  mu_mc_pos,  sig_mc_pos  = params[9],  params[10], params[11]
    amp_hiv,     mu_hiv,     sig_hiv     = params[12], params[13], params[14]

    # CORRECTION : toutes les fonctions utilisent la même gaussienne
    # lambda_HPV HIV- : pic à ~21 ans
    f_lambda_HPV = lambda a: gaussian(a, amp_hpv, mu_hpv, sig_hpv)

    # lambda_HPV HIV+ : pic plus précoce (~19 ans) et amplitude plus haute
    f_lambda_HPV_pos = lambda a: gaussian(a, amp_hpv_pos, mu_hpv_pos, sig_hpv_pos)

    # mu_c HIV- : mortalité cancer, pic ~52 ans
    f_mu_c = lambda a: gaussian(a, amp_mc, mu_mc, sig_mc)

    # mu_c HIV+ : mortalité cancer, pic ~45 ans, amplitude plus haute
    f_mu_c_pos = lambda a: gaussian(a, amp_mc_pos, mu_mc_pos, sig_mc_pos)

    # lambda_HIV : incidence HIV, pic ~28–33 ans
    f_lambda_HIV = lambda a: gaussian(a, amp_hiv, mu_hiv, sig_hiv)

    return f_lambda_HPV, f_lambda_HPV_pos, f_mu_c, f_mu_c_pos, f_lambda_HIV

def validate_params_detailed(
    params,
    params_fixed,
    vax_params,
    tol=1e-12,
    report=False,
):
    """Check that transition controls and probabilities remain within [0, 1]."""
    lambda_hpv, lambda_hpv_pos, mu_c, mu_c_pos, lambda_hiv = create_age_functions(params)
    ages_step = ages[:N_YEARS]
    mu_arr = np.asarray(get_mortality_rate(ages_step), dtype=float)
    mu_hiv_arr = np.asarray(get_mortality_HIV(ages_step), dtype=float)
    mu_c_arr = np.asarray(mu_c(ages_step), dtype=float)
    mu_c_pos_arr = np.asarray(mu_c_pos(ages_step), dtype=float)
    lambda_hpv_arr = np.asarray(lambda_hpv(ages_step), dtype=float)
    lambda_hpv_pos_arr = np.asarray(lambda_hpv_pos(ages_step), dtype=float)
    lambda_hiv_arr = np.asarray(lambda_hiv(ages_step), dtype=float)
    gamma_arr = np.asarray(vax_params["gamma"], dtype=float)[:N_YEARS]
    gamma_h_arr = np.asarray(vax_params["gamma_h"], dtype=float)[:N_YEARS]
    rho_arr = np.asarray(vax_params["rho"], dtype=float)[:N_YEARS]
    delta_arr = np.asarray(vax_params["delta"], dtype=float)[:N_YEARS]
    delta_h_arr = np.asarray(vax_params["delta_h"], dtype=float)[:N_YEARS]
    delta_boost_arr = np.asarray(vax_params["delta_boost"], dtype=float)[:N_YEARS]
    rows = []
    any_fail = False
    def check_01(name, values, context=""):
        nonlocal any_fail
        values = np.atleast_1d(np.asarray(values, dtype=float))
        for index, value in enumerate(values):
            if np.isnan(value):
                ok = False
                reason = "NaN"
                severity = np.nan
            elif value < -tol or value > 1 + tol:
                ok = False
                reason = "outside [0, 1]"
                severity = -value if value < 0 else value - 1
            else:
                ok = True
                reason = "ok"
                severity = 0.0
            if not ok:
                any_fail = True
            if report:
                rows.append({
                    "check_name": name,
                    "index": index,
                    "value": value,
                    "ok": ok,
                    "reason": reason,
                    "severity": severity,
                    "context": context,
                })
    # Vaccination controls and residual infection risks
    check_01("gamma", gamma_arr, "HIV-negative vaccination")
    check_01("gamma_h", gamma_h_arr, "HIV-positive vaccination")
    check_01("rho", rho_arr, "booster vaccination")
    check_01("delta", delta_arr, "HIV-negative residual infection risk")
    check_01("delta_h", delta_h_arr, "HIV-positive residual infection risk")
    check_01("delta_boost", delta_boost_arr, "booster residual infection risk")
    # Annual transition probabilities derived from rates
    rate_arrays = {
        "natural_mortality": mu_arr,
        "HIV_mortality": mu_hiv_arr,
        "cancer_mortality": mu_c_arr,
        "cancer_mortality_pos": mu_c_pos_arr,
        "HPV_acquisition": lambda_hpv_arr,
        "HPV_acquisition_pos": lambda_hpv_pos_arr,
        "HIV_acquisition": lambda_hiv_arr,
    }
    for name, rates in rate_arrays.items():
        probabilities = 1 - np.exp(-rates)
        check_01(name, probabilities, "rate converted to annual probability")
    # Fixed transition rates converted back to annual probabilities
    for name, rate in params_fixed.items():
        probability = 1 - np.exp(-float(rate))
        check_01(name, probability, "fixed natural-history parameter")
    ok = not any_fail
    if not report:
        return ok
    report_df = pd.DataFrame(rows)
    if ok:
        report_df = report_df.iloc[0:0]
    else:
        report_df = report_df.loc[~report_df["ok"]].reset_index(drop=True)
    return ok, report_df

def run_model(params, params_fixed, vax_params, return_eff=False, run_tests=False, test_tol=1e-12, dt=1.0):
    """
    Multiplicative / competing-risks version of the joint HPV/HIV model.

    All updates are based on:

        stay = exp(-R * dt)

        flow_j_to_i =
            (rate_j_to_i / R)
            * (1 - exp(-R * dt))
            * X_j

    where R is the total outgoing hazard from the source compartment.
    """
    lambda_HPV, lambda_HPV_pos, mu_c, mu_c_pos, lambda_HIV_interp = create_age_functions(params)
    state_neg = np.zeros((N_YEARS + 1, len(NEG_NAMES)), dtype=float)
    state_pos = np.zeros((N_YEARS + 1, len(POS_NAMES)), dtype=float)
    state_neg[0, 0] = n_HIV_neg
    state_pos[0, 0] = n_HIV_pos

    # -------------------------------------------------------------------------
    # Precompute age-dependent quantities
    # -------------------------------------------------------------------------
    ages_step = ages[:N_YEARS]
    mu_arr = np.array([float(get_mortality_rate(a)) for a in ages_step], dtype=float)
    mu_hiv_arr = np.array([float(get_mortality_HIV(a)) for a in ages_step], dtype=float)
    lam_HPV_arr = np.array([float(lambda_HPV(a)) for a in ages_step], dtype=float)
    lam_HPV_pos_arr = np.array([float(lambda_HPV_pos(a)) for a in ages_step], dtype=float)
    mu_c_arr = np.array([float(mu_c(a)) for a in ages_step], dtype=float)
    mu_c_pos_arr = np.array([float(mu_c_pos(a)) for a in ages_step], dtype=float)
    lambda_HIV_arr = np.array([float(lambda_HIV_interp(a)) for a in ages_step], dtype=float)

    # -------------------------------------------------------------------------
    # Time-varying vaccination/protection parameters
    # -------------------------------------------------------------------------
    gamma_arr = np.asarray(vax_params["gamma"], dtype=float)[:N_YEARS]
    gamma_h_arr = np.asarray(vax_params["gamma_h"], dtype=float)[:N_YEARS]
    rho_arr = np.asarray(vax_params["rho"], dtype=float)[:N_YEARS]
    waning_arr = np.asarray(vax_params["delta"], dtype=float)[:N_YEARS]
    waning_p_arr = np.asarray(vax_params["delta_h"], dtype=float)[:N_YEARS]
    waning_b_arr = np.asarray(vax_params["delta_boost"], dtype=float)[:N_YEARS]

    # -------------------------------------------------------------------------
    # Extract fixed hazards
    # -------------------------------------------------------------------------
    c_HPV = float(params_fixed["c_HPV"])
    c_HPV_pos = float(params_fixed["c_HPV_pos"])
    r_c = float(params_fixed["r_c"])
    r_c_pos = float(params_fixed["r_c_pos"])
    lam_ip1 = float(params_fixed["lambda_IP1"])
    lam_ip1_pos = float(params_fixed["lambda_IP1_pos"])
    lam_p1p2 = float(params_fixed["lambda_P1P2"])
    lam_p1p2_pos = float(params_fixed["lambda_P1P2_pos"])
    lam_p2p3 = float(params_fixed["lambda_P2P3"])
    lam_p2p3_pos = float(params_fixed["lambda_P2P3_pos"])
    lam_p3c = float(params_fixed["lambda_P3C"])
    lam_p3c_pos = float(params_fixed["lambda_P3C_pos"])
    c_P1 = float(params_fixed["c_P1"])
    c_P2 = float(params_fixed["c_P2"])
    c_P3 = float(params_fixed["c_P3"])
    c_P1_pos = float(params_fixed["c_P1_pos"])
    c_P2_pos = float(params_fixed["c_P2_pos"])
    c_P3_pos = float(params_fixed["c_P3_pos"])

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    for t in SIMULATION_YEARS:
        i = t - 1
        mu = mu_arr[i]
        mu_HIV = mu_hiv_arr[i]
        lam_hpv = lam_HPV_arr[i]
        lam_hpv_pos = lam_HPV_pos_arr[i]
        mu_c_val = mu_c_arr[i]
        mu_c_p_val = mu_c_pos_arr[i]
        lambda_HIV = lambda_HIV_arr[i]
        waning_t = float(waning_arr[i])
        waning_p_t = float(waning_p_arr[i])
        waning_b_t = float(waning_b_arr[i])
        rho_t = float(rho_arr[i])
        gamma_t = float(gamma_arr[i])
        gamma_h_t = float(gamma_h_arr[i])
        prev_state_neg = state_neg[t - 1]
        prev_state_pos = state_pos[t - 1]
        S, V, I, P1, P2, P3, C, R, D_c, D, new_cases_prev, new_vax_prev = prev_state_neg
        Sp, Vp, Vp_b, Ip, Pp1, Pp2, Pp3, Cp, Rp, D_cp, Dp, Dhp, new_cases_p_prev, new_vax_p_prev, new_boost_p_prev = prev_state_pos

        # =====================================================================
        # TRANSITION PROBABILITIES - HIV negative
        # =====================================================================
        R_S = gamma_t + lam_hpv + lambda_HIV + mu
        stay_S, p_S = split_competing_risks(R_S, {"to_V": gamma_t, "to_I": lam_hpv, "to_Sp": lambda_HIV, "to_D": mu}, dt=dt)

        R_V = waning_t * lam_hpv + lambda_HIV + mu
        stay_V, p_V = split_competing_risks(R_V, {"to_I": waning_t * lam_hpv, "to_Vp": lambda_HIV, "to_D": mu}, dt=dt)

        R_I = c_HPV + lam_ip1 + lambda_HIV + mu
        stay_I, p_I = split_competing_risks(R_I, {"to_S": c_HPV, "to_P1": lam_ip1, "to_Ip": lambda_HIV, "to_D": mu}, dt=dt)

        R_P1 = c_P1 + lam_p1p2 + lambda_HIV + mu
        stay_P1, p_P1 = split_competing_risks(R_P1, {"to_S": c_P1, "to_P2": lam_p1p2, "to_Pp1": lambda_HIV, "to_D": mu}, dt=dt)

        R_P2 = c_P2 + lam_p2p3 + lambda_HIV + mu
        stay_P2, p_P2 = split_competing_risks(R_P2, {"to_S": c_P2, "to_P3": lam_p2p3, "to_Pp2": lambda_HIV, "to_D": mu}, dt=dt)

        R_P3 = c_P3 + lam_p3c + lambda_HIV + mu
        stay_P3, p_P3 = split_competing_risks(R_P3, {"to_S": c_P3, "to_C": lam_p3c, "to_Pp3": lambda_HIV, "to_D": mu}, dt=dt)

        R_C = r_c + mu_c_val + lambda_HIV + mu
        stay_C, p_C = split_competing_risks(R_C, {"to_R": r_c, "to_Dc": mu_c_val, "to_Cp": lambda_HIV, "to_D": mu}, dt=dt)

        R_R = lambda_HIV + mu
        stay_R, p_R = split_competing_risks(R_R, {"to_Rp": lambda_HIV, "to_D": mu}, dt=dt)

        # =====================================================================
        # TRANSITION PROBABILITIES - HIV positive
        # =====================================================================
        R_Sp = gamma_h_t + lam_hpv_pos + mu_HIV + mu
        stay_Sp, p_Sp = split_competing_risks(R_Sp, {"to_Vp": gamma_h_t, "to_Ip": lam_hpv_pos, "to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        R_Vp = rho_t + waning_p_t * lam_hpv_pos + mu_HIV + mu
        stay_Vp, p_Vp = split_competing_risks(R_Vp, {"to_Vp_b": rho_t, "to_Ip": waning_p_t * lam_hpv_pos, "to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        R_Vp_b = waning_b_t * lam_hpv_pos + mu_HIV + mu
        stay_Vp_b, p_Vp_b = split_competing_risks(R_Vp_b, {"to_Ip": waning_b_t * lam_hpv_pos, "to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        R_Ip = c_HPV_pos + lam_ip1_pos + mu_HIV + mu
        stay_Ip, p_Ip = split_competing_risks(R_Ip, {"to_Sp": c_HPV_pos, "to_Pp1": lam_ip1_pos, "to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        R_Pp1 = c_P1_pos + lam_p1p2_pos + mu_HIV + mu
        stay_Pp1, p_Pp1 = split_competing_risks(R_Pp1, {"to_Sp": c_P1_pos, "to_Pp2": lam_p1p2_pos, "to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        R_Pp2 = c_P2_pos + lam_p2p3_pos + mu_HIV + mu
        stay_Pp2, p_Pp2 = split_competing_risks(R_Pp2, {"to_Sp": c_P2_pos, "to_Pp3": lam_p2p3_pos, "to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        R_Pp3 = c_P3_pos + lam_p3c_pos + mu_HIV + mu
        stay_Pp3, p_Pp3 = split_competing_risks(R_Pp3, {"to_Sp": c_P3_pos, "to_Cp": lam_p3c_pos, "to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        R_Cp = r_c_pos + mu_c_p_val + mu_HIV + mu
        stay_Cp, p_Cp = split_competing_risks(R_Cp, {"to_Rp": r_c_pos, "to_Dcp": mu_c_p_val, "to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        R_Rp = mu_HIV + mu
        stay_Rp, p_Rp = split_competing_risks(R_Rp, {"to_Dhp": mu_HIV, "to_Dp": mu}, dt=dt)

        # =====================================================================
        # FLOWS
        # =====================================================================
        flow_S_to_V = p_S["to_V"] * S
        flow_P3_to_C = p_P3["to_C"] * P3
        flow_Pp3_to_Cp = p_Pp3["to_Cp"] * Pp3
        flow_Sp_to_Vp = p_Sp["to_Vp"] * Sp
        flow_Vp_to_Vpb = p_Vp["to_Vp_b"] * Vp
        new_cases = flow_P3_to_C
        new_cases_p = flow_Pp3_to_Cp
        new_vax = flow_S_to_V
        new_vax_p = flow_Sp_to_Vp
        new_boost_p = flow_Vp_to_Vpb

        # =====================================================================
        # UPDATED STATES - HIV negative
        # =====================================================================
        new_S = stay_S * S + p_I["to_S"] * I + p_P1["to_S"] * P1 + p_P2["to_S"] * P2 + p_P3["to_S"] * P3
        new_V = stay_V * V + p_S["to_V"] * S
        new_I = stay_I * I + p_S["to_I"] * S + p_V["to_I"] * V
        new_P1 = stay_P1 * P1 + p_I["to_P1"] * I
        new_P2 = stay_P2 * P2 + p_P1["to_P2"] * P1
        new_P3 = stay_P3 * P3 + p_P2["to_P3"] * P2
        new_C = stay_C * C + flow_P3_to_C
        new_R = stay_R * R + p_C["to_R"] * C
        new_D_c = D_c + p_C["to_Dc"] * C
        new_D = D + p_S["to_D"] * S + p_V["to_D"] * V + p_I["to_D"] * I + p_P1["to_D"] * P1 + p_P2["to_D"] * P2 + p_P3["to_D"] * P3 + p_C["to_D"] * C + p_R["to_D"] * R

        # =====================================================================
        # UPDATED STATES - HIV positive
        # =====================================================================
        new_Sp = stay_Sp * Sp + p_S["to_Sp"] * S + p_Ip["to_Sp"] * Ip + p_Pp1["to_Sp"] * Pp1 + p_Pp2["to_Sp"] * Pp2 + p_Pp3["to_Sp"] * Pp3
        new_Vp = stay_Vp * Vp + p_Sp["to_Vp"] * Sp + p_V["to_Vp"] * V
        new_Vp_b = stay_Vp_b * Vp_b + p_Vp["to_Vp_b"] * Vp
        new_Ip = stay_Ip * Ip + p_Sp["to_Ip"] * Sp + p_Vp["to_Ip"] * Vp + p_Vp_b["to_Ip"] * Vp_b + p_I["to_Ip"] * I
        new_Pp1 = stay_Pp1 * Pp1 + p_Ip["to_Pp1"] * Ip + p_P1["to_Pp1"] * P1
        new_Pp2 = stay_Pp2 * Pp2 + p_Pp1["to_Pp2"] * Pp1 + p_P2["to_Pp2"] * P2
        new_Pp3 = stay_Pp3 * Pp3 + p_Pp2["to_Pp3"] * Pp2 + p_P3["to_Pp3"] * P3
        new_Cp = stay_Cp * Cp + flow_Pp3_to_Cp + p_C["to_Cp"] * C
        new_Rp = stay_Rp * Rp + p_Cp["to_Rp"] * Cp + p_R["to_Rp"] * R
        new_D_cp = D_cp + p_Cp["to_Dcp"] * Cp
        new_Dp = Dp + p_Sp["to_Dp"] * Sp + p_Vp["to_Dp"] * Vp + p_Vp_b["to_Dp"] * Vp_b + p_Ip["to_Dp"] * Ip + p_Pp1["to_Dp"] * Pp1 + p_Pp2["to_Dp"] * Pp2 + p_Pp3["to_Dp"] * Pp3 + p_Cp["to_Dp"] * Cp + p_Rp["to_Dp"] * Rp
        new_Dhp = Dhp + p_Sp["to_Dhp"] * Sp + p_Vp["to_Dhp"] * Vp + p_Vp_b["to_Dhp"] * Vp_b + p_Ip["to_Dhp"] * Ip + p_Pp1["to_Dhp"] * Pp1 + p_Pp2["to_Dhp"] * Pp2 + p_Pp3["to_Dhp"] * Pp3 + p_Cp["to_Dhp"] * Cp + p_Rp["to_Dhp"] * Rp

        state_neg[t] = [new_S, new_V, new_I, new_P1, new_P2, new_P3, new_C, new_R, new_D_c, new_D, new_cases, new_vax]
        state_pos[t] = [new_Sp, new_Vp, new_Vp_b, new_Ip, new_Pp1, new_Pp2, new_Pp3, new_Cp, new_Rp, new_D_cp, new_Dp, new_Dhp, new_cases_p, new_vax_p, new_boost_p]

        if run_tests:
            try:
                age_i = ages_step[i]
                tot_prev = S + V + I + P1 + P2 + P3 + C + R + Sp + Vp + Vp_b + Ip + Pp1 + Pp2 + Pp3 + Cp + Rp + D + D_c + Dp + Dhp + D_cp
                tot_next = new_S + new_V + new_I + new_P1 + new_P2 + new_P3 + new_C + new_R + new_Sp + new_Vp + new_Vp_b + new_Ip + new_Pp1 + new_Pp2 + new_Pp3 + new_Cp + new_Rp + new_D + new_D_c + new_Dp + new_Dhp + new_D_cp
                _mass_conservation_check(tot_prev, tot_next, t=t, i=i, age=age_i, tol=test_tol)
            except Exception as e:
                ctx = {
                    "t": t,
                    "i": i,
                    "age": age_i,
                    "mu": float(mu),
                    "mu_HIV": float(mu_HIV),
                    "lambda_HIV": float(lambda_HIV),
                    "lam_hpv": float(lam_hpv),
                    "lam_hpv_pos": float(lam_hpv_pos),
                    "gamma_t": float(gamma_t),
                    "gamma_h_t": float(gamma_h_t),
                    "rho_t": float(rho_t),
                    "delta_t": float(waning_t),
                    "delta_h_t": float(waning_p_t),
                    "delta_b_t": float(waning_b_t),
                    "dt": float(dt),
                }
                msg = f"[run_model TEST FAILURE] {type(e).__name__}: {e}\nContext: {ctx}"
                raise ValueError(msg) from e

    if return_eff:
        return state_neg, state_pos, {"gamma_eff": gamma_arr, "gamma_h_eff": gamma_h_arr, "rho_eff": rho_arr}
    return state_neg, state_pos

def var_output(
    state_neg: np.ndarray,
    state_pos: np.ndarray,
    ages: np.ndarray,
    states_neg: list = NEG_NAMES,
    states_pos: list = POS_NAMES,
    dw: float =DISABILITY_WEIGHT,
    discount_rate: float =DISCOUNT_RATE_HEALTH,
    life_expectancy: float = 50.13,
    remaining_life_fn=None,   # pass functools.partial(demography.remainlife_expectancy, df_demography=df)
    cancer_cost_unit: float = CANCER_COST,
    vax_cost_per_dose: float = DOSE_PRICE,
    boost_cost_per_dose: float = DOSE_PRICE_BOOST,
) -> pd.DataFrame:
    """
    Build a per-age, per-year summary table aggregating core epidemiologic states,
    DALYs (YLL + YLD), and direct costs (cancer + vaccination) for HIV-negative and
    HIV-positive compartments.

    Parameters
    ----------
    state_neg : np.ndarray, shape (T, K_neg)
        Time-by-state array for HIV-negative individuals. Must contain the columns
        named in `states_neg` (see below).
    state_pos : np.ndarray, shape (T, K_pos)
        Time-by-state array for HIV-positive individuals. Must contain the columns
        named in `states_pos` (see below).
    ages : np.ndarray, shape (T,)
        Age in years for each time step (used to compute remaining life years).
    states_neg : list[str], optional
        Column names for `state_neg`. Must include:
        - "C"              : cervical cancer compartment size (or proxy)
        - "new_cases"      : incident cervical cancer cases this period
        - "D_c"            : cumulative cervical-cancer deaths
        - "D"              : other deaths among HIV-negative
        - "V"             : vaccinated stock (HIV−)
        - "new_vax"        : new primary doses (HIV−)
        (Plus any additional states you track; indices are resolved via names.index)
    states_pos : list[str], optional
        Column names for `state_pos`. Must include the HIV+ analogs:
        - "Cp", "new_cases_p", "D_cp", "Dp", "Dhp", "Spv", "Vp_b",
          "new_vax_p", "new_boost_p".
    dw : float, default 0.451
        Disability weight used to approximate YLD as `dw * incident cases`.
    discount_rate : float, default 0.03
        Annual discount rate used in `discounted_years()` for YLL.
    life_expectancy : float, default 50.13
        Mean life expectancy used if `remaining_life_fn` is not provided.
    remaining_life_fn : callable, optional
        If provided, a function mapping ages (np.ndarray) -> remaining life years
        (np.ndarray) to compute undiscounted YLL. If None, uses
        `max(life_expectancy - age, 0)`.
    cancer_cost_unit : float, default CANCER_COST
        Unit (per case) direct medical cost applied to `new_cases + new_cases_p`.
    vax_cost_per_dose : float, default DOSE_PRICE
        Unit cost per primary dose. Total primary vaccination cost is
        `2 * vax_cost_per_dose * (new_vax + new_vax_p)`.
    boost_cost_per_dose : float, default DOSE_PRICE
        Unit cost per booster dose. Booster cost is
        `boost_cost_per_dose * new_boost_p`.

    Returns
    -------
    pd.DataFrame
        One row per time step (age), with:
        - Core counts: cancer compartments/cases/deaths (neg/pos), all-cause deaths,
          alive population (neg/pos/all), HIV prevalence proportion.
        - Vaccination flows: stocks and new doses (primary/booster).
        - YLL (undiscounted/discounted), YLD, and total DALYs (undisc/disc).
        - Costs: cancer and vaccination (USD).

    Notes
    -----
    - Lower DALYs indicate better health outcomes (DALYs = YLL + YLD).
    - YLD is approximated as `dw * incident cases` for simplicity; replace with a more
      detailed duration/severity model if available.
    - Discounting of YLL uses `discounted_years()` applied to remaining life years.
    - The function assumes consistent time length T across `state_neg`, `state_pos`,
      and `ages`.
    - Sums for "alive_neg" (first 6 columns) and "alive_pos" (first 7 columns) are
      layout-dependent placeholders; adjust slices to match your state layout.
    """
    # Checks
    assert state_neg.shape[0] == state_pos.shape[0], "state_neg/state_pos length mismatch"
    # N_YEARS = state_neg.shape[0]
   # assert len(ages) == N_YEARS-1, "ages length must equal N_YEARS"

    states_neg = states_neg or []
    states_pos = states_pos or []

    def col(state, names, name):
        return state[:, names.index(name)]

    # --- Extract base variables ---
    s_neg= col(state_neg, states_neg, "S")
    s_pos= col(state_pos, states_pos, "Sp")
    i_neg= col(state_neg, states_neg, "I")
    i_pos= col(state_pos, states_pos, "Ip")

    c_neg = col(state_neg, states_neg, "C")
    c_pos = col(state_pos, states_pos, "Cp")

    cc_cases_new_neg = col(state_neg, states_neg, "new_cases")
    cc_cases_new_pos = col(state_pos, states_pos, "new_cases_p")

    # Death without natural mortality
    cc_deaths_cum_neg = col(state_neg, states_neg, "D_c")
    cc_deaths_cum_pos = col(state_pos, states_pos, "D_cp")

    # Death with natural mortality
    d_neg  = col(state_neg, states_neg, "D")
    d_pos  = col(state_pos, states_pos, "Dp")
    d_hiv  = col(state_pos, states_pos, "Dhp")

    vacc_stock_neg   = col(state_neg, states_neg, "V")
    vacc_stock_pos   = col(state_pos, states_pos, "Vp")
    vacc_stock_boost = col(state_pos, states_pos, "Vp_b")

    vacc_doses_new_neg   = col(state_neg, states_neg, "new_vax")
    vacc_doses_new_pos   = col(state_pos, states_pos, "new_vax_p")
    vacc_doses_new_boost = col(state_pos, states_pos, "new_boost_p")

    # Alive population (adapt [:6]/[:7] to your layout)
    alive_neg = state_neg[:, :8].sum(axis=1)
    alive_pos = state_pos[:, :9].sum(axis=1)
    alive_all = alive_neg + alive_pos

    # Total deaths (example)
    cc_deaths_cum_all = cc_deaths_cum_neg + cc_deaths_cum_pos
    deaths_total = cc_deaths_cum_all + d_neg + d_pos + d_hiv

    # HIV prevalence (proportion)
    with np.errstate(divide="ignore", invalid="ignore"):
        hiv_prev_prop = np.where(alive_all > 0, alive_pos / alive_all, 0.0)

    # --- Yearly rows ---
    rows = []
    for t, age in enumerate(ages):
        rows.append({
            "scope": "yearly",
            "age_years": float(age),
            
            "s_neg": float(s_neg[t]),
            "s_pos": float(s_pos[t]),
            "i_neg": float(i_neg[t]),
            "i_pos": float(i_pos[t]),

            "c_neg": c_neg[t],
            "c_pos": c_pos[t],

            "cc_cases_new_neg": cc_cases_new_neg[t],
            "cc_cases_new_pos": cc_cases_new_pos[t],

            "cc_deaths_cum_neg": cc_deaths_cum_neg[t],
            "cc_deaths_cum_pos": cc_deaths_cum_pos[t],

            "d_neg": d_neg[t],
            "d_pos": d_pos[t],
            "d_hiv": d_hiv[t],

            "deaths_total": deaths_total[t],

            "alive_neg": alive_neg[t],
            "alive_pos": alive_pos[t],
            "alive_all": alive_all[t],

            "hiv_prev_prop": hiv_prev_prop[t],

            "vacc_stock_neg":   vacc_stock_neg[t],
            "vacc_stock_pos":   vacc_stock_pos[t],
            "vacc_stock_boost": vacc_stock_boost[t],

            "vacc_doses_new_neg":   float(vacc_doses_new_neg[t]),
            "vacc_doses_new_pos":   float(vacc_doses_new_pos[t]),
            "vacc_doses_new_boost": float(vacc_doses_new_boost[t]),
        })

    df_yearly = pd.DataFrame(rows)

    # --- DALYs & costs on yearly only ---
    # Remaining life (undiscounted or via callable). If not provided, use a simple
    # life-expectancy minus age approximation, floored at 0.
    if remaining_life_fn is not None:
        rem_years = remaining_life_fn(df_yearly["age_years"].to_numpy())
    else:
        rem_years = (life_expectancy - df_yearly["age_years"]).clip(lower=0)

   # ------------------------------------------------------------
    # DALYs
    # ------------------------------------------------------------
    df_yearly["remaining_life_years"] = rem_years
    df_yearly["remaining_life_disc_years"] = discounted_years(df_yearly["remaining_life_years"], r_annual=discount_rate)
    health_discount_factor = 1.0 / ((1.0 + discount_rate) ** np.arange(len(df_yearly)))
    
    # New cervical-cancer deaths during each year
    cc_deaths_new_neg = np.diff(cc_deaths_cum_neg, prepend=0)
    cc_deaths_new_pos = np.diff(cc_deaths_cum_pos, prepend=0)
    df_yearly["cc_deaths_new_neg"] = cc_deaths_new_neg
    df_yearly["cc_deaths_new_pos"] = cc_deaths_new_pos
    
    # YLL — undiscounted
    df_yearly["yll_neg_undisc"] = df_yearly["remaining_life_years"] * cc_deaths_new_neg
    df_yearly["yll_pos_undisc"] = df_yearly["remaining_life_years"] * cc_deaths_new_pos
    df_yearly["yll_all_undisc"] = df_yearly["yll_neg_undisc"] + df_yearly["yll_pos_undisc"]
    
    # YLL — discounted
    df_yearly["yll_neg_disc"] = df_yearly["remaining_life_disc_years"] * cc_deaths_new_neg * health_discount_factor
    df_yearly["yll_pos_disc"] = df_yearly["remaining_life_disc_years"] * cc_deaths_new_pos * health_discount_factor
    df_yearly["yll_all_disc"] = df_yearly["yll_neg_disc"] + df_yearly["yll_pos_disc"]
    
    # YLD — undiscounted
    df_yearly["yld_neg_undisc"] = dw * df_yearly["cc_cases_new_neg"]
    df_yearly["yld_pos_undisc"] = dw * df_yearly["cc_cases_new_pos"]
    df_yearly["yld_all_undisc"] = df_yearly["yld_neg_undisc"] + df_yearly["yld_pos_undisc"]
    
    # YLD — discounted
    df_yearly["yld_neg_disc"] = df_yearly["yld_neg_undisc"] * health_discount_factor
    df_yearly["yld_pos_disc"] = df_yearly["yld_pos_undisc"] * health_discount_factor
    df_yearly["yld_all_disc"] = df_yearly["yld_neg_disc"] + df_yearly["yld_pos_disc"]
    
    # Keep old column name
    df_yearly["yld_all"] = df_yearly["yld_all_undisc"]
    
    # DALYs
    df_yearly["dalys_all_undisc"] = df_yearly["yll_all_undisc"] + df_yearly["yld_all_undisc"]
    df_yearly["dalys_all_disc"] = df_yearly["yll_all_disc"] + df_yearly["yld_all_disc"]
    # Costs
    y = cancer_cost_unit * (df_yearly["cc_cases_new_neg"] + df_yearly["cc_cases_new_pos"])
    df_yearly["cost_cancer_usd"]=y* discountVectCost
    y= (
         vax_cost_per_dose   * (df_yearly["vacc_doses_new_neg"] + df_yearly["vacc_doses_new_pos"]) +
        boost_cost_per_dose *    df_yearly["vacc_doses_new_boost"]
    )
    df_yearly["cost_vaccination_usd"] = y * discountVectCost

    return df_yearly

def apply_hiv_pos_catchup_dose_adjustment(
    df_year,
    scenario_spec,
    *,
    routine_primary_doses=DOSE_NUMBERS,
    cost_per_physical_dose=COST_PER_PHYSICAL_DOSE,
):
    """
    Add the third physical dose and its cost for HIV-positive
    catch-up recipients in S3--S6.

    The epidemiological model records one primary-vaccination event
    per vaccinated person. Routine primary vaccination is already
    costed as a two-dose course in var_output.

    Therefore, one additional physical dose and its cost are added
    for each HIV-positive catch-up recipient.
    """

    df_year = df_year.copy()

    # Initialise the new columns for every scenario
    df_year["hiv_pos_catchup_recipients"] = 0.0
    df_year["extra_catchup_physical_doses"] = 0.0
    df_year["extra_catchup_cost_usd"] = 0.0

    catchup_age = scenario_spec.get("catchup_age")
    catchup_doses = scenario_spec.get(
        "catchup_primary_doses_hiv_pos",
        0,
    )

    # S0, S1, S2 and no-vaccination have no catch-up course
    if catchup_age is None or catchup_doses <= routine_primary_doses:
        return df_year

    # In run_model, the campaign hazard at age a is applied during
    # the interval [a, a+1), and the resulting flow is stored in
    # the next output row.
    catchup_output_age = catchup_age + 1

    catchup_mask = (
        df_year["age_years"] == catchup_output_age
    )

    catchup_recipients = (
        df_year.loc[
            catchup_mask,
            "vacc_doses_new_pos"
        ]
        .astype(float)
    )

    extra_doses_per_recipient = (
        catchup_doses
        - routine_primary_doses
    )

    extra_physical_doses = (
        catchup_recipients
        * extra_doses_per_recipient
    )

    # discountVectCost has the same row order as df_year
    row_discount_factors = np.asarray(
        discountVectCost[:len(df_year)],
        dtype=float,
    )

    extra_cost = (
        extra_physical_doses.to_numpy(dtype=float)
        * cost_per_physical_dose
        * row_discount_factors[catchup_mask.to_numpy()]
    )

    df_year.loc[
        catchup_mask,
        "hiv_pos_catchup_recipients"
    ] = catchup_recipients.to_numpy(dtype=float)

    df_year.loc[
        catchup_mask,
        "extra_catchup_physical_doses"
    ] = extra_physical_doses.to_numpy(dtype=float)

    df_year.loc[
        catchup_mask,
        "extra_catchup_cost_usd"
    ] = extra_cost

    # Add the discounted cost of the third dose
    df_year.loc[
        catchup_mask,
        "cost_vaccination_usd"
    ] += extra_cost

    return df_year

def agg_ageclasses_from_yearly(
    df_yearly: pd.DataFrame,
    cc_age_bins: list[tuple[int, int]] = CC_AGE_BINS,
) -> pd.DataFrame:
    """
    Build rows aggregated by age classes (scope == 'age_class')
    from the yearly DataFrame.

    cc_age_bins: list of tuples (age_min, age_max) in years.
    """
    if df_yearly.empty:
        return pd.DataFrame(columns=[
            "scope",
            "age",
            "inc_cc_ageclass_per100k",
            "mort_cc_ageclass_per100k",
            "hiv_prev_ageclass_pct",
            "alive_ageclass"
        ])

    ages = df_yearly["age_years"].to_numpy(dtype=float)

    alive_neg = df_yearly["alive_neg"].to_numpy(dtype=float)
    alive_pos = df_yearly["alive_pos"].to_numpy(dtype=float)
    alive_all = alive_neg + alive_pos

    cases_new_neg = df_yearly["cc_cases_new_neg"].to_numpy(dtype=float)
    cases_new_pos = df_yearly["cc_cases_new_pos"].to_numpy(dtype=float)
    cases_new_all = cases_new_neg + cases_new_pos

    deaths_cum_neg = df_yearly["cc_deaths_cum_neg"].to_numpy(dtype=float)
    deaths_cum_pos = df_yearly["cc_deaths_cum_pos"].to_numpy(dtype=float)
    deaths_cum_all = deaths_cum_neg + deaths_cum_pos

    rows = []
    cc_age_bins = cc_age_bins or []
    for (age_min, age_max) in cc_age_bins:
        idx = np.where((ages >= age_min) & (ages <= age_max))[0]
        if idx.size == 0:
            inc_rate = mort_rate = hiv_prev_pct = popalive = 0.0
        else:
            popalive = float(alive_neg[idx].sum() + alive_pos[idx].sum())
            new_cc_cases = float(cases_new_all[idx].sum())

            # new CC deaths in the bin = difference of cumulative values at edges
            new_cc_deaths = float(
                deaths_cum_all[idx[-1]] - (deaths_cum_all[idx[0] - 1] if idx[0] > 0 else 0.0)
            )

            # approximate HIV person-time = sum of alive HIV+ over the bin
            hiv_person_time = float(alive_pos[idx].sum())

            inc_rate     = (new_cc_cases  / popalive) * 1e5 if popalive > 0 else 0.0
            mort_rate    = (new_cc_deaths / popalive) * 1e5 if popalive > 0 else 0.0
            hiv_prev_pct = (hiv_person_time / popalive)   if popalive > 0 else 0.0

        rows.append({
            "scope": "age_class",
            "age": f"{age_min}-{age_max}",
            "inc_cc_ageclass_per100k": inc_rate,
            "mort_cc_ageclass_per100k": mort_rate,
            "hiv_prev_ageclass_pct": hiv_prev_pct,
            "alive_ageclass": popalive,
        })

    return pd.DataFrame(rows)

