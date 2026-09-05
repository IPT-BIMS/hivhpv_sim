"""
HPV/HIV Cervical Cancer Model — Shiny for Python
Faithful interactive implementation of:
02_Cameroon_calibration_template_multi_country_corrected.ipynb

Workflow exposed in the app:
1. Country input / data upload
2. Demography + HIV mortality construction and checks
3. Initial parameter validation + baseline model tests
4. Full 17-parameter calibration (including 2 tunnel durations + RR penalty)
5. S-1 ... S6 vaccination strategies
6. Scenario simulation + model-output plots
7. CEA vs S0 and vs S-1 + consistency checks
8. One-way sensitivity / tornado
9. S6 stability sensitivity (primary and booster)
10. PSA + CE plane + CEAC

Run:
    source ~/hpv_env/bin/activate
    cd /home/bassem/Downloads/HIV-HPV-Prob.1/hpv_app_1/app
    shiny run --reload app.py
"""
from __future__ import annotations

import io
import traceback
import functools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget

import hivhpv_sim as h
from hivhpv_sim.constants import (
    COUNTRY_NAME, N_COHORT, HIV_PREVALENCE, YEAR_START, MAX_AGE, N_YEARS,
    ages, TIME_HORIZON, SIMULATION_YEARS,
    DISCOUNT_RATE_HEALTH, DISABILITY_WEIGHT, DISCOUNT_RATE_COSTS,
    DOSE_NUMBERS, BASE_COVERAGE, OPT_COVERAGE, AGE_VACC,
    DURATION_NEG, DELTA_NEG, VACC_EFF_NEG,
    DURATION_POS, DELTA_POS, VACC_EFF_POS,
    DURATION_BOOST, DELTA_BOOST, VACC_EFF_BOOST,
    DOSE_PRICE, DOSE_PRICE_BOOST, CANCER_COST,
    NEG_NAMES, POS_NAMES, NEG_NAMES_CORE, POS_NAMES_CORE,
    CC_AGE_BINS, CC_AGE_LABELS, discountVectCost,
    COST_PER_PHYSICAL_DOSE, CATCHUP_DOSE_NUMBERS_HIV_POS,
)
from hivhpv_sim.model import (
    run_model as _run_model,
    var_output as _var_output,
    agg_ageclasses_from_yearly,
    configure_mortality,
    validate_params_detailed,
    create_age_functions,
)
from hivhpv_sim.calibration import (
    params_fixed as DEFAULT_PARAMS_FIXED,
    params as DEFAULT_PARAMS,
    params_vector as DEFAULT_PARAMS_VECTOR,
    AGE_PARAMETER_NAMES,
    EXPECTED_CALIBRATION_VECTOR_SIZE,
    RR_TARGET, RR_MIN, RR_MAX,
    probability_to_rate, rate_to_probability,
    build_calibration_vector,
    build_params_fixed_from_durations,
    calculate_incidence_rr_table,
    incidence_rr_penalty,
    validate_candidate_vector,
)
from hivhpv_sim.calibration_runner import (
    normalized_mse,
    notebook_calibration_setup,
    vector_to_components,
    calculate_calibration_error_components,
    calibration_objective,
    run_notebook_calibration,
    calibration_environment,
)
from hivhpv_sim.vaccination import (
    build_vax_vectors_by_age,
    build_baseline_scenarios,
    vax_params as NOVAX,
)
from hivhpv_sim.demography import (
    remainlife_expectancy,
    build_mortality_functions,
)
from hivhpv_sim.data_io import (
    load_life_tables,
    make_project_dirs,
)
from hivhpv_sim.sensitivity import compute_ceac


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
COUNTRY_INPUT_DIR = PROJECT_ROOT / "02_country_inputs"
DEFAULT_COUNTRY_FILE = COUNTRY_INPUT_DIR / f"{COUNTRY_NAME}_life_table.xlsx"

try:
    OUTPUT_DIRS = make_project_dirs(COUNTRY_NAME, PROJECT_ROOT)
except Exception:
    OUTPUT_DIRS = {}


# ---------------------------------------------------------------------------
# Exact notebook strategy configuration
# Cell 83 / vaccination.py
# ---------------------------------------------------------------------------
STRATEGY_DESCRIPTIONS = {
    "S-1": "No vaccination (coverage = 0)",
    "S0": "Baseline: primary vaccination at age 14 for HIV−/HIV+; 50% coverage",
    "S1": "S0 + HIV+ booster at age 18; 50% coverage",
    "S2": "S0 + HIV+ booster at age 24; 50% coverage",
    "S3": "S1 + HIV+ catch-up primary + booster at age 18; 50% coverage",
    "S4": "S2 + HIV+ catch-up primary + booster at age 24; 50% coverage",
    "S5": "S3 with OPT_COVERAGE = 98% for HIV+ catch-up",
    "S6": "S4 with OPT_COVERAGE = 98% for HIV+ catch-up",
}
STRATEGY_ORDER = ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6"]
SCENARIO_COLORS = {
    "S-1": "#6b7280", "S0": "#3b82f6", "S1": "#8b5cf6",
    "S2": "#10b981", "S3": "#f59e0b", "S4": "#ef4444",
    "S5": "#06b6d4", "S6": "#ec4899",
}


def build_notebook_scenarios():
    """Use the package's S-1..S6 builder with the notebook values explicitly."""
    return build_baseline_scenarios(
        ages,
        age_vacc=AGE_VACC,
        base_coverage=0.50,
        opt_coverage=0.98,
        delta_neg=0.96,
        duration_neg=20,
        delta_pos=0.87,
        duration_pos=15,  # notebook cell 83
        delta_boost=0.96,
        duration_boost=65,  # notebook: 50 + DURATION_POS = 65
    )


SCENARIO_BUNDLE = build_notebook_scenarios()
SCENARIOS_SO = SCENARIO_BUNDLE["SCENARIOS_SO"]
SCENARIO_NOVAX = SCENARIO_BUNDLE["SCENARIO_novax"]
SCENARIO_SPECS = SCENARIO_BUNDLE["scenario_specs"].copy()

# Add the explicit 3-dose correction metadata required by the CEA logic.
for sc in ["S3", "S4", "S5", "S6"]:
    SCENARIO_SPECS[sc] = dict(
        SCENARIO_SPECS.get(sc, {}),
        catchup_age=18 if sc in ["S3", "S5"] else 24,
        catchup_primary_doses_hiv_pos=3,
    )


# ---------------------------------------------------------------------------
# Robust Excel / country loading
# ---------------------------------------------------------------------------
def parse_country_excel(path: str | Path) -> dict[str, Any]:
    """Read the exact Cameroon workbook structure used by the notebook."""
    path = Path(path)
    xl = pd.ExcelFile(path)
    sheets = xl.sheet_names

    required = {"demography", "cc", "qx_HIV"}
    missing = required.difference(sheets)
    if missing:
        raise ValueError(
            f"Workbook is missing required sheets: {sorted(missing)}. "
            f"Available sheets: {sheets}"
        )

    df_demography = pd.read_excel(path, sheet_name="demography")
    df_field = pd.read_excel(path, sheet_name="cc")
    df_qx_hiv = pd.read_excel(path, sheet_name="qx_HIV")

    # Preserve notebook column names.
    if "age" not in df_field.columns and "age_group" in df_field.columns:
        df_field["age"] = df_field["age_group"].astype(str)
    if "age_group" not in df_field.columns and "age" in df_field.columns:
        df_field["age_group"] = df_field["age"].astype(str)

    # Validate calibration columns.
    required_cc = {
        "cc_incidence_per100k",
        "cc_mortality_per100k",
        "hiv_prevalence",
    }
    miss_cc = required_cc.difference(df_field.columns)
    if miss_cc:
        raise ValueError(
            f"Sheet 'cc' is missing calibration columns: {sorted(miss_cc)}"
        )

    # The notebook interpolates remaining life from the demography sheet.
    if "Remaining life years" not in df_demography.columns:
        raise ValueError(
            "Sheet 'demography' must contain 'Remaining life years'."
        )

    # Configure the package mortality functions exactly from the workbook.
    gmr, gmh = build_mortality_functions(
        df_demography,
        df_qx_hiv,
    )
    configure_mortality(gmr, gmh)

    # Re-use package loader where possible for consistency.
    try:
        loaded = load_life_tables(COUNTRY_NAME, str(path.parent))
        # load_life_tables expects the canonical filename. If this is an upload
        # under another name, the manually-read tables above remain authoritative.
        _ = loaded
    except Exception:
        pass

    # Notebook's age-13/15-19 prevalence input.
    hiv_prevalence_13 = float(df_field["hiv_prevalence"].iloc[0])

    # Life expectancy at age 0.
    row0 = df_demography[df_demography["age"] == 0]
    life_expectancy = (
        float(row0["Remaining life years"].iloc[0])
        if not row0.empty else float(df_demography["Remaining life years"].iloc[0])
    )

    return {
        "path": str(path),
        "demography": df_demography,
        "df_demography_full": df_demography.copy(),
        "qx_hiv": df_qx_hiv,
        "df_cc": df_field,
        "field_data": df_field.copy(),
        "hiv_prevalence_13": hiv_prevalence_13,
        "life_expectancy": life_expectancy,
    }


def default_country_data():
    if DEFAULT_COUNTRY_FILE.exists():
        try:
            return parse_country_excel(DEFAULT_COUNTRY_FILE)
        except Exception as exc:
            return {"_error": f"{type(exc).__name__}: {exc}"}
    return None


# ---------------------------------------------------------------------------
# Notebook calibration
# ---------------------------------------------------------------------------
# The calibration implementation lives in hivhpv_sim.calibration_runner and is
# shared with Jupyter.  The Shiny app calls that module directly so there is
# only one scientific calibration code path.


# ---------------------------------------------------------------------------
# CEA helpers — includes the 3-dose HIV+ catch-up correction
# ---------------------------------------------------------------------------
def run_scenarios_exact(
    scenarios,
    params_vector,
    params_fixed,
    cancer_cost=CANCER_COST,
    dose_price=DOSE_PRICE,
    boost_price=DOSE_PRICE_BOOST,
):
    yearly = {}
    summary_rows = []

    for scenario, spec in scenarios.items():
        sn, sp = _run_model(
            params_vector,
            params_fixed=params_fixed,
            vax_params=spec["vec"],
            return_eff=False,
        )
        df = _var_output(
            sn, sp, ages,
            dw=DISABILITY_WEIGHT,
            discount_rate=DISCOUNT_RATE_HEALTH,
            cancer_cost_unit=cancer_cost,
            vax_cost_per_dose=dose_price,
            boost_cost_per_dose=boost_price,
        )

        # Correct HIV+ catch-up from the model's 2-dose routine accounting to
        # the notebook's intended 3 physical doses.
        extra_doses = np.zeros(len(df), dtype=float)
        catchup_age = SCENARIO_SPECS.get(scenario, {}).get("catchup_age")
        if catchup_age is not None:
            output_age = catchup_age + 1
            mask = df["age_years"].to_numpy() == float(output_age)
            recipients = df.loc[mask, "vacc_doses_new_pos"].to_numpy(float)
            extra = recipients * (CATCHUP_DOSE_NUMBERS_HIV_POS - DOSE_NUMBERS)
            extra_doses[mask] = extra
            df.loc[mask, "cost_vaccination_usd"] += (
                extra * COST_PER_PHYSICAL_DOSE *
                discountVectCost[np.where(mask)[0]]
            )

        df["extra_catchup_physical_doses"] = extra_doses
        df["total_physical_vaccine_doses"] = (
            DOSE_NUMBERS *
            (df["vacc_doses_new_neg"] + df["vacc_doses_new_pos"])
            + df["vacc_doses_new_boost"]
            + extra_doses
        )

        yearly[scenario] = df

        total_cases = float(
            df["cc_cases_new_neg"].sum() + df["cc_cases_new_pos"].sum()
        )
        total_deaths = float(
            df["cc_deaths_cum_neg"].iloc[-1] +
            df["cc_deaths_cum_pos"].iloc[-1]
        )
        dalys = float(df["dalys_all_disc"].sum())
        cancer = float(df["cost_cancer_usd"].sum())
        vaccination = float(df["cost_vaccination_usd"].sum())
        primary_doses = float(
            DOSE_NUMBERS *
            (df["vacc_doses_new_neg"].sum() + df["vacc_doses_new_pos"].sum())
            + df["extra_catchup_physical_doses"].sum()
        )
        boost_doses = float(df["vacc_doses_new_boost"].sum())

        summary_rows.append({
            "Scenario": scenario,
            "Description": spec.get("description", STRATEGY_DESCRIPTIONS.get(scenario, "")),
            "Total cases": total_cases,
            "CC Deaths": total_deaths,
            "YLD": float(df["yld_all"].sum()),
            "YLL (disc)": float(df["yll_all_disc"].sum()),
            "DALYs (disc)": dalys,
            "Cancer cost (USD)": cancer,
            "Vaccination cost (USD)": vaccination,
            "Total cost (USD)": cancer + vaccination,
            "Total vaccination doses": primary_doses,
            "Total boost doses": boost_doses,
            "Total physical vaccine doses": primary_doses + boost_doses,
        })

    return pd.DataFrame(summary_rows), yearly


def add_incremental_cea(summary, comparator):
    df = summary.copy()
    ref = df.loc[df["Scenario"] == comparator].iloc[0]
    df["ΔCost_vs_baseline"] = df["Total cost (USD)"] - float(ref["Total cost (USD)"])
    df["DALYs_averted_vs_baseline"] = float(ref["DALYs (disc)"]) - df["DALYs (disc)"]

    with np.errstate(divide="ignore", invalid="ignore"):
        df["ICER"] = np.where(
            df["DALYs_averted_vs_baseline"] > 0,
            df["ΔCost_vs_baseline"] / df["DALYs_averted_vs_baseline"],
            np.nan,
        )
    return df


def add_dominance_flags(df):
    out = df.copy()
    out["Strictly dominated"] = False
    out["Extendedly dominated"] = False

    # Strict dominance: another strategy is at least as effective and no more
    # costly, with one strict inequality.
    for i, row in out.iterrows():
        dominates = (
            (out["DALYs (disc)"] <= row["DALYs (disc)"]) &
            (out["Total cost (USD)"] <= row["Total cost (USD)"]) &
            (
                (out["DALYs (disc)"] < row["DALYs (disc)"]) |
                (out["Total cost (USD)"] < row["Total cost (USD)"])
            )
        )
        if dominates.any():
            out.loc[i, "Strictly dominated"] = True

    # Extended dominance: sequential ICER must increase along the frontier.
    work = out.loc[~out["Strictly dominated"]].copy()
    work = work.sort_values(["DALYs (disc)", "Total cost (USD)"]).reset_index(drop=True)

    changed = True
    while changed and len(work) >= 3:
        changed = False
        icer = []
        for i in range(1, len(work)):
            de = work.loc[i, "DALYs (disc)"] - work.loc[i-1, "DALYs (disc)"]
            dc = work.loc[i, "Total cost (USD)"] - work.loc[i-1, "Total cost (USD)"]
            icer.append(dc / de if de > 0 else np.inf)

        for j in range(1, len(icer)):
            if icer[j] <= icer[j-1] + 1e-10:
                scen = work.loc[j, "Scenario"]
                out.loc[out["Scenario"] == scen, "Extendedly dominated"] = True
                work = work.drop(index=j).reset_index(drop=True)
                changed = True
                break

    return out


def cEA_verification_tables(cea_s0, yearly_s0):
    """Reproduce notebook cells 92, 94 and 96."""
    check = cea_s0[
        [
            "Scenario", "Cancer cost (USD)", "Vaccination cost (USD)",
            "Total cost (USD)", "DALYs (disc)",
            "ΔCost_vs_baseline", "DALYs_averted_vs_baseline",
        ]
    ].copy().set_index("Scenario")

    cancer_ref = float(check.loc["S0", "Cancer cost (USD)"])
    vax_ref = float(check.loc["S0", "Vaccination cost (USD)"])
    total_ref = float(check.loc["S0", "Total cost (USD)"])

    check["Cancer cost avoided"] = cancer_ref - check["Cancer cost (USD)"]
    check["Additional vaccination cost"] = check["Vaccination cost (USD)"] - vax_ref
    check["Expected net cost change"] = (
        check["Additional vaccination cost"] - check["Cancer cost avoided"]
    )
    check["Calculated net cost change"] = check["Total cost (USD)"] - total_ref
    check["Cost calculation correct"] = np.isclose(
        check["Expected net cost change"],
        check["Calculated net cost change"],
        rtol=1e-6, atol=1e-6,
    )

    cases = cea_s0[
        ["Scenario", "Total cases", "Cancer cost (USD)",
         "Vaccination cost (USD)", "Total cost (USD)"]
    ].copy().set_index("Scenario")
    cases_ref = float(cases.loc["S0", "Total cases"])
    cancer_ref2 = float(cases.loc["S0", "Cancer cost (USD)"])
    cases["Cancer cases prevented"] = cases_ref - cases["Total cases"]
    cases["Cancer cost avoided"] = cancer_ref2 - cases["Cancer cost (USD)"]

    rows = []
    df_ref = yearly_s0["S0"]
    for scen, df in yearly_s0.items():
        prevented = (
            df_ref["cc_cases_new_neg"] + df_ref["cc_cases_new_pos"]
            - df["cc_cases_new_neg"] - df["cc_cases_new_pos"]
        )
        expected = (
            prevented.to_numpy(float)
            * CANCER_COST
            * discountVectCost[:len(df)]
        ).sum()
        calculated = (
            df_ref["cost_cancer_usd"].sum() - df["cost_cancer_usd"].sum()
        )
        rows.append({
            "Scenario": scen,
            "Expected discounted cancer cost avoided": expected,
            "Calculated cancer cost avoided": calculated,
            "Correct": bool(np.isclose(expected, calculated, rtol=1e-6, atol=1e-6)),
        })
    discounted = pd.DataFrame(rows).set_index("Scenario")
    return check, cases, discounted


# ---------------------------------------------------------------------------
# Sensitivity functions: exact notebook concept
# ---------------------------------------------------------------------------
def generate_psa_samples_exact(params, n_sims=1000, variation=0.20,
                               long_vector_mode="scale", seed=None):
    rng = np.random.default_rng(seed)
    samples = []
    for sim_id in range(int(n_sims)):
        new = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
               for k, v in params.items()}
        new["sim_id"] = sim_id
        new["param_name"] = "PSA"
        new["param_value"] = np.nan
        for key, value in params.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                base = np.asarray(value, dtype=float)
                if len(base) <= 3 or long_vector_mode == "elementwise":
                    new[key] = rng.uniform(base*(1-variation), base*(1+variation)).tolist()
                    new[f"{key}_mode"] = "elementwise"
                else:
                    scale = float(rng.uniform(1-variation, 1+variation))
                    new[key] = (base*scale).tolist()
                    new[f"{key}_scale"] = scale
                    new[f"{key}_mode"] = "scale"
            else:
                new[key] = float(rng.uniform(value*(1-variation), value*(1+variation)))
                new[f"{key}_mode"] = "scalar"
        samples.append(new)
    return samples


def generate_param_variants_exact(params, variation=0.1, n_points=5):
    variants = []
    for key, value in params.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            base = np.asarray(value, dtype=float)
            scales = np.linspace(1-variation, 1+variation, n_points)
            for s in scales:
                new = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
                       for k, v in params.items()}
                new[key] = (base*s).tolist()
                new["param_name"] = key
                new["param_value"] = float(s)
                new["param_mode"] = "scale"
                variants.append(new)
        else:
            grid = np.linspace(float(value)*(1-variation), float(value)*(1+variation), n_points)
            for v in grid:
                new = {k: (list(vv) if isinstance(vv, (list, tuple, np.ndarray)) else vv)
                       for k, vv in params.items()}
                new[key] = float(v)
                new["param_name"] = key
                new["param_value"] = float(v)
                new["param_mode"] = "scalar"
                variants.append(new)
    return variants


def params_dict_to_vector(p):
    return np.concatenate([
        np.asarray(p["lambda_HPV"], dtype=float),
        np.asarray(p["lambda_HPV_pos"], dtype=float),
        np.asarray(p["mu_c"], dtype=float),
        np.asarray(p["mu_c_pos"], dtype=float),
        np.asarray(p["lambda_HIV"], dtype=float),
    ])


def run_sensitivity_simulations_exact(
    params,
    scenarios,
    fixed_params,
    mode="oneway",
    variation=0.1,
    n_points=5,
    n_sims=2000,
    seed=None,
):
    plan = (
        generate_psa_samples_exact(params, n_sims=n_sims, variation=variation, seed=seed)
        if mode == "psa"
        else generate_param_variants_exact(params, variation=variation, n_points=n_points)
    )
    rows = []
    for sim_id, p in enumerate(plan):
        pv = params_dict_to_vector(p)
        summary, _ = run_scenarios_exact(
            scenarios, pv, fixed_params
        )
        cea = add_incremental_cea(summary, "S0")
        for _, selected in cea.iterrows():
            primary = float(selected["Total vaccination doses"])
            boost = float(selected["Total boost doses"])
            rows.append({
                "Scenario": str(selected["Scenario"]),
                "param_name": p.get("param_name", "PSA" if mode == "psa" else "unknown"),
                "param_value": p.get("param_value", np.nan),
                "sim_id": sim_id,
                "lambda_HPV_amp": float(p["lambda_HPV"][0]),
                "lambda_HPV_m": float(p["lambda_HPV"][1]),
                "lambda_HPV_sp": float(p["lambda_HPV"][2]),
                "lambda_HPV_pos_amp": float(p["lambda_HPV_pos"][0]),
                "lambda_HPV_pos_m": float(p["lambda_HPV_pos"][1]),
                "lambda_HPV_pos_sp": float(p["lambda_HPV_pos"][2]),
                "mu_c": float(np.linalg.norm(p["mu_c"])),
                "mu_c_pos": float(np.linalg.norm(p["mu_c_pos"])),
                "lambda_HIV_amp": float(p["lambda_HIV"][0]),
                "lambda_HIV_m": float(p["lambda_HIV"][1]),
                "lambda_HIV_sp": float(p["lambda_HIV"][2]),
                "DALYs_disc": float(selected["DALYs (disc)"]),
                "vax_dose_total": primary + boost,
                "total_cost": float(selected["Total cost (USD)"]),
                "ΔCost_vs_baseline": float(selected["ΔCost_vs_baseline"]),
                "DALYs_averted_vs_baseline": float(selected["DALYs_averted_vs_baseline"]),
            })
    df = pd.DataFrame(rows)

    if mode == "oneway" and not df.empty:
        df = df.sort_values(
            ["Scenario", "param_name", "param_value", "sim_id"],
            kind="mergesort",
        )
        dparam = df.groupby(["Scenario", "param_name"])["param_value"].diff()
        metrics = [
            "DALYs_disc", "vax_dose_total", "total_cost",
            "ΔCost_vs_baseline", "DALYs_averted_vs_baseline",
        ]
        for metric in metrics:
            df[f"{metric}_diff"] = (
                df.groupby(["Scenario", "param_name"])[metric].diff() / dparam
            )
    return df


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
THEME_CSS = """
:root{--bg:#0a0e1a;--surface:#111827;--surface2:#1a2235;--border:#1e293b;
--border2:#2d3f5a;--accent:#00d4ff;--accent2:#7c3aed;--accent3:#10b981;
--accent4:#f59e0b;--danger:#ef4444;--text:#e2e8f0;--text2:#94a3b8;
--text3:#64748b;--font-body:'IBM Plex Sans',system-ui,sans-serif;
--font-mono:'IBM Plex Mono','Courier New',monospace}
html,body,.container-fluid{background:var(--bg)!important;color:var(--text)!important}
.bslib-sidebar-layout>.sidebar,aside.sidebar,.sidebar{background:var(--surface)!important;border-right:1px solid var(--border)!important}
.app-logo{padding:18px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.logo-icon{width:36px;height:36px;background:linear-gradient(135deg,var(--accent),var(--accent2));
border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;color:#000}
.logo-text{font-size:13px;font-weight:700}.logo-sub{font-size:10px;color:var(--text3)}
.nav-section-title{font-size:9px;font-weight:700;letter-spacing:.15em;color:var(--text3);padding:14px 16px 4px}
.sidebar-footer{padding:14px 16px;border-top:1px solid var(--border);font-size:10px;color:var(--text2)}
.page-header{border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:20px}
.page-title{font-size:22px;font-weight:700}.page-desc{font-size:13px;color:var(--text2);margin-top:4px;max-width:900px}
.app-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:18px;margin-bottom:14px}
.card-title{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text2);margin-bottom:14px}
.run-log{background:var(--bg);border:1px solid var(--border);border-radius:5px;padding:12px;
font-family:var(--font-mono);font-size:11px;color:var(--text2);max-height:360px;overflow:auto;white-space:pre-wrap}
table.table{color:var(--text)!important;font-size:12px}table.table th{background:var(--surface2)!important;color:var(--text)!important}
table.table td{border-bottom:1px solid var(--border)!important}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:14px}
.kpi-label{font-size:10px;color:var(--text3);text-transform:uppercase}.kpi-value{font-size:21px;font-weight:700}
"""

PLOTLY_TEMPLATE = dict(layout=dict(
    paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
    font=dict(family="IBM Plex Sans", color="#e2e8f0", size=12),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#2d3f5a"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#2d3f5a"),
    colorway=["#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444","#38bdf8","#f87171","#3b82f6"],
    legend=dict(bgcolor="rgba(17,24,39,.75)",bordercolor="#1e293b",borderwidth=1),
    margin=dict(l=60,r=20,t=50,b=55),
))


def card(title, content):
    return ui.div(
        ui.div(title, class_="card-title"),
        content,
        class_="app-card",
    )


def dataframe_html(df, max_rows=None):
    if df is None:
        return ui.div("—", style="color:var(--text3)")
    d = df.copy()
    if max_rows:
        d = d.head(max_rows)
    return ui.HTML(d.to_html(index=False, classes="table", border=0, na_rep="—"))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_data():
    return ui.div(
        ui.div(ui.div("1 · Data & Demography", class_="page-title"),
               ui.div("Same country-input workflow as the notebook: demography, remaining life, HIV mortality and cervical-cancer calibration targets.",
                      class_="page-desc"), class_="page-header"),
        ui.layout_columns(
            card("Country Excel", ui.div(
                ui.input_file("xlsx_file","Upload country life-table (.xlsx)",
                              accept=[".xlsx"], multiple=False),
                ui.input_action_button("btn_load_default","↻ Load Cameroon default", class_="btn-primary"),
                ui.output_ui("data_status"),
            )),
            card("Workbook structure", ui.output_ui("data_summary")),
            col_widths=(6,6),
        ),
        ui.layout_columns(
            card("Natural mortality m(x)", output_widget("plot_mortality")),
            card("HIV mortality m(x)", output_widget("plot_hiv_mortality")),
            col_widths=(6,6),
        ),
        card("Field calibration data — cervical cancer incidence, mortality and HIV prevalence",
             ui.output_ui("field_table")),
    )


def page_initial_tests():
    return ui.div(
        ui.div(ui.div("2 · Initial Model & Tests", class_="page-title"),
               ui.div("Reproduces the notebook's parameter validation, no-vaccination run, competing-risk / conservation tests and initial health-output diagnostics.",
                      class_="page-desc"), class_="page-header"),
        ui.layout_columns(
            card("Run tests", ui.div(
                ui.input_action_button("btn_initial_test","▶ Run initial tests",class_="btn-primary"),
                ui.output_ui("initial_test_status"),
            )),
            card("Initial parameter vector", ui.output_ui("initial_vector_table")),
            col_widths=(5,7),
        ),
        ui.layout_columns(
            card("Gaussian age-dependent parameters", output_widget("plot_initial_gaussian")),
            card("Field vs model — incidence / mortality / HIV", output_widget("plot_initial_fit")),
            col_widths=(6,6),
        ),
        card("Detailed validation report", ui.output_ui("initial_validation_table")),
        card("Model results — key outputs", output_widget("plot_initial_results")),
    )


def page_calibration():
    return ui.div(
        ui.div(ui.div("3 · Calibration", class_="page-title"),
               ui.div("Full notebook calibration: 17 parameters = 15 age-structured parameters + HIV−/HIV+ precancer tunnel durations. L-BFGS-B with the notebook bounds, normalized MSE, regularisation and soft RR penalty.",
                      class_="page-desc"), class_="page-header"),
        ui.layout_columns(
            card("Calibration controls", ui.div(
                ui.input_numeric("calib_maxiter","Maximum iterations",10000,min=100,max=20000,step=100),
                ui.input_action_button("btn_calibrate","🎯 Run full calibration",class_="btn-primary"),
                ui.output_ui("calib_status"),
            )),
            card("Objective diagnostics", ui.output_ui("calib_objective_table")),
            col_widths=(5,7),
        ),
        ui.layout_columns(
            card("Optimised Gaussian curves", output_widget("plot_calib_curves")),
            card("Calibrated model vs field", output_widget("plot_calib_fit")),
            col_widths=(6,6),
        ),
        card("Optimised 17-parameter vector", ui.output_ui("calib_vector_table")),
        card("Error components + RR diagnostics", ui.output_ui("calib_error_table")),
        card("RR by age group", ui.output_ui("calib_rr_table")),
    )


def page_strategies():
    return ui.div(
        ui.div(ui.div("4 · Vaccination Strategies", class_="page-title"),
               ui.div("Exact S-1 … S6 strategy family used in the notebook, including the HIV-positive 3-dose catch-up definition.",
                      class_="page-desc"), class_="page-header"),
        card("Strategy definitions", ui.output_ui("strategy_table")),
        ui.layout_columns(
            card("Coverage vectors", output_widget("plot_strategy_coverage")),
            card("Residual-risk / efficacy vectors", output_widget("plot_strategy_efficacy")),
            col_widths=(6,6),
        ),
    )


def page_run():
    return ui.div(
        ui.div(ui.div("5 · Run Scenarios", class_="page-title"),
               ui.div("Run S-1, S0, S1, S2, S3, S4, S5 and S6 with the calibrated or initial 15-element age-parameter vector.",
                      class_="page-desc"), class_="page-header"),
        ui.layout_columns(
            card("Run configuration", ui.div(
                ui.input_switch("use_calibrated","Use calibrated parameters",True),
                ui.input_numeric("run_cancer_cost","Cancer cost (USD/case)",CANCER_COST,min=0,step=10),
                ui.input_action_button("btn_run_scenarios","▶ Run all 8 scenarios",class_="btn-primary"),
                ui.output_ui("run_status"),
            )),
            card("Scenario summary", ui.output_ui("run_summary_table")),
            col_widths=(5,7),
        ),
        card("Scenario comparison — annual health/economic outputs",
             output_widget("plot_scenario_outputs")),
    )


def page_cea():
    return ui.div(
        ui.div(ui.div("6 · CEA & Verification", class_="page-title"),
               ui.div("Reproduces the notebook CEA against S0 and against S-1, with cost decomposition, ICERs, dominance and the three consistency checks.",
                      class_="page-desc"), class_="page-header"),
        ui.layout_columns(
            card("CEA vs S0", ui.output_ui("cea_s0_table")),
            card("CEA vs S-1", ui.output_ui("cea_novax_table")),
            col_widths=(6,6),
        ),
        ui.layout_columns(
            card("Cost-effectiveness plane vs S0", output_widget("plot_cea_s0")),
            card("Cost-effectiveness plane vs S-1", output_widget("plot_cea_novax")),
            col_widths=(6,6),
        ),
        ui.layout_columns(
            card("Cost components", output_widget("plot_cost_components")),
            card("Physical vaccine doses", output_widget("plot_physical_doses")),
            col_widths=(6,6),
        ),
        card("Verification 1 — expected net cost change", ui.output_ui("verify_costs")),
        card("Verification 2 — cancer cases prevented / cancer cost avoided", ui.output_ui("verify_cases")),
        card("Verification 3 — discounted cancer cost consistency", ui.output_ui("verify_discounted")),
    )


def page_owsa():
    return ui.div(
        ui.div(ui.div("7 · One-way Sensitivity", class_="page-title"),
               ui.div("Notebook OWSA: ±10%, five points, long parameter vectors scaled together. Runs all eight strategies and computes local derivatives.",
                      class_="page-desc"), class_="page-header"),
        ui.layout_columns(
            card("OWSA controls", ui.div(
                ui.input_slider("owsa_variation","Variation",0.05,0.30,0.10,step=0.05),
                ui.input_slider("owsa_points","Grid points",3,9,5,step=1),
                ui.input_action_button("btn_owsa","🔬 Run OWSA",class_="btn-primary"),
                ui.output_ui("owsa_status"),
            )),
            card("Results", ui.output_ui("owsa_summary")),
            col_widths=(5,7),
        ),
        ui.layout_columns(
            card("Tornado — DALYs", output_widget("plot_tornado_dalys")),
            card("Tornado — total cost", output_widget("plot_tornado_cost")),
            col_widths=(6,6),
        ),
        card("OWSA result table", ui.output_ui("owsa_table")),
    )


def page_s6():
    return ui.div(
        ui.div(ui.div("8 · S6 Stability Sensitivity", class_="page-title"),
               ui.div("Exact notebook grids for the optimal-point stability checks: primary effect/duration and booster effect/duration.",
                      class_="page-desc"), class_="page-header"),
        ui.layout_columns(
            card("Primary vaccine effect", ui.div(
                ui.input_text("s6_primary_T","Durations T", "7,15,20,25"),
                ui.input_text("s6_primary_eff","Efficacies", "0.70,0.85,0.95"),
                ui.input_action_button("btn_s6_primary","Run primary stability",class_="btn-primary"),
                ui.output_ui("s6_primary_status"),
            )),
            card("Booster effect", ui.div(
                ui.input_text("s6_boost_T","Durations T", "25,35,50,80"),
                ui.input_text("s6_boost_eff","Efficacies", "0.75,0.95,1.0"),
                ui.input_action_button("btn_s6_boost","Run booster stability",class_="btn-primary"),
                ui.output_ui("s6_boost_status"),
            )),
            col_widths=(6,6),
        ),
        ui.layout_columns(
            card("Primary stability heatmap", output_widget("plot_s6_primary")),
            card("Booster stability heatmap", output_widget("plot_s6_boost")),
            col_widths=(6,6),
        ),
        card("Stability result tables", ui.output_ui("s6_tables")),
    )


def page_psa():
    return ui.div(
        ui.div(ui.div("9 · PSA / CEAC", class_="page-title"),
               ui.div("Notebook PSA: 2,000 simulations, ±10% parameter variation, long vectors scaled together; CE plane and CEAC over WTP = 0…5,000 USD/DALY.",
                      class_="page-desc"), class_="page-header"),
        ui.layout_columns(
            card("PSA controls", ui.div(
                ui.input_numeric("psa_n","Simulations",2000,min=20,max=5000,step=20),
                ui.input_slider("psa_variation","Variation",0.05,0.30,0.10,step=0.05),
                ui.input_numeric("psa_seed","Seed (blank = random)",123,min=0,max=999999,step=1),
                ui.input_action_button("btn_psa","📈 Run PSA",class_="btn-primary"),
                ui.output_ui("psa_status"),
            )),
            card("PSA summary", ui.output_ui("psa_summary")),
            col_widths=(5,7),
        ),
        card("PSA cost-effectiveness plane", output_widget("plot_psa_plane")),
        card("CEAC — WTP 0 to 5,000 USD/DALY", output_widget("plot_ceac")),
    )


SIDEBAR = ui.sidebar(
    ui.div(
        ui.div("⚕",class_="logo-icon"),
        ui.div(ui.div("HPV·HIV",class_="logo-text"),
               ui.div("CAMEROON · v2",class_="logo-sub")),
        class_="app-logo",
    ),
    ui.div("Notebook workflow",class_="nav-section-title"),
    ui.input_radio_buttons(
        "nav_page",None,
        choices={
            "data":"1 · Data & Demography",
            "initial":"2 · Initial Model & Tests",
            "calibration":"3 · Calibration",
            "strategies":"4 · Vaccination Strategies",
            "run":"5 · Run Scenarios",
            "cea":"6 · CEA & Verification",
            "owsa":"7 · One-way Sensitivity",
            "s6":"8 · S6 Stability",
            "psa":"9 · PSA / CEAC",
        },
        selected="data",
    ),
    ui.div(
        ui.HTML("Pasteur Institute of Tunis<br>BIMS<br>S. BenMiled · O. Laraj · B. Razgui · 2026"),
        class_="sidebar-footer",
    ),
    width=280,
)


APP_UI = ui.page_sidebar(
    SIDEBAR,
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap",
        ),
        ui.tags.style(THEME_CSS),
    ),
    ui.output_ui("active_page"),
    title="HPV/HIV Cervical Cancer Model — Cameroon",
    fillable=False,
)


def safe_input(inp, name, default):
    try:
        fn = getattr(inp, name)
        value = fn()
        return default if value is None else value
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):
    rv_data = reactive.Value(default_country_data())
    rv_initial = reactive.Value(None)
    rv_calibration = reactive.Value(None)
    rv_scenario_run = reactive.Value(None)
    rv_cea = reactive.Value(None)
    rv_owsa = reactive.Value(None)
    rv_s6_primary = reactive.Value(None)
    rv_s6_boost = reactive.Value(None)
    rv_psa = reactive.Value(None)

    @output
    @render.ui
    def active_page():
        p = safe_input(input,"nav_page","data")
        pages = {
            "data":page_data, "initial":page_initial_tests,
            "calibration":page_calibration, "strategies":page_strategies,
            "run":page_run, "cea":page_cea, "owsa":page_owsa,
            "s6":page_s6, "psa":page_psa,
        }
        return pages.get(p,page_data)()

    # ------------------------- data
    @reactive.effect
    @reactive.event(input.xlsx_file)
    def _upload():
        f = input.xlsx_file()
        if not f:
            return
        try:
            rv_data.set(parse_country_excel(f[0]["datapath"]))
            rv_initial.set(None)
            rv_calibration.set(None)
            rv_scenario_run.set(None)
            rv_cea.set(None)
        except Exception as exc:
            rv_data.set({"_error":f"{type(exc).__name__}: {exc}"})

    @reactive.effect
    @reactive.event(input.btn_load_default)
    def _load_default():
        rv_data.set(default_country_data())

    @output
    @render.ui
    def data_status():
        d=rv_data()
        if d is None:
            return ui.div("No workbook loaded.",style="color:var(--text3)")
        if "_error" in d:
            return ui.div(f"❌ {d['_error']}",class_="alert alert-danger")
        return ui.div(f"✅ Loaded: {Path(d['path']).name}",class_="alert alert-success")

    @output
    @render.ui
    def data_summary():
        d=rv_data()
        if not d or "_error" in d:
            return ui.div("Load the Cameroon workbook first.",style="color:var(--text3)")
        rows=[]
        for key,label in [
            ("demography","Demography"),("qx_hiv","HIV mortality"),
            ("df_cc","Cervical-cancer calibration"),
        ]:
            df=d.get(key)
            rows.append(f"<tr><td>{label}</td><td>{len(df)}</td><td>{len(df.columns)}</td></tr>")
        rows.append(
            f"<tr><td>HIV prevalence input</td><td colspan='2'>{d['hiv_prevalence_13']:.6f}</td></tr>"
        )
        rows.append(
            f"<tr><td>Life expectancy @ age 0</td><td colspan='2'>{d['life_expectancy']:.3f}</td></tr>"
        )
        return ui.HTML(
            "<table class='table'><thead><tr><th>Table</th><th>Rows</th><th>Columns / value</th></tr></thead>"
            + "<tbody>"+"".join(rows)+"</tbody></table>"
        )

    @output
    @render.ui
    def field_table():
        d=rv_data()
        if not d or "_error" in d: return ui.div("—")
        return dataframe_html(d["field_data"])

    def mortality_plot(df, age_col, rate_col, title, color):
        fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=350,title=title,
                          xaxis_title="Age",yaxis_title="m(x)")
        if df is not None and rate_col in df.columns:
            fig.add_trace(go.Scatter(x=df[age_col],y=df[rate_col],mode="lines+markers",
                                     line=dict(color=color,width=2),name="Original"))
        return fig

    @output
    @render_widget
    def plot_mortality():
        d=rv_data()
        if not d or "_error" in d:
            return go.Figure().update_layout(template=PLOTLY_TEMPLATE)
        df=d["demography"]
        return mortality_plot(df,"age","m(x)","Natural mortality m(x)","#38bdf8")

    @output
    @render_widget
    def plot_hiv_mortality():
        d=rv_data()
        if not d or "_error" in d:
            return go.Figure().update_layout(template=PLOTLY_TEMPLATE)
        df=d["qx_hiv"]
        return mortality_plot(df,"age","mx","HIV mortality m(x)","#f87171")

    # ------------------------- initial tests
    @reactive.effect
    @reactive.event(input.btn_initial_test)
    def _initial_test():
        d=rv_data()
        if not d or "_error" in d:
            rv_initial.set({"error":"Load workbook first."}); return
        try:
            initial, bounds, lo, hi = notebook_calibration_setup()
            pf=dict(DEFAULT_PARAMS_FIXED)
            # Match notebook initial prevalence input when present.
            if "hiv_prevalence_13" in d:
                pf["hiv_prevalence_13"]=float(d["hiv_prevalence_13"])
            validation=validate_params_detailed(
                params=DEFAULT_PARAMS_VECTOR,
                params_fixed=pf,
                vax_params=NOVAX,
                tol=1e-12,
                report=True,
            )
            sn,sp=_run_model(DEFAULT_PARAMS_VECTOR,pf,NOVAX,run_tests=True)
            df=_var_output(sn,sp,ages)
            agg=agg_ageclasses_from_yearly(df,CC_AGE_BINS)
            rr=calculate_incidence_rr_table(sn,sp,ages[:sn.shape[0]])
            rv_initial.set({
                "initial_vector":initial,
                "validation":validation,
                "state_neg":sn,"state_pos":sp,
                "df_yearly":df,"df_agg":agg,"df_rr":rr,
            })
        except Exception as exc:
            rv_initial.set({"error":f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})

    @output
    @render.ui
    def initial_test_status():
        r=rv_initial()
        if r is None: return ui.div("Not run.",style="color:var(--text3)")
        if "error" in r: return ui.div("❌ "+r["error"],class_="alert alert-danger")
        ok=bool(r["validation"][0])
        return ui.div(
            ("✅ All constraints satisfied; model run completed with run_tests=True."
             if ok else "⚠ Parameter validation reported violations."),
            class_="alert alert-success" if ok else "alert alert-warning",
        )

    @output
    @render.ui
    def initial_vector_table():
        r=rv_initial()
        if not r or "initial_vector" not in r: return ui.div("Run initial tests first.")
        names=[
            "λ_HPV amp","λ_HPV mean","λ_HPV sigma",
            "λ_HPV+ amp","λ_HPV+ mean","λ_HPV+ sigma",
            "μ_c amp","μ_c mean","μ_c sigma",
            "μ_c+ amp","μ_c+ mean","μ_c+ sigma",
            "λ_HIV amp","λ_HIV mean","λ_HIV sigma",
            "HIV− tunnel total","HIV+ tunnel total",
        ]
        return dataframe_html(pd.DataFrame({"Parameter":names,"Value":r["initial_vector"]}))

    @output
    @render.ui
    def initial_validation_table():
        r=rv_initial()
        if not r or "validation" not in r: return ui.div("Run initial tests first.")
        rep=r["validation"][1]
        if isinstance(rep,pd.DataFrame):
            return dataframe_html(rep,100)
        return ui.div(str(rep))

    def plot_gaussians(vec,title):
        x=np.linspace(0,100,500)
        fig=go.Figure()
        labels=[
            ("λ_HPV (HIV−)",0),("λ_HPV (HIV+)",3),
            ("μ_c (HIV−)",6),("μ_c (HIV+)",9),("λ_HIV",12)
        ]
        for label,i in labels:
            A,mu,s=map(float,vec[i:i+3])
            y=A*np.exp(-((x-mu)**2)/(2*s**2))
            fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name=label))
        fig.update_layout(template=PLOTLY_TEMPLATE,height=350,title=title,
                          xaxis_title="Age (years)",yaxis_title="Annual rate")
        return fig

    @output
    @render_widget
    def plot_initial_gaussian():
        r=rv_initial()
        vec=DEFAULT_PARAMS_VECTOR if not r or "initial_vector" not in r else r["initial_vector"][:15]
        return plot_gaussians(vec,"Initial Gaussian age-dependent parameters")

    @output
    @render_widget
    def plot_initial_fit():
        r=rv_initial(); d=rv_data()
        fig=go.Figure(); fig.update_layout(template=PLOTLY_TEMPLATE,height=350,
                                           xaxis_title="Age group")
        if not r or "df_agg" not in r or not d or "_error" in d:
            fig.add_annotation(text="Run initial tests first",showarrow=False); return fig
        obs=d["field_data"]
        mod=r["df_agg"]
        labels=mod["age"].astype(str)
        for col,name,dash in [
            ("cc_incidence_per100k","Incidence — field","solid"),
        ]:
            if col in obs:
                fig.add_trace(go.Bar(x=labels,y=obs[col].to_numpy()[:len(labels)],name=name))
        fig.add_trace(go.Scatter(x=labels,y=mod["inc_cc_ageclass_per100k"],name="Incidence — model",
                                 mode="lines+markers"))
        fig.update_layout(title="Initial model vs field incidence")
        return fig

    @output
    @render_widget
    def plot_initial_results():
        r=rv_initial(); fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=350,xaxis_title="Age (years)")
        if not r or "df_yearly" not in r:
            fig.add_annotation(text="Run initial tests first",showarrow=False); return fig
        df=r["df_yearly"]
        for c,label in [
            ("cc_cases_new_neg","CC cases HIV−"),
            ("cc_cases_new_pos","CC cases HIV+"),
            ("dalys_all_disc","Discounted DALYs"),
        ]:
            fig.add_trace(go.Scatter(x=df["age_years"],y=df[c],mode="lines",name=label))
        return fig

    # ------------------------- calibration
    @reactive.effect
    @reactive.event(input.btn_calibrate)
    def _calibrate():
        d=rv_data()
        if not d or "_error" in d:
            rv_calibration.set({"error":"Load workbook first."}); return
        try:
            maxiter=int(safe_input(input,"calib_maxiter",10000))
            result=run_notebook_calibration(d["field_data"], DEFAULT_PARAMS_FIXED, maxiter=maxiter)
            rv_calibration.set(result)
        except Exception as exc:
            rv_calibration.set({"error":f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})

    @output
    @render.ui
    def calib_status():
        r=rv_calibration()
        if r is None: return ui.div("Not run.",style="color:var(--text3)")
        if "error" in r: return ui.div("❌ "+r["error"],class_="alert alert-danger")
        rr=r["result"]
        env=r.get("environment", calibration_environment())
        return ui.div(
            ui.div(
                f"✅ success={rr.success} · objective={rr.fun:.8f} · iterations={rr.nit} · evaluations={rr.nfev}"
            ),
            ui.div(
                f"Python {env.get('python','?')} · NumPy {env.get('numpy','?')} · "
                f"SciPy {env.get('scipy','?')} · pandas {env.get('pandas','?')}",
                style="font-size:0.82rem; margin-top:4px; opacity:0.85;",
            ),
            class_="alert alert-success" if rr.success else "alert alert-warning",
        )

    @output
    @render.ui
    def calib_objective_table():
        r=rv_calibration()
        if not r or "result" not in r: return ui.div("Run calibration first.")
        rr=r["result"]; ic=r["initial_components"]; fc=r["final_components"]
        t=pd.DataFrame([
            ["Initial objective",r["initial_objective"]],
            ["Final objective",rr.fun],
            ["Incidence error",fc["incidence"]],
            ["Mortality error",fc["mortality"]],
            ["HIV error",fc["hiv"]],
            ["Data error",fc["data_error"]],
            ["Regularisation",fc["regularization"]],
            ["RR penalty",fc["rr_penalty"]],
            ["HIV− tunnel total",fc["duration_neg_total"]],
            ["HIV+ tunnel total",fc["duration_pos_total"]],
        ],columns=["Metric","Value"])
        return dataframe_html(t)

    @output
    @render.ui
    def calib_vector_table():
        r=rv_calibration()
        if not r or "optimized_vector" not in r: return ui.div("Run calibration first.")
        names=[
            "λ_HPV amp","λ_HPV mean","λ_HPV sigma",
            "λ_HPV+ amp","λ_HPV+ mean","λ_HPV+ sigma",
            "μ_c amp","μ_c mean","μ_c sigma",
            "μ_c+ amp","μ_c+ mean","μ_c+ sigma",
            "λ_HIV amp","λ_HIV mean","λ_HIV sigma",
            "HIV− tunnel total","HIV+ tunnel total",
        ]
        return dataframe_html(pd.DataFrame({"Parameter":names,"Value":r["optimized_vector"]}))

    @output
    @render.ui
    def calib_error_table():
        r=rv_calibration()
        if not r or "final_components" not in r: return ui.div("Run calibration first.")
        fc=r["final_components"]
        t=pd.DataFrame([{
            "Incidence":fc["incidence"],"Mortality":fc["mortality"],
            "HIV":fc["hiv"],"Data error":fc["data_error"],
            "Regularisation":fc["regularization"],"RR penalty":fc["rr_penalty"],
            "Total error":fc["total_error"],
        }])
        return dataframe_html(t)

    @output
    @render.ui
    def calib_rr_table():
        r=rv_calibration()
        if not r or "final_components" not in r: return ui.div("Run calibration first.")
        return dataframe_html(r["final_components"]["df_rr"])

    @output
    @render_widget
    def plot_calib_curves():
        r=rv_calibration()
        vec=DEFAULT_PARAMS_VECTOR if not r or "optimized_vector" not in r else r["optimized_vector"][:15]
        return plot_gaussians(vec,"Optimised Gaussian curves")

    @output
    @render_widget
    def plot_calib_fit():
        r=rv_calibration(); d=rv_data()
        fig=go.Figure(); fig.update_layout(template=PLOTLY_TEMPLATE,height=350,xaxis_title="Age group")
        if not r or "final_components" not in r or not d or "_error" in d:
            fig.add_annotation(text="Run calibration first",showarrow=False); return fig
        mod=r["final_components"]["df_agg"]; obs=d["field_data"]
        labels=mod["age"].astype(str)
        for obs_col,mod_col,label in [
            ("cc_incidence_per100k","inc_cc_ageclass_per100k","Incidence"),
            ("cc_mortality_per100k","mort_cc_ageclass_per100k","Mortality"),
            ("hiv_prevalence","hiv_prev_ageclass_pct","HIV prevalence"),
        ]:
            # Notebook's plotting code accepts either fraction or percent.
            ov=obs[obs_col].to_numpy(float)[:len(labels)]
            mv=mod[mod_col].to_numpy(float)
            if np.nanmax(np.abs(ov)) <= 1 and np.nanmax(np.abs(mv)) > 1:
                ov=ov*100
            fig.add_trace(go.Scatter(x=labels,y=ov,mode="markers",name=f"{label} — field"))
            fig.add_trace(go.Scatter(x=labels,y=mv,mode="lines+markers",name=f"{label} — model"))
        fig.update_layout(title="Calibrated model vs field data")
        return fig

    # ------------------------- strategies
    @output
    @render.ui
    def strategy_table():
        rows=[]
        for sc in STRATEGY_ORDER:
            spec=SCENARIO_NOVAX[sc]
            v=spec["vec"]
            ages_arr=np.asarray(ages)
            cov_h=np.where(v["gamma_h"]>0,1-np.exp(-v["gamma_h"]),0)
            rho=np.where(v["rho"]>0,1-np.exp(-v["rho"]),0)
            rows.append({
                "Scenario":sc,
                "Description":spec.get("description",STRATEGY_DESCRIPTIONS[sc]),
                "Primary HIV+ @14":float(cov_h[AGE_VACC]),
                "Booster HIV+ @18":float(rho[18]),
                "Booster HIV+ @24":float(rho[24]),
                "Extra catch-up physical doses":3 if sc in ["S3","S4","S5","S6"] else 0,
            })
        return dataframe_html(pd.DataFrame(rows))

    def strategy_plot(kind):
        fig=go.Figure()
        for sc in STRATEGY_ORDER:
            v=SCENARIO_NOVAX[sc]["vec"]
            if kind=="coverage":
                y=1-np.exp(-np.asarray(v["gamma_h"],float))
                y2=1-np.exp(-np.asarray(v["rho"],float))
                fig.add_trace(go.Scatter(x=ages,y=y,mode="lines",name=f"{sc} primary HIV+"))
                fig.add_trace(go.Scatter(x=ages,y=y2,mode="lines",name=f"{sc} booster HIV+",
                                         line=dict(dash="dot")))
            else:
                y=1-np.asarray(v["delta_h"],float)
                y2=1-np.asarray(v["delta_boost"],float)
                fig.add_trace(go.Scatter(x=ages,y=y,mode="lines",name=f"{sc} primary VE HIV+"))
                fig.add_trace(go.Scatter(x=ages,y=y2,mode="lines",name=f"{sc} booster VE",
                                         line=dict(dash="dot")))
        fig.update_layout(template=PLOTLY_TEMPLATE,height=350,xaxis_title="Age",
                          yaxis_title="Coverage" if kind=="coverage" else "Vaccine efficacy")
        return fig

    @output
    @render_widget
    def plot_strategy_coverage(): return strategy_plot("coverage")

    @output
    @render_widget
    def plot_strategy_efficacy(): return strategy_plot("efficacy")

    # ------------------------- run scenarios
    @reactive.effect
    @reactive.event(input.btn_run_scenarios)
    def _run_scenarios():
        try:
            cal=rv_calibration()
            if safe_input(input,"use_calibrated",True) and cal and "optimized_vector" in cal:
                pv=cal["optimized_vector"][:15]
                pf=cal["optimized_fixed"]
            else:
                pv=DEFAULT_PARAMS_VECTOR
                pf=DEFAULT_PARAMS_FIXED
            cost=float(safe_input(input,"run_cancer_cost",CANCER_COST))
            summary, yearly=run_scenarios_exact(SCENARIO_NOVAX,pv,pf,cancer_cost=cost)
            cea_s0=add_incremental_cea(summary,"S0")
            cea_s0=add_dominance_flags(cea_s0)
            cea_novax=add_incremental_cea(summary,"S-1")
            cea_novax=add_dominance_flags(cea_novax)
            rv_scenario_run.set({"summary":summary,"yearly":yearly,"params":pv,"fixed":pf,
                                 "cea_s0":cea_s0,"cea_novax":cea_novax})
            rv_cea.set({"summary":summary,"yearly":yearly,"cea_s0":cea_s0,"cea_novax":cea_novax})
        except Exception as exc:
            rv_scenario_run.set({"error":f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})

    @output
    @render.ui
    def run_status():
        r=rv_scenario_run()
        if not r: return ui.div("Not run.",style="color:var(--text3)")
        if "error" in r: return ui.div("❌ "+r["error"],class_="alert alert-danger")
        return ui.div("✅ All eight scenarios completed.",class_="alert alert-success")

    @output
    @render.ui
    def run_summary_table():
        r=rv_scenario_run()
        if not r or "summary" not in r: return ui.div("Run scenarios first.")
        return dataframe_html(r["summary"])

    @output
    @render_widget
    def plot_scenario_outputs():
        r=rv_scenario_run(); fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=500,xaxis_title="Age (years)")
        if not r or "yearly" not in r:
            fig.add_annotation(text="Run scenarios first",showarrow=False); return fig
        for sc in STRATEGY_ORDER:
            df=r["yearly"][sc]
            fig.add_trace(go.Scatter(
                x=df["age_years"],y=df["cc_cases_new_neg"]+df["cc_cases_new_pos"],
                mode="lines",name=f"{sc} CC cases",
                line=dict(color=SCENARIO_COLORS[sc],width=2),
            ))
        fig.update_layout(title="Annual cervical-cancer incident cases by scenario",
                          legend=dict(orientation="h",yanchor="bottom",y=1.02))
        return fig

    # ------------------------- CEA
    def cea_table_ui(key):
        r=rv_cea()
        if not r or key not in r: return ui.div("Run scenarios first.")
        df=r[key].copy()
        cols=[
            "Scenario","Description","Total cases","CC Deaths","DALYs (disc)",
            "Cancer cost (USD)","Vaccination cost (USD)","Total cost (USD)",
            "Total physical vaccine doses","ΔCost_vs_baseline",
            "DALYs_averted_vs_baseline","ICER","Strictly dominated","Extendedly dominated",
        ]
        cols=[c for c in cols if c in df.columns]
        return dataframe_html(df[cols])

    @output
    @render.ui
    def cea_s0_table(): return cea_table_ui("cea_s0")

    @output
    @render.ui
    def cea_novax_table(): return cea_table_ui("cea_novax")

    def ce_plane(key,title):
        r=rv_cea(); fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=400,
                          title=title,xaxis_title="DALYs averted vs comparator",
                          yaxis_title="Δ Cost (USD)")
        if not r or key not in r:
            fig.add_annotation(text="Run scenarios first",showarrow=False); return fig
        df=r[key]
        for _,row in df.iterrows():
            x=float(row["DALYs_averted_vs_baseline"]); y=float(row["ΔCost_vs_baseline"])
            if np.isfinite(x) and np.isfinite(y):
                fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",
                    text=[row["Scenario"]],textposition="top center",name=row["Scenario"],
                    marker=dict(size=14,color=SCENARIO_COLORS.get(row["Scenario"],"#00d4ff"),
                                line=dict(color="#fff",width=1.5))))
        fig.add_hline(y=0,line=dict(color="#94a3b8",dash="dot"))
        fig.add_vline(x=0,line=dict(color="#94a3b8",dash="dot"))
        return fig

    @output
    @render_widget
    def plot_cea_s0(): return ce_plane("cea_s0","CEA plane vs S0")

    @output
    @render_widget
    def plot_cea_novax(): return ce_plane("cea_novax","CEA plane vs S-1")

    @output
    @render_widget
    def plot_cost_components():
        r=rv_cea(); fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=350,barmode="stack",
                          xaxis_title="Scenario",yaxis_title="USD")
        if r and "summary" in r:
            s=r["summary"]
            fig.add_trace(go.Bar(x=s["Scenario"],y=s["Cancer cost (USD)"],name="Cancer cost"))
            fig.add_trace(go.Bar(x=s["Scenario"],y=s["Vaccination cost (USD)"],name="Vaccination cost"))
        return fig

    @output
    @render_widget
    def plot_physical_doses():
        r=rv_cea(); fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=350,
                          xaxis_title="Scenario",yaxis_title="Physical doses")
        if r and "summary" in r:
            s=r["summary"]
            fig.add_trace(go.Bar(x=s["Scenario"],y=s["Total physical vaccine doses"],name="Primary + catch-up"))
            fig.add_trace(go.Bar(x=s["Scenario"],y=s["Total boost doses"],name="Boosters"))
        return fig

    @output
    @render.ui
    def verify_costs():
        r=rv_cea()
        if not r: return ui.div("Run scenarios first.")
        check,_,_=cEA_verification_tables(r["cea_s0"],r["yearly"])
        return dataframe_html(check.reset_index())

    @output
    @render.ui
    def verify_cases():
        r=rv_cea()
        if not r: return ui.div("Run scenarios first.")
        _,cases,_=cEA_verification_tables(r["cea_s0"],r["yearly"])
        return dataframe_html(cases.reset_index())

    @output
    @render.ui
    def verify_discounted():
        r=rv_cea()
        if not r: return ui.div("Run scenarios first.")
        _,_,disc=cEA_verification_tables(r["cea_s0"],r["yearly"])
        return dataframe_html(disc.reset_index())

    # ------------------------- OWSA
    @reactive.effect
    @reactive.event(input.btn_owsa)
    def _owsa():
        try:
            cal=rv_calibration()
            pv=(cal["optimized_vector"][:15] if cal and "optimized_vector" in cal else DEFAULT_PARAMS_VECTOR)
            pf=(cal["optimized_fixed"] if cal and "optimized_fixed" in cal else DEFAULT_PARAMS_FIXED)
            params_dict={
                "lambda_HPV":list(pv[0:3]),
                "lambda_HPV_pos":list(pv[3:6]),
                "mu_c":list(pv[6:9]),
                "mu_c_pos":list(pv[9:12]),
                "lambda_HIV":list(pv[12:15]),
            }
            var=float(safe_input(input,"owsa_variation",0.1))
            pts=int(safe_input(input,"owsa_points",5))
            df=run_sensitivity_simulations_exact(params_dict,SCENARIOS_SO,pf,
                                                 mode="oneway",variation=var,n_points=pts)
            rv_owsa.set(df)
        except Exception as exc:
            rv_owsa.set(pd.DataFrame({"error":[f"{type(exc).__name__}: {exc}"]}))

    @output
    @render.ui
    def owsa_status():
        d=rv_owsa()
        if d is None: return ui.div("Not run.")
        if "error" in d.columns: return ui.div("❌ "+str(d["error"].iloc[0]),class_="alert alert-danger")
        return ui.div(f"✅ {len(d):,} long-format rows.",class_="alert alert-success")

    @output
    @render.ui
    def owsa_summary():
        d=rv_owsa()
        if d is None or "error" in d.columns: return ui.div("Run OWSA first.")
        return ui.div(f"Parameters/scenario rows: {len(d):,}<br>Scenarios: {d['Scenario'].nunique()}",
                       class_="alert alert-info")

    def tornado(df, metric, diff, title):
        fig=go.Figure()
        if df is None or "error" in df.columns:
            fig.add_annotation(text="Run OWSA first",showarrow=False)
            return fig
        z=(df[df["Scenario"]=="S6"].groupby("param_name")[diff].mean()
           .abs().sort_values(ascending=False).head(12))
        fig.add_trace(go.Bar(x=z.values,y=z.index,orientation="h",name=metric))
        fig.update_layout(template=PLOTLY_TEMPLATE,height=420,title=title,
                          xaxis_title="Mean absolute local derivative")
        return fig

    @output
    @render_widget
    def plot_tornado_dalys():
        return tornado(rv_owsa(),"DALYs","DALYs_disc_diff","Local sensitivity — DALYs")

    @output
    @render_widget
    def plot_tornado_cost():
        return tornado(rv_owsa(),"Total cost","total_cost_diff","Local sensitivity — total cost")

    @output
    @render.ui
    def owsa_table():
        d=rv_owsa()
        if d is None: return ui.div("Run OWSA first.")
        if "error" in d.columns: return ui.div(str(d["error"].iloc[0]))
        cols=["Scenario","param_name","param_value","sim_id","DALYs_disc","vax_dose_total",
              "total_cost","ΔCost_vs_baseline","DALYs_averted_vs_baseline"]
        return dataframe_html(d[cols],300)

    # ------------------------- S6 stability
    def parse_float_list(text, defaults):
        try:
            vals=[float(x.strip()) for x in str(text).split(",") if x.strip()]
            return vals or defaults
        except Exception:
            return defaults

    def run_s6_stability(mode, Ts, effs):
        cal=rv_calibration()
        pv=(cal["optimized_vector"][:15] if cal and "optimized_vector" in cal else DEFAULT_PARAMS_VECTOR)
        pf=(cal["optimized_fixed"] if cal and "optimized_fixed" in cal else DEFAULT_PARAMS_FIXED)
        rows=[]
        for T in Ts:
            for eff in effs:
                # Exact notebook idea: vary HIV+ primary or booster while keeping S6 structure.
                bundle=build_baseline_scenarios(
                    ages,age_vacc=AGE_VACC,base_coverage=0.50,opt_coverage=0.98,
                    delta_neg=0.96,duration_neg=20,delta_pos=(eff if mode=="primary" else 0.87),
                    duration_pos=(T if mode=="primary" else 15),
                    delta_boost=(eff if mode=="booster" else 0.96),
                    duration_boost=(T if mode=="booster" else 65),
                )
                # S6 compared with S0, exactly the stability criterion used by notebook.
                s6_only={"S0":bundle["SCENARIOS_SO"]["S0"],"S6":bundle["SCENARIOS_SO"]["S6"]}
                summary,yearly=run_scenarios_exact(s6_only,pv,pf)
                cea=add_incremental_cea(summary,"S0")
                row=cea.loc[cea["Scenario"]=="S6"].iloc[0]
                strict=bool(add_dominance_flags(cea)["Strictly dominated"].iloc[
                    list(cea["Scenario"]).index("S6")
                ])
                not_dom=not strict
                dominant=(float(row["ΔCost_vs_baseline"])<0 and
                           float(row["DALYs_averted_vs_baseline"])>0 and not strict)
                rows.append({
                    "T":float(T),"eff":float(eff),
                    "S6_not_dominated":not_dom,
                    "S6_dominant_vs_S0":dominant,
                    "DALYs_averted":float(row["DALYs_averted_vs_baseline"]),
                    "DeltaCost":float(row["ΔCost_vs_baseline"]),
                })
        return pd.DataFrame(rows)

    @reactive.effect
    @reactive.event(input.btn_s6_primary)
    def _s6p():
        try:
            Ts=parse_float_list(safe_input(input,"s6_primary_T","7,15,20,25"),[7,15,20,25])
            es=parse_float_list(safe_input(input,"s6_primary_eff","0.70,0.85,0.95"),[.7,.85,.95])
            rv_s6_primary.set(run_s6_stability("primary",Ts,es))
        except Exception as exc: rv_s6_primary.set(pd.DataFrame({"error":[str(exc)]}))

    @reactive.effect
    @reactive.event(input.btn_s6_boost)
    def _s6b():
        try:
            Ts=parse_float_list(safe_input(input,"s6_boost_T","25,35,50,80"),[25,35,50,80])
            es=parse_float_list(safe_input(input,"s6_boost_eff","0.75,0.95,1.0"),[.75,.95,1.0])
            rv_s6_boost.set(run_s6_stability("booster",Ts,es))
        except Exception as exc: rv_s6_boost.set(pd.DataFrame({"error":[str(exc)]}))

    @output
    @render.ui
    def s6_primary_status():
        d=rv_s6_primary()
        if d is None:return ui.div("Not run.")
        return ui.div(f"✅ {len(d)} combinations.",class_="alert alert-success")

    @output
    @render.ui
    def s6_boost_status():
        d=rv_s6_boost()
        if d is None:return ui.div("Not run.")
        return ui.div(f"✅ {len(d)} combinations.",class_="alert alert-success")

    def stability_plot(df,title):
        fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=350,title=title,
                          xaxis_title="Efficacy",yaxis_title="Duration T")
        if df is None or "error" in df.columns:
            fig.add_annotation(text="Run stability analysis",showarrow=False); return fig
        z=df.pivot(index="T",columns="eff",values="S6_dominant_vs_S0").astype(int)
        fig.add_trace(go.Heatmap(z=z.values,x=z.columns,y=z.index,
                                  colorscale=[[0,"#1a2235"],[1,"#10b981"]],
                                  showscale=False))
        return fig

    @output
    @render_widget
    def plot_s6_primary(): return stability_plot(rv_s6_primary(),"S6 dominant vs S0 — primary sensitivity")

    @output
    @render_widget
    def plot_s6_boost(): return stability_plot(rv_s6_boost(),"S6 dominant vs S0 — booster sensitivity")

    @output
    @render.ui
    def s6_tables():
        a=rv_s6_primary(); b=rv_s6_boost()
        out=[]
        if a is not None and "error" not in a.columns: out.append(dataframe_html(a))
        if b is not None and "error" not in b.columns: out.append(dataframe_html(b))
        return ui.div(*out) if out else ui.div("Run one or both stability analyses.")

    # ------------------------- PSA
    @reactive.effect
    @reactive.event(input.btn_psa)
    def _psa():
        try:
            cal=rv_calibration()
            pv=(cal["optimized_vector"][:15] if cal and "optimized_vector" in cal else DEFAULT_PARAMS_VECTOR)
            pf=(cal["optimized_fixed"] if cal and "optimized_fixed" in cal else DEFAULT_PARAMS_FIXED)
            params_dict={
                "lambda_HPV":list(pv[0:3]),"lambda_HPV_pos":list(pv[3:6]),
                "mu_c":list(pv[6:9]),"mu_c_pos":list(pv[9:12]),
                "lambda_HIV":list(pv[12:15]),
            }
            n=int(safe_input(input,"psa_n",2000))
            var=float(safe_input(input,"psa_variation",0.10))
            seed=int(safe_input(input,"psa_seed",123))
            df=run_sensitivity_simulations_exact(
                params_dict,SCENARIOS_SO,pf,mode="psa",
                variation=var,n_sims=n,seed=seed,
            )
            rv_psa.set(df)
        except Exception as exc:
            rv_psa.set(pd.DataFrame({"error":[f"{type(exc).__name__}: {exc}"]}))

    @output
    @render.ui
    def psa_status():
        d=rv_psa()
        if d is None:return ui.div("Not run.")
        if "error" in d.columns:return ui.div("❌ "+str(d["error"].iloc[0]),class_="alert alert-danger")
        return ui.div(f"✅ {len(d):,} long-format rows.",class_="alert alert-success")

    @output
    @render.ui
    def psa_summary():
        d=rv_psa()
        if d is None or "error" in d.columns:return ui.div("Run PSA first.")
        s=d.groupby("Scenario").agg(
            mean_DALYs=("DALYs_disc","mean"),sd_DALYs=("DALYs_disc","std"),
            mean_cost=("total_cost","mean"),sd_cost=("total_cost","std"),
            mean_dCost=("ΔCost_vs_baseline","mean"),
            mean_DALYs_averted=("DALYs_averted_vs_baseline","mean"),
        ).round(2).reset_index()
        return dataframe_html(s)

    @output
    @render_widget
    def plot_psa_plane():
        d=rv_psa(); fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=500,
                          title="Cost-effectiveness plane — PSA",
                          xaxis_title="DALYs averted vs S0",yaxis_title="Δ Cost (USD)")
        if d is None or "error" in d.columns:
            fig.add_annotation(text="Run PSA first",showarrow=False);return fig
        for sc in STRATEGY_ORDER:
            if sc=="S0": continue
            sub=d[d["Scenario"]==sc]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["DALYs_averted_vs_baseline"],
                y=sub["ΔCost_vs_baseline"],
                mode="markers",name=sc,
                marker=dict(size=4,opacity=.45,color=SCENARIO_COLORS[sc]),
            ))
        fig.add_hline(y=0,line=dict(color="#94a3b8",dash="dot"))
        fig.add_vline(x=0,line=dict(color="#94a3b8",dash="dot"))
        return fig

    @output
    @render_widget
    def plot_ceac():
        d=rv_psa(); fig=go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,height=430,
                          title="Cost-effectiveness acceptability curves",
                          xaxis_title="WTP (USD/DALY averted)",
                          yaxis_title="Probability cost-effective",yaxis_range=[0,1.05])
        if d is None or "error" in d.columns:
            fig.add_annotation(text="Run PSA first",showarrow=False);return fig
        try:
            wtp=np.linspace(0,5000,50)
            ceac=compute_ceac(d,wtp,baseline="S0")
            for sc in ceac["Scenario"].unique():
                sub=ceac[ceac["Scenario"]==sc]
                fig.add_trace(go.Scatter(x=sub["WTP"],y=sub["Prob_cost_effective"],
                                         mode="lines",name=sc,
                                         line=dict(color=SCENARIO_COLORS.get(sc,"#00d4ff"),width=2)))
        except Exception as exc:
            fig.add_annotation(text=f"CEAC error: {exc}",showarrow=False)
        return fig


app=App(APP_UI,server)

if __name__=="__main__":
    import shiny
    shiny.run_app("app:app",reload=True,launch_browser=False)