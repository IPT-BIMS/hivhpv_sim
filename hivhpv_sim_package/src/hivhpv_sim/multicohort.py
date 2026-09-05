"""Multi-scenario orchestration: run every vaccination scenario through
`model.run_model`, build the cost-effectiveness summary table, and (optionally)
export to Excel and draw comparison plots via `hivhpv_sim.viz`.

`run_scenarios_to_summary_and_plots` was missing from the cell range originally
inspected (truncated mid-extraction) and was supplied directly afterward -- it
lives in notebook cell 66, alongside `_plot_scenario_comparisons`. Both are
transcribed verbatim below; the plotting half is re-exported from `viz.py` and
imported here so this module stays focused on orchestration, per your
decision to give plotting its own module.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .model import run_model, var_output
from .viz import _plot_scenario_comparisons
from .constants import (
    NEG_NAMES, POS_NAMES, DISABILITY_WEIGHT, DISCOUNT_RATE_HEALTH,
    CANCER_COST, DOSE_PRICE, DOSE_PRICE_BOOST,
)


def run_scenarios_to_summary_and_plots(
    scenarios: dict,
    optimized_params,
    params_fixed,
    ages,
    *,
    states_neg=NEG_NAMES,
    states_pos=POS_NAMES,
    output_opts=None,
    cea_opts=None,
    save_outputs=False,
    output_xlsx_path="Results.xlsx",
    make_plots=True,
    save_plots=False,
    plots_dir="scenario_plots",
    log_auto=True,
):
    output_opts = {} if output_opts is None else dict(output_opts)
    cea_opts = {} if cea_opts is None else dict(cea_opts)

    dw = output_opts.get("dw", DISABILITY_WEIGHT)
    discount_rate = output_opts.get("discount_rate", DISCOUNT_RATE_HEALTH)
    cancer_cost_unit = output_opts.get("cancer_cost_unit", CANCER_COST)
    vax_cost_per_dose = output_opts.get("vax_cost_per_dose", DOSE_PRICE)
    boost_cost_per_dose = output_opts.get("boost_cost_per_dose", DOSE_PRICE_BOOST)

    comparator = cea_opts.get("comparator", "S0")
    cost_col = cea_opts.get("cost_col", "Total cost (USD)")
    effect_col = cea_opts.get("effect_col", "DALYs (disc)")

    summary_df = pd.DataFrame()
    varoutput_by_scenario = {}

    writer = None

    if save_plots:
        Path(plots_dir).mkdir(parents=True, exist_ok=True)

    try:
        if save_outputs:
            writer = pd.ExcelWriter(output_xlsx_path, engine="openpyxl", mode="w")

        # =====================================================
        # 1. RUN SCENARIOS
        # =====================================================
        for name, spec in scenarios.items():
            vax_vectors = spec["vec"]
            desc = spec.get("description", "")

            state_neg, state_pos = run_model(
                optimized_params,
                params_fixed=params_fixed,
                vax_params=vax_vectors,
                return_eff=False,
            )

            df_year = var_output(
                state_neg=state_neg,
                state_pos=state_pos,
                ages=ages,
                states_neg=states_neg,
                states_pos=states_pos,
                dw=dw,
                discount_rate=discount_rate,
                cancer_cost_unit=cancer_cost_unit,
                vax_cost_per_dose=vax_cost_per_dose,
                boost_cost_per_dose=boost_cost_per_dose,
            )

            varoutput_by_scenario[str(name)] = df_year

            cc_cases_new = df_year["cc_cases_new_neg"] + df_year["cc_cases_new_pos"]
            vacc_doses_new = df_year["vacc_doses_new_neg"] + df_year["vacc_doses_new_pos"]

            summary_df.loc[name, "Description"] = desc
            summary_df.loc[name, "Total cases"] = cc_cases_new.sum()
            summary_df.loc[name, "Total vaccination doses"] = vacc_doses_new.sum()
            summary_df.loc[name, "Total boost doses"] = df_year["vacc_doses_new_boost"].sum()

            summary_df.loc[name, "CC Death"] = (
                df_year["cc_deaths_cum_neg"].iat[-1]
                + df_year["cc_deaths_cum_pos"].iat[-1]
            )

            summary_df.loc[name, "YLD"] = df_year["yld_all"].sum()
            summary_df.loc[name, "YLL (disc)"] = df_year["yll_all_disc"].sum()
            summary_df.loc[name, effect_col] = df_year["dalys_all_disc"].sum()

            cost_cancer = df_year["cost_cancer_usd"].sum()
            cost_vax = df_year["cost_vaccination_usd"].sum()

            summary_df.loc[name, "Cancer cost (USD)"] = cost_cancer
            summary_df.loc[name, "Vaccination cost (USD)"] = cost_vax
            summary_df.loc[name, cost_col] = cost_cancer + cost_vax

        # =====================================================
        # 2. CEA
        # =====================================================
        C_ref = float(summary_df.loc[comparator, cost_col])
        E_ref = float(summary_df.loc[comparator, effect_col])

        summary_df["ΔCost_vs_baseline"] = summary_df[cost_col] - C_ref
        summary_df["DALYs_averted_vs_baseline"] = E_ref - summary_df[effect_col]

        with np.errstate(divide="ignore", invalid="ignore"):
            summary_df["ICER"] = np.where(
                summary_df["DALYs_averted_vs_baseline"] > 0,
                summary_df["ΔCost_vs_baseline"] / summary_df["DALYs_averted_vs_baseline"],
                np.nan,
            )

        summary_df["Strictly dominated"] = False
        summary_df["Extendedly dominated"] = False

        # =====================================================
        # 3. EXPORT EXCEL
        # =====================================================
        if writer is not None:
            summary_df.to_excel(writer, sheet_name="SUMMARY")

            for name, df in varoutput_by_scenario.items():
                df.to_excel(writer, sheet_name=str(name)[:31], index=False)

            writer.close()
            writer = None

        # =====================================================
        # 4. PLOTS
        # =====================================================
        if make_plots:
            _plot_scenario_comparisons(
                varoutput_by_scenario,
                summary_df,
                comparator=comparator,
                cost_col=cost_col,
                effect_col=effect_col,
                save_plots=save_plots,
                plots_dir=plots_dir,
                log_auto=log_auto,
            )

        return (
            summary_df
            .reset_index()
            .rename(columns={"index": "Scenario"}),
            varoutput_by_scenario
        )

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

