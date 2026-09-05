"""All visualisation functions — Cameroon Updated Framework."""
import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns
from IPython.display import display, clear_output
from ipywidgets import Checkbox, Output, VBox, HBox, Button, Layout, Label

from .model import agg_ageclasses_from_yearly
from .data_io import save_figure

def plot_results(
    df,
    x_col="age_years",
    scope="yearly",
    hide=None,
    save_prefix=None,
    dpi=150,
    log_auto=False,
    max_cols_per_fig=6,
):
    hide = set(hide or [])
    dfp = df[df["scope"] == scope].copy() if "scope" in df.columns else df.copy()
    x = dfp[x_col].to_numpy(dtype=float)
    exclude = {"scope", x_col, "age_class"}
    numeric_cols = [c for c in dfp.columns if c not in exclude and c not in hide and np.issubdtype(dfp[c].dtype, np.number)]
    families = {
        "Population states": ["s_", "i_", "p1", "p2", "p3", "c_", "r_", "v"],
        "Cancer cases": ["cc_cases", "cases"],
        "Deaths": ["death", "deaths", "mortality", "d_", "dhp", "dcp", "d_c", "d_cp"],
        "Vaccination": ["vacc", "boost", "dose"],
        "YLL": ["yll"],
        "YLD": ["yld"],
        "DALYs": ["daly"],
        "Costs": ["cost"],
    }
    grouped = {fam: [] for fam in families}
    grouped["Other"] = []
    for col in numeric_cols:
        col_low = col.lower()
        matched = False
        for fam, patterns in families.items():
            if any(p in col_low for p in patterns):
                grouped[fam].append(col)
                matched = True
                break
        if not matched:
            grouped["Other"].append(col)
    def style_for(col):
        c = col.lower()
        if "neg" in c or c.endswith("_n"):
            color = "tab:blue"
            linestyle = "-"
        elif "pos" in c or "_p" in c:
            color = "tab:red"
            linestyle = "--"
        elif "all" in c or "total" in c:
            color = "black"
            linestyle = "-"
        elif "boost" in c:
            color = "tab:purple"
            linestyle = "-."
        else:
            color = None
            linestyle = "-"
        return color, linestyle
    def clean_label(col):
        return col.replace("_", " ").replace("neg", "HIV−").replace("pos", "HIV+").replace("all", "Total").replace("cc", "CC").title()
    saved = []
    for fam, cols in grouped.items():
        cols = [c for c in cols if c in dfp.columns]
        if not cols:
            continue
        for start in range(0, len(cols), max_cols_per_fig):
            chunk = cols[start:start + max_cols_per_fig]
            n = len(chunk)
            ncols = 2 if n > 1 else 1
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.2 * nrows), constrained_layout=True)
            axes = np.array(axes).reshape(-1)
            for ax, col in zip(axes, chunk):
                y = dfp[col].to_numpy(dtype=float)
                color, linestyle = style_for(col)
                ax.plot(x, y, linewidth=2, color=color, linestyle=linestyle, label=clean_label(col))
                ax.set_title(clean_label(col), fontsize=11)
                ax.set_xlabel("Age years")
                ax.set_ylabel("Value")
                ax.grid(alpha=0.3)
                if log_auto:
                    positive_y = y[np.isfinite(y) & (y > 0)]
                    if len(positive_y) > 0:
                        ratio = positive_y.max() / max(positive_y.min(), 1e-12)
                        if ratio > 100:
                            ax.set_yscale("log")
                            ax.set_ylabel("Value - log scale")
                ax.legend(frameon=False, fontsize=9)
            for ax in axes[len(chunk):]:
                ax.axis("off")
            fig.suptitle(fam, fontsize=16, fontweight="bold")
            if save_prefix:
                fname = f"{save_prefix}_{fam.lower().replace(' ', '_')}_{start // max_cols_per_fig + 1}.png"
                path = Path(fname)
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                saved.append(path)
            plt.show()
    return saved

def plot_field_vs_model(df_field_data, df_yearly):
    """Compare age-class field data with aggregated model outputs."""
    colors = {"Model": "#1f77b4", "Field": "#ff7f0e"}
    if "age" not in df_field_data.columns:
        raise KeyError("df_field_data must contain an 'age' column such as '25-29'.")
    if "cc_incidence_per100k" not in df_field_data.columns:
        raise KeyError("df_field_data must contain 'cc_incidence_per100k'.")
    if "age_years" not in df_yearly.columns:
        raise KeyError("df_yearly must contain 'age_years'.")
    def parse_age_bin(label):
        match = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*$", str(label))
        if not match:
            raise ValueError(f"Cannot parse age-bin label {label!r}. Expected format '25-29'.")
        age_min, age_max = int(match.group(1)), int(match.group(2))
        return (min(age_min, age_max), max(age_min, age_max))
    age_bins = [parse_age_bin(age) for age in df_field_data["age"]]
    model_ageclasses = agg_ageclasses_from_yearly(df_yearly, age_bins)
    if model_ageclasses.empty:
        raise ValueError("Model age-class aggregation produced an empty DataFrame.")
    field_data = df_field_data.copy()
    if "hiv_prevalence" in field_data.columns:
        hiv_field = field_data["hiv_prevalence"].astype(float)
        field_data["hiv_prevalence_pct"] = np.where(hiv_field <= 1.5, hiv_field * 100, hiv_field)
    else:
        field_data["hiv_prevalence_pct"] = np.nan
    has_mort = "mort_cc_ageclass_per100k" in model_ageclasses.columns and "cc_mortality_per100k" in field_data.columns
    has_hiv = "hiv_prev_ageclass_pct" in model_ageclasses.columns and "hiv_prevalence_pct" in field_data.columns
    model_columns = ["age", "inc_cc_ageclass_per100k"]
    field_columns = ["age", "cc_incidence_per100k"]
    if has_mort:
        model_columns.append("mort_cc_ageclass_per100k")
        field_columns.append("cc_mortality_per100k")
    if has_hiv:
        model_columns.append("hiv_prev_ageclass_pct")
        field_columns.append("hiv_prevalence_pct")
    comparison = pd.merge(model_ageclasses[model_columns], field_data[field_columns], on="age", how="inner")
    if comparison.empty:
        raise ValueError("No overlapping age classes were found between field and model data.")
    comparison["_order"] = comparison["age"].str.split("-").str[0].astype(int)
    comparison = comparison.sort_values("_order", kind="stable").drop(columns="_order")
    age_labels = comparison["age"].tolist()
    x = np.arange(len(age_labels))
    width = 0.38
    n_panels = 1 + int(has_mort) + int(has_hiv)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    ax.bar(x - width / 2, comparison["inc_cc_ageclass_per100k"], width, label="Model", color=colors["Model"])
    ax.bar(x + width / 2, comparison["cc_incidence_per100k"], width, label="Field", color=colors["Field"])
    ax.set_title("Incidence per 100,000")
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels, rotation=45, ha="right")
    ax.set_ylabel("Rate per 100,000")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    panel = 1
    if has_mort:
        ax = axes[panel]
        ax.bar(x - width / 2, comparison["mort_cc_ageclass_per100k"], width, label="Model", color=colors["Model"])
        ax.bar(x + width / 2, comparison["cc_mortality_per100k"], width, label="Field", color=colors["Field"])
        ax.set_title("Mortality per 100,000")
        ax.set_xticks(x)
        ax.set_xticklabels(age_labels, rotation=45, ha="right")
        ax.set_ylabel("Rate per 100,000")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left")
        panel += 1
    if has_hiv:
        ax = axes[panel]
        ax.bar(x - width / 2, comparison["hiv_prev_ageclass_pct"] * 100, width, label="Model", color=colors["Model"])
        ax.bar(x + width / 2, comparison["hiv_prevalence_pct"], width, label="Field", color=colors["Field"])
        ax.set_title("HIV prevalence (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(age_labels, rotation=45, ha="right")
        ax.set_ylabel("Percent (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left")
    plt.show()

def save_figures_created_by_function(
    function_to_run,
    filename_prefix,
):
    old_show = plt.show
    saved_paths = []
    counter = {
        "value": 0,
    }

    def saving_show(*args, **kwargs):
        figure_numbers = plt.get_fignums()
        for fig_number in figure_numbers:
            counter["value"] += 1
            fig = plt.figure(fig_number)
            figure_path = (
                CALIBRATION_FIGURES_PATH
                / f"{country_file_prefix}_{filename_prefix}_{counter['value']:02d}.png"
            )
            fig.savefig(
                figure_path,
                dpi=300,
                bbox_inches="tight",
            )
            saved_paths.append(figure_path)
            print("Saved:", figure_path)
        old_show(*args, **kwargs)
        plt.close("all")

    plt.show = saving_show

    try:
        function_to_run()
    finally:
        plt.show = old_show

    return saved_paths

def _plot_scenario_comparisons(
    varoutput_by_scenario,
    summary_df,
    *,
    comparator="S0",
    cost_col="Total cost (USD)",
    effect_col="DALYs (disc)",
    save_plots=False,
    plots_dir="scenario_plots",
    log_auto=True,
):
    families = {
        "Cancer cases": [
            "cc_cases_new_neg",
            "cc_cases_new_pos",
        ],
        "Cancer deaths": [
            "cc_deaths_cum_neg",
            "cc_deaths_cum_pos",
        ],
        "Vaccination": [
            "vacc_doses_new_neg",
            "vacc_doses_new_pos",
            "vacc_doses_new_boost",
            "vacc_stock_neg",
            "vacc_stock_pos",
            "vacc_stock_boost",
        ],
        "DALYs": [
            "yld_all",
            "yll_all_disc",
            "dalys_all_disc",
            "dalys_all_undisc",
        ],
        "Costs": [
            "cost_cancer_usd",
            "cost_vaccination_usd",
        ],
        "States HIV negative": [
            "s_neg",
            "i_neg",
            "c_neg",
        ],
        "States HIV positive": [
            "s_pos",
            "i_pos",
            "c_pos",
        ],
    }

    def clean_label(col):
        return (
            col.replace("_", " ")
               .replace("neg", "HIV−")
               .replace("pos", "HIV+")
               .replace("cc", "CC")
               .replace("cum", "cumulative")
               .title()
        )

    def should_log(y):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y) & (y > 0)]

        if len(y) == 0:
            return False

        ratio = y.max() / max(y.min(), 1e-12)
        return ratio > 100

    scenario_names = list(varoutput_by_scenario.keys())

    for family_name, cols in families.items():
        available_cols = []

        for col in cols:
            if any(col in df.columns for df in varoutput_by_scenario.values()):
                available_cols.append(col)

        if not available_cols:
            continue

        n = len(available_cols)
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(14, 4.5 * nrows),
            constrained_layout=True
        )

        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, available_cols):
            all_y = []

            for scen in scenario_names:
                df = varoutput_by_scenario[scen]

                if col not in df.columns:
                    continue

                dfp = df[df["scope"] == "yearly"].copy() if "scope" in df.columns else df.copy()

                x = dfp["age_years"].to_numpy(dtype=float)
                y = dfp[col].to_numpy(dtype=float)

                all_y.extend(y[np.isfinite(y)])

                linewidth = 2.7 if scen == comparator else 1.8
                linestyle = "-" if scen == comparator else "--"
                alpha = 1.0 if scen == comparator else 0.85

                ax.plot(
                    x,
                    y,
                    label=scen,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=alpha,
                )

            ax.set_title(clean_label(col), fontsize=11)
            ax.set_xlabel("Age years")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.3)

            if log_auto and should_log(all_y):
                ax.set_yscale("log")
                ax.set_ylabel("Value - log scale")

            ax.legend(
                frameon=False,
                fontsize=8,
                ncols=2,
                loc="best"
            )

        for ax in axes[len(available_cols):]:
            ax.axis("off")

        fig.suptitle(family_name, fontsize=16, fontweight="bold")

        if save_plots:
            path = Path(plots_dir) / f"{family_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")

        plt.show()

    # =====================================================
    # CEA scatter plot
    # =====================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    for scen, row in summary_df.iterrows():
        x = row["DALYs_averted_vs_baseline"]
        y = row["ΔCost_vs_baseline"]

        ax.scatter(x, y, s=80)

        ax.text(
            x,
            y,
            f" {scen}",
            fontsize=9,
            va="center"
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    ax.set_title("Cost-effectiveness plane")
    ax.set_xlabel("DALYs averted vs baseline")
    ax.set_ylabel("Incremental cost vs baseline")
    ax.grid(alpha=0.3)

    if save_plots:
        path = Path(plots_dir) / "cea_plane.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")

    plt.show()

    # =====================================================
    # Summary bar plots
    # =====================================================
    summary_metrics = [
        "Total cases",
        "CC Death",
        effect_col,
        cost_col,
        "ICER",
    ]

    available_metrics = [m for m in summary_metrics if m in summary_df.columns]

    n = len(available_metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 4.5 * nrows),
        constrained_layout=True
    )

    axes = np.array(axes).reshape(-1)

    for ax, metric in zip(axes, available_metrics):
        y = summary_df[metric].to_numpy(dtype=float)
        x = np.arange(len(summary_df.index))

        ax.bar(x, y)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df.index, rotation=45, ha="right")
        ax.set_title(metric)
        ax.grid(alpha=0.3, axis="y")

        if log_auto and should_log(y):
            ax.set_yscale("log")
            ax.set_ylabel("Value - log scale")

    for ax in axes[len(available_metrics):]:
        ax.axis("off")

    fig.suptitle("Scenario summary", fontsize=16, fontweight="bold")

    if save_plots:
        path = Path(plots_dir) / "scenario_summary_bars.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")

    plt.show()

def plot_strategy_grouped_legend(
    ages,
    vectors,
    title="Strategy"
):
    """
    Unique legend grouped into two sections:

    Coverage:
        gamma   = primary vaccination coverage HIV-
        gamma_h = primary vaccination coverage HIV+
        rho     = booster coverage HIV+

    Efficacy:
        1 - delta
        1 - delta_h
        1 - delta_boost

    Two checkboxes control the visibility of each section.
    """

    ages = np.asarray(ages, dtype=float)

    # Vaccination variables are stored as rates.
    gamma_rate = np.asarray(
        vectors.get("gamma", np.zeros_like(ages)),
        dtype=float
    )

    gamma_h_rate = np.asarray(
        vectors.get("gamma_h", np.zeros_like(ages)),
        dtype=float
    )

    rho_rate = np.asarray(
        vectors.get("rho", np.zeros_like(ages)),
        dtype=float
    )

    # Convert rates back to annual probabilities for plotting.
    gamma = 1.0 - np.exp(-gamma_rate)
    gamma_h = 1.0 - np.exp(-gamma_h_rate)
    rho = 1.0 - np.exp(-rho_rate)

    # Delta represents residual HPV infection risk.
    delta = np.asarray(
        vectors.get("delta", np.ones_like(ages)),
        dtype=float
    )

    delta_h = np.asarray(
        vectors.get("delta_h", np.ones_like(ages)),
        dtype=float
    )

    delta_boost = np.asarray(
        vectors.get("delta_boost", np.ones_like(ages)),
        dtype=float
    )

    # Convert residual risk into vaccine efficacy.
    efficacy_neg = 1.0 - delta
    efficacy_pos = 1.0 - delta_h
    efficacy_boost = 1.0 - delta_boost

    cb_cov = Checkbox(value=True, description="Coverage")
    cb_eff = Checkbox(value=False, description="Efficacy")

    out = Output()

    # Header handles used inside the legend.
    header_cov = Line2D(
        [],
        [],
        color="none",
        label="— Coverage —"
    )

    header_eff = Line2D(
        [],
        [],
        color="none",
        label="— Efficacy —"
    )

    def redraw(_=None):
        with out:
            clear_output(wait=True)

            fig, ax = plt.subplots(
                1,
                1,
                figsize=(10, 5),
                constrained_layout=True
            )

            handles = []
            labels = []

            # ---------------------------------------------
            # Coverage curves
            # ---------------------------------------------

            if cb_cov.value:
                h1, = ax.plot(
                    ages,
                    gamma,
                    label="Primary coverage HIV−"
                )

                h2, = ax.plot(
                    ages,
                    gamma_h,
                    label="Primary coverage HIV+"
                )

                h3, = ax.plot(
                    ages,
                    rho,
                    label="Booster coverage HIV+",
                    linestyle="--"
                )

                handles += [header_cov, h1, h2, h3]

                labels += [
                    header_cov.get_label(),
                    h1.get_label(),
                    h2.get_label(),
                    h3.get_label()
                ]

            # ---------------------------------------------
            # Efficacy curves
            # ---------------------------------------------

            if cb_eff.value:
                h4, = ax.plot(
                    ages,
                    efficacy_neg,
                    label="Vaccine efficacy HIV−",
                    linewidth=2,
                    alpha=0.85
                )

                h5, = ax.plot(
                    ages,
                    efficacy_pos,
                    label="Vaccine efficacy HIV+",
                    linewidth=2,
                    alpha=0.85
                )

                h6, = ax.plot(
                    ages,
                    efficacy_boost,
                    label="Booster efficacy HIV+",
                    linestyle=":",
                    linewidth=2
                )

                if cb_cov.value:
                    spacer = Line2D([], [], color="none", label="")

                    handles.append(spacer)
                    labels.append("")

                handles += [header_eff, h4, h5, h6]

                labels += [
                    header_eff.get_label(),
                    h4.get_label(),
                    h5.get_label(),
                    h6.get_label()
                ]

            if not (cb_cov.value or cb_eff.value):
                ax.text(
                    0.5,
                    0.5,
                    "Select Coverage and/or Efficacy",
                    ha="center",
                    va="center",
                    transform=ax.transAxes
                )

            ax.set_title(f"Coverage and efficacy by age — {title}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Proportion")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)

            if handles:
                legend = ax.legend(
                    handles,
                    labels,
                    loc="best",
                    frameon=True
                )

                for text, handle in zip(legend.get_texts(), handles):
                    if handle is header_cov or handle is header_eff:
                        text.set_fontweight("bold")

            plt.show()

    cb_cov.observe(redraw, names=["value"])
    cb_eff.observe(redraw, names=["value"])

    redraw()

    display(
        HBox([
            VBox([
                cb_cov,
                cb_eff
            ]),
            out
        ])
    )

def plot_cea_planes_side_by_side(
    cea_vs_novax: pd.DataFrame,
    cea_vs_s0: pd.DataFrame,
    comparator1: str = "S-1",
    comparator2: str = "S0",
    figsize: tuple = (16, 8),
    dpi: int = 100,
    shade_quadrants: bool = True,
    quad_alpha: float = 0.08
):
    """
    Plot two cost-effectiveness planes side by side.

    Quadrants:
      SE: more effective and less costly
      NE: more effective and more costly
      SW: less effective and less costly
      NW: less effective and more costly
    """

    comparator1 = str(comparator1)
    comparator2 = str(comparator2)

    x_col = "DALYs_averted_vs_comparator"
    y_col = "ΔCost_vs_comparator"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=dpi,
        sharey=False
    )

    distinct_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5"
    ]

    def _shade_quadrants(ax):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        y_zero = (0 - y_min) / (y_max - y_min)

        ax.axvspan(
            0,
            x_max,
            ymin=y_zero,
            ymax=1.0,
            color="tab:orange",
            alpha=quad_alpha,
            lw=0
        )

        ax.axvspan(
            x_min,
            0,
            ymin=y_zero,
            ymax=1.0,
            color="tab:red",
            alpha=quad_alpha,
            lw=0
        )

        ax.axvspan(
            0,
            x_max,
            ymin=0.0,
            ymax=y_zero,
            color="tab:green",
            alpha=quad_alpha,
            lw=0
        )

        ax.axvspan(
            x_min,
            0,
            ymin=0.0,
            ymax=y_zero,
            color="tab:blue",
            alpha=quad_alpha,
            lw=0
        )

    # =====================================================
    # Left: comparison with no vaccination
    # =====================================================

    ax = axes[0]

    ax.scatter(
        0,
        0,
        s=300,
        color="black",
        marker="*",
        label=f"{comparator1} (reference)",
        zorder=10,
        edgecolors="white",
        linewidth=1.5
    )

    ax.text(
        0,
        0,
        comparator1,
        fontsize=12,
        ha="right",
        va="bottom",
        weight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8
        )
    )

    scenarios = [
        str(scen)
        for scen in cea_vs_novax.index
        if str(scen) != comparator1
    ]

    for i, scen in enumerate(scenarios):
        row = cea_vs_novax.loc[scen]

        x = row[x_col]
        y = row[y_col]

        if pd.isna(x) or pd.isna(y):
            continue

        color = distinct_colors[i % len(distinct_colors)]

        ax.scatter(
            x,
            y,
            s=180,
            color=color,
            label=scen,
            alpha=0.9,
            edgecolors="white",
            linewidth=1.2,
            zorder=5
        )

        ax.annotate(
            scen,
            (x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            ha="left",
            va="bottom",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8
            )
        )

    ax.axhline(
        0,
        color="grey",
        linestyle="-",
        alpha=0.8,
        linewidth=1.2
    )

    ax.axvline(
        0,
        color="grey",
        linestyle="-",
        alpha=0.8,
        linewidth=1.2
    )

    ax.set_xlabel(
        f"DALYs averted vs {comparator1}",
        fontsize=13,
        weight="bold"
    )

    ax.set_ylabel(
        f"Incremental cost vs {comparator1} (USD)",
        fontsize=13,
        weight="bold"
    )

    ax.set_title(
        f"Cost-Effectiveness Plane\n"
        f"vs {comparator1} (No Vaccination)",
        fontsize=14,
        weight="bold",
        pad=20
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.margins(x=0.10, y=0.15)
    ax.autoscale_view()

    if shade_quadrants:
        _shade_quadrants(ax)

    ax.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.05, 1)
    )

    # =====================================================
    # Right: comparison with S0
    # =====================================================

    ax = axes[1]

    ax.scatter(
        0,
        0,
        s=300,
        color="black",
        marker="*",
        label=f"{comparator2} (reference)",
        zorder=10,
        edgecolors="white",
        linewidth=1.5
    )

    ax.text(
        0,
        0,
        comparator2,
        fontsize=12,
        ha="right",
        va="bottom",
        weight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8
        )
    )

    scenarios = [
        str(scen)
        for scen in cea_vs_s0.index
        if str(scen) != comparator2
    ]

    for i, scen in enumerate(scenarios):
        row = cea_vs_s0.loc[scen]

        x = row[x_col]
        y = row[y_col]

        if pd.isna(x) or pd.isna(y):
            continue

        color = distinct_colors[i % len(distinct_colors)]

        ax.scatter(
            x,
            y,
            s=180,
            color=color,
            label=scen,
            alpha=0.9,
            edgecolors="white",
            linewidth=1.2,
            zorder=5
        )

        ax.annotate(
            scen,
            (x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            ha="left",
            va="bottom",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8
            )
        )

    ax.axhline(
        0,
        color="grey",
        linestyle="-",
        alpha=0.8,
        linewidth=1.2
    )

    ax.axvline(
        0,
        color="grey",
        linestyle="-",
        alpha=0.8,
        linewidth=1.2
    )

    ax.set_xlabel(
        f"DALYs averted vs {comparator2}",
        fontsize=13,
        weight="bold"
    )

    ax.set_ylabel(
        f"Incremental cost vs {comparator2} (USD)",
        fontsize=13,
        weight="bold"
    )

    ax.set_title(
        f"Cost-Effectiveness Plane\n"
        f"vs {comparator2} (Baseline)",
        fontsize=14,
        weight="bold",
        pad=20
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.margins(x=0.10, y=0.15)
    ax.autoscale_view()

    if shade_quadrants:
        _shade_quadrants(ax)

    ax.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.05, 1)
    )

    plt.tight_layout()
    plt.show()

    return fig, axes

def plot_hiv_positive_cancer_outcomes_publication(
    scenario_results,
    country_name,
    base_output_folder,
    include_no_vaccination=False
):
    output_folder = (
        Path(base_output_folder)
        / "publication_figures"
        / country_name
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    scenario_labels = {
        "S-1": "No vaccination",
        "S0": "S0: Baseline",
        "S1": "S1: Baseline + booster at 18 (HIV+)",
        "S2": "S2: Baseline + booster at 24 (HIV+)",
        "S3": "S3: S1 + catch-up at 18 (HIV+)",
        "S4": "S4: S2 + catch-up at 24 (HIV+)",
        "S5": "S5: S3 with high HIV+ coverage vac.",
        "S6": "S6: S4 with high HIV+ coverage vac."
    }

    # Fixed colours matching the reference figure
    scenario_colors = {
        "S-1": "#0072B2",
        "S0": "#0072B2",
        "S1": "#E69F00",
        "S2": "#56B4E9",
        "S3": "#D55E00",
        "S4": "#CC79A7",
        "S5": "#F0E442",
        "S6": "#00BFC4"
    }

    if include_no_vaccination:
        scenario_order = ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    else:
        scenario_order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]

    available_scenarios = [
        scenario
        for scenario in scenario_order
        if scenario in scenario_results
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5.2),
        dpi=150
    )

    ax_cases, ax_deaths = axes

    for scenario in available_scenarios:
        df = scenario_results[scenario].copy()

        if "scope" in df.columns:
            yearly_mask = (
                df["scope"]
                .astype(str)
                .str.lower()
                .eq("yearly")
            )
            if yearly_mask.any():
                df = df.loc[yearly_mask].copy()

        df = df.sort_values("age_years")

        ages = df["age_years"].to_numpy()
        annual_cases = df["cc_cases_new_pos"].to_numpy()

        if "cc_deaths_cum_pos" in df.columns:
            cumulative_deaths = df["cc_deaths_cum_pos"].to_numpy()
        elif "cc_deaths_new_pos" in df.columns:
            cumulative_deaths = np.cumsum(
                df["cc_deaths_new_pos"].to_numpy()
            )
        else:
            raise KeyError(
                "The dataframe must contain either "
                "'cc_deaths_cum_pos' or 'cc_deaths_new_pos'."
            )

        linewidth = 1.2 if scenario == "S0" else 0.9

        ax_cases.plot(
            ages,
            annual_cases,
            color=scenario_colors[scenario],
            linewidth=linewidth,
            linestyle="-",
            label=scenario_labels[scenario]
        )

        ax_deaths.plot(
            ages,
            cumulative_deaths,
            color=scenario_colors[scenario],
            linewidth=linewidth,
            linestyle="-",
            label=scenario_labels[scenario]
        )

    ax_cases.set_title(
        "Annual Cervical Cancer Cases (HIV+)",
        fontsize=11
    )
    ax_cases.set_xlabel("Age (years)")
    ax_cases.set_ylabel("New cases per year")
    ax_cases.set_xlim(0, 100)
    ax_cases.set_ylim(bottom=0)
    ax_cases.grid(True, alpha=0.25)

    ax_deaths.set_title(
        "Cumulative Cervical Cancer Deaths (HIV+)",
        fontsize=11
    )
    ax_deaths.set_xlabel("Age (years)")
    ax_deaths.set_ylabel("Cumulative deaths")
    ax_deaths.set_xlim(0, 100)
    ax_deaths.set_ylim(bottom=0)
    ax_deaths.grid(True, alpha=0.25)

    handles, labels = ax_cases.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=7.5,
        frameon=True
    )

    # No large figure title
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.25,
        wspace=0.28
    )

    clean_country_name = (
        country_name
        .replace(" ", "_")
        .replace("ô", "o")
        .replace("’", "")
        .replace("'", "")
    )

    output_path = (
        output_folder
        / f"{clean_country_name}_HIV_positive_cancer_outcomes_publication.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()

    print(f"Figure saved to:\n{output_path}")

    return fig, output_path

