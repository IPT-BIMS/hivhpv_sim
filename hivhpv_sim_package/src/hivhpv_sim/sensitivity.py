"""Deterministic sensitivity analysis (OWSA) and PSA utilities.

Section 5 OWSA functions (new in updated framework):
  maximum_valid_tunnel_duration, run_validated_sensitivity_scenarios,
  create_parameter_values, run_calibrated_parameter_owsa,
  run_cancer_cost_owsa, calculate_global_sensitivity,
  calculate_local_sensitivity, get_strategy_plot_colours,
  format_axis_number, plot_global_and_local_sensitivity

Section 6 PSA functions (new in updated framework):
  sample_full_calibration_vector, convert_full_vector, run_psa_parameter_vector

NOTE: run_validated_sensitivity_scenarios (cell 204) references
SCENARIOS_SO and scenario_vectors defined in the notebook scope at runtime.
It is transcribed verbatim; the caller must ensure those globals exist
(they are produced by vaccination.build_baseline_scenarios).

NOTE: run_calibrated_parameter_owsa (cell 210) and run_cancer_cost_owsa
(cell 212) reference several notebook-level globals (OWSA_DESIGN,
SCENARIOS_SO, scenario_vectors, etc.) and are also transcribed verbatim.
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .multicohort import run_scenarios_to_summary_and_plots
from .calibration import params_fixed as _DEFAULT_PARAMS_FIXED
from .constants import ages as _DEFAULT_AGES

def maximum_valid_tunnel_duration(
    clearance_rates,
    number_of_stages=3,
    safety_fraction=0.99,
):
    """
    Progression rates are derived as:

        lambda = number_of_stages / total_duration
                 - clearance_rate

    The total duration must remain below:

        number_of_stages / clearance_rate
    """

    clearance_rates = np.asarray(
        clearance_rates,
        dtype=float,
    )

    if np.any(clearance_rates <= 0):
        raise ValueError(
            "Clearance rates must be positive."
        )

    strict_upper_bound = np.min(
        number_of_stages
        / clearance_rates
    )

    return float(
        safety_fraction
        * strict_upper_bound
    )

def run_validated_sensitivity_scenarios(
    calibration_vector,
    cancer_treatment_cost=None,
):
    """
    Run S0-S6 using the existing validated scenario vectors.

    Parameters
    ----------
    calibration_vector
        Complete 17-element calibrated parameter vector.

    cancer_treatment_cost
        Country-specific cervical cancer treatment cost.

    Returns
    -------
    summary : pandas.DataFrame
        Scenario-level outcomes.

    yearly_results : dict
        Yearly outputs for S0-S6.
    """

    calibration_vector = np.asarray(
        calibration_vector,
        dtype=float,
    )

    if calibration_vector.size != 17:
        raise ValueError(
            "calibration_vector must contain 17 values."
        )

    if cancer_treatment_cost is None:
        cancer_treatment_cost = BASE_CANCER_COST

    cancer_treatment_cost = float(
        cancer_treatment_cost
    )

    (
        candidate_params_fixed,
        candidate_age_parameters,
        candidate_age_vector,
        candidate_duration_neg,
        candidate_duration_pos,
    ) = calibration_vector_to_parameters(
        calibration_vector
    )

    summary_rows = []
    yearly_results = {}

    for strategy in STRATEGY_ORDER:

        scenario_specification = BASE_SCENARIOS[
            strategy
        ]

        # ----------------------------------------------------
        # Run the epidemiological model
        # ----------------------------------------------------

        state_neg, state_pos = run_model(
            candidate_age_vector,
            params_fixed=candidate_params_fixed,
            vax_params=scenario_specification["vec"],
            return_eff=False,
        )

        # ----------------------------------------------------
        # Construct annual health and cost outcomes
        # ----------------------------------------------------

        yearly_output = var_output(
            state_neg=state_neg,
            state_pos=state_pos,
            ages=ages,
            states_neg=NEG_NAMES,
            states_pos=POS_NAMES,
            dw=DISABILITY_WEIGHT,
            discount_rate=DISCOUNT_RATE_HEALTH,
            cancer_cost_unit=cancer_treatment_cost,
            vax_cost_per_dose=(
                int(DOSE_NUMBERS)
                * COST_PER_PHYSICAL_DOSE
            ),
            boost_cost_per_dose=(
                COST_PER_PHYSICAL_DOSE
            ),
        )
        # ----------------------------------------------------
        # Correct HIV-positive catch-up vaccination cost
        #
        # The original output costs primary vaccination as a
        # two-dose course. This function adds the cost of the
        # third physical dose for S3-S6.
        # ----------------------------------------------------

        yearly_output = (
            apply_hiv_pos_catchup_dose_adjustment(
                df_year=yearly_output,
                scenario_spec=scenario_specification,
                routine_primary_doses=int(
                    DOSE_NUMBERS
                ),
                cost_per_physical_dose=(
                    COST_PER_PHYSICAL_DOSE
                ),
            )
        )

        yearly_results[strategy] = yearly_output

        # ----------------------------------------------------
        # Vaccination recipients
        #
        # Important:
        # Vaccination performed at model age a appears in the
        # yearly output at age_years = a + 1.
        #
        # Therefore:
        # routine age 14 appears at age 15;
        # catch-up age 18 appears at age 19;
        # catch-up age 24 appears at age 25.
        # ----------------------------------------------------

        primary_recipients_hiv_neg = float(
            yearly_output[
                "vacc_doses_new_neg"
            ].sum()
        )

        recorded_routine_age = float(
            AGE_VACC + 1
        )

        routine_hiv_pos_mask = (
            yearly_output["age_years"]
            == recorded_routine_age
        )

        primary_recipients_hiv_pos_routine = float(
            yearly_output.loc[
                routine_hiv_pos_mask,
                "vacc_doses_new_pos",
            ].sum()
        )

        catchup_age = scenario_specification.get(
            "catchup_age",
            None,
        )

        if catchup_age is not None:

            recorded_catchup_age = float(
                catchup_age + 1
            )

            catchup_hiv_pos_mask = (
                yearly_output["age_years"]
                == recorded_catchup_age
            )

            primary_recipients_hiv_pos_catchup = float(
                yearly_output.loc[
                    catchup_hiv_pos_mask,
                    "vacc_doses_new_pos",
                ].sum()
            )

        else:
            recorded_catchup_age = None

            primary_recipients_hiv_pos_catchup = 0.0

        # Each booster recipient receives one physical dose
        booster_doses_hiv_pos = float(
            yearly_output[
                "vacc_doses_new_boost"
            ].sum()
        )

        # ----------------------------------------------------
        # Convert recipients into physical vaccine doses
        # ----------------------------------------------------

        # HIV-negative routine primary vaccination:
        # two physical doses per recipient
        physical_doses_hiv_neg = (
            int(DOSE_NUMBERS)
            * primary_recipients_hiv_neg
        )

        # HIV-positive routine primary vaccination:
        # two physical doses per recipient
        physical_doses_hiv_pos_routine = (
            int(DOSE_NUMBERS)
            * primary_recipients_hiv_pos_routine
        )

        # HIV-positive catch-up vaccination:
        # three physical doses per recipient in S3-S6
        catchup_course_doses = int(
            scenario_specification.get(
                "catchup_primary_doses_hiv_pos",
                0,
            )
        )

        physical_doses_hiv_pos_catchup = (
            catchup_course_doses
            * primary_recipients_hiv_pos_catchup
        )

        primary_physical_doses = (
            physical_doses_hiv_neg
            + physical_doses_hiv_pos_routine
            + physical_doses_hiv_pos_catchup
        )

        total_physical_doses = (
            primary_physical_doses
            + booster_doses_hiv_pos
        )

        # ----------------------------------------------------
        # Health outcomes
        # ----------------------------------------------------

        total_cases = float(
            (
                yearly_output[
                    "cc_cases_new_neg"
                ]
                + yearly_output[
                    "cc_cases_new_pos"
                ]
            ).sum()
        )

        total_deaths = float(
            yearly_output[
                "cc_deaths_cum_neg"
            ].iat[-1]
            + yearly_output[
                "cc_deaths_cum_pos"
            ].iat[-1]
        )

        total_dalys = float(
            yearly_output[
                "dalys_all_disc"
            ].sum()
        )

        # ----------------------------------------------------
        # Cost outcomes
        # ----------------------------------------------------

        cancer_cost = float(
            yearly_output[
                "cost_cancer_usd"
            ].sum()
        )

        vaccination_cost = float(
            yearly_output[
                "cost_vaccination_usd"
            ].sum()
        )

        total_cost = (
            cancer_cost
            + vaccination_cost
        )

        # ----------------------------------------------------
        # Save scenario-level results
        # ----------------------------------------------------

        summary_rows.append(
            {
                "Scenario": strategy,

                "Description": (
                    scenario_specification.get(
                        "description",
                        "",
                    )
                ),

                "Total cases": total_cases,

                "CC deaths": total_deaths,

                "DALYs (disc)": total_dalys,

                "Cancer cost (USD)": cancer_cost,

                "Vaccination cost (USD)": vaccination_cost,

                "Total cost (USD)": total_cost,

                "Primary recipients HIV-negative": (
                    primary_recipients_hiv_neg
                ),

                "Primary recipients HIV-positive routine": (
                    primary_recipients_hiv_pos_routine
                ),

                "Primary recipients HIV-positive catch-up": (
                    primary_recipients_hiv_pos_catchup
                ),

                "Physical doses HIV-negative": (
                    physical_doses_hiv_neg
                ),

                "Physical doses HIV-positive routine": (
                    physical_doses_hiv_pos_routine
                ),

                "Physical doses HIV-positive catch-up": (
                    physical_doses_hiv_pos_catchup
                ),

                "Booster physical doses": (
                    booster_doses_hiv_pos
                ),

                "Primary physical vaccine doses": (
                    primary_physical_doses
                ),

                "Total physical vaccine doses": (
                    total_physical_doses
                ),
            }
        )

    # --------------------------------------------------------
    # Convert all scenario rows into one summary table
    # --------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    baseline = summary.loc[
        summary["Scenario"]
        == COMPARATOR_STRATEGY
    ].iloc[0]

    # --------------------------------------------------------
    # Incremental outcomes relative to S0
    # --------------------------------------------------------

    summary[
        "DALYs averted vs S0"
    ] = (
        float(
            baseline[
                "DALYs (disc)"
            ]
        )
        - summary[
            "DALYs (disc)"
        ]
    )

    summary[
        "Incremental cost vs S0 (USD)"
    ] = (
        summary[
            "Total cost (USD)"
        ]
        - float(
            baseline[
                "Total cost (USD)"
            ]
        )
    )

    summary[
        "Additional physical doses vs S0"
    ] = (
        summary[
            "Total physical vaccine doses"
        ]
        - float(
            baseline[
                "Total physical vaccine doses"
            ]
        )
    )

    return summary, yearly_results

def create_parameter_values(
    parameter_name,
    base_value,
):
    """
    Create five sensitivity values around the base case.

    Tunnel-duration upper bounds are restricted to values that
    produce positive progression rates.
    """

    values = (
        float(base_value)
        * OWSA_MULTIPLIERS
    )

    if parameter_name == "duration_precancer_hiv_neg":
        values = np.minimum(
            values,
            MAX_DURATION_HIV_NEG,
        )

    elif parameter_name == "duration_precancer_hiv_pos":
        values = np.minimum(
            values,
            MAX_DURATION_HIV_POS,
        )

    # Amplitudes and spreads must remain positive
    if (
        "amplitude" in parameter_name
        or "spread" in parameter_name
    ):
        values = np.maximum(
            values,
            1e-8,
        )

    # Peak ages must remain within the modelled age range
    if "peak_age" in parameter_name:
        values = np.clip(
            values,
            float(np.min(ages)),
            float(np.max(ages)),
        )

    values = np.unique(
        values
    )

    return values

def run_calibrated_parameter_owsa():
    result_rows = []

    number_of_runs = len(
        OWSA_DESIGN
    )

    print(
        f"Running {number_of_runs} calibrated "
        "parameter sets."
    )

    for run_number, design_row in OWSA_DESIGN.iterrows():
        parameter_name = design_row[
            "parameter"
        ]

        parameter_index = int(
            design_row[
                "parameter_index"
            ]
        )

        tested_value = float(
            design_row[
                "tested_value"
            ]
        )

        print(
            f"{run_number + 1}/{number_of_runs} | "
            f"{parameter_name} = {tested_value:.6g}"
        )

        candidate_vector = (
            BASE_CALIBRATION_VECTOR.copy()
        )

        candidate_vector[
            parameter_index
        ] = tested_value

        try:
            candidate_summary, _ = (
                run_validated_sensitivity_scenarios(
                    calibration_vector=(
                        candidate_vector
                    ),
                    cancer_treatment_cost=(
                        BASE_CANCER_COST
                    ),
                )
            )

        except ValueError as error:
            print(
                "Skipped invalid parameter set:",
                error,
            )
            continue

        for _, scenario_row in candidate_summary.iterrows():
            result_rows.append(
                {
                    "run_id": int(run_number),
                    "parameter": parameter_name,
                    "parameter_index": parameter_index,
                    "base_value": float(
                        design_row["base_value"]
                    ),
                    "tested_value": tested_value,
                    "multiplier": float(
                        design_row["multiplier"]
                    ),
                    "Scenario": scenario_row[
                        "Scenario"
                    ],
                    "DALYs_disc": float(
                        scenario_row[
                            "DALYs (disc)"
                        ]
                    ),
                    "DALYs_averted_vs_S0": float(
                        scenario_row[
                            "DALYs averted vs S0"
                        ]
                    ),
                    "Total_cost_USD": float(
                        scenario_row[
                            "Total cost (USD)"
                        ]
                    ),
                    "Incremental_cost_vs_S0_USD": float(
                        scenario_row[
                            "Incremental cost vs S0 (USD)"
                        ]
                    ),
                    "Total_physical_doses": float(
                        scenario_row[
                            "Total physical vaccine doses"
                        ]
                    ),
                    "Additional_physical_doses_vs_S0": float(
                        scenario_row[
                            "Additional physical doses vs S0"
                        ]
                    ),
                }
            )

    results = pd.DataFrame(
        result_rows
    )

    output_path = (
        SENSITIVITY_OUTPUT_DIR
        / (
            f"{COUNTRY_KEY}_calibrated_parameter_"
            "OWSA_corrected_3dose.csv"
        )
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        "Calibrated-parameter OWSA completed."
    )

    print(
        "Saved:",
        output_path.resolve(),
    )

    return results

def calculate_global_sensitivity(
    results,
    strategy="S6",  # default: most comprehensive strategy
):
    strategy_results = results.loc[
        results["Scenario"] == strategy
    ].copy()

    base_outcomes = {
        "DALYs_averted_vs_S0": float(
            TARGET_BASE_ROW[
                "DALYs averted vs S0"
            ]
        ),
        "Incremental_cost_vs_S0_USD": float(
            TARGET_BASE_ROW[
                "Incremental cost vs S0 (USD)"
            ]
        ),
        "Additional_physical_doses_vs_S0": float(
            TARGET_BASE_ROW[
                "Additional physical doses vs S0"
            ]
        ),
    }

    output_rows = []

    for parameter_name, parameter_data in (
        strategy_results.groupby(
            "parameter"
        )
    ):
        for outcome_label, outcome_column in (
            OUTCOME_COLUMNS.items()
        ):
            base_outcome = base_outcomes[
                outcome_column
            ]

            minimum_outcome = float(
                parameter_data[
                    outcome_column
                ].min()
            )

            maximum_outcome = float(
                parameter_data[
                    outcome_column
                ].max()
            )

            output_rows.append(
                {
                    "parameter": parameter_name,
                    "outcome": outcome_label,
                    "outcome_column": outcome_column,
                    "base_outcome": base_outcome,
                    "minimum_outcome": minimum_outcome,
                    "maximum_outcome": maximum_outcome,
                    "minimum_change": (
                        minimum_outcome
                        - base_outcome
                    ),
                    "maximum_change": (
                        maximum_outcome
                        - base_outcome
                    ),
                    "total_range": (
                        maximum_outcome
                        - minimum_outcome
                    ),
                }
            )

    return pd.DataFrame(
        output_rows
    )

def calculate_local_sensitivity(
    results,
    strategy="S6",
):
    strategy_results = results.loc[
        results["Scenario"] == strategy
    ].copy()

    base_outcomes = {
        "DALYs_averted_vs_S0": float(
            TARGET_BASE_ROW[
                "DALYs averted vs S0"
            ]
        ),
        "Incremental_cost_vs_S0_USD": float(
            TARGET_BASE_ROW[
                "Incremental cost vs S0 (USD)"
            ]
        ),
        "Additional_physical_doses_vs_S0": float(
            TARGET_BASE_ROW[
                "Additional physical doses vs S0"
            ]
        ),
    }

    output_rows = []

    for parameter_name, parameter_data in (
        strategy_results.groupby(
            "parameter"
        )
    ):
        parameter_data = parameter_data.sort_values(
            "tested_value"
        )

        base_parameter = float(
            parameter_data[
                "base_value"
            ].iloc[0]
        )

        lower_candidates = parameter_data.loc[
            parameter_data[
                "tested_value"
            ] < base_parameter
        ]

        upper_candidates = parameter_data.loc[
            parameter_data[
                "tested_value"
            ] > base_parameter
        ]

        if (
            lower_candidates.empty
            or upper_candidates.empty
        ):
            continue

        lower_row = lower_candidates.iloc[-1]
        upper_row = upper_candidates.iloc[0]

        lower_parameter = float(
            lower_row["tested_value"]
        )

        upper_parameter = float(
            upper_row["tested_value"]
        )

        parameter_difference = (
            upper_parameter
            - lower_parameter
        )

        if parameter_difference == 0:
            continue

        for outcome_label, outcome_column in (
            OUTCOME_COLUMNS.items()
        ):
            lower_outcome = float(
                lower_row[outcome_column]
            )

            upper_outcome = float(
                upper_row[outcome_column]
            )

            local_derivative = (
                upper_outcome
                - lower_outcome
            ) / parameter_difference

            base_outcome = base_outcomes[
                outcome_column
            ]

            if base_outcome != 0:
                elasticity = (
                    local_derivative
                    * base_parameter
                    / base_outcome
                )
            else:
                elasticity = np.nan

            output_rows.append(
                {
                    "parameter": parameter_name,
                    "outcome": outcome_label,
                    "outcome_column": outcome_column,
                    "base_parameter": base_parameter,
                    "lower_parameter": lower_parameter,
                    "upper_parameter": upper_parameter,
                    "lower_outcome": lower_outcome,
                    "upper_outcome": upper_outcome,
                    "local_derivative": local_derivative,
                    "absolute_derivative": abs(
                        local_derivative
                    ),
                    "normalised_elasticity": elasticity,
                    "absolute_elasticity": abs(
                        elasticity
                    )
                    if np.isfinite(elasticity)
                    else np.nan,
                }
            )

    return pd.DataFrame(
        output_rows
    )

def get_strategy_plot_colours(strategy):
    """
    Select the colour scheme according to the strategy group.
    """

    if strategy in ["S1", "S2"]:
        return STRATEGY_GROUP_COLOURS["booster"]

    if strategy in ["S3", "S4"]:
        return STRATEGY_GROUP_COLOURS["standard"]

    if strategy in ["S5", "S6"]:
        return STRATEGY_GROUP_COLOURS["high"]

    return {
        "main": "#777777",
        "light": "#D9D9D9",
        "dark": "#333333",
    }

def format_axis_number(value, position=None):
    """
    Format large axis values without scientific notation.
    """

    absolute_value = abs(value)

    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if absolute_value >= 1_000:
        return f"{value / 1_000:.0f}k"

    if absolute_value >= 100:
        return f"{value:.0f}"

    if absolute_value >= 10:
        return f"{value:.1f}"

    return f"{value:.2f}"

def plot_global_and_local_sensitivity(
    global_results,
    local_results,
    outcome_label,
    filename,
    top_n=12,
):
    """
    Create a two-panel publication-style sensitivity figure.

    Left panel
    ----------
    Minimum and maximum changes relative to the base case
    across the complete tested OWSA range.

    Right panel
    -----------
    Absolute normalised local elasticity around the base case.
    """

    if outcome_label not in OUTCOME_PLOT_SETTINGS:
        raise KeyError(
            f"No plotting settings found for {outcome_label!r}."
        )

    colours = get_strategy_plot_colours(
        TARGET_STRATEGY
    )

    settings = OUTCOME_PLOT_SETTINGS[
        outcome_label
    ]

    # --------------------------------------------------------
    # Select and rank global sensitivity results
    # --------------------------------------------------------

    global_data = global_results.loc[
        global_results["outcome"] == outcome_label
    ].copy()

    global_data = global_data.sort_values(
        "total_range",
        ascending=False,
    )

    if top_n is not None:
        global_data = global_data.head(
            int(top_n)
        )

    # Reverse so the largest bar appears at the top
    global_data = (
        global_data
        .iloc[::-1]
        .reset_index(drop=True)
    )

    ordered_parameters = (
        global_data["parameter"].tolist()
    )

    display_labels = [
        PARAMETER_DISPLAY_LABELS.get(
            parameter,
            parameter,
        )
        for parameter in ordered_parameters
    ]

    # --------------------------------------------------------
    # Match local sensitivity data to the same order
    # --------------------------------------------------------

    local_data = local_results.loc[
        local_results["outcome"] == outcome_label
    ].copy()

    local_data = (
        local_data
        .set_index("parameter")
        .reindex(ordered_parameters)
        .reset_index()
    )

    local_data["absolute_elasticity"] = (
        local_data["absolute_elasticity"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    positions = np.arange(
        len(global_data)
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, 9),
        gridspec_kw={
            "width_ratios": [1.50, 0.78],
            "wspace": 0.34,
        },
    )

    figure.patch.set_facecolor(
        "white"
    )

    for axis in axes:
        axis.set_facecolor(
            "#F4F7F9"
        )

        axis.grid(
            axis="x",
            linestyle=":",
            linewidth=0.8,
            alpha=0.50,
            zorder=0,
        )

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        axis.spines["left"].set_color(
            "#888888"
        )

        axis.spines["bottom"].set_color(
            "#888888"
        )

        axis.tick_params(
            axis="x",
            labelsize=10,
        )

    # --------------------------------------------------------
    # LEFT PANEL: overall min–max sensitivity
    # --------------------------------------------------------

    axes[0].barh(
        positions,
        global_data["minimum_change"],
        height=0.62,
        color=colours["light"],
        edgecolor=colours["dark"],
        linewidth=0.8,
        label="Minimum",
        zorder=3,
    )

    axes[0].barh(
        positions,
        global_data["maximum_change"],
        height=0.62,
        color=colours["main"],
        edgecolor=colours["dark"],
        linewidth=0.8,
        label="Maximum",
        zorder=3,
    )

    axes[0].axvline(
        0,
        color="#222222",
        linewidth=1.5,
        zorder=4,
    )

    axes[0].set_yticks(
        positions
    )

    axes[0].set_yticklabels(
        display_labels,
        fontsize=10.5,
    )

    axes[0].set_xlabel(
        settings["global_xlabel"],
        fontsize=11.5,
    )

    axes[0].set_title(
        "Overall sensitivity: minimum–maximum impact",
        fontsize=13,
        fontweight="bold",
        pad=13,
    )

    axes[0].xaxis.set_major_formatter(
        FuncFormatter(
            format_axis_number
        )
    )

    axes[0].legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=9.5,
    )

    # --------------------------------------------------------
    # RIGHT PANEL: local normalised elasticity
    # --------------------------------------------------------

    axes[1].barh(
        positions,
        local_data["absolute_elasticity"],
        height=0.62,
        color=colours["main"],
        edgecolor=colours["dark"],
        linewidth=0.8,
        zorder=3,
    )

    axes[1].set_yticks(
        positions
    )

    # Do not repeat parameter labels in the right panel
    axes[1].set_yticklabels(
        []
    )

    axes[1].tick_params(
        axis="y",
        length=0,
    )

    axes[1].set_xlabel(
        settings["local_xlabel"],
        fontsize=11.5,
    )

    axes[1].set_title(
        "Local sensitivity near the base case",
        fontsize=13,
        fontweight="bold",
        pad=13,
    )

    axes[1].xaxis.set_major_formatter(
        FuncFormatter(
            format_axis_number
        )
    )

    # --------------------------------------------------------
    # Add optional value labels to local bars
    # --------------------------------------------------------

    local_values = (
        local_data["absolute_elasticity"]
        .to_numpy(dtype=float)
    )

    maximum_local_value = (
        np.nanmax(local_values)
        if len(local_values) > 0
        else 0.0
    )

    if maximum_local_value > 0:
        value_offset = (
            0.015 * maximum_local_value
        )

        axes[1].set_xlim(
            0,
            maximum_local_value * 1.14,
        )

        for position, value in zip(
            positions,
            local_values,
        ):
            if value <= 0:
                continue

            axes[1].text(
                value + value_offset,
                position,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#444444",
            )

    # --------------------------------------------------------
    # Main title and subtitle
    # --------------------------------------------------------

    figure.suptitle(
        (
            f"Sensitivity analysis of {TARGET_STRATEGY}: "
            f"{settings['title']}"
        ),
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )

    figure.text(
        0.5,
        0.943,
        (
            f"{str(COUNTRY_NAME).title()} · "
            f"{TARGET_STRATEGY} relative to "
            f"{COMPARATOR_STRATEGY}"
        ),
        ha="center",
        va="center",
        fontsize=12.5,
        color="#555555",
    )

    # --------------------------------------------------------
    # Explanatory note
    # --------------------------------------------------------

    figure.text(
        0.5,
        0.022,
        (
            "Longer bars indicate greater influence on the "
            "model outcome. Local sensitivity is shown as "
            "absolute normalised elasticity."
        ),
        ha="center",
        va="center",
        fontsize=9.7,
        color="#666666",
    )

    # --------------------------------------------------------
    # Final spacing
    # --------------------------------------------------------

    figure.subplots_adjust(
        left=0.28,
        right=0.98,
        top=0.88,
        bottom=0.11,
        wspace=0.34,
    )

    # --------------------------------------------------------
    # Save and display
    # --------------------------------------------------------

    output_path = (
        SENSITIVITY_OUTPUT_DIR
        / filename
    )

    figure.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    plt.close(
        figure
    )

    print(
        "Saved:",
        output_path.resolve(),
    )


# NOTE: convert_full_vector and run_psa_parameter_vector reference
# `calibration_vector_to_parameters` which is an application-level function
# defined in the calibration notebook cell (not a library function).
# Before calling either function from outside the notebook, inject it:
#   from hivhpv_sim import sensitivity
#   sensitivity.calibration_vector_to_parameters = your_fn
calibration_vector_to_parameters = None  # injected at runtime by notebook

def sample_full_calibration_vector(
    base_vector,
    variation,
    rng,
):
    """
    Independently sample all 17 calibrated parameters from
    Uniform(base × 0.90, base × 1.10).
    """

    base_vector = np.asarray(
        base_vector,
        dtype=float,
    )

    lower = (
        base_vector
        * (1 - variation)
    )

    upper = (
        base_vector
        * (1 + variation)
    )

    sampled = rng.uniform(
        low=lower,
        high=upper,
    )

    # Amplitudes, spreads and durations must remain positive.
    positive_indices = [
        0, 2,
        3, 5,
        6, 8,
        9, 11,
        12, 14,
        15, 16,
    ]

    sampled[
        positive_indices
    ] = np.maximum(
        sampled[
            positive_indices
        ],
        1e-12,
    )

    return sampled

def convert_full_vector(full_vector):
    """
    Convert the complete 17-element calibration vector into:

    - sampled fixed parameters, reconstructed from durations;
    - sampled 15-element age-dependent vector;
    - sampled HIV-negative precancer duration;
    - sampled HIV-positive precancer duration.
    """

    (
        sampled_params_fixed,
        sampled_age_parameter_dictionary,
        sampled_age_vector,
        sampled_duration_negative,
        sampled_duration_positive,
    ) = calibration_vector_to_parameters(
        np.asarray(
            full_vector,
            dtype=float,
        )
    )

    return {
        "params_fixed":
            sampled_params_fixed,

        "age_parameter_dictionary":
            sampled_age_parameter_dictionary,

        "age_vector":
            np.asarray(
                sampled_age_vector,
                dtype=float,
            ),

        "duration_negative":
            float(
                sampled_duration_negative
            ),

        "duration_positive":
            float(
                sampled_duration_positive
            ),
    }

def run_psa_parameter_vector(
    full_vector,
):
    """
    Rebuild duration-dependent progression rates and then
    run S0-S6 using the corresponding 15-element age vector.
    """

    converted = convert_full_vector(
        full_vector
    )

    summary_df, yearly_outputs = (
        run_scenarios_to_summary_and_plots(
            scenarios=SCENARIOS_SO,

            # The model receives the first 15 components.
            optimized_params=converted[
                "age_vector"
            ],

            # These fixed parameters have been rebuilt using
            # the two sampled precancer durations.
            params_fixed=converted[
                "params_fixed"
            ],

            ages=ages,
            states_neg=NEG_NAMES,
            states_pos=POS_NAMES,

            output_opts={
                "dw":
                    DISABILITY_WEIGHT,

                "discount_rate":
                    DISCOUNT_RATE_HEALTH,

                "cancer_cost_unit":
                    CANCER_COST,

                "vax_cost_per_dose":
                    PSA_PRIMARY_COURSE_COST,

                "boost_cost_per_dose":
                    PSA_BOOSTER_COST,
            },

            cea_opts={
                "comparator":
                    PSA_COMPARATOR,

                "cost_col":
                    "Total cost (USD)",

                "effect_col":
                    "DALYs (disc)",
            },

            save_outputs=False,
            make_plots=False,
            save_plots=False,
            log_auto=False,
        )
    )

    return (
        summary_df.copy(),
        converted,
    )

def run_cancer_cost_owsa():
    result_rows = []

    for run_id, cancer_cost_value in enumerate(
        CANCER_COST_VALUES
    ):
        print(
            f"Cancer treatment cost = "
            f"USD {cancer_cost_value:,.2f}"
        )

        summary, _ = (
            run_validated_sensitivity_scenarios(
                calibration_vector=(
                    BASE_CALIBRATION_VECTOR
                ),
                cancer_treatment_cost=(
                    cancer_cost_value
                ),
            )
        )

        for _, scenario_row in summary.iterrows():
            result_rows.append(
                {
                    "run_id": run_id,
                    "parameter": (
                        "cancer_treatment_cost"
                    ),
                    "base_value": (
                        BASE_CANCER_COST
                    ),
                    "tested_value": float(
                        cancer_cost_value
                    ),
                    "multiplier": float(
                        cancer_cost_value
                        / BASE_CANCER_COST
                    ),
                    "Scenario": scenario_row[
                        "Scenario"
                    ],
                    "DALYs_disc": float(
                        scenario_row[
                            "DALYs (disc)"
                        ]
                    ),
                    "DALYs_averted_vs_S0": float(
                        scenario_row[
                            "DALYs averted vs S0"
                        ]
                    ),
                    "Total_cost_USD": float(
                        scenario_row[
                            "Total cost (USD)"
                        ]
                    ),
                    "Incremental_cost_vs_S0_USD": float(
                        scenario_row[
                            "Incremental cost vs S0 (USD)"
                        ]
                    ),
                    "Total_physical_doses": float(
                        scenario_row[
                            "Total physical vaccine doses"
                        ]
                    ),
                    "Additional_physical_doses_vs_S0": float(
                        scenario_row[
                            "Additional physical doses vs S0"
                        ]
                    ),
                }
            )

    results = pd.DataFrame(
        result_rows
    )

    output_path = (
        SENSITIVITY_OUTPUT_DIR
        / (
            f"{COUNTRY_KEY}_cancer_cost_"
            "OWSA_corrected_3dose.csv"
        )
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        "Cancer-cost sensitivity completed."
    )

    print(
        "Saved:",
        output_path.resolve(),
    )

    return results


def compute_ceac(df, wtp_values, baseline='S0'):
    import numpy as np, pandas as pd
    rows = []
    for sc in df['Scenario'].unique():
        sub = df[df['Scenario'] == sc]
        for wtp in wtp_values:
            nb = df[df['Scenario'] == baseline]
            delta_e = sub['DALYs_averted_vs_baseline'].values
            delta_c = sub['ΔCost_vs_baseline'].values
            prob = float(np.mean(delta_c - wtp * delta_e < 0))
            rows.append({'Scenario': sc, 'WTP': wtp, 'Prob_cost_effective': prob})
    return pd.DataFrame(rows)

