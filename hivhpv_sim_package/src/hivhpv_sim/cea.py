"""CEA utility functions extracted from Section 4.3 of the notebook.

These are pure DataFrame-manipulation functions; they do not call run_model.
New vs previous Cameroon version: this module did not exist. The updated
framework (Section 4.3) introduces a fully structured CEA workflow with
sequential ICERs, dominance detection, and multi-perspective tables.

Application-only code (cells 158-193 that drive the actual CEA run and
call run_scenarios_to_summary_and_plots) stays in the notebook.
"""
import numpy as np
import pandas as pd

def get_yearly_rows(df):
    result = df.copy()

    if "scope" in result.columns:
        yearly_mask = (
            result["scope"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yearly")
        )

        if yearly_mask.any():
            result = result.loc[
                yearly_mask
            ].copy()

    if "age_years" in result.columns:
        result = result.sort_values(
            "age_years"
        )

    return result.reset_index(
        drop=True
    )

def correct_cea_physical_doses_3dose(
    cea_table,
    yearly_results,
    scenario_specs,
    comparator="S0",
):
    """
    Correct physical-dose totals without rerunning the model.

    Routine primary:
        2 doses per recipient.

    HIV-positive catch-up:
        3 doses per recipient.

    Booster:
        1 dose per recipient.
    """

    corrected_table = cea_table.copy()

    if "Scenario" not in corrected_table.columns:
        corrected_table = (
            corrected_table
            .reset_index()
        )

        if "Scenario" not in corrected_table.columns:
            first_column = (
                corrected_table.columns[0]
            )

            corrected_table = (
                corrected_table.rename(
                    columns={
                        first_column: "Scenario"
                    }
                )
            )

    corrected_table["Scenario"] = (
        corrected_table["Scenario"]
        .astype(str)
        .str.strip()
    )

    dose_rows = []

    for strategy in STRATEGY_ORDER:
        if strategy not in yearly_results:
            raise KeyError(
                f"{strategy} is missing from yearly_results."
            )

        if strategy not in scenario_specs:
            raise KeyError(
                f"{strategy} is missing from scenario_specs."
            )

        yearly_df = get_yearly_rows(
            yearly_results[strategy]
        )

        specification = (
            scenario_specs[strategy]
        )

        primary_recipients_hiv_neg = float(
            yearly_df[
                "vacc_doses_new_neg"
            ].sum()
        )

        primary_recipients_hiv_pos_total = float(
            yearly_df[
                "vacc_doses_new_pos"
            ].sum()
        )

        booster_recipients_hiv_pos = float(
            yearly_df[
                "vacc_doses_new_boost"
            ].sum()
        )

        if (
            "hiv_pos_catchup_recipients"
            in yearly_df.columns
        ):
            catchup_recipients_hiv_pos = float(
                yearly_df[
                    "hiv_pos_catchup_recipients"
                ].sum()
            )
        else:
            catchup_age = specification.get(
                "catchup_age",
                None,
            )

            if catchup_age is None:
                catchup_recipients_hiv_pos = 0.0
            else:
                catchup_output_age = float(
                    catchup_age + 1
                )

                catchup_recipients_hiv_pos = float(
                    yearly_df.loc[
                        yearly_df["age_years"]
                        == catchup_output_age,
                        "vacc_doses_new_pos",
                    ].sum()
                )

        routine_recipients_hiv_pos = (
            primary_recipients_hiv_pos_total
            - catchup_recipients_hiv_pos
        )

        if (
            routine_recipients_hiv_pos < 0
            and np.isclose(
                routine_recipients_hiv_pos,
                0.0,
                atol=1e-8,
            )
        ):
            routine_recipients_hiv_pos = 0.0

        if routine_recipients_hiv_pos < 0:
            raise ValueError(
                f"{strategy}: calculated routine HIV-positive "
                "recipients are negative."
            )

        catchup_course_doses = int(
            specification.get(
                "catchup_primary_doses_hiv_pos",
                0,
            )
        )

        if catchup_recipients_hiv_pos > 0:
            if catchup_course_doses != 3:
                raise ValueError(
                    f"{strategy}: catch-up recipients were "
                    f"found, but the scenario specifies "
                    f"{catchup_course_doses} doses."
                )
        else:
            catchup_course_doses = 0

        physical_doses_hiv_neg = (
            int(DOSE_NUMBERS)
            * primary_recipients_hiv_neg
        )

        physical_doses_hiv_pos_routine = (
            int(DOSE_NUMBERS)
            * routine_recipients_hiv_pos
        )

        physical_doses_hiv_pos_catchup = (
            catchup_course_doses
            * catchup_recipients_hiv_pos
        )

        booster_physical_doses = (
            booster_recipients_hiv_pos
        )

        primary_physical_doses = (
            physical_doses_hiv_neg
            + physical_doses_hiv_pos_routine
            + physical_doses_hiv_pos_catchup
        )

        total_physical_doses = (
            primary_physical_doses
            + booster_physical_doses
        )

        dose_rows.append(
            {
                "Scenario": strategy,

                "Primary recipients HIV-negative": (
                    primary_recipients_hiv_neg
                ),

                "Routine recipients HIV-positive": (
                    routine_recipients_hiv_pos
                ),

                "Catch-up recipients HIV-positive": (
                    catchup_recipients_hiv_pos
                ),

                "Booster recipients HIV-positive": (
                    booster_recipients_hiv_pos
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
                    booster_physical_doses
                ),

                "Primary physical vaccine doses": (
                    primary_physical_doses
                ),

                "Total physical vaccine doses": (
                    total_physical_doses
                ),
            }
        )

    dose_correction_table = pd.DataFrame(
        dose_rows
    )

    baseline_total = float(
        dose_correction_table.loc[
            dose_correction_table[
                "Scenario"
            ] == comparator,
            "Total physical vaccine doses",
        ].iloc[0]
    )

    dose_correction_table[
        "Additional physical doses vs S0"
    ] = (
        dose_correction_table[
            "Total physical vaccine doses"
        ]
        - baseline_total
    )

    dose_columns_to_replace = [
        "Primary recipients HIV-negative",
        "Routine recipients HIV-positive",
        "Catch-up recipients HIV-positive",
        "Booster recipients HIV-positive",
        "Physical doses HIV-negative",
        "Physical doses HIV-positive routine",
        "Physical doses HIV-positive catch-up",
        "Booster physical doses",
        "Primary physical vaccine doses",
        "Total physical vaccine doses",
        "Additional physical doses vs S0",
    ]

    corrected_table = corrected_table.drop(
        columns=dose_columns_to_replace,
        errors="ignore",
    )

    corrected_table = corrected_table.merge(
        dose_correction_table,
        on="Scenario",
        how="left",
        validate="one_to_one",
    )

    return (
        corrected_table,
        dose_correction_table,
    )

def add_direct_cost_effectiveness(
    table,
):
    result = table.copy()

    result[
        "ICER vs S0 (USD/DALY averted)"
    ] = np.where(
        result[
            "DALYs averted vs S0"
        ] > 0,

        result[
            "Incremental cost vs S0 (USD)"
        ]
        / result[
            "DALYs averted vs S0"
        ],

        np.nan,
    )

    classifications = []

    for _, row in result.iterrows():
        scenario = row["Scenario"]

        if scenario == "S0":
            classification = (
                "Baseline comparator"
            )

        else:
            delta_cost = float(
                row[
                    "Incremental cost vs S0 (USD)"
                ]
            )

            delta_effect = float(
                row[
                    "DALYs averted vs S0"
                ]
            )

            if (
                delta_effect > 0
                and delta_cost < 0
            ):
                classification = "Cost-saving"

            elif (
                delta_effect > 0
                and delta_cost >= 0
            ):
                classification = (
                    "More effective and more costly"
                )

            elif (
                delta_effect <= 0
                and delta_cost > 0
            ):
                classification = "Dominated"

            elif (
                delta_effect <= 0
                and delta_cost <= 0
            ):
                classification = (
                    "Less effective and less costly"
                )

            else:
                classification = (
                    "No material difference"
                )

        classifications.append(
            classification
        )

    result[
        "Cost-effectiveness classification vs S0"
    ] = classifications

    result[
        "Marginal net cost per 100k additional doses"
    ] = np.where(
        result[
            "Additional physical doses vs S0"
        ] > 0,

        result[
            "Incremental cost vs S0 (USD)"
        ]
        / result[
            "Additional physical doses vs S0"
        ]
        * 100_000,

        np.nan,
    )

    return result

def make_cost_outcome_table(
    table,
):
    return table[
        [
            "Country",
            "Scenario",
            "Strategy description",
            "Cost perspective",
            "Applied cancer cost per case (USD)",
            "Cancer cost (USD)",
            "Vaccination cost (USD)",
            "Economic total cost (USD)",
            "Incremental cost vs S0 (USD)",
            "DALYs (disc)",
            "DALYs averted vs S0",
            "Total physical vaccine doses",
            "Additional physical doses vs S0",
            "ICER vs S0 (USD/DALY averted)",
            "Marginal net cost per 100k additional doses",
            "Cost-effectiveness classification vs S0",
        ]
    ].copy()

def make_incremental_table(
    table,
):
    return table[
        [
            "Country",
            "Scenario",
            "Strategy description",
            "Incremental cost vs S0 (USD)",
            "DALYs averted vs S0",
            "Additional physical doses vs S0",
            "ICER vs S0 (USD/DALY averted)",
            "Marginal net cost per 100k additional doses",
            "Cost-effectiveness classification vs S0",
        ]
    ].copy()

def calculate_sequential_icers(
    input_df,
    cost_column,
    perspective_name,
):
    working = input_df[
        [
            "Scenario",
            "Strategy description",
            "DALYs averted vs S0",
            cost_column,
        ]
    ].copy()

    working = working.rename(
        columns={
            "DALYs averted vs S0": "Effect",
            cost_column: "Cost",
        }
    )

    working["Perspective"] = (
        perspective_name
    )

    working = working.sort_values(
        [
            "Effect",
            "Cost",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Remove strategies with the same effect but higher cost
    # --------------------------------------------------------

    duplicate_effect_removed = {}

    retained_indices = []

    for _, group in working.groupby(
        "Effect",
        sort=True,
    ):
        best_index = group[
            "Cost"
        ].idxmin()

        retained_indices.append(
            best_index
        )

        best_scenario = working.loc[
            best_index,
            "Scenario",
        ]

        for index in group.index:
            if index != best_index:
                duplicate_effect_removed[
                    working.loc[
                        index,
                        "Scenario",
                    ]
                ] = best_scenario

    unique_effects = (
        working.loc[
            retained_indices
        ]
        .sort_values(
            [
                "Effect",
                "Cost",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove strictly dominated strategies
    # --------------------------------------------------------

    strict_dominated = {}

    non_dominated_rows = []

    for i, row in unique_effects.iterrows():
        dominators = unique_effects.loc[
            (
                unique_effects[
                    "Effect"
                ] >= row["Effect"]
            )
            & (
                unique_effects[
                    "Cost"
                ] <= row["Cost"]
            )
            & (
                (
                    unique_effects[
                        "Effect"
                    ] > row["Effect"]
                )
                | (
                    unique_effects[
                        "Cost"
                    ] < row["Cost"]
                )
            )
        ]

        if not dominators.empty:
            strict_dominated[
                row["Scenario"]
            ] = (
                dominators[
                    "Scenario"
                ].tolist()
            )
        else:
            non_dominated_rows.append(
                row
            )

    frontier_candidates = pd.DataFrame(
        non_dominated_rows
    ).reset_index(drop=True)

    frontier_candidates = (
        frontier_candidates
        .sort_values(
            "Effect"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove extendedly dominated strategies
    # --------------------------------------------------------

    extended_dominated = {}

    changed = True

    while (
        changed
        and len(
            frontier_candidates
        ) >= 3
    ):
        changed = False

        sequential_values = []

        for i in range(
            1,
            len(frontier_candidates),
        ):
            delta_cost = (
                frontier_candidates.loc[
                    i,
                    "Cost",
                ]
                - frontier_candidates.loc[
                    i - 1,
                    "Cost",
                ]
            )

            delta_effect = (
                frontier_candidates.loc[
                    i,
                    "Effect",
                ]
                - frontier_candidates.loc[
                    i - 1,
                    "Effect",
                ]
            )

            sequential_values.append(
                delta_cost
                / delta_effect
            )

        for i in range(
            1,
            len(sequential_values),
        ):
            if (
                sequential_values[i]
                <= sequential_values[i - 1]
                + 1e-10
            ):
                removed_row = (
                    frontier_candidates
                    .iloc[i]
                )

                previous_row = (
                    frontier_candidates
                    .iloc[i - 1]
                )

                next_row = (
                    frontier_candidates
                    .iloc[i + 1]
                )

                extended_dominated[
                    removed_row["Scenario"]
                ] = {
                    "Previous strategy": (
                        previous_row[
                            "Scenario"
                        ]
                    ),
                    "Next strategy": (
                        next_row[
                            "Scenario"
                        ]
                    ),
                }

                frontier_candidates = (
                    frontier_candidates
                    .drop(
                        frontier_candidates.index[i]
                    )
                    .reset_index(drop=True)
                )

                changed = True
                break

    frontier_df = (
        frontier_candidates.copy()
    )

    frontier_df[
        "Sequential comparator"
    ] = None

    frontier_df[
        "Sequential ICER"
    ] = np.nan

    for i in range(
        1,
        len(frontier_df),
    ):
        previous = (
            frontier_df.iloc[
                i - 1
            ]
        )

        current = (
            frontier_df.iloc[i]
        )

        delta_cost = (
            current["Cost"]
            - previous["Cost"]
        )

        delta_effect = (
            current["Effect"]
            - previous["Effect"]
        )

        if delta_effect <= 0:
            raise ValueError(
                "Frontier effects are not increasing."
            )

        frontier_df.loc[
            frontier_df.index[i],
            "Sequential comparator",
        ] = previous["Scenario"]

        frontier_df.loc[
            frontier_df.index[i],
            "Sequential ICER",
        ] = (
            delta_cost
            / delta_effect
        )

    # --------------------------------------------------------
    # Attach frontier status to all strategies
    # --------------------------------------------------------

    result = working.copy()

    result[
        "Sequential comparator"
    ] = None

    result[
        "Sequential ICER"
    ] = np.nan

    result[
        "Frontier status"
    ] = ""

    frontier_scenarios = set(
        frontier_df["Scenario"]
    )

    for index, row in result.iterrows():
        scenario = row["Scenario"]

        if scenario in duplicate_effect_removed:
            result.loc[
                index,
                "Frontier status",
            ] = (
                "Excluded: same effectiveness "
                "but more costly than "
                + duplicate_effect_removed[
                    scenario
                ]
            )

        elif scenario in strict_dominated:
            result.loc[
                index,
                "Frontier status",
            ] = (
                "Strictly dominated by "
                + ", ".join(
                    strict_dominated[
                        scenario
                    ]
                )
            )

        elif scenario in extended_dominated:
            details = (
                extended_dominated[
                    scenario
                ]
            )

            result.loc[
                index,
                "Frontier status",
            ] = (
                "Extendedly dominated between "
                + details[
                    "Previous strategy"
                ]
                + " and "
                + details[
                    "Next strategy"
                ]
            )

        elif scenario in frontier_scenarios:
            frontier_row = (
                frontier_df.loc[
                    frontier_df[
                        "Scenario"
                    ] == scenario
                ].iloc[0]
            )

            result.loc[
                index,
                "Sequential comparator",
            ] = frontier_row[
                "Sequential comparator"
            ]

            result.loc[
                index,
                "Sequential ICER",
            ] = frontier_row[
                "Sequential ICER"
            ]

            if pd.isna(
                frontier_row["Sequential ICER"]
            ):
                if len(frontier_df) == 1:
                    result.loc[
                        index,
                        "Frontier status",
                    ] = "Only non-dominated strategy"
                else:
                    result.loc[
                        index,
                        "Frontier status",
                    ] = "Frontier: least-effective strategy"
            else:
                result.loc[
                    index,
                    "Frontier status",
                ] = (
                    "On cost-effectiveness frontier"
                )

        else:
            result.loc[
                index,
                "Frontier status",
            ] = "Excluded from frontier"

    strategy_rank = {
        strategy: position
        for position, strategy
        in enumerate(
            STRATEGY_ORDER
        )
    }

    result["_strategy_rank"] = (
        result["Scenario"]
        .map(strategy_rank)
    )

    result = (
        result
        .sort_values(
            "_strategy_rank"
        )
        .drop(
            columns="_strategy_rank"
        )
        .reset_index(drop=True)
    )

    return result, frontier_df

def merge_direct_and_sequential(
    sequential_results,
    direct_results,
):
    direct_columns = direct_results[
        [
            "Scenario",
            "Incremental cost vs S0 (USD)",
            "DALYs averted vs S0",
            "ICER vs S0 (USD/DALY averted)",
            "Cost-effectiveness classification vs S0",
        ]
    ].copy()

    result = sequential_results.drop(
        columns=[
            "Strategy description",
        ],
        errors="ignore",
    )

    result = result.merge(
        direct_columns,
        on="Scenario",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        direct_results[
            [
                "Scenario",
                "Strategy description",
            ]
        ],
        on="Scenario",
        how="left",
        validate="one_to_one",
    )

    return result

def make_cea_ratio_table(
    table,
):
    return table[
        [
            "Scenario",
            "Strategy description",
            "DALYs averted vs S0",
            "Incremental cost vs S0 (USD)",
            "ICER vs S0 (USD/DALY averted)",
            "Cost-effectiveness classification vs S0",
            "Sequential comparator",
            "Sequential ICER",
            "Frontier status",
        ]
    ].copy()

def check_increasing_sequential_icers(
    frontier_df,
    perspective,
):
    sequential_icers = (
        frontier_df[
            "Sequential ICER"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(sequential_icers) > 1:
        if not np.all(
            np.diff(
                sequential_icers
            ) >= -1e-8
        ):
            raise ValueError(
                "Sequential ICERs are not increasing for "
                f"{perspective}: {sequential_icers}"
            )

    print(
        "Sequential ICER check passed:",
        perspective,
    )

