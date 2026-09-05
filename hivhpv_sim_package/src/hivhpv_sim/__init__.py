"""hivhpv_sim — Cameroon Updated Framework (v0.2.0).

New modules vs v0.1.0:
  cea.py — sequential ICER, dominance, multi-perspective CEA tables
  sensitivity.py — OWSA + PSA (full 17-parameter support)
"""
from . import constants, data_io, demography, competing_risks
from . import model, calibration, vaccination, multicohort, cea, sensitivity, viz

from .model import (
    run_model, var_output, agg_ageclasses_from_yearly, configure_mortality,
    build_params_vector, apply_hiv_pos_catchup_dose_adjustment,
)
from .calibration import (
    extract_calibration_targets, build_calibration_vector,
    probability_to_rate, rate_to_probability,
    derive_progression_rate, build_params_fixed_from_durations,
    calculate_incidence_rr_table, incidence_rr_penalty,
    validate_candidate_vector,
    params_fixed, params, params_vector,
    AGE_PARAMETER_NAMES, EXPECTED_CALIBRATION_VECTOR_SIZE,
    RR_TARGET, RR_MIN, RR_MAX, RR_AGE_GROUPS,
    CALIBRATION_WEIGHTS, REGULARIZATION_WEIGHT, RR_REGULARIZATION_WEIGHT,
)
from .vaccination import build_vax_vectors_by_age, build_baseline_scenarios, vax_params
from .multicohort import run_scenarios_to_summary_and_plots
from .cea import (
    get_yearly_rows, correct_cea_physical_doses_3dose,
    add_direct_cost_effectiveness, make_cost_outcome_table,
    make_incremental_table, calculate_sequential_icers,
    merge_direct_and_sequential, make_cea_ratio_table,
    check_increasing_sequential_icers,
)
from .sensitivity import (
    maximum_valid_tunnel_duration, run_validated_sensitivity_scenarios,
    create_parameter_values, run_calibrated_parameter_owsa,
    run_cancer_cost_owsa, calculate_global_sensitivity,
    calculate_local_sensitivity, plot_global_and_local_sensitivity,
    sample_full_calibration_vector, convert_full_vector, run_psa_parameter_vector,
)


def load_country_context(country_name=None, country_input_dir=None):
    """Load Excel, build mortality functions, bind into model."""
    if country_name is None:
        country_name = constants.COUNTRY_NAME
    tables = data_io.load_life_tables(country_name, country_input_dir)
    gmr, gmh = demography.build_mortality_functions(tables["demography"], tables["qx_hiv"])
    model.configure_mortality(gmr, gmh)
    return {"demography_data": tables, "get_mortality_rate": gmr, "get_mortality_HIV": gmh,
            "params_fixed": calibration.params_fixed, "params": calibration.params,
            "params_vector": calibration.params_vector, "vax_params": vaccination.vax_params}

# Shared notebook/Shiny calibration runner.  Import these helpers from either
# ``hivhpv_sim`` or ``hivhpv_sim.calibration_runner`` so both interfaces use
# exactly the same calibration implementation.
from .calibration_runner import (
    normalized_mse, notebook_calibration_setup, vector_to_components,
    calculate_calibration_error_components, calibration_objective,
    run_notebook_calibration, calibration_environment,
)
