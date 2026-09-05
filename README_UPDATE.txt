HPV/HIV Shiny app — shared notebook/Shiny calibration update

GOAL
The Shiny interface now uses one shared calibration implementation from the
hivhpv_sim package, so the app is a graphical wrapper around the same
scientific calibration code that can also be imported by Jupyter notebooks.

RUNNABLE SHINY ENTRY POINT
    app/app.py

SHARED CALIBRATION ENGINE
    hivhpv_sim_package/src/hivhpv_sim/calibration_runner.py

The following functions were moved out of app/app.py and into the shared runner:
    normalized_mse
    notebook_calibration_setup
    vector_to_components
    calculate_calibration_error_components
    calibration_objective
    run_notebook_calibration

The Shiny app imports and calls these functions directly. A Jupyter notebook can
do the same, for example:

    from hivhpv_sim.calibration_runner import run_notebook_calibration
    result = run_notebook_calibration(field_data, params_fixed)

CALIBRATION SETTINGS PRESERVED
    method   = L-BFGS-B
    maxiter  = 10000
    maxfun   = 150000
    ftol     = 1e-10
    gtol     = 1e-7
    maxls    = 50

No explicit eps option is used.

The 17-parameter formulation, initial vector, bounds, 1/1/5 data weights,
0.05 regularisation weight, 0.05 RR penalty weight, model equations and fixed
natural-history parameters are preserved.

REPRODUCIBILITY
Each Shiny calibration result now records and displays the Python, NumPy,
SciPy and pandas versions used for that run. This helps distinguish scientific
code reproducibility from numerical optimizer/environment differences.

IMPORTANT
This archive does not hard-code a historical Zimbabwe calibration result and
does not load a saved calibrated vector as if it were a new calibration.
Calibration still starts from the notebook initial vector and runs L-BFGS-B.

Keep your existing 02_country_inputs and result/output folders. This archive
contains the code structure; country Excel inputs are not bundled here.
