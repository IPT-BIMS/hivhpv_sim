"""Natural mortality and remaining-life-expectancy utilities."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .constants import DISCOUNT_RATE_HEALTH

def remainlife_expectancy(age_array, df_demography):
    """Interpolate remaining life expectancy by age, with linear extrapolation outside the data range."""
    x = df_demography["age"].to_numpy(dtype=float)
    y = df_demography["Remaining life years"].to_numpy(dtype=float)
    ages_input = np.atleast_1d(np.asarray(age_array, dtype=float))
    remaining_life = np.interp(ages_input, x, y)
    left_mask = ages_input < x[0]
    slope_left = (y[1] - y[0]) / (x[1] - x[0])
    remaining_life[left_mask] = y[0] + slope_left * (ages_input[left_mask] - x[0])
    right_mask = ages_input > x[-1]
    slope_right = (y[-1] - y[-2]) / (x[-1] - x[-2])
    remaining_life[right_mask] = y[-1] + slope_right * (ages_input[right_mask] - x[-1])
    return float(remaining_life[0]) if np.isscalar(age_array) else remaining_life

def discounted_years(remaining_years, r_annual=DISCOUNT_RATE_HEALTH, annuite_debut=True):
    """Calculate discounted remaining life years for a scalar or array."""
    remaining_years = np.asarray(remaining_years, dtype=float)
    if r_annual < 0:
        raise ValueError("The annual discount rate cannot be negative.")
    if r_annual == 0:
        discounted = remaining_years
    else:
        discounted = (1 - (1 + r_annual) ** (-remaining_years)) / r_annual
        if annuite_debut:
            discounted *= 1 + r_annual
    return float(discounted) if np.isscalar(remaining_years) else discounted

def make_get_mortality_rate(
    df,
    age_col='age',
    mortality_col="m(x)",
    interpolate=True,
    log_scale=False,
    fit=None  # None, "gompertz", or "makeham"
):
    """
    Factory function to create a mortality rate function m(x)
    with optional interpolation, extrapolation, and parametric fitting.

    Parameters
    ----------
    df : pd.DataFrame
        Mortality table with at least two columns: age and m(x)
    age_col : str
        Column name for ages
    qx_col : str
        Column name for mortality probabilities
    interpolate : bool, default=True
        Whether to interpolate between known data points
    log_scale : bool, default=False
        Perform interpolation/extrapolation in log space (realistic for rates)
    fit : str or None, default=None
        Optional functional fit. Options:
          - None → direct table-based (interpolation/extrapolation)
          - "gompertz" → fit m(x) = a * exp(b * x)
          - "makeham" → fit m(x) = a + b * exp(c * x)

    Returns
    -------
    function
        get_mortality_rate( age): returns m(x) for given age(s)
    """

    # Prepare and clean data
    df_sorted = df.sort_values(by=age_col)
    AGE_DATA = df_sorted[age_col].values.astype(float)
    MORTALITY_DATA = df_sorted[mortality_col].values.astype(float)
    MORTALITY_DATA = np.maximum(MORTALITY_DATA, 1e-9)  # avoid log(0)

    # ---------------------------------------------------
    # Optional: Fit Gompertz or Makeham model
    # ---------------------------------------------------
    if fit is not None:
        if fit.lower() == "gompertz":
            def gompertz(x, a, b):
                return a * np.exp(b * x)
            popt, _ = curve_fit(gompertz, AGE_DATA, MORTALITY_DATA, p0=[1e-4, 0.05])
            a, b = popt

            def get_mortality_rate(age):
                age_np = np.atleast_1d(age).astype(float)
                qx = gompertz(age_np, a, b)
                return qx[0] if np.isscalar(age) else qx

        elif fit.lower() == "makeham":
            def makeham(x, a, b, c):
                return a + b * np.exp(c * x)
            popt, _ = curve_fit(makeham, AGE_DATA, MORTALITY_DATA, p0=[1e-4, 1e-4, 0.05])
            a, b, c = popt

            def get_mortality_rate(age):
                age_np = np.atleast_1d(age).astype(float)
                qx = makeham(age_np, a, b, c)
                return qx[0] if np.isscalar(age) else qx

        else:
            raise ValueError("fit must be one of: None, 'gompertz', or 'makeham'")

        return get_mortality_rate

    # ---------------------------------------------------
    # Otherwise, interpolation/extrapolation-based approach
    # ---------------------------------------------------
    if log_scale:
        Y_DATA = np.log(MORTALITY_DATA)
    else:
        Y_DATA = MORTALITY_DATA

    def get_mortality_rate(age):
        age_np = np.atleast_1d(age).astype(float)

        # Interpolation (or nearest lookup)
        if interpolate:
            qx_vals = np.interp(age_np, AGE_DATA, Y_DATA)
        else:
            idx_closest = np.abs(AGE_DATA[:, None] - age_np).argmin(axis=0)
            qx_vals = Y_DATA[idx_closest]

        # Extrapolation: linear or log-linear depending on log_scale
        left_mask = age_np < AGE_DATA[0]
        right_mask = age_np > AGE_DATA[-1]

        if np.any(left_mask):
            slope_left = (Y_DATA[1] - Y_DATA[0]) / (AGE_DATA[1] - AGE_DATA[0])
            qx_vals[left_mask] = Y_DATA[0] + slope_left * (age_np[left_mask] - AGE_DATA[0])

        if np.any(right_mask):
            slope_right = (Y_DATA[-1] - Y_DATA[-2]) / (AGE_DATA[-1] - AGE_DATA[-2])
            qx_vals[right_mask] = Y_DATA[-1] + slope_right * (age_np[right_mask] - AGE_DATA[-1])

        # Convert back from log if necessary
        if log_scale:
            qx_vals = np.exp(qx_vals)

        return qx_vals[0] if np.isscalar(age) else qx_vals

    return get_mortality_rate


def build_mortality_functions(df_demography: pd.DataFrame, df_qx_hiv: pd.DataFrame):
    """Build natural-mortality and HIV-mortality rate functions.

    NOTE: make_get_mortality_rate in this notebook version (cell 27) does not
    have the collapse_repeated_groups parameter (that was the previous Cameroon
    version, cell 27). Uses log_scale=True to match cells 28 and 33.
    """
    get_mortality_rate = make_get_mortality_rate(
        df=df_demography, age_col="age", mortality_col="m(x)",
        interpolate=True, log_scale=True, fit=None,
    )
    get_mortality_HIV = make_get_mortality_rate(
        df=df_qx_hiv, age_col="age", mortality_col="mx",
        interpolate=True, log_scale=True, fit=None,
    )
    return get_mortality_rate, get_mortality_HIV
