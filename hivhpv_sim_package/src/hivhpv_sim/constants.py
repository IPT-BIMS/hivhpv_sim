"""Static configuration constants — Cameroon.

This file keeps the colleague app/package interface unchanged.
Only model-input values are set to the final analysis assumptions.
"""
import numpy as np

COUNTRY_NAME = "Cameroon"

# ------------------------------------------------------------
# 1. Population and country settings
# ------------------------------------------------------------
N_COHORT = 100000
HIV_PREVALENCE = 0.00236

# ------------------------------------------------------------
# 2. Age and simulation settings
# ------------------------------------------------------------
YEAR_START = 0
MAX_AGE = 100
N_YEARS = MAX_AGE - YEAR_START
ages = np.arange(YEAR_START, MAX_AGE + 1)
TIME_HORIZON = 100
SIMULATION_YEARS = np.arange(1, TIME_HORIZON + 1)

# ------------------------------------------------------------
# 3. Health outcome settings
# ------------------------------------------------------------
DISCOUNT_RATE_HEALTH = 0.03
DISABILITY_WEIGHT = 0.451

# ============================================================
# VACCINATION CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# 1. Vaccination strategy settings
# ------------------------------------------------------------
DOSE_NUMBERS = 2
BASE_COVERAGE = 0.50
OPT_COVERAGE = 0.98
AGE_VACC = 14

# ------------------------------------------------------------
# 2. Vaccine protection for HIV-negative women
# ------------------------------------------------------------
DURATION_NEG = 20
VACC_EFF_NEG = 0.96
DELTA_NEG = 0.96

# ------------------------------------------------------------
# 3. Vaccine protection for HIV-positive women
# ------------------------------------------------------------
DURATION_POS = 10
VACC_EFF_POS = 0.87
DELTA_POS = 0.87

# ------------------------------------------------------------
# 4. Booster protection
# ------------------------------------------------------------
DURATION_BOOST = 30
VACC_EFF_BOOST = 0.96
DELTA_BOOST = 0.96

# ============================================================
# COST AND DISCOUNTING CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# 5. Vaccination and treatment costs
# ------------------------------------------------------------
DELIVERY_COST_PER_DOSE = 1.50
VACCINE_COST_PER_DOSE = 4.50

COST_PER_PHYSICAL_DOSE = (
    VACCINE_COST_PER_DOSE
    + DELIVERY_COST_PER_DOSE
)

DOSE_PRICE = (
    DOSE_NUMBERS
    * COST_PER_PHYSICAL_DOSE
)

CATCHUP_DOSE_NUMBERS_HIV_POS = 3

DOSE_PRICE_CATCHUP_HIV_POS = (
    CATCHUP_DOSE_NUMBERS_HIV_POS
    * COST_PER_PHYSICAL_DOSE
)

DOSE_PRICE_BOOST = COST_PER_PHYSICAL_DOSE

CANCER_COST = 3777.89

# ------------------------------------------------------------
# 6. Cost discounting
# ------------------------------------------------------------
DISCOUNT_RATE_COSTS = 0.03

discountVectCost = np.array([
    1 / ((1 + DISCOUNT_RATE_COSTS) ** (year + YEAR_START))
    for year in range(N_YEARS + 1)
])

# ============================================================
# GLOBAL STATE NAMES
# ============================================================
NEG_NAMES_CORE = [
    "S", "V", "I", "P1", "P2", "P3", "C", "R", "D_c", "D"
]

POS_NAMES_CORE = [
    "Sp", "Vp", "Vp_b", "Ip", "Pp1", "Pp2", "Pp3",
    "Cp", "Rp", "D_cp", "Dp", "Dhp"
]

NEG_NAMES = NEG_NAMES_CORE + [
    "new_cases", "new_vax"
]

POS_NAMES = POS_NAMES_CORE + [
    "new_cases_p", "new_vax_p", "new_boost_p"
]

# ============================================================
# CERVICAL CANCER AGE GROUPS
# ============================================================
CC_AGE_BINS = [
    (0, 14), (15, 19), (20, 24), (25, 29),
    (30, 34), (35, 39), (40, 44), (45, 49),
    (50, 54), (55, 59), (60, 64), (65, 69),
    (70, 74), (75, 79), (80, 84), (85, 100)
]

CC_AGE_LABELS = [
    "<15", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69",
    "70-74", "75-79", "80-84", "85+"
]

assert len(CC_AGE_BINS) == len(CC_AGE_LABELS), (
    "Age bins and labels do not match."
)

cc_age_bins = CC_AGE_BINS
cc_age_labels = CC_AGE_LABELS

# ============================================================
# INITIAL COHORT DISTRIBUTION BY HIV STATUS
# ============================================================
n_HIV_pos = HIV_PREVALENCE * N_COHORT
n_HIV_neg = (1 - HIV_PREVALENCE) * N_COHORT
