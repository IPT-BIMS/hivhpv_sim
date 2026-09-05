"""Competing-risks splitting and the mass-conservation test helper.

Generic, model-agnostic mechanics: split a total hazard into named
flows, and check that population mass is conserved across a model step.
"""

import numpy as np


def split_competing_risks(total_rate, rate_dict, dt=1.0):
    rates = {k: float(v) for k, v in rate_dict.items()}

    if dt <= 0:
        raise ValueError("dt must be positive")

    if any(v < 0 for v in rates.values()):
        raise ValueError(f"Negative rate found: {rates}")

    expected_total = sum(rates.values())

    if not np.isclose(total_rate, expected_total):
        raise ValueError(
            f"total_rate={total_rate} does not equal "
            f"sum(rate_dict)={expected_total}"
        )

    if total_rate == 0:
        return 1.0, {k: 0.0 for k in rates}

    stay = np.exp(-total_rate * dt)
    out = 1.0 - stay

    probs = {
        k: (v / total_rate) * out
        for k, v in rates.items()
    }

    return stay, probs


def _mass_conservation_check(tot_prev, tot_next, *, t=None, i=None, age=None, tol=1e-12):
    """Mass conservation check with explicit error."""
    if not np.isclose(tot_next, tot_prev, rtol=tol, atol=tol):
        diff = tot_next - tot_prev
        raise ValueError(
            f"[TEST] Mass not conserved at t={t}, i={i}, age={age}: "
            f"tot_prev={tot_prev:.12g}, tot_next={tot_next:.12g}, diff={diff:.12g}"
        )

