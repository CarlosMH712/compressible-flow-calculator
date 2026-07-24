"""Normal-shock relations for a calorically perfect gas."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate(mach1: float, gamma: float) -> None:
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1.")
    if mach1 <= 1.0:
        raise ValueError("A finite normal shock requires M1 > 1.")


@dataclass(frozen=True)
class NormalShockResult:
    mach1: float
    mach2: float
    p2_over_p1: float
    rho2_over_rho1: float
    t2_over_t1: float
    velocity2_over_velocity1: float
    p02_over_p01: float
    entropy_change_over_r: float


def downstream_mach(mach1: float, gamma: float = 1.4) -> float:
    _validate(mach1, gamma)
    numerator = 1.0 + 0.5 * (gamma - 1.0) * mach1**2
    denominator = gamma * mach1**2 - 0.5 * (gamma - 1.0)
    return math.sqrt(numerator / denominator)


def pressure_ratio(mach1: float, gamma: float = 1.4) -> float:
    _validate(mach1, gamma)
    return 1.0 + (2.0 * gamma / (gamma + 1.0)) * (mach1**2 - 1.0)


def density_ratio(mach1: float, gamma: float = 1.4) -> float:
    _validate(mach1, gamma)
    return ((gamma + 1.0) * mach1**2) / (
        (gamma - 1.0) * mach1**2 + 2.0
    )


def temperature_ratio(mach1: float, gamma: float = 1.4) -> float:
    return pressure_ratio(mach1, gamma) / density_ratio(mach1, gamma)


def total_pressure_ratio(mach1: float, gamma: float = 1.4) -> float:
    _validate(mach1, gamma)
    term1 = ((gamma + 1.0) * mach1**2) / (
        (gamma - 1.0) * mach1**2 + 2.0
    )
    term2 = (gamma + 1.0) / (
        2.0 * gamma * mach1**2 - (gamma - 1.0)
    )
    return term1 ** (gamma / (gamma - 1.0)) * term2 ** (
        1.0 / (gamma - 1.0)
    )


def entropy_change_over_r(mach1: float, gamma: float = 1.4) -> float:
    """Return (s2-s1)/R for the physical normal-shock branch."""
    p_ratio = pressure_ratio(mach1, gamma)
    t_ratio = temperature_ratio(mach1, gamma)
    return (gamma / (gamma - 1.0)) * math.log(t_ratio) - math.log(p_ratio)


def all_relations(mach1: float, gamma: float = 1.4) -> NormalShockResult:
    rho_ratio = density_ratio(mach1, gamma)
    return NormalShockResult(
        mach1=mach1,
        mach2=downstream_mach(mach1, gamma),
        p2_over_p1=pressure_ratio(mach1, gamma),
        rho2_over_rho1=rho_ratio,
        t2_over_t1=temperature_ratio(mach1, gamma),
        velocity2_over_velocity1=1.0 / rho_ratio,
        p02_over_p01=total_pressure_ratio(mach1, gamma),
        entropy_change_over_r=entropy_change_over_r(mach1, gamma),
    )
