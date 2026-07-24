"""Funciones de Prandtl-Meyer y relaciones de expansión."""

from __future__ import annotations

import math
from dataclasses import dataclass
from scipy.optimize import brentq


@dataclass(frozen=True)
class ExpansionResult:
    mach1: float
    mach2: float
    theta_deg: float
    nu1_deg: float
    nu2_deg: float
    mu1_deg: float
    mu2_deg: float
    p2_over_p1: float
    rho2_over_rho1: float
    t2_over_t1: float
    p02_over_p01: float


def nu_max_deg(gamma: float = 1.4) -> float:
    if gamma <= 1.0:
        raise ValueError("gamma debe ser mayor que 1.")
    return math.degrees(
        0.5
        * math.pi
        * (math.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
    )


def nu_deg(mach: float, gamma: float = 1.4) -> float:
    if gamma <= 1.0:
        raise ValueError("gamma debe ser mayor que 1.")
    if mach < 1.0:
        raise ValueError("La función de Prandtl-Meyer requiere M >= 1.")
    if math.isclose(mach, 1.0, abs_tol=1e-14):
        return 0.0

    term = math.sqrt(mach**2 - 1.0)
    a = math.sqrt((gamma + 1.0) / (gamma - 1.0))
    b = math.sqrt(((gamma - 1.0) / (gamma + 1.0)) * (mach**2 - 1.0))
    nu = a * math.atan(b) - math.atan(term)
    return math.degrees(nu)


def mach_from_nu(nu_target_deg: float, gamma: float = 1.4) -> float:
    if nu_target_deg < 0.0:
        raise ValueError("nu debe ser no negativo.")
    maximum = nu_max_deg(gamma)
    if nu_target_deg >= maximum:
        raise ValueError(
            f"nu debe ser menor que nu_max={maximum:.6f}° para un Mach finito."
        )
    if math.isclose(nu_target_deg, 0.0, abs_tol=1e-14):
        return 1.0

    objective = lambda m: nu_deg(m, gamma) - nu_target_deg
    upper = 2.0
    while objective(upper) < 0.0 and upper < 1e6:
        upper *= 2.0
    return brentq(objective, 1.0 + 1e-12, upper, maxiter=300)


def expansion(
    mach1: float,
    theta_deg: float,
    gamma: float = 1.4,
) -> ExpansionResult:
    if mach1 <= 1.0:
        raise ValueError("Una expansión de Prandtl-Meyer requiere M1 > 1.")
    if theta_deg <= 0.0:
        raise ValueError("El giro de expansión debe ser positivo.")

    nu1 = nu_deg(mach1, gamma)
    nu2 = nu1 + theta_deg
    mach2 = mach_from_nu(nu2, gamma)

    a = 0.5 * (gamma - 1.0)
    t2_over_t1 = (1.0 + a * mach1**2) / (1.0 + a * mach2**2)
    p2_over_p1 = t2_over_t1 ** (gamma / (gamma - 1.0))
    rho2_over_rho1 = t2_over_t1 ** (1.0 / (gamma - 1.0))

    return ExpansionResult(
        mach1=mach1,
        mach2=mach2,
        theta_deg=theta_deg,
        nu1_deg=nu1,
        nu2_deg=nu2,
        mu1_deg=math.degrees(math.asin(1.0 / mach1)),
        mu2_deg=math.degrees(math.asin(1.0 / mach2)),
        p2_over_p1=p2_over_p1,
        rho2_over_rho1=rho2_over_rho1,
        t2_over_t1=t2_over_t1,
        p02_over_p01=1.0,
    )
