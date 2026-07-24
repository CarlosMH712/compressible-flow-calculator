"""Relaciones isoentrópicas para un gas caloríficamente perfecto."""

from __future__ import annotations

import math
from dataclasses import dataclass
from scipy.optimize import brentq


def _validate_gamma(gamma: float) -> None:
    if gamma <= 1.0:
        raise ValueError("gamma debe ser mayor que 1.")


def _validate_mach(mach: float, *, positive: bool = True) -> None:
    if positive and mach <= 0.0:
        raise ValueError("El número de Mach debe ser positivo.")


@dataclass(frozen=True)
class IsentropicResult:
    mach: float
    gamma: float
    t0_over_t: float
    t_over_t0: float
    p0_over_p: float
    p_over_p0: float
    rho0_over_rho: float
    rho_over_rho0: float
    area_over_astar: float
    mach_star: float
    mach_angle_deg: float | None


def temperature_ratio(mach: float, gamma: float = 1.4) -> float:
    _validate_gamma(gamma)
    _validate_mach(mach)
    return 1.0 + 0.5 * (gamma - 1.0) * mach**2


def pressure_ratio(mach: float, gamma: float = 1.4) -> float:
    tr = temperature_ratio(mach, gamma)
    return tr ** (gamma / (gamma - 1.0))


def density_ratio(mach: float, gamma: float = 1.4) -> float:
    tr = temperature_ratio(mach, gamma)
    return tr ** (1.0 / (gamma - 1.0))


def area_ratio(mach: float, gamma: float = 1.4) -> float:
    _validate_gamma(gamma)
    _validate_mach(mach)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    bracket = (2.0 / (gamma + 1.0)) * (
        1.0 + 0.5 * (gamma - 1.0) * mach**2
    )
    return (1.0 / mach) * bracket**exponent


def mach_angle_deg(mach: float) -> float | None:
    _validate_mach(mach)
    if mach < 1.0:
        return None
    return math.degrees(math.asin(1.0 / mach))


def characteristic_mach(mach: float, gamma: float = 1.4) -> float:
    """Return M* = V/a* for the same total-enthalpy line."""
    _validate_gamma(gamma)
    _validate_mach(mach)
    return math.sqrt(
        ((gamma + 1.0) * mach**2)
        / (2.0 + (gamma - 1.0) * mach**2)
    )


def critical_ratios(gamma: float = 1.4) -> dict[str, float]:
    """Return sonic-to-total property ratios."""
    _validate_gamma(gamma)
    t_star_over_t0 = 2.0 / (gamma + 1.0)
    return {
        "t_star_over_t0": t_star_over_t0,
        "p_star_over_p0": t_star_over_t0 ** (gamma / (gamma - 1.0)),
        "rho_star_over_rho0": t_star_over_t0 ** (1.0 / (gamma - 1.0)),
    }


def all_relations(mach: float, gamma: float = 1.4) -> IsentropicResult:
    t0_t = temperature_ratio(mach, gamma)
    p0_p = pressure_ratio(mach, gamma)
    rho0_rho = density_ratio(mach, gamma)
    return IsentropicResult(
        mach=mach,
        gamma=gamma,
        t0_over_t=t0_t,
        t_over_t0=1.0 / t0_t,
        p0_over_p=p0_p,
        p_over_p0=1.0 / p0_p,
        rho0_over_rho=rho0_rho,
        rho_over_rho0=1.0 / rho0_rho,
        area_over_astar=area_ratio(mach, gamma),
        mach_star=characteristic_mach(mach, gamma),
        mach_angle_deg=mach_angle_deg(mach),
    )


def mach_from_area_ratio(
    area_over_astar: float,
    branch: str = "supersonic",
    gamma: float = 1.4,
) -> float:
    _validate_gamma(gamma)
    if area_over_astar < 1.0:
        raise ValueError("A/A* debe ser mayor o igual que 1.")

    if math.isclose(area_over_astar, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return 1.0

    branch_normalized = branch.strip().lower()
    objective = lambda m: area_ratio(m, gamma) - area_over_astar

    if branch_normalized in {"subsonic", "subsónica", "subsonica"}:
        return brentq(objective, 1e-8, 1.0 - 1e-10, maxiter=200)
    if branch_normalized in {"supersonic", "supersónica", "supersonica"}:
        upper = 2.0
        while objective(upper) < 0.0 and upper < 1e4:
            upper *= 2.0
        if upper >= 1e4:
            raise ValueError("No se pudo acotar la solución supersónica.")
        return brentq(objective, 1.0 + 1e-10, upper, maxiter=200)

    raise ValueError("La rama debe ser 'subsonic' o 'supersonic'.")
