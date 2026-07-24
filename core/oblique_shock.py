"""Ondas de choque oblicuas mediante la relación theta-beta-M."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .normal_shock import all_relations as normal_shock_relations


@dataclass(frozen=True)
class ObliqueShockResult:
    mach1: float
    theta_deg: float
    beta_deg: float
    mach2: float
    mn1: float
    mn2: float
    p2_over_p1: float
    rho2_over_rho1: float
    t2_over_t1: float
    p02_over_p01: float
    entropy_change_over_r: float
    branch: str
    theta_max_deg: float


def _validate(mach1: float, theta_deg: float, gamma: float) -> None:
    if gamma <= 1.0:
        raise ValueError("gamma debe ser mayor que 1.")
    if mach1 <= 1.0:
        raise ValueError("Un choque oblicuo requiere M1 > 1.")
    if theta_deg <= 0.0:
        raise ValueError("El ángulo de deflexión debe ser positivo.")


def theta_from_beta(mach1: float, beta_rad: float, gamma: float = 1.4) -> float:
    sin_b = math.sin(beta_rad)
    numerator = 2.0 * (mach1**2 * sin_b**2 - 1.0)
    denominator = math.tan(beta_rad) * (
        mach1**2 * (gamma + math.cos(2.0 * beta_rad)) + 2.0
    )
    return math.atan(numerator / denominator)


def theta_max(mach1: float, gamma: float = 1.4) -> tuple[float, float]:
    if mach1 <= 1.0:
        raise ValueError("M1 debe ser mayor que 1.")
    mu = math.asin(1.0 / mach1)
    eps = 1e-9

    result = minimize_scalar(
        lambda beta: -theta_from_beta(mach1, beta, gamma),
        bounds=(mu + eps, 0.5 * math.pi - eps),
        method="bounded",
        options={"xatol": 1e-13},
    )
    beta_peak = float(result.x)
    theta_peak = theta_from_beta(mach1, beta_peak, gamma)
    return math.degrees(theta_peak), math.degrees(beta_peak)


def shock_angle(
    mach1: float,
    theta_deg: float,
    gamma: float = 1.4,
    branch: str = "weak",
) -> float:
    _validate(mach1, theta_deg, gamma)
    theta_target = math.radians(theta_deg)
    theta_max_deg, beta_peak_deg = theta_max(mach1, gamma)

    if theta_deg > theta_max_deg + 1e-10:
        raise ValueError(
            f"Choque desprendido: theta={theta_deg:.4f}° excede "
            f"theta_max={theta_max_deg:.4f}°."
        )

    mu = math.asin(1.0 / mach1)
    beta_peak = math.radians(beta_peak_deg)
    upper = 0.5 * math.pi - 1e-9
    f = lambda beta: theta_from_beta(mach1, beta, gamma) - theta_target

    branch_key = branch.strip().lower()
    if branch_key in {"weak", "débil", "debil"}:
        beta = brentq(f, mu + 1e-9, beta_peak, maxiter=200)
    elif branch_key in {"strong", "fuerte"}:
        beta = brentq(f, beta_peak, upper, maxiter=200)
    else:
        raise ValueError("La rama debe ser 'weak' o 'strong'.")

    return math.degrees(beta)


def all_relations(
    mach1: float,
    theta_deg: float,
    gamma: float = 1.4,
    branch: str = "weak",
) -> ObliqueShockResult:
    beta_deg = shock_angle(mach1, theta_deg, gamma, branch)
    beta = math.radians(beta_deg)
    theta = math.radians(theta_deg)

    mn1 = mach1 * math.sin(beta)
    normal = normal_shock_relations(mn1, gamma)
    mn2 = normal.mach2
    denominator = math.sin(beta - theta)
    if denominator <= 0.0:
        raise ValueError("Configuración no física: beta - theta debe ser positivo.")
    mach2 = mn2 / denominator
    theta_max_deg, _ = theta_max(mach1, gamma)

    return ObliqueShockResult(
        mach1=mach1,
        theta_deg=theta_deg,
        beta_deg=beta_deg,
        mach2=mach2,
        mn1=mn1,
        mn2=mn2,
        p2_over_p1=normal.p2_over_p1,
        rho2_over_rho1=normal.rho2_over_rho1,
        t2_over_t1=normal.t2_over_t1,
        p02_over_p01=normal.p02_over_p01,
        entropy_change_over_r=normal.entropy_change_over_r,
        branch=branch,
        theta_max_deg=theta_max_deg,
    )
