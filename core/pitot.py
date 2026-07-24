"""Ideal compressible Pitot-probe relations.

For supersonic flow, the centerline model is:

1. a locally normal shock ahead of the probe;
2. isentropic deceleration of the post-shock subsonic state to M = 0;
3. the measured pressure is p02, not the post-shock static pressure p2.
"""

from __future__ import annotations

from dataclasses import dataclass

from .isentropic import pressure_ratio as isentropic_pressure_ratio
from .normal_shock import all_relations as normal_shock_relations


@dataclass(frozen=True)
class PitotResult:
    mach1: float
    p_measured_over_p1: float
    regime: str
    normal_shock_present: bool
    mach2: float | None = None
    p2_over_p1: float = 1.0
    rho2_over_rho1: float = 1.0
    t2_over_t1: float = 1.0
    p02_over_p01: float = 1.0
    p01_over_p1: float = 1.0
    p02_over_p2: float | None = None
    entropy_change_over_r: float = 0.0

    @property
    def p_measured_over_p_static(self) -> float:
        """Backward-compatible alias."""
        return self.p_measured_over_p1


def pitot_pressure_ratio(mach: float, gamma: float = 1.4) -> PitotResult:
    """Calculate the ideal Pitot reading for a local Mach number."""
    if mach <= 0.0:
        raise ValueError("Mach must be positive.")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1.")

    p01_over_p1 = isentropic_pressure_ratio(mach, gamma)

    if mach <= 1.0:
        return PitotResult(
            mach1=mach,
            p_measured_over_p1=p01_over_p1,
            p01_over_p1=p01_over_p1,
            p02_over_p2=p01_over_p1,
            normal_shock_present=False,
            regime="subsonic_isentropic_stagnation",
        )

    shock = normal_shock_relations(mach, gamma)
    p02_over_p1 = shock.p02_over_p01 * p01_over_p1
    p02_over_p2 = isentropic_pressure_ratio(shock.mach2, gamma)

    return PitotResult(
        mach1=mach,
        mach2=shock.mach2,
        p2_over_p1=shock.p2_over_p1,
        rho2_over_rho1=shock.rho2_over_rho1,
        t2_over_t1=shock.t2_over_t1,
        p02_over_p01=shock.p02_over_p01,
        p01_over_p1=p01_over_p1,
        p02_over_p2=p02_over_p2,
        p_measured_over_p1=p02_over_p1,
        normal_shock_present=True,
        entropy_change_over_r=shock.entropy_change_over_r,
        regime="supersonic_normal_shock_then_isentropic_stagnation",
    )
