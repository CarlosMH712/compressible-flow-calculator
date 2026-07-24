"""Sequential solver for an open, piecewise-linear 2-D surface.

Conventions
-----------
- Freestream travels from left to right.
- ``angle_deg`` is the absolute inclination of each wall segment.
- For a lower surface, increasing wall angle is a compression turn.
- For an upper surface, decreasing wall angle is a compression turn.
- Wave interactions, boundary-layer separation, and viscous effects are excluded.
- A Pitot probe creates a local diagnostic branch; its normal shock does not
  modify the external main-flow solution on subsequent panels.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

from core.isentropic import pressure_ratio as isentropic_p0_over_p
from core.isentropic import temperature_ratio as isentropic_t0_over_t
from core.oblique_shock import all_relations as oblique_shock
from core.pitot import pitot_pressure_ratio
from core.prandtl_meyer import expansion


@dataclass
class FlowState:
    region: int
    segment: int
    event_code: str
    event: str
    turn_deg: float
    wall_delta_deg: float
    wall_angle_deg: float
    mach: float
    pressure_pa: float
    pressure_ratio_to_freestream: float
    temperature_k: float
    temperature_ratio_to_freestream: float
    density_ratio_to_freestream: float
    total_pressure_pa: float
    total_pressure_ratio_to_freestream: float
    cp: float
    beta_deg: float | None = None
    mn1: float | None = None
    mn2: float | None = None
    mu1_deg: float | None = None
    mu2_deg: float | None = None
    branch: str | None = None
    note: str = ""


def build_vertices(segments: Iterable[dict]) -> list[tuple[float, float]]:
    points = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    for row in segments:
        length = float(row["length"])
        angle = math.radians(float(row["angle_deg"]))
        if length <= 0.0:
            raise ValueError("All segment lengths must be positive.")
        x += length * math.cos(angle)
        y += length * math.sin(angle)
        points.append((x, y))
    return points


def solve_surface(
    segments: Iterable[dict],
    *,
    mach_inf: float,
    pressure_inf_pa: float,
    temperature_inf_k: float,
    gamma: float = 1.4,
    side: str = "lower",
    branch: str = "weak",
    pitot_segment: int | None = None,
) -> tuple[pd.DataFrame, list[dict], list[tuple[float, float]], dict | None]:
    """Solve the external flow over one open polygonal surface."""
    rows = list(segments)
    if not rows:
        raise ValueError("At least one segment is required.")
    if mach_inf <= 1.0:
        raise ValueError("The geometry solver currently requires M∞ > 1.")
    if pressure_inf_pa <= 0.0 or temperature_inf_k <= 0.0:
        raise ValueError("Pressure and temperature must be positive.")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1.")

    side_key = side.strip().lower()
    if side_key not in {"lower", "upper", "inferior", "superior"}:
        raise ValueError("side must be lower/inferior or upper/superior.")

    is_lower = side_key in {"lower", "inferior"}
    toward_flow_sign = 1.0 if is_lower else -1.0
    field_sign = 1.0 if is_lower else -1.0

    vertices = build_vertices(rows)
    q_inf = 0.5 * gamma * pressure_inf_pa * mach_inf**2
    p0_inf = pressure_inf_pa * isentropic_p0_over_p(mach_inf, gamma)

    mach = mach_inf
    pressure = pressure_inf_pa
    temperature = temperature_inf_k
    rho_ratio = 1.0
    p0 = p0_inf
    previous_angle = 0.0

    states: list[FlowState] = []
    waves: list[dict] = []
    pitot_output: dict | None = None

    for i, row in enumerate(rows, start=1):
        angle = float(row["angle_deg"])
        wall_delta = angle - previous_angle
        physical_turn = toward_flow_sign * wall_delta

        event_code = "no_turn"
        event = "sin giro"
        beta: float | None = None
        mn1: float | None = None
        mn2: float | None = None
        mu1_deg: float | None = None
        mu2_deg: float | None = None
        wave_branch: str | None = None
        note = ""

        if abs(physical_turn) < 1e-10:
            pass
        elif physical_turn > 0.0:
            event_code = "oblique_shock"
            event = "choque oblicuo"
            try:
                result = oblique_shock(
                    mach,
                    abs(physical_turn),
                    gamma=gamma,
                    branch=branch,
                )
            except ValueError as exc:
                event_code = "detached_shock"
                event = "choque desprendido"
                note = str(exc)
                states.append(
                    FlowState(
                        region=i + 1,
                        segment=i,
                        event_code=event_code,
                        event=event,
                        turn_deg=physical_turn,
                        wall_delta_deg=wall_delta,
                        wall_angle_deg=angle,
                        mach=mach,
                        pressure_pa=pressure,
                        pressure_ratio_to_freestream=pressure / pressure_inf_pa,
                        temperature_k=temperature,
                        temperature_ratio_to_freestream=temperature / temperature_inf_k,
                        density_ratio_to_freestream=rho_ratio,
                        total_pressure_pa=p0,
                        total_pressure_ratio_to_freestream=p0 / p0_inf,
                        cp=(pressure - pressure_inf_pa) / q_inf,
                        beta_deg=None,
                        mn1=None,
                        mn2=None,
                        mu1_deg=None,
                        mu2_deg=None,
                        branch=branch,
                        note=note,
                    )
                )
                break

            upstream_angle = previous_angle
            beta = result.beta_deg
            mn1 = result.mn1
            mn2 = result.mn2
            wave_branch = branch
            mach = result.mach2
            pressure *= result.p2_over_p1
            temperature *= result.t2_over_t1
            rho_ratio *= result.rho2_over_rho1
            p0 *= result.p02_over_p01
            waves.append(
                {
                    "kind": "shock",
                    "vertex_index": i - 1,
                    "direction_deg": upstream_angle + field_sign * beta,
                    "beta_deg": beta,
                    "turn_deg": abs(physical_turn),
                    "branch": branch,
                    "source_region": i,
                    "target_region": i + 1,
                }
            )
        else:
            event_code = "prandtl_meyer_expansion"
            event = "expansión Prandtl-Meyer"
            result = expansion(mach, abs(physical_turn), gamma=gamma)
            upstream_angle = previous_angle
            mu1 = math.degrees(math.asin(1.0 / mach))
            mu2 = math.degrees(math.asin(1.0 / result.mach2))
            mu1_deg = mu1
            mu2_deg = mu2
            mach = result.mach2
            pressure *= result.p2_over_p1
            temperature *= result.t2_over_t1
            rho_ratio *= result.rho2_over_rho1
            waves.append(
                {
                    "kind": "expansion",
                    "vertex_index": i - 1,
                    "direction_start_deg": upstream_angle + field_sign * mu1,
                    "direction_end_deg": angle + field_sign * mu2,
                    "turn_deg": abs(physical_turn),
                    "source_region": i,
                    "target_region": i + 1,
                }
            )

        state = FlowState(
            region=i + 1,
            segment=i,
            event_code=event_code,
            event=event,
            turn_deg=physical_turn,
            wall_delta_deg=wall_delta,
            wall_angle_deg=angle,
            mach=mach,
            pressure_pa=pressure,
            pressure_ratio_to_freestream=pressure / pressure_inf_pa,
            temperature_k=temperature,
            temperature_ratio_to_freestream=temperature / temperature_inf_k,
            density_ratio_to_freestream=rho_ratio,
            total_pressure_pa=p0,
            total_pressure_ratio_to_freestream=p0 / p0_inf,
            cp=(pressure - pressure_inf_pa) / q_inf,
            beta_deg=beta,
            mn1=mn1,
            mn2=mn2,
            mu1_deg=mu1_deg,
            mu2_deg=mu2_deg,
            branch=wave_branch,
            note=note,
        )
        states.append(state)

        if pitot_segment == i:
            pitot = pitot_pressure_ratio(mach, gamma)
            t0 = temperature * isentropic_t0_over_t(mach, gamma)

            if pitot.normal_shock_present:
                p2 = pressure * pitot.p2_over_p1
                t2 = temperature * pitot.t2_over_t1
                rho2_ratio = rho_ratio * pitot.rho2_over_rho1
                p02 = p0 * pitot.p02_over_p01
            else:
                p2 = pressure
                t2 = temperature
                rho2_ratio = rho_ratio
                p02 = p0

            pitot_output = {
                "segment": i,
                "source_region": i + 1,
                "wall_angle_deg": angle,
                "fluid_normal_sign": 1.0 if is_lower else -1.0,
                "normal_shock_present": pitot.normal_shock_present,
                "mach1": mach,
                "p1_pa": pressure,
                "t1_k": temperature,
                "rho1_ratio_to_freestream": rho_ratio,
                "p01_pa": p0,
                "mach2": pitot.mach2,
                "p2_pa": p2,
                "t2_k": t2,
                "rho2_ratio_to_freestream": rho2_ratio,
                "p2_over_p1": pitot.p2_over_p1,
                "t2_over_t1": pitot.t2_over_t1,
                "rho2_over_rho1": pitot.rho2_over_rho1,
                "p02_over_p01": pitot.p02_over_p01,
                "p01_over_p1": pitot.p01_over_p1,
                "p02_over_p2": pitot.p02_over_p2,
                "p02_over_p1": pitot.p_measured_over_p1,
                "p02_pa": p02,
                "p_measured_pa": p02,
                "t0_k": t0,
                "p2_ratio_to_freestream": p2 / pressure_inf_pa,
                "t2_ratio_to_freestream": t2 / temperature_inf_k,
                "p02_ratio_to_freestream_static": p02 / pressure_inf_pa,
                "p02_ratio_to_freestream_total": p02 / p0_inf,
                "t0_ratio_to_freestream": t0 / temperature_inf_k,
                "rho0_ratio_to_freestream": (
                    (p02 / pressure_inf_pa) / (t0 / temperature_inf_k)
                ),
                "entropy_change_over_r": pitot.entropy_change_over_r,
                "regime_code": pitot.regime,
            }

        previous_angle = angle

    return pd.DataFrame(asdict(state) for state in states), waves, vertices, pitot_output


def build_region_table(
    surface_df: pd.DataFrame,
    *,
    mach_inf: float,
    pressure_inf_pa: float,
    temperature_inf_k: float,
    gamma: float = 1.4,
    pitot_data: dict | None = None,
) -> pd.DataFrame:
    """Build a presentation/CSV table with freestream and local Pitot branch rows.

    Pitot rows are inserted after the selected main-flow region. They are explicitly
    tagged as a diagnostic branch so they are not mistaken for the state propagated
    to subsequent wall panels.
    """
    q_inf = 0.5 * gamma * pressure_inf_pa * mach_inf**2
    p0_inf = pressure_inf_pa * isentropic_p0_over_p(mach_inf, gamma)

    rows: list[dict] = [
        {
            "region": "1",
            "path_code": "main_flow",
            "event_code": "freestream",
            "turn_deg": math.nan,
            "wall_angle_deg": 0.0,
            "mach": mach_inf,
            "pressure_pa": pressure_inf_pa,
            "pressure_ratio_to_freestream": 1.0,
            "temperature_k": temperature_inf_k,
            "temperature_ratio_to_freestream": 1.0,
            "density_ratio_to_freestream": 1.0,
            "total_pressure_pa": p0_inf,
            "total_pressure_ratio_to_freestream": 1.0,
            "cp": 0.0,
            "beta_deg": math.nan,
            "mn1": math.nan,
            "mn2": math.nan,
            "mu1_deg": math.nan,
            "mu2_deg": math.nan,
            "branch": "",
            "note_code": "",
        }
    ]

    for state in surface_df.to_dict(orient="records"):
        rows.append(
            {
                "region": str(int(state["region"])),
                "path_code": "main_flow",
                "event_code": state["event_code"],
                "turn_deg": state["turn_deg"],
                "wall_angle_deg": state["wall_angle_deg"],
                "mach": state["mach"],
                "pressure_pa": state["pressure_pa"],
                "pressure_ratio_to_freestream": state["pressure_ratio_to_freestream"],
                "temperature_k": state["temperature_k"],
                "temperature_ratio_to_freestream": state["temperature_ratio_to_freestream"],
                "density_ratio_to_freestream": state["density_ratio_to_freestream"],
                "total_pressure_pa": state["total_pressure_pa"],
                "total_pressure_ratio_to_freestream": state["total_pressure_ratio_to_freestream"],
                "cp": state["cp"],
                "beta_deg": state["beta_deg"],
                "mn1": state.get("mn1"),
                "mn2": state.get("mn2"),
                "mu1_deg": state.get("mu1_deg"),
                "mu2_deg": state.get("mu2_deg"),
                "branch": state.get("branch") or "",
                "note_code": "detached_shock" if state["event_code"] == "detached_shock" else "",
            }
        )

        if pitot_data and int(state["segment"]) == int(pitot_data["segment"]):
            source_region = int(pitot_data["source_region"])

            if pitot_data["normal_shock_present"]:
                p2 = float(pitot_data["p2_pa"])
                t2 = float(pitot_data["t2_k"])
                p02 = float(pitot_data["p02_pa"])
                rows.append(
                    {
                        "region": f"P{source_region}a",
                        "path_code": "pitot_branch",
                        "event_code": "pitot_normal_shock",
                        "turn_deg": math.nan,
                        "wall_angle_deg": pitot_data["wall_angle_deg"],
                        "mach": pitot_data["mach2"],
                        "pressure_pa": p2,
                        "pressure_ratio_to_freestream": p2 / pressure_inf_pa,
                        "temperature_k": t2,
                        "temperature_ratio_to_freestream": t2 / temperature_inf_k,
                        "density_ratio_to_freestream": pitot_data["rho2_ratio_to_freestream"],
                        "total_pressure_pa": p02,
                        "total_pressure_ratio_to_freestream": p02 / p0_inf,
                        "cp": (p2 - pressure_inf_pa) / q_inf,
                        "beta_deg": 90.0,
                        "mn1": pitot_data["mach1"],
                        "mn2": pitot_data["mach2"],
                        "mu1_deg": math.degrees(math.asin(1.0 / pitot_data["mach1"])) if pitot_data["mach1"] >= 1.0 else math.nan,
                        "mu2_deg": math.nan,
                        "branch": "normal",
                        "note_code": "pitot_local_branch",
                    }
                )
                stagnation_region = f"P{source_region}b"
            else:
                stagnation_region = f"P{source_region}"

            p02 = float(pitot_data["p02_pa"])
            t0 = float(pitot_data["t0_k"])
            rows.append(
                {
                    "region": stagnation_region,
                    "path_code": "pitot_branch",
                    "event_code": "pitot_stagnation",
                    "turn_deg": math.nan,
                    "wall_angle_deg": pitot_data["wall_angle_deg"],
                    "mach": 0.0,
                    "pressure_pa": p02,
                    "pressure_ratio_to_freestream": p02 / pressure_inf_pa,
                    "temperature_k": t0,
                    "temperature_ratio_to_freestream": t0 / temperature_inf_k,
                    "density_ratio_to_freestream": pitot_data["rho0_ratio_to_freestream"],
                    "total_pressure_pa": p02,
                    "total_pressure_ratio_to_freestream": p02 / p0_inf,
                    "cp": (p02 - pressure_inf_pa) / q_inf,
                    "beta_deg": math.nan,
                    "mn1": math.nan,
                    "mn2": math.nan,
                    "mu1_deg": math.nan,
                    "mu2_deg": math.nan,
                    "branch": "",
                    "note_code": "pitot_measured_pressure",
                }
            )

    return pd.DataFrame(rows)
