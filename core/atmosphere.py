"""U.S. Standard Atmosphere 1976, lower atmosphere through 84.852 km.

The implementation uses geopotential altitude and the piecewise-linear
standard temperature profile.  Geometric altitude is converted internally to
geopotential altitude before the hydrostatic equations are evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


G0 = 9.80665
R_AIR = 287.05287
GAMMA_AIR = 1.4
EARTH_RADIUS_M = 6_356_766.0
SEA_LEVEL_T_K = 288.15
SEA_LEVEL_P_PA = 101_325.0
SEA_LEVEL_RHO_KG_M3 = 1.225000018

# Base geopotential altitudes [m] and lapse rates [K/m].
_LAYER_BASES_M = [0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]
_LAPSE_RATES = [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020]


@dataclass(frozen=True)
class AtmosphereResult:
    geometric_altitude_m: float
    geopotential_altitude_m: float
    layer_index: int
    lapse_rate_k_per_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float
    dynamic_viscosity_pa_s: float
    kinematic_viscosity_m2_s: float
    gravity_m_s2: float
    temperature_ratio: float
    pressure_ratio: float
    density_ratio: float


def geometric_to_geopotential(geometric_altitude_m: float) -> float:
    h = float(geometric_altitude_m)
    if h <= -EARTH_RADIUS_M:
        raise ValueError("Geometric altitude is outside the model domain.")
    return EARTH_RADIUS_M * h / (EARTH_RADIUS_M + h)


def geopotential_to_geometric(geopotential_altitude_m: float) -> float:
    h = float(geopotential_altitude_m)
    if h >= EARTH_RADIUS_M:
        raise ValueError("Geopotential altitude is outside the model domain.")
    return EARTH_RADIUS_M * h / (EARTH_RADIUS_M - h)


def _base_states() -> tuple[list[float], list[float]]:
    temperatures = [SEA_LEVEL_T_K]
    pressures = [SEA_LEVEL_P_PA]
    for index, lapse in enumerate(_LAPSE_RATES):
        h_b = _LAYER_BASES_M[index]
        h_top = _LAYER_BASES_M[index + 1]
        t_b = temperatures[-1]
        p_b = pressures[-1]
        delta_h = h_top - h_b
        if abs(lapse) < 1e-15:
            t_top = t_b
            p_top = p_b * math.exp(-G0 * delta_h / (R_AIR * t_b))
        else:
            t_top = t_b + lapse * delta_h
            p_top = p_b * (t_b / t_top) ** (G0 / (R_AIR * lapse))
        temperatures.append(t_top)
        pressures.append(p_top)
    return temperatures, pressures


_BASE_TEMPERATURES_K, _BASE_PRESSURES_PA = _base_states()


def standard_atmosphere(
    geometric_altitude_m: float,
    *,
    gamma: float = GAMMA_AIR,
) -> AtmosphereResult:
    """Evaluate the 1976 standard atmosphere from -610 m to 84.852 km geopotential."""
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1.")

    h_geom = float(geometric_altitude_m)
    h_geo = geometric_to_geopotential(h_geom)
    if h_geo < -610.0:
        raise ValueError("The implemented atmosphere is limited to altitudes above about -610 m.")
    if h_geo > _LAYER_BASES_M[-1] + 1e-9:
        raise ValueError("The implemented atmosphere is limited to 84.852 km geopotential altitude.")

    if h_geo < 0.0:
        layer_index = 0
    else:
        layer_index = len(_LAPSE_RATES) - 1
        for idx in range(len(_LAPSE_RATES)):
            if _LAYER_BASES_M[idx] <= h_geo <= _LAYER_BASES_M[idx + 1]:
                layer_index = idx
                break

    h_b = _LAYER_BASES_M[layer_index]
    t_b = _BASE_TEMPERATURES_K[layer_index]
    p_b = _BASE_PRESSURES_PA[layer_index]
    lapse = _LAPSE_RATES[layer_index]
    delta_h = h_geo - h_b

    if abs(lapse) < 1e-15:
        temperature = t_b
        pressure = p_b * math.exp(-G0 * delta_h / (R_AIR * t_b))
    else:
        temperature = t_b + lapse * delta_h
        pressure = p_b * (t_b / temperature) ** (G0 / (R_AIR * lapse))

    density = pressure / (R_AIR * temperature)
    speed_of_sound = math.sqrt(gamma * R_AIR * temperature)
    dynamic_viscosity = 1.458e-6 * temperature**1.5 / (temperature + 110.4)
    kinematic_viscosity = dynamic_viscosity / density
    gravity = G0 * (EARTH_RADIUS_M / (EARTH_RADIUS_M + h_geom)) ** 2

    return AtmosphereResult(
        geometric_altitude_m=h_geom,
        geopotential_altitude_m=h_geo,
        layer_index=layer_index,
        lapse_rate_k_per_m=lapse,
        temperature_k=temperature,
        pressure_pa=pressure,
        density_kg_m3=density,
        speed_of_sound_m_s=speed_of_sound,
        dynamic_viscosity_pa_s=dynamic_viscosity,
        kinematic_viscosity_m2_s=kinematic_viscosity,
        gravity_m_s2=gravity,
        temperature_ratio=temperature / SEA_LEVEL_T_K,
        pressure_ratio=pressure / SEA_LEVEL_P_PA,
        density_ratio=density / SEA_LEVEL_RHO_KG_M3,
    )
