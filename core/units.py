"""Unit conversions used by the Streamlit interface.

All numerical solvers operate internally in SI units.  The user interface may
present SI/practical metric units or U.S. customary units without changing the
underlying gas-dynamics equations.
"""

from __future__ import annotations

from dataclasses import dataclass


PRESSURE_TO_PA = {
    "Pa": 1.0,
    "kPa": 1.0e3,
    "MPa": 1.0e6,
    "bar": 1.0e5,
    "atm": 101325.0,
    "psi": 6894.757293168,
    "psf": 47.88025898033584,
}

ALTITUDE_TO_M = {
    "m": 1.0,
    "km": 1000.0,
    "ft": 0.3048,
    "kft": 304.8,
}

DENSITY_FROM_KG_M3 = {
    "kg/m³": 1.0,
    "slug/ft³": 0.001940320331979716,
    "lbm/ft³": 0.062427960576145,
}

SPEED_FROM_M_S = {
    "m/s": 1.0,
    "ft/s": 3.280839895013123,
}

VISCOSITY_FROM_PA_S = {
    "Pa·s": 1.0,
    "lbf·s/ft²": 0.020885434273039,
}

KINEMATIC_VISCOSITY_FROM_M2_S = {
    "m²/s": 1.0,
    "ft²/s": 10.7639104167097,
}


@dataclass(frozen=True)
class UnitSelection:
    system: str
    pressure: str
    temperature: str
    altitude: str
    density: str
    speed: str
    dynamic_viscosity: str
    kinematic_viscosity: str


def defaults_for_system(system: str) -> UnitSelection:
    key = system.strip().lower()
    if key in {"si", "metric", "métrico", "metrico"}:
        return UnitSelection(
            system="SI",
            pressure="kPa",
            temperature="K",
            altitude="m",
            density="kg/m³",
            speed="m/s",
            dynamic_viscosity="Pa·s",
            kinematic_viscosity="m²/s",
        )
    if key in {"us", "english", "u.s. customary", "ingles", "inglés"}:
        return UnitSelection(
            system="US customary",
            pressure="psi",
            temperature="°R",
            altitude="ft",
            density="slug/ft³",
            speed="ft/s",
            dynamic_viscosity="lbf·s/ft²",
            kinematic_viscosity="ft²/s",
        )
    raise ValueError("Unknown unit system.")


def pressure_to_pa(value: float, unit: str) -> float:
    try:
        return float(value) * PRESSURE_TO_PA[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported pressure unit: {unit}") from exc


def pressure_from_pa(value_pa: float, unit: str) -> float:
    try:
        return float(value_pa) / PRESSURE_TO_PA[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported pressure unit: {unit}") from exc


def temperature_to_k(value: float, unit: str) -> float:
    value = float(value)
    if unit == "K":
        result = value
    elif unit == "°C":
        result = value + 273.15
    elif unit == "°R":
        result = value * 5.0 / 9.0
    elif unit == "°F":
        result = (value + 459.67) * 5.0 / 9.0
    else:
        raise ValueError(f"Unsupported temperature unit: {unit}")
    if result <= 0.0:
        raise ValueError("Absolute temperature must be greater than zero.")
    return result


def temperature_from_k(value_k: float, unit: str) -> float:
    value_k = float(value_k)
    if unit == "K":
        return value_k
    if unit == "°C":
        return value_k - 273.15
    if unit == "°R":
        return value_k * 9.0 / 5.0
    if unit == "°F":
        return value_k * 9.0 / 5.0 - 459.67
    raise ValueError(f"Unsupported temperature unit: {unit}")


def altitude_to_m(value: float, unit: str) -> float:
    try:
        return float(value) * ALTITUDE_TO_M[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported altitude unit: {unit}") from exc


def altitude_from_m(value_m: float, unit: str) -> float:
    try:
        return float(value_m) / ALTITUDE_TO_M[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported altitude unit: {unit}") from exc


def density_from_kg_m3(value: float, unit: str) -> float:
    try:
        return float(value) * DENSITY_FROM_KG_M3[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported density unit: {unit}") from exc


def speed_from_m_s(value: float, unit: str) -> float:
    try:
        return float(value) * SPEED_FROM_M_S[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported speed unit: {unit}") from exc


def dynamic_viscosity_from_pa_s(value: float, unit: str) -> float:
    try:
        return float(value) * VISCOSITY_FROM_PA_S[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported dynamic-viscosity unit: {unit}") from exc


def kinematic_viscosity_from_m2_s(value: float, unit: str) -> float:
    try:
        return float(value) * KINEMATIC_VISCOSITY_FROM_M2_S[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported kinematic-viscosity unit: {unit}") from exc
