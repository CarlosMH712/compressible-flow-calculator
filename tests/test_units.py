import math

from core.units import (
    pressure_from_pa,
    pressure_to_pa,
    temperature_from_k,
    temperature_to_k,
)


def test_pressure_round_trip_atmospheres():
    value = 2.75
    assert math.isclose(pressure_from_pa(pressure_to_pa(value, "atm"), "atm"), value)


def test_temperature_round_trip_fahrenheit():
    value = 59.0
    assert math.isclose(temperature_from_k(temperature_to_k(value, "°F"), "°F"), value, abs_tol=1e-12)
