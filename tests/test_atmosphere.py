import math

from core.atmosphere import standard_atmosphere


def test_standard_atmosphere_sea_level():
    state = standard_atmosphere(0.0)
    assert math.isclose(state.temperature_k, 288.15, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(state.pressure_pa, 101325.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(state.density_kg_m3, 1.2250000, rel_tol=2e-6)
    assert math.isclose(state.speed_of_sound_m_s, 340.294, rel_tol=2e-5)


def test_standard_atmosphere_11_km_geometric_is_reasonable():
    state = standard_atmosphere(11_000.0)
    # Geometric 11 km corresponds to slightly less than 11 km geopotential.
    assert 216.7 < state.temperature_k < 217.0
    assert 22_690.0 < state.pressure_pa < 22_710.0
    assert 0.364 < state.density_kg_m3 < 0.365
