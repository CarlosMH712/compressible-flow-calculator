import math

import pytest

from core.atmosphere import (
    geometric_to_geopotential,
    geopotential_to_geometric,
    standard_atmosphere,
)


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


def test_layer_boundaries_match_the_published_1976_tables():
    """Bases de capa tabuladas, en altitud *geopotencial*.

    Cada valor es el que aparece en NOAA-S/T-76-1562 para H = 11, 20, 32 y 47 km.
    Se entra por la altitud geométrica equivalente porque es lo que recibe la
    función: confundir ambas es el error clásico y aquí quedaría a la vista.
    """
    published = {
        11_000.0: (216.650, 22_632.06, 0.363918),
        20_000.0: (216.650, 5_474.89, 0.088035),
        32_000.0: (228.650, 868.02, 0.0132250),
        47_000.0: (270.650, 110.91, 0.00142753),
    }
    for geopotential_m, (t_k, p_pa, rho) in published.items():
        state = standard_atmosphere(geopotential_to_geometric(geopotential_m))
        assert state.temperature_k == pytest.approx(t_k, abs=1e-3)
        assert state.pressure_pa == pytest.approx(p_pa, rel=5e-5)
        assert state.density_kg_m3 == pytest.approx(rho, rel=5e-5)


def test_geopotential_conversion_round_trips():
    for altitude_m in (0.0, 5_000.0, 11_000.0, 40_000.0, 84_852.0):
        assert geopotential_to_geometric(
            geometric_to_geopotential(altitude_m)
        ) == pytest.approx(altitude_m, rel=1e-12, abs=1e-9)


def test_geometric_altitude_always_exceeds_geopotential_above_sea_level():
    """La gravedad decrece con la altura, así que H < h salvo en el suelo."""
    assert geometric_to_geopotential(0.0) == pytest.approx(0.0, abs=1e-12)
    for altitude_m in (1_000.0, 20_000.0, 80_000.0):
        assert geometric_to_geopotential(altitude_m) < altitude_m


def test_pressure_and_density_decrease_monotonically():
    previous_p = math.inf
    previous_rho = math.inf
    for altitude_m in range(0, 84_000, 2_000):
        state = standard_atmosphere(float(altitude_m))
        assert state.pressure_pa < previous_p
        assert state.density_kg_m3 < previous_rho
        previous_p, previous_rho = state.pressure_pa, state.density_kg_m3


def test_altitude_above_the_model_ceiling_is_rejected():
    with pytest.raises(ValueError):
        standard_atmosphere(120_000.0)
