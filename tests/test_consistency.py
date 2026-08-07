"""Invariantes que ligan unos módulos con otros.

Las demás pruebas comparan cada módulo por separado contra las tablas de
Anderson. Estas comprueban que los módulos son coherentes *entre sí*, que es
donde aparecen los errores que una tabla no detecta: un choque oblicuo que no
degenera en el normal, una expansión que no es reversible, o una relación que
deja de valer al cambiar γ.
"""

from __future__ import annotations

import inspect
import math

import pytest

from core import isentropic, normal_shock, oblique_shock, pitot, prandtl_meyer
from core.isentropic import all_relations as iso
from core.normal_shock import all_relations as shock
from core.oblique_shock import all_relations as oblique
from core.oblique_shock import theta_max
from core.pitot import pitot_pressure_ratio
from core.prandtl_meyer import expansion, nu_deg, nu_max_deg


GAMMAS = [1.2, 1.3, 1.4, 5.0 / 3.0]


def test_strong_oblique_shock_degenerates_into_the_normal_shock():
    """Con θ → 0 la rama fuerte es un choque normal, y debe coincidir con él."""
    strong = oblique(2.0, 1e-6, 1.4, "strong")
    normal = shock(2.0, 1.4)
    assert strong.beta_deg == pytest.approx(90.0, abs=1e-5)
    assert strong.p2_over_p1 == pytest.approx(normal.p2_over_p1, rel=1e-9)
    assert strong.mn2 == pytest.approx(normal.mach2, rel=1e-9)
    assert strong.p02_over_p01 == pytest.approx(normal.p02_over_p01, rel=1e-9)


def test_weak_oblique_shock_degenerates_into_a_mach_wave():
    """Con θ → 0 la rama débil se desvanece: β tiende al ángulo de Mach."""
    weak = oblique(2.0, 1e-6, 1.4, "weak")
    assert weak.beta_deg == pytest.approx(math.degrees(math.asin(0.5)), abs=1e-5)
    assert weak.p2_over_p1 == pytest.approx(1.0, abs=1e-5)
    assert weak.p02_over_p01 == pytest.approx(1.0, abs=1e-9)


def test_theta_max_matches_anderson_detachment_values():
    """Ángulo máximo de desviación antes de que el choque se desprenda."""
    published = {2.0: 22.97, 3.0: 34.07, 4.0: 38.77, 5.0: 41.12}
    for mach, expected in published.items():
        computed, _ = theta_max(mach, 1.4)
        assert computed == pytest.approx(expected, abs=5e-3)

    # El límite M → ∞ es finito: ni siquiera una velocidad infinita
    # deja al choque adherirse a una cuña arbitrariamente abierta.
    limit, _ = theta_max(1e6, 1.4)
    assert limit == pytest.approx(45.58, abs=5e-3)


def test_beyond_theta_max_the_solver_refuses_instead_of_guessing():
    limit, _ = theta_max(2.0, 1.4)
    with pytest.raises(ValueError):
        oblique(2.0, limit + 0.5, 1.4, "weak")


def test_expansion_is_reversible_because_it_is_isentropic():
    """Expandir θ y volver −θ devuelve el Mach de partida."""
    out = expansion(2.0, 15.0, 1.4)
    assert nu_deg(out.mach2, 1.4) == pytest.approx(nu_deg(2.0, 1.4) + 15.0, rel=1e-10)
    assert out.p02_over_p01 == pytest.approx(1.0, rel=1e-12)

    # p2/p1 reconstruido desde las relaciones isoentrópicas de cada estado.
    expected = iso(2.0, 1.4).p0_over_p / iso(out.mach2, 1.4).p0_over_p
    assert out.p2_over_p1 == pytest.approx(expected, rel=1e-10)


def test_nu_max_is_the_closed_form_limit():
    """νmax = 90°(√((γ+1)/(γ−1)) − 1): el giro que llevaría a M → ∞."""
    for gamma in GAMMAS:
        closed_form = 90.0 * (math.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
        assert nu_max_deg(gamma) == pytest.approx(closed_form, rel=1e-12)

    # Para un gas monoatómico el límite cae justo en 90°.
    assert nu_max_deg(5.0 / 3.0) == pytest.approx(90.0, rel=1e-12)


def test_subsonic_pitot_measures_the_isentropic_stagnation_pressure():
    """Sin choque, la sonda lee p01 y nada más."""
    result = pitot_pressure_ratio(0.6, 1.4)
    assert not result.normal_shock_present
    assert result.p_measured_over_p1 == pytest.approx(iso(0.6, 1.4).p0_over_p, rel=1e-12)
    assert result.p02_over_p01 == pytest.approx(1.0, rel=1e-12)


def test_supersonic_pitot_reading_is_the_shock_followed_by_stagnation():
    """p_medida = p01 · (p02/p01): la sonda no mide p2, mide p02."""
    mach = 3.0
    probe = pitot_pressure_ratio(mach, 1.4)
    normal = shock(mach, 1.4)
    expected = iso(mach, 1.4).p0_over_p * normal.p02_over_p01
    assert probe.p_measured_over_p1 == pytest.approx(expected, rel=1e-10)
    assert probe.p_measured_over_p1 > probe.p2_over_p1


@pytest.mark.parametrize("gamma", GAMMAS)
def test_shock_relations_hold_for_any_gamma(gamma: float):
    """γ es un parámetro de la interfaz, así que las relaciones deben aguantarlo."""
    mach = 2.5
    result = shock(mach, gamma)

    assert result.mach2 < 1.0 < mach
    assert result.p2_over_p1 == pytest.approx(
        1.0 + 2.0 * gamma / (gamma + 1.0) * (mach**2 - 1.0), rel=1e-12
    )
    # T2/T1 debe ser consistente con la ecuación de estado.
    assert result.t2_over_t1 == pytest.approx(
        result.p2_over_p1 / result.rho2_over_rho1, rel=1e-12
    )
    # La entalpía de estancamiento se conserva: T02 = T01.
    assert iso(result.mach2, gamma).t0_over_t * result.t2_over_t1 == pytest.approx(
        iso(mach, gamma).t0_over_t, rel=1e-12
    )
    # Segunda ley: un choque comprime y disipa.
    assert result.entropy_change_over_r > 0.0
    assert result.p02_over_p01 < 1.0


@pytest.mark.parametrize("gamma", GAMMAS)
def test_area_ratio_inversion_round_trips_for_any_gamma(gamma: float):
    for mach in (0.3, 0.8, 1.6, 4.0):
        ratio = isentropic.area_ratio(mach, gamma)
        branch = "subsonic" if mach < 1.0 else "supersonic"
        assert isentropic.mach_from_area_ratio(ratio, branch, gamma) == pytest.approx(
            mach, rel=1e-7
        )


def test_core_modules_stay_dimensionless():
    """Contrato: el núcleo devuelve cocientes, nunca magnitudes con unidades.

    Por eso R vive en la interfaz —donde el usuario la ajusta para pasar de aire
    a helio— y no en la física. Si alguien añade aquí una constante del gas, esta
    prueba avisa de que la R de la barra lateral se quedó sin efecto en ese módulo.
    """
    for module in (isentropic, normal_shock, oblique_shock, pitot, prandtl_meyer):
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_") or function.__module__ != module.__name__:
                continue
            parameters = set(inspect.signature(function).parameters)
            assert not parameters & {"r_gas", "gas_constant", "r_air"}, (
                f"{module.__name__}.{name} recibe una constante del gas; "
                "revisa el cableado de R en app.py"
            )
