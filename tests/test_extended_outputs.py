import math

from core.isentropic import all_relations
from core.oblique_shock import all_relations as oblique
from core.prandtl_meyer import expansion


def test_isentropic_area_input_precision_reference():
    # The solver must preserve the full entered ratio, not a UI-rounded surrogate.
    result = all_relations(0.428613, 1.4)
    assert result.area_over_astar > 1.5
    assert math.isclose(result.t0_over_t * result.t_over_t0, 1.0, rel_tol=1e-12)


def test_oblique_contains_both_normal_mach_numbers():
    result = oblique(2.5, 15.0, 1.4, "weak")
    assert result.mn1 > 1.0
    assert result.mn2 < 1.0
    assert math.isclose(result.mach2, result.mn2 / math.sin(math.radians(result.beta_deg - 15.0)), rel_tol=1e-12)


def test_expansion_contains_mach_angles():
    result = expansion(2.0, 15.0, 1.4)
    assert math.isclose(result.mu1_deg, 30.0, abs_tol=1e-12)
    assert result.mu2_deg < result.mu1_deg
