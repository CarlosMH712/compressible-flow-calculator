import pytest

from core.oblique_shock import all_relations


def test_oblique_shock_m2_theta10():
    r = all_relations(2.0, 10.0, 1.4, "weak")
    assert r.beta_deg == pytest.approx(39.3139, rel=1e-4)
    assert r.mach2 == pytest.approx(1.6405, rel=2e-4)
    assert r.p2_over_p1 == pytest.approx(1.7066, rel=2e-4)
