import pytest

from core.normal_shock import all_relations


def test_normal_shock_mach_2():
    r = all_relations(2.0, 1.4)
    assert r.mach2 == pytest.approx(0.577350269, rel=1e-8)
    assert r.p2_over_p1 == pytest.approx(4.5, rel=1e-12)
    assert r.rho2_over_rho1 == pytest.approx(2.666666667, rel=1e-8)
    assert r.velocity2_over_velocity1 == pytest.approx(0.375, rel=1e-12)
    assert r.t2_over_t1 == pytest.approx(1.6875, rel=1e-12)
    assert r.p02_over_p01 == pytest.approx(0.720873861, rel=1e-8)
    assert r.entropy_change_over_r > 0.0
