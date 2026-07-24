import pytest

from core.prandtl_meyer import expansion, nu_deg, mach_from_nu


def test_nu_inversion():
    nu = nu_deg(2.0, 1.4)
    assert mach_from_nu(nu, 1.4) == pytest.approx(2.0, rel=1e-9)


def test_expansion_from_mach_2_by_10_deg():
    r = expansion(2.0, 10.0, 1.4)
    assert r.mach2 == pytest.approx(2.3849, rel=3e-4)
    assert r.p2_over_p1 < 1.0
    assert r.t2_over_t1 < 1.0
    assert r.p02_over_p01 == 1.0
