import pytest

from core.pitot import pitot_pressure_ratio


def test_supersonic_pitot_mach_2():
    r = pitot_pressure_ratio(2.0, 1.4)
    assert r.mach2 == pytest.approx(0.577350269, rel=1e-8)
    assert r.p2_over_p1 == pytest.approx(4.5, rel=1e-12)
    assert r.p02_over_p01 == pytest.approx(0.720873861, rel=1e-8)
    assert r.p_measured_over_p1 == pytest.approx(5.640440813, rel=1e-8)
    assert r.p02_over_p2 == pytest.approx(1.253431292, rel=1e-8)
