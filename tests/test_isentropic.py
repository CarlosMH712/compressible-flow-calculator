import pytest

from core.isentropic import all_relations, mach_from_area_ratio


def test_isentropic_mach_2():
    r = all_relations(2.0, 1.4)
    assert r.t0_over_t == pytest.approx(1.8, rel=1e-12)
    assert r.p0_over_p == pytest.approx(7.824449, rel=1e-6)
    assert r.area_over_astar == pytest.approx(1.6875, rel=1e-12)


def test_area_inversion():
    assert mach_from_area_ratio(1.6875, "supersonic", 1.4) == pytest.approx(
        2.0, rel=1e-9
    )
    assert mach_from_area_ratio(1.6875, "subsonic", 1.4) == pytest.approx(
        0.372244, rel=1e-5
    )
