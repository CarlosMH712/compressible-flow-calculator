import pytest

from geometry.solver import build_region_table, solve_surface


def test_region_table_includes_pitot_normal_shock_and_stagnation_rows():
    segments = [
        {"length": 1.0, "angle_deg": 5.0},
        {"length": 1.0, "angle_deg": -5.0},
    ]
    surface_df, waves, vertices, pitot = solve_surface(
        segments,
        mach_inf=3.0,
        pressure_inf_pa=101325.0,
        temperature_inf_k=288.15,
        gamma=1.4,
        side="lower",
        branch="weak",
        pitot_segment=2,
    )
    table = build_region_table(
        surface_df,
        mach_inf=3.0,
        pressure_inf_pa=101325.0,
        temperature_inf_k=288.15,
        gamma=1.4,
        pitot_data=pitot,
    )

    assert table.iloc[0]["event_code"] == "freestream"
    assert "pitot_normal_shock" in set(table["event_code"])
    assert "pitot_stagnation" in set(table["event_code"])

    shock_row = table.loc[table["event_code"] == "pitot_normal_shock"].iloc[0]
    stagnation_row = table.loc[table["event_code"] == "pitot_stagnation"].iloc[0]

    assert shock_row["path_code"] == "pitot_branch"
    assert shock_row["mach"] < 1.0
    assert shock_row["beta_deg"] == pytest.approx(90.0)
    assert stagnation_row["mach"] == pytest.approx(0.0)
    assert stagnation_row["pressure_pa"] > shock_row["pressure_pa"]
    assert stagnation_row["total_pressure_pa"] == pytest.approx(
        shock_row["total_pressure_pa"]
    )


def test_lecture_validation_cases():
    # Oblique shock lecture case.
    surface_df, _, _, _ = solve_surface(
        [{"length": 1.0, "angle_deg": 15.0}],
        mach_inf=2.5,
        pressure_inf_pa=80000.0,
        temperature_inf_k=250.0,
        gamma=1.4,
        side="lower",
        branch="weak",
    )
    state = surface_df.iloc[0]
    assert state["mach"] == pytest.approx(1.8735, rel=5e-4)
    assert state["pressure_ratio_to_freestream"] == pytest.approx(2.4675, rel=5e-4)

    # Diamond half-surface: leading shock followed by a 10-degree expansion.
    diamond_df, _, _, _ = solve_surface(
        [
            {"length": 1.0, "angle_deg": 5.0},
            {"length": 1.0, "angle_deg": -5.0},
        ],
        mach_inf=3.0,
        pressure_inf_pa=101325.0,
        temperature_inf_k=288.15,
        gamma=1.4,
        side="lower",
        branch="weak",
    )
    assert diamond_df.iloc[0]["event_code"] == "oblique_shock"
    assert diamond_df.iloc[1]["event_code"] == "prandtl_meyer_expansion"
    assert diamond_df.iloc[0]["pressure_ratio_to_freestream"] == pytest.approx(1.454, rel=3e-3)
    assert diamond_df.iloc[1]["pressure_ratio_to_freestream"] == pytest.approx(0.668, rel=4e-3)
