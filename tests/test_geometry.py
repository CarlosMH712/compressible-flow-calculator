from geometry.solver import solve_surface


def test_lower_surface_compression_then_expansion():
    segments = [
        {"length": 1.0, "angle_deg": 0.0},
        {"length": 1.0, "angle_deg": 8.0},
        {"length": 1.0, "angle_deg": -2.0},
    ]
    df, waves, vertices, pitot = solve_surface(
        segments,
        mach_inf=2.5,
        pressure_inf_pa=101325.0,
        temperature_inf_k=288.15,
        gamma=1.4,
        side="lower",
        pitot_segment=2,
    )
    assert df.iloc[1]["event"] == "choque oblicuo"
    assert df.iloc[2]["event"] == "expansión Prandtl-Meyer"
    assert len(vertices) == 4
    assert pitot is not None
    assert pitot["mach1"] > 1.0
    assert pitot["mach2"] < 1.0
    assert pitot["p2_over_p1"] > 1.0
    assert pitot["p_measured_pa"] > pitot["p2_pa"]
