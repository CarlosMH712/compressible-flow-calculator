# Compressible Flow Calculator — Version 3

Academic Streamlit application for Aerodynamics II, developed to accompany the lecture collection at Carlos Molina's academic portal.

## Version 3 improvements

- Global **SI / U.S. customary** unit selector.
- Practical pressure units: Pa, kPa, bar, atm, psi, and psf.
- Temperature display in K, °C, °R, or °F.
- New **U.S. Standard Atmosphere 1976** calculator through 84.852 km geopotential altitude.
- More complete isentropic equation summary and critical-property ratios.
- Six-decimal area-ratio input so values such as `1.507` remain visible and are not displayed as `1.51`.
- Mach angle shown only for supersonic states, with its physical interpretation.
- Normal-shock module now reports absolute downstream pressure, temperature, density, and speed of sound.
- Pitot measurement moved to an independent module to distinguish `p2` from the measured stagnation pressure `p02`.
- Oblique-shock outputs now include both `Mn1` and `Mn2`.
- Anderson-style theta-beta-M chart: theta on the horizontal axis, beta on the vertical axis, and constant-Mach curves.
- Prandtl-Meyer outputs now include Mach angles `mu1` and `mu2` and explain how property ratios are obtained from isentropic relations.
- Geometry results separated into a state table and a wave-details table.
- Pitot placement selected by **flow region** rather than a raw segment number.
- Separate Pitot diagnostic schematic to avoid confusing the local probe shock with the external wave pattern.
- 17 automated tests.

## Quick start on macOS

After extracting the ZIP, double-click:

```text
run_calculator.command
```

The script creates `.venv`, installs missing packages, and starts Streamlit.

If macOS blocks the first launch, right-click the file, choose **Open**, and confirm.

## Manual start

```bash
cd "/path/to/compressible_flow_calculator_v3"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
```

Expected result:

```text
17 passed
```

## Main modules

1. U.S. Standard Atmosphere 1976
2. Isentropic flow
3. Normal shock
4. Compressible Pitot measurement
5. Oblique shock
6. Prandtl-Meyer expansion
7. Sequential shock-expansion geometry

## Technical assumptions

- Calorically perfect gas.
- Default air gas constant: `R = 287.05287 J/(kg K)`.
- Geometry module: two-dimensional, inviscid, piecewise-linear surface.
- Attached oblique-shock model unless the solver reports detachment.
- No boundary-layer interaction or wave-wave interaction.
- A Pitot shock is a local diagnostic branch and does not alter subsequent external-flow regions.

## Standard-atmosphere source

U.S. Standard Atmosphere, 1976, NOAA-S/T-76-1562 and NASA-TM-X-74335.
