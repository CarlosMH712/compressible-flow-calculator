# Compressible Flow Calculator

Academic Streamlit application for **Aerodynamics II**, developed to accompany
the lecture collection at Carlos Molina's academic portal. It covers the
standard atmosphere, isentropic flow, shocks, Prandtl–Meyer expansions, Pitot
measurement, and sequential shock–expansion geometry.

This is the repository whose structure, unit system, and visual language the
rest of the portal's calculators follow:
[nozzle-calculator](https://github.com/CarlosMH712/nozzle-calculator),
[turbofan-calculator](https://github.com/CarlosMH712/turbofan-calculator),
[turbojet-calculator](https://github.com/CarlosMH712/turbojet-calculator) and
[ramjet-calculator](https://github.com/CarlosMH712/ramjet-calculator).

## Main modules

1. U.S. Standard Atmosphere 1976
2. Isentropic flow
3. Normal shock
4. Compressible Pitot measurement
5. Oblique shock
6. Prandtl–Meyer expansion
7. Sequential shock–expansion geometry

## Two things that are easy to get wrong

**A Pitot probe does not measure `p₂`.** In supersonic flow it reads `p₀₂`, after
the local normal shock and the subsequent isentropic deceleration. The Pitot
calculation lives in its own module for exactly this reason, and in the geometry
module the probe's shock is a local diagnostic branch that does not alter the
external state propagated to later panels.

**The Mach angle is a supersonic quantity.** It is shown only for `M ≥ 1`,
because it describes the inclination of an infinitesimal Mach wave. A subsonic
disturbance is not confined to a Mach cone, so a number there would be
meaningless rather than merely unused.

## Adjustable gas constant

`R` is a sidebar parameter, default 287.05287 J/(kg·K), so the same relations can
be run on helium, argon, or combustion products — remembering to change γ too.

It reaches only what actually depends on it: density and speed of sound. Every
core module returns **ratios**, so β, `Mₙ₁`, `p₂/p₁`, `ν` and the rest are
untouched by `R` — and a test enforces that, so that adding a gas constant to a
core module cannot silently bypass the sidebar.

The atmosphere tab is the deliberate exception: it always uses 287.05287,
because that value is part of the definition of the 1976 standard rather than a
property of the air being modelled. Changing it there would contradict the model
rather than generalise it. The app says so on the tab.

## Quick start on macOS

Double-click:

```text
run_calculator.command
```

The script creates `.venv`, installs missing packages, and starts Streamlit.

If macOS blocks the first launch, right-click the file, choose **Open**, and
confirm.

## Manual start

```bash
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
39 passed
```

Each module is checked against published tables — Anderson's appendices for the
isentropic, normal-shock and Prandtl–Meyer relations, and NOAA-S/T-76-1562 for
the four atmosphere layer boundaries. `tests/test_consistency.py` then checks
the modules against **each other**, which is where the errors a table cannot
catch appear:

- With θ → 0 the strong oblique-shock branch must reproduce the normal shock,
  and the weak branch must fade into a Mach wave.
- `θmax` must match Anderson's detachment values, including the finite `M → ∞`
  limit of 45.58°.
- An expansion must be reversible, and `νmax` must equal its closed form —
  which lands exactly on 90° for a monatomic gas.
- The shock relations must survive γ ≠ 1.4, since γ is a user-facing parameter.

## Technical assumptions

- Calorically perfect gas.
- Geometry module: two-dimensional, inviscid, piecewise-linear surface.
- Attached oblique-shock model unless the solver reports detachment.
- No boundary-layer interaction or wave–wave interaction.

## Credits

Developed at the Universidad Autónoma de Chihuahua (UACH) by Carlos Alberto
Molina Holguín.

## Sources

- J. D. Anderson, *Modern Compressible Flow*, appendices A, B, and C.
- U.S. Standard Atmosphere, 1976, NOAA-S/T-76-1562 and NASA-TM-X-74335.

See `DEPLOYMENT.md` for GitHub and Streamlit Community Cloud instructions.
