from __future__ import annotations

import math
from typing import Iterable

import pandas as pd
import streamlit as st

from core.atmosphere import R_AIR, standard_atmosphere
from core.isentropic import all_relations as isentropic_relations
from core.isentropic import critical_ratios, mach_from_area_ratio
from core.normal_shock import all_relations as normal_shock_relations
from core.oblique_shock import all_relations as oblique_shock_relations
from core.oblique_shock import theta_max
from core.pitot import pitot_pressure_ratio
from core.prandtl_meyer import expansion, mach_from_nu, nu_max_deg
from core.units import (
    altitude_from_m,
    altitude_to_m,
    density_from_kg_m3,
    dynamic_viscosity_from_pa_s,
    kinematic_viscosity_from_m2_s,
    pressure_from_pa,
    pressure_to_pa,
    speed_from_m_s,
    temperature_from_k,
    temperature_to_k,
)
from geometry.solver import build_region_table, solve_surface
from visualization.plots import geometry_figure, pitot_schematic_figure, theta_beta_m_figure

SITE_URL = "https://carlosmh712.github.io/"
ATMOSPHERE_SOURCE = "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770009539.pdf"
CREDITS = "Carlos Alberto Molina Holguín · UACH"
LECTURES = {
    "isentropic": f"{SITE_URL}aerodynamics-ii/isentropic-flow.html",
    "normal": f"{SITE_URL}aerodynamics-ii/normal-shocks.html",
    "oblique": f"{SITE_URL}aerodynamics-ii/oblique-shocks.html",
    "pm": f"{SITE_URL}aerodynamics-ii/prandtl-meyer.html",
    "geometry": f"{SITE_URL}aerodynamics-ii/shock-expansion.html",
}

st.set_page_config(
    page_title="Compressible Flow Calculator | Carlos Molina",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEXT = {
    "en": {
        "hero_eyebrow": "AERODYNAMICS II · COMPUTATIONAL TOOL",
        "hero_title": "Compressible Flow Calculator",
        "byline": "by M.C. Carlos Molina",
        "hero_copy": (
            "Standard atmosphere, isentropic flow, shocks, Prandtl–Meyer expansions, "
            "Pitot measurements, and sequential shock–expansion geometry."
        ),
        "global": "Global configuration",
        "gamma": "Specific-heat ratio γ",
        "r_gas": "Specific gas constant R [J/(kg·K)]",
        "r_gas_help": (
            "Air is 287.05287. Change it to run the same relations on another gas — "
            "helium, argon, or combustion products — remembering to change γ as well."
        ),
        "credits": "Credits",
        "decimals": "Displayed decimals",
        "units": "Unit system",
        "pressure_unit": "Pressure unit",
        "temperature_unit": "Temperature unit",
        "altitude_unit": "Altitude unit",
        "assumptions": (
            "Calorically perfect air. The geometry module is two-dimensional, inviscid, "
            "and neglects wave interactions and boundary-layer effects."
        ),
        "resources": "Academic resources",
        "website": "Academic website",
        "course": "Aerodynamics II lectures",
        "theory": "Read the lecture",
        "download": "Download results CSV",
        "main_flow": "Main flow",
        "pitot_branch": "Pitot diagnostic",
        "events": {
            "freestream": "Freestream",
            "no_turn": "No new wave",
            "oblique_shock": "Oblique shock",
            "prandtl_meyer_expansion": "Prandtl–Meyer expansion",
            "detached_shock": "Detached shock · calculation stopped",
            "pitot_normal_shock": "↳ Pitot normal shock",
            "pitot_stagnation": "↳ Pitot stagnation / measured pressure",
        },
        "tabs": [
            "01 · Atmosphere",
            "02 · Isentropic",
            "03 · Normal shock",
            "04 · Pitot",
            "05 · Oblique shock",
            "06 · Prandtl–Meyer",
            "07 · Geometry",
        ],
    },
    "es": {
        "hero_eyebrow": "AERODINÁMICA II · HERRAMIENTA COMPUTACIONAL",
        "hero_title": "Calculadora de Flujo Compresible",
        "byline": "por M.C. Carlos Molina",
        "hero_copy": (
            "Atmósfera estándar, flujo isoentrópico, ondas de choque, expansiones de "
            "Prandtl–Meyer, medición Pitot y geometría choque–expansión."
        ),
        "global": "Configuración global",
        "gamma": "Relación de calores específicos γ",
        "r_gas": "Constante específica del gas R [J/(kg·K)]",
        "r_gas_help": (
            "El aire vale 287.05287. Cámbiala para aplicar las mismas relaciones a otro "
            "gas —helio, argón o productos de combustión— recordando cambiar también γ."
        ),
        "credits": "Créditos",
        "decimals": "Decimales mostrados",
        "units": "Sistema de unidades",
        "pressure_unit": "Unidad de presión",
        "temperature_unit": "Unidad de temperatura",
        "altitude_unit": "Unidad de altitud",
        "assumptions": (
            "Aire calóricamente perfecto. El módulo geométrico es bidimensional, inviscido "
            "y no incluye interacciones entre ondas ni efectos de capa límite."
        ),
        "resources": "Recursos académicos",
        "website": "Sitio académico",
        "course": "Lectures de Aerodinámica II",
        "theory": "Leer la lecture",
        "download": "Descargar resultados CSV",
        "main_flow": "Flujo principal",
        "pitot_branch": "Diagnóstico Pitot",
        "events": {
            "freestream": "Corriente libre",
            "no_turn": "Sin nueva onda",
            "oblique_shock": "Choque oblicuo",
            "prandtl_meyer_expansion": "Expansión de Prandtl–Meyer",
            "detached_shock": "Choque desprendido · cálculo detenido",
            "pitot_normal_shock": "↳ Choque normal del Pitot",
            "pitot_stagnation": "↳ Estancamiento Pitot / presión medida",
        },
        "tabs": [
            "01 · Atmósfera",
            "02 · Isoentrópicas",
            "03 · Choque normal",
            "04 · Pitot",
            "05 · Choque oblicuo",
            "06 · Prandtl–Meyer",
            "07 · Geometría",
        ],
    },
}


def inject_css() -> None:
    """Inyecta la hoja de estilos del portal y fuerza el aspecto claro.

    El bloque final anula el tema oscuro. ``.streamlit/config.toml`` fija
    ``base = "light"``, pero eso no basta: el navegador del visitante puede
    pedir esquema oscuro y Streamlit recolorea entonces etiquetas, tablas y
    campos de entrada por su cuenta.

    No poner comentarios CSS dentro del bloque: Streamlit pasa la cadena por
    su procesador de markdown, que se come el ``/*`` de apertura y vuelca el
    resto de la hoja de estilos como texto visible en la página.
    """
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] { font-family: "Inter", sans-serif; }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f5f8fa 0%, #ffffff 34%, #f5f8fa 100%);
        }
        [data-testid="stHeader"] { background: rgba(245,248,250,.90); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eaf4fb 0%, #f5f8fa 100%);
            border-right: 1px solid #d8e2ea;
        }
        .block-container { max-width: 1380px; padding-top: 1.2rem; padding-bottom: 4rem; }
        .portal-hero {
            overflow:hidden; margin:0 0 1.25rem; padding:2.2rem 2.4rem; border-radius:20px;
            color:white; background:radial-gradient(circle at 82% 12%, rgba(73,171,235,.32), transparent 22rem),
            linear-gradient(135deg,#061729,#0a3558 62%,#0c4f83); box-shadow:0 22px 60px rgba(7,29,51,.16);
        }
        .portal-hero h1 { margin:.2rem 0 .65rem; font-family:"Source Serif 4",Georgia,serif;
            font-size:clamp(2.5rem,5.4vw,4.7rem); line-height:1; letter-spacing:-.035em; }
        .portal-hero p { max-width:900px; margin:0; color:#d8e9f4; font-size:1.04rem; }
        .portal-eyebrow { color:#9fd6fa; font-size:.73rem; font-weight:700; letter-spacing:.13em; }
        .hero-byline { margin:-.2rem 0 .8rem; color:#9fd6fa; font-size:.94rem;
            font-weight:600; letter-spacing:.01em; }
        .hero-chips { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1.25rem; }
        .hero-chip { padding:.3rem .62rem; border:1px solid rgba(255,255,255,.22); border-radius:999px;
            color:#eaf4fb; background:rgba(255,255,255,.06); font-size:.74rem; font-weight:600; }
        div[data-testid="stMetric"] { min-height:112px; padding:.9rem 1rem; border:1px solid #d8e2ea;
            border-top:4px solid #0d568d; border-radius:12px; background:#fff; box-shadow:0 10px 28px rgba(18,50,77,.055); }
        div[data-testid="stMetric"] label { color:#607181 !important; }
        div[data-testid="stMetricValue"] { color:#071d33; font-family:"Source Serif 4",Georgia,serif; }
        .stTabs [data-baseweb="tab-list"] { gap:.28rem; padding:.35rem; border:1px solid #d8e2ea;
            border-radius:12px; background:#fff; overflow-x:auto; }
        .stTabs [data-baseweb="tab"] { height:46px; padding:0 .8rem; border-radius:8px; color:#607181; font-weight:650; white-space:nowrap; }
        .stTabs [aria-selected="true"] { background:#071d33 !important; color:white !important; }
        .stTabs [data-baseweb="tab-highlight"] { display:none; }
        .section-banner { margin:.8rem 0 1.2rem; padding:1.1rem 1.25rem; border-left:5px solid #d67b28;
            border-radius:0 10px 10px 0; background:#fff; box-shadow:0 8px 24px rgba(18,50,77,.04); }
        .section-banner h2 { margin:0; color:#071d33; font-family:"Source Serif 4",Georgia,serif; }
        .section-banner p { margin:.25rem 0 0; color:#607181; }
        .info-card { margin:.8rem 0; padding:1rem 1.15rem; border-left:4px solid #0d568d;
            border-radius:0 9px 9px 0; background:#eaf4fb; color:#344454; }
        .warning-card { margin:.8rem 0; padding:1rem 1.15rem; border-left:4px solid #d67b28;
            border-radius:0 9px 9px 0; background:#fff7ed; color:#344454; }
        .formula-note { color:#607181; font-size:.9rem; margin-bottom:.5rem; }
        .credits { color:#344454; font-size:.84rem; line-height:1.55; margin:.2rem 0 0; }
        div.stButton > button, div.stDownloadButton > button, a[data-testid="stLinkButton"] {
            border-radius:8px !important; border:1px solid #d8e2ea !important; font-weight:700 !important; }
        div.stDownloadButton > button { color:white !important; background:#d67b28 !important; border-color:#d67b28 !important; }
        [data-testid="stDataFrame"] { border:1px solid #d8e2ea; border-radius:12px; overflow:hidden; }
        @media (max-width:700px) { .portal-hero { padding:1.55rem 1.3rem; } .portal-hero h1 { font-size:2.55rem; } }

        :root, [data-testid="stApp"] { color-scheme: light !important; }
        [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
        [data-testid="stSidebarContent"], [data-testid="stHeader"] {
            color:#172333 !important; }
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary *,
        [data-testid="stCaptionContainer"], label, .stRadio div, .stSlider label {
            color:#172333 !important; }
        [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
            color:#344454 !important; }
        [data-testid="stDataFrame"], [data-baseweb="select"] > div,
        [data-baseweb="popover"] div, [data-testid="stExpanderDetails"] {
            background-color:#ffffff !important; color:#172333 !important; }
        [data-testid="stAlertContainer"] { color:#344454 !important; }
        [data-testid="stNumberInputField"], [data-baseweb="input"] input {
            background-color:#ffffff !important; color:#172333 !important; }
        .portal-hero h1, .portal-hero .hero-chip { color:#ffffff !important; }
        .portal-hero p { color:#d8e9f4 !important; }
        .hero-byline { color:#9fd6fa !important; }
        .portal-eyebrow { color:#9fd6fa !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_banner(title: str, copy: str) -> None:
    st.markdown(f'<div class="section-banner"><h2>{title}</h2><p>{copy}</p></div>', unsafe_allow_html=True)


def metric_grid(items: Iterable[tuple[str, float | str, str | None]], decimals: int) -> None:
    values = list(items)
    for start in range(0, len(values), 4):
        chunk = values[start:start + 4]
        cols = st.columns(len(chunk))
        for col, (label, value, suffix) in zip(cols, chunk):
            if isinstance(value, float):
                display = f"{value:.{decimals}f}{suffix or ''}"
            else:
                display = f"{value}{suffix or ''}"
            col.metric(label, display)


def pressure_input(label: str, default_pa: float, key: str, pressure_unit: str) -> float:
    default = pressure_from_pa(default_pa, pressure_unit)
    value = st.number_input(label, min_value=1e-12, value=float(default), step=max(abs(default) * 0.01, 1e-6), format="%.8f", key=key)
    return pressure_to_pa(value, pressure_unit)


def temperature_input(label: str, default_k: float, key: str, temperature_unit: str) -> float:
    default = temperature_from_k(default_k, temperature_unit)
    value = st.number_input(label, value=float(default), step=1.0, format="%.6f", key=key)
    return temperature_to_k(value, temperature_unit)


def state_metrics(p_pa: float, t_k: float, gamma: float, r_gas: float, pressure_unit: str, temperature_unit: str, density_unit: str, speed_unit: str) -> list[tuple[str, float | str, str | None]]:
    rho = p_pa / (r_gas * t_k)
    a = math.sqrt(gamma * r_gas * t_k)
    return [
        (f"p [{pressure_unit}]", pressure_from_pa(p_pa, pressure_unit), None),
        (f"T [{temperature_unit}]", temperature_from_k(t_k, temperature_unit), None),
        (f"ρ [{density_unit}]", density_from_kg_m3(rho, density_unit), None),
        (f"a [{speed_unit}]", speed_from_m_s(a, speed_unit), None),
    ]


def localized_region_table(df: pd.DataFrame, language: str, branch: str, r_gas: float, pressure_unit: str, temperature_unit: str, density_unit: str) -> pd.DataFrame:
    text = TEXT[language]
    result = df.copy()
    result["Path"] = result["path_code"].map({"main_flow": text["main_flow"], "pitot_branch": text["pitot_branch"]})
    result["Event"] = result["event_code"].map(text["events"]).fillna(result["event_code"])
    branch_name = "strong" if branch == "strong" else "weak"
    result.loc[result["event_code"] == "oblique_shock", "Event"] += f" · {branch_name}"
    result[f"p [{pressure_unit}]"] = result["pressure_pa"].map(lambda v: pressure_from_pa(v, pressure_unit))
    result[f"T [{temperature_unit}]"] = result["temperature_k"].map(lambda v: temperature_from_k(v, temperature_unit))
    density_si = result["pressure_pa"] / (r_gas * result["temperature_k"])
    result[f"ρ [{density_unit}]"] = density_si.map(lambda v: density_from_kg_m3(v, density_unit))
    result = result.rename(columns={
        "region": "Region", "turn_deg": "Turn [deg]", "mach": "M",
        "pressure_ratio_to_freestream": "p/p∞", "temperature_ratio_to_freestream": "T/T∞",
        "density_ratio_to_freestream": "ρ/ρ∞", "total_pressure_ratio_to_freestream": "p₀/p₀∞",
        "cp": "Cₚ", "beta_deg": "β [deg]", "mn1": "Mₙ₁", "mn2": "Mₙ₂",
        "mu1_deg": "μ₁ [deg]", "mu2_deg": "μ₂ [deg]", "branch": "Branch",
    })
    return result


def display_state_table(df: pd.DataFrame, decimals: int, pressure_unit: str, temperature_unit: str, density_unit: str) -> None:
    columns = ["Region", "Path", "Event", "Turn [deg]", "M", "p/p∞", f"p [{pressure_unit}]", "T/T∞", f"T [{temperature_unit}]", "ρ/ρ∞", f"ρ [{density_unit}]", "p₀/p₀∞", "Cₚ"]
    st.dataframe(df[columns], width="stretch", hide_index=True, column_config={
        col: st.column_config.NumberColumn(format=f"%.{decimals}f")
        for col in columns if col not in {"Region", "Path", "Event"}
    })


def display_wave_table(df: pd.DataFrame, decimals: int) -> None:
    columns = ["Region", "Event", "Turn [deg]", "β [deg]", "Mₙ₁", "Mₙ₂", "μ₁ [deg]", "μ₂ [deg]", "Branch"]
    st.dataframe(df[columns], width="stretch", hide_index=True, column_config={
        col: st.column_config.NumberColumn(format=f"%.{decimals}f")
        for col in columns if col not in {"Region", "Event", "Branch"}
    })


def equations_isentropic() -> None:
    st.markdown('<p class="formula-note">Total–static, critical, area–Mach, and characteristic relations.</p>', unsafe_allow_html=True)
    st.latex(r"\frac{T_0}{T}=1+\frac{\gamma-1}{2}M^2")
    st.latex(r"\frac{p_0}{p}=\left(1+\frac{\gamma-1}{2}M^2\right)^{\gamma/(\gamma-1)}")
    st.latex(r"\frac{\rho_0}{\rho}=\left(1+\frac{\gamma-1}{2}M^2\right)^{1/(\gamma-1)}")
    st.latex(r"\frac{A}{A^*}=\frac{1}{M}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{(\gamma+1)/[2(\gamma-1)]}")
    st.latex(r"M^*=\sqrt{\frac{(\gamma+1)M^2}{2+(\gamma-1)M^2}}")
    st.latex(r"\mu=\sin^{-1}\!\left(\frac{1}{M}\right),\qquad M\ge 1")
    st.latex(r"\frac{T^*}{T_0}=\frac{2}{\gamma+1},\quad \frac{p^*}{p_0}=\left(\frac{2}{\gamma+1}\right)^{\gamma/(\gamma-1)}")


def equations_normal_shock() -> None:
    st.latex(r"M_2^2=\frac{1+\frac{\gamma-1}{2}M_1^2}{\gamma M_1^2-\frac{\gamma-1}{2}}")
    st.latex(r"\frac{p_2}{p_1}=1+\frac{2\gamma}{\gamma+1}(M_1^2-1)")
    st.latex(r"\frac{\rho_2}{\rho_1}=\frac{(\gamma+1)M_1^2}{2+(\gamma-1)M_1^2},\qquad \frac{T_2}{T_1}=\frac{p_2/p_1}{\rho_2/\rho_1}")
    st.latex(r"\frac{p_{0,2}}{p_{0,1}}=\left[\frac{(\gamma+1)M_1^2}{2+(\gamma-1)M_1^2}\right]^{\gamma/(\gamma-1)}\left[\frac{\gamma+1}{2\gamma M_1^2-(\gamma-1)}\right]^{1/(\gamma-1)}")
    st.latex(r"\frac{s_2-s_1}{R}=\frac{\gamma}{\gamma-1}\ln\frac{T_2}{T_1}-\ln\frac{p_2}{p_1}")


def equations_pitot() -> None:
    st.markdown("**Subsonic:**")
    st.latex(r"p_{\mathrm{Pitot}}=p_{0,1},\qquad \frac{p_{0,1}}{p_1}=\left(1+\frac{\gamma-1}{2}M_1^2\right)^{\gamma/(\gamma-1)}")
    st.markdown("**Supersonic (Rayleigh–Pitot):**")
    st.latex(r"p_{\mathrm{Pitot}}=p_{0,2}\ne p_2")
    st.latex(r"\frac{p_{0,2}}{p_1}=\left[\frac{(\gamma+1)^2M_1^2}{4\gamma M_1^2-2(\gamma-1)}\right]^{\gamma/(\gamma-1)}\left[\frac{1-\gamma+2\gamma M_1^2}{\gamma+1}\right]")


def equations_oblique() -> None:
    st.latex(r"M_{n1}=M_1\sin\beta")
    st.latex(r"\tan\theta=2\cot\beta\,\frac{M_1^2\sin^2\beta-1}{M_1^2[\gamma+\cos(2\beta)]+2}")
    st.latex(r"M_{n2}^2=\frac{1+\frac{\gamma-1}{2}M_{n1}^2}{\gamma M_{n1}^2-\frac{\gamma-1}{2}},\qquad M_2=\frac{M_{n2}}{\sin(\beta-\theta)}")
    st.latex(r"\frac{p_2}{p_1}=1+\frac{2\gamma}{\gamma+1}(M_{n1}^2-1)")
    st.markdown(r"The remaining density, temperature, entropy, and total-pressure ratios are the normal-shock relations evaluated at \(M_{n1}\).")


def equations_pm() -> None:
    st.latex(r"\nu(M)=\sqrt{\frac{\gamma+1}{\gamma-1}}\tan^{-1}\sqrt{\frac{\gamma-1}{\gamma+1}(M^2-1)}-\tan^{-1}\sqrt{M^2-1}")
    st.latex(r"\nu(M_2)=\nu(M_1)+\theta")
    st.latex(r"\mu_i=\sin^{-1}\!\left(\frac{1}{M_i}\right)")
    st.latex(r"\frac{T_2}{T_1}=\frac{1+\frac{\gamma-1}{2}M_1^2}{1+\frac{\gamma-1}{2}M_2^2}")
    st.latex(r"\frac{p_2}{p_1}=\left(\frac{T_2}{T_1}\right)^{\gamma/(\gamma-1)},\qquad \frac{\rho_2}{\rho_1}=\left(\frac{T_2}{T_1}\right)^{1/(\gamma-1)}")
    st.latex(r"p_{0,2}=p_{0,1},\qquad T_{0,2}=T_{0,1}")


inject_css()

with st.sidebar:
    language_label = st.selectbox("Language / Idioma", ["English", "Español"], index=0)
    language = "en" if language_label == "English" else "es"
    text = TEXT[language]

    st.markdown(f"### {text['global']}")
    unit_system = st.radio(text["units"], ["SI", "US customary"], horizontal=True)
    if unit_system == "SI":
        pressure_options = ["Pa", "kPa", "bar", "atm"]
        temperature_options = ["K", "°C"]
        altitude_options = ["m", "km"]
        density_unit, speed_unit = "kg/m³", "m/s"
        dynamic_viscosity_unit, kinematic_viscosity_unit = "Pa·s", "m²/s"
    else:
        pressure_options = ["psi", "psf", "atm"]
        temperature_options = ["°R", "°F"]
        altitude_options = ["ft", "kft"]
        density_unit, speed_unit = "slug/ft³", "ft/s"
        dynamic_viscosity_unit, kinematic_viscosity_unit = "lbf·s/ft²", "ft²/s"

    pressure_unit = st.selectbox(text["pressure_unit"], pressure_options, index=1 if unit_system == "SI" else 0)
    temperature_unit = st.selectbox(text["temperature_unit"], temperature_options, index=0)
    altitude_unit = st.selectbox(text["altitude_unit"], altitude_options, index=0)
    gamma = st.number_input(text["gamma"], min_value=1.01, max_value=2.0, value=1.4, step=0.01, format="%.4f")
    r_gas = st.number_input(
        text["r_gas"], min_value=20.0, max_value=4200.0, value=R_AIR,
        step=0.00001, format="%.5f", help=text["r_gas_help"],
    )
    decimals = st.slider(text["decimals"], 3, 7, 5)
    st.info(text["assumptions"])

    st.markdown(f"### {text['resources']}")
    st.link_button(text["website"], SITE_URL, width="stretch")
    st.link_button(text["course"], f"{SITE_URL}aerodynamics-ii/", width="stretch")
    st.markdown(f"### {text['credits']}")
    st.markdown(f'<p class="credits">{CREDITS}</p>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="portal-hero">
      <div class="portal-eyebrow">{text['hero_eyebrow']}</div>
      <h1>{text['hero_title']}</h1>
      <div class="hero-byline">{text['byline']}</div>
      <p>{text['hero_copy']}</p>
      <div class="hero-chips">
        <span class="hero-chip">U.S. Standard Atmosphere 1976</span>
        <span class="hero-chip">{unit_system}</span>
        <span class="hero-chip">γ = {gamma:.4g}</span>
        <span class="hero-chip">R = {r_gas:.5g}</span>
        <span class="hero-chip">Python · Streamlit</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(text["tabs"])

# 01 Atmosphere
with tabs[0]:
    section_banner(
        "U.S. Standard Atmosphere 1976" if language == "en" else "Atmósfera Estándar de EE. UU. 1976",
        "Temperature, pressure, density, speed of sound, and viscosity as functions of geometric altitude."
        if language == "en" else
        "Temperatura, presión, densidad, velocidad del sonido y viscosidad en función de la altitud geométrica.",
    )
    default_alt = altitude_from_m(0.0, altitude_unit)
    altitude_value = st.number_input(f"Geometric altitude [{altitude_unit}] / Altitud geométrica", value=float(default_alt), step=1.0, format="%.6f")
    try:
        atm = standard_atmosphere(altitude_to_m(altitude_value, altitude_unit), gamma=gamma)
        metric_grid([
            (f"h [{altitude_unit}]", altitude_from_m(atm.geometric_altitude_m, altitude_unit), None),
            (f"H geopotential [{altitude_unit}]", altitude_from_m(atm.geopotential_altitude_m, altitude_unit), None),
            (f"T [{temperature_unit}]", temperature_from_k(atm.temperature_k, temperature_unit), None),
            (f"p [{pressure_unit}]", pressure_from_pa(atm.pressure_pa, pressure_unit), None),
            (f"ρ [{density_unit}]", density_from_kg_m3(atm.density_kg_m3, density_unit), None),
            (f"a [{speed_unit}]", speed_from_m_s(atm.speed_of_sound_m_s, speed_unit), None),
            (f"μ [{dynamic_viscosity_unit}]", f"{dynamic_viscosity_from_pa_s(atm.dynamic_viscosity_pa_s, dynamic_viscosity_unit):.4e}", None),
            (f"ν [{kinematic_viscosity_unit}]", f"{kinematic_viscosity_from_m2_s(atm.kinematic_viscosity_m2_s, kinematic_viscosity_unit):.4e}", None),
        ], decimals)
        metric_grid([
            ("T/T₀,SL", atm.temperature_ratio, None),
            ("p/pSL", atm.pressure_ratio, None),
            ("ρ/ρSL", atm.density_ratio, None),
            ("g [m/s²]", atm.gravity_m_s2, None),
        ], decimals)
        with st.expander("Model equations / Ecuaciones del modelo"):
            st.latex(r"H=\frac{r_0 h}{r_0+h}")
            st.latex(r"T=T_b+L_b(H-H_b)")
            st.latex(r"\frac{p}{p_b}=\left(\frac{T_b}{T}\right)^{g_0/(RL_b)}\quad (L_b\ne0)")
            st.latex(r"\frac{p}{p_b}=\exp\left[-\frac{g_0(H-H_b)}{RT_b}\right]\quad (L_b=0)")
            st.latex(r"\rho=\frac{p}{RT},\qquad a=\sqrt{\gamma RT}")
        st.markdown(
            '<div class="info-card">This tab always uses <strong>R = 287.05287 J/(kg·K)</strong>, '
            'whatever the sidebar says. That value is part of the definition of the 1976 standard, '
            'not a property of the air being modelled, so changing it would not generalise the model '
            '— it would contradict it. The sidebar R applies to the flow modules.<br>'
            'Esta pestaña usa siempre <strong>R = 287.05287 J/(kg·K)</strong>, diga lo que diga la barra lateral. '
            'Ese valor forma parte de la definición de la norma de 1976, así que cambiarlo no generalizaría '
            'el modelo: lo contradiría. La R de la barra lateral aplica a los módulos de flujo.</div>',
            unsafe_allow_html=True,
        )
        st.link_button("NASA/NOAA source", ATMOSPHERE_SOURCE)
    except ValueError as exc:
        st.error(str(exc))

# 02 Isentropic
with tabs[1]:
    section_banner(
        "Isentropic flow" if language == "en" else "Flujo isoentrópico",
        "Total–static relations, critical properties, area–Mach inversion, and supersonic characteristic angle."
        if language == "en" else
        "Relaciones total–estática, propiedades críticas, inversión área–Mach y ángulo característico supersónico.",
    )
    mode_options = ["Mach number", "Area ratio A/A*"] if language == "en" else ["Número de Mach", "Relación de área A/A*"]
    mode = st.radio("Input variable / Variable de entrada", mode_options, horizontal=True)
    try:
        if mode == mode_options[0]:
            mach = st.number_input("Mach", min_value=0.000001, value=2.0, step=0.001, format="%.6f", key="iso_mach_v3")
        else:
            c1, c2 = st.columns(2)
            area = c1.number_input("A/A*", min_value=1.0, value=1.507, step=0.001, format="%.6f", key="area_ratio_v3")
            area_branch = c2.selectbox("Branch / Rama", ["subsonic", "supersonic"], index=1)
            mach = mach_from_area_ratio(area, branch=area_branch, gamma=gamma)
            st.success(f"Mach = {mach:.{decimals}f}")

        result = isentropic_relations(mach, gamma)
        metric_grid([
            ("T₀/T", result.t0_over_t, None), ("T/T₀", result.t_over_t0, None),
            ("p₀/p", result.p0_over_p, None), ("p/p₀", result.p_over_p0, None),
            ("ρ₀/ρ", result.rho0_over_rho, None), ("ρ/ρ₀", result.rho_over_rho0, None),
            ("A/A*", result.area_over_astar, None), ("M*", result.mach_star, None),
        ], decimals)
        critical = critical_ratios(gamma)
        st.caption(f"T*/T₀ = {critical['t_star_over_t0']:.{decimals}f} · p*/p₀ = {critical['p_star_over_p0']:.{decimals}f} · ρ*/ρ₀ = {critical['rho_star_over_rho0']:.{decimals}f}")
        if result.mach_angle_deg is not None:
            st.markdown(
                f'<div class="info-card"><strong>Mach angle μ = {result.mach_angle_deg:.{decimals}f}°.</strong> '
                'It is included only for M ≥ 1 because it describes the inclination of an infinitesimal Mach wave or characteristic. '
                'A subsonic disturbance is not confined to a Mach cone.</div>', unsafe_allow_html=True)
        with st.expander("Equation summary / Resumen de ecuaciones"):
            equations_isentropic()
        st.link_button(text["theory"], LECTURES["isentropic"])
    except ValueError as exc:
        st.error(str(exc))

# 03 Normal shock
with tabs[2]:
    section_banner(
        "Normal shock" if language == "en" else "Onda de choque normal",
        "Dimensionless jump relations and the complete downstream thermodynamic state."
        if language == "en" else
        "Relaciones adimensionales de salto y estado termodinámico completo aguas abajo.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        mach1 = st.number_input("M₁", min_value=1.000001, value=2.0, step=0.001, format="%.6f", key="normal_mach_v3")
    with c2:
        p1 = pressure_input(f"p₁ [{pressure_unit}]", 50_000.0, "normal_p1_v3", pressure_unit)
    with c3:
        t1 = temperature_input(f"T₁ [{temperature_unit}]", 220.0, "normal_t1_v3", temperature_unit)
    try:
        result = normal_shock_relations(mach1, gamma)
        p2, t2 = p1 * result.p2_over_p1, t1 * result.t2_over_t1
        rho1 = p1 / (r_gas * t1)
        rho2 = rho1 * result.rho2_over_rho1
        metric_grid([
            ("M₂", result.mach2, None), ("p₂/p₁", result.p2_over_p1, None),
            ("ρ₂/ρ₁", result.rho2_over_rho1, None), ("T₂/T₁", result.t2_over_t1, None),
            ("u₂/u₁", result.velocity2_over_velocity1, None), ("p₀₂/p₀₁", result.p02_over_p01, None),
            ("Δs/R", result.entropy_change_over_r, None), ("T₀₂/T₀₁", 1.0, None),
        ], decimals)
        st.markdown("#### Downstream state / Estado aguas abajo")
        metric_grid(state_metrics(p2, t2, gamma, r_gas, pressure_unit, temperature_unit, density_unit, speed_unit), decimals)
        st.caption(f"ρ₁ = {density_from_kg_m3(rho1, density_unit):.{decimals}f} {density_unit} · ρ₂ = {density_from_kg_m3(rho2, density_unit):.{decimals}f} {density_unit}")
        st.markdown('<div class="info-card">The Pitot calculation has been separated into its own module so that the post-shock static state is not confused with the pressure measured by a probe.</div>', unsafe_allow_html=True)
        with st.expander("Key equations / Fórmulas clave"):
            equations_normal_shock()
        st.link_button(text["theory"], LECTURES["normal"])
    except ValueError as exc:
        st.error(str(exc))

# 04 Pitot
with tabs[3]:
    section_banner(
        "Compressible Pitot measurement" if language == "en" else "Medición Pitot compresible",
        "A separate diagnostic path for subsonic stagnation or a supersonic normal shock followed by stagnation."
        if language == "en" else
        "Trayectoria diagnóstica separada: estancamiento subsónico o choque normal supersónico seguido de estancamiento.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        pitot_mach = st.number_input("Local M₁", min_value=0.000001, value=3.0, step=0.001, format="%.6f", key="pitot_mach_v3")
    with c2:
        pitot_p1 = pressure_input(f"Local p₁ [{pressure_unit}]", 30_000.0, "pitot_p1_v3", pressure_unit)
    with c3:
        pitot_t1 = temperature_input(f"Local T₁ [{temperature_unit}]", 220.0, "pitot_t1_v3", temperature_unit)
    try:
        pitot = pitot_pressure_ratio(pitot_mach, gamma)
        t0 = pitot_t1 * (1.0 + 0.5 * (gamma - 1.0) * pitot_mach**2)
        p_measured = pitot_p1 * pitot.p_measured_over_p1
        if pitot.normal_shock_present:
            p2 = pitot_p1 * pitot.p2_over_p1
            t2 = pitot_t1 * pitot.t2_over_t1
            rows = [
                ["1", "Local upstream", pitot_mach, pitot_p1, pitot_t1, pitot.p01_over_p1],
                ["2", "Normal shock", pitot.mach2, p2, t2, pitot.p02_over_p01 * pitot.p01_over_p1],
                ["0,2", "Stagnation / measured", 0.0, p_measured, t0, pitot.p02_over_p01 * pitot.p01_over_p1],
            ]
        else:
            p2, t2 = pitot_p1, pitot_t1
            rows = [
                ["1", "Local upstream", pitot_mach, pitot_p1, pitot_t1, pitot.p01_over_p1],
                ["0,1", "Isentropic stagnation / measured", 0.0, p_measured, t0, pitot.p01_over_p1],
            ]
        metric_grid([
            (f"p measured [{pressure_unit}]", pressure_from_pa(p_measured, pressure_unit), None),
            ("p measured/p₁", pitot.p_measured_over_p1, None),
            ("p₀₁/p₁", pitot.p01_over_p1, None),
            ("p₀₂/p₀₁", pitot.p02_over_p01, None),
            ("M₂ after shock", pitot.mach2 if pitot.mach2 is not None else "—", None),
            ("p₂/p₁", pitot.p2_over_p1, None),
            ("T₂/T₁", pitot.t2_over_t1, None),
            ("Δs/R", pitot.entropy_change_over_r, None),
        ], decimals)
        table = pd.DataFrame(rows, columns=["State", "Event", "M", "p_pa", "T_k", "p0_over_p1"])
        table[f"p [{pressure_unit}]"] = table["p_pa"].map(lambda value: pressure_from_pa(value, pressure_unit))
        table[f"T [{temperature_unit}]"] = table["T_k"].map(lambda value: temperature_from_k(value, temperature_unit))
        table[f"ρ [{density_unit}]"] = (table["p_pa"] / (r_gas * table["T_k"])).map(lambda value: density_from_kg_m3(value, density_unit))
        st.dataframe(table[["State", "Event", "M", f"p [{pressure_unit}]", f"T [{temperature_unit}]", f"ρ [{density_unit}]", "p0_over_p1"]], width="stretch", hide_index=True)
        st.markdown('<div class="warning-card"><strong>p₂ is not the Pitot reading.</strong> In supersonic flow the probe measures p₀₂ after the local normal shock and the final isentropic deceleration.</div>', unsafe_allow_html=True)
        with st.expander("Key equations / Fórmulas clave"):
            equations_pitot()
        st.link_button(text["theory"], LECTURES["normal"])
    except ValueError as exc:
        st.error(str(exc))

# 05 Oblique shock
with tabs[4]:
    section_banner(
        "Oblique shock" if language == "en" else "Onda de choque oblicua",
        "Weak/strong solutions, complete normal-component information, absolute downstream state, and an Anderson-style θ–β–M chart."
        if language == "en" else
        "Soluciones débil/fuerte, componentes normales completas, estado absoluto aguas abajo y diagrama θ–β–M al estilo de Anderson.",
    )
    c1, c2, c3 = st.columns(3)
    oblique_mach = c1.number_input("M₁", min_value=1.000001, value=2.5, step=0.001, format="%.6f", key="oblique_mach_v3")
    theta = c2.number_input("θ [deg]", min_value=0.000001, value=15.0, step=0.1, format="%.6f", key="oblique_theta_v3")
    branch = c3.selectbox("Branch / Rama", ["weak", "strong"], key="oblique_branch_v3")
    c4, c5 = st.columns(2)
    with c4:
        oblique_p1 = pressure_input(f"p₁ [{pressure_unit}]", 80_000.0, "oblique_p1_v3", pressure_unit)
    with c5:
        oblique_t1 = temperature_input(f"T₁ [{temperature_unit}]", 250.0, "oblique_t1_v3", temperature_unit)
    try:
        result = oblique_shock_relations(oblique_mach, theta, gamma=gamma, branch=branch)
        metric_grid([
            ("β", result.beta_deg, "°"), ("θmax", result.theta_max_deg, "°"),
            ("Mₙ₁", result.mn1, None), ("Mₙ₂", result.mn2, None),
            ("M₂", result.mach2, None), ("p₂/p₁", result.p2_over_p1, None),
            ("ρ₂/ρ₁", result.rho2_over_rho1, None), ("T₂/T₁", result.t2_over_t1, None),
            ("p₀₂/p₀₁", result.p02_over_p01, None), ("Δs/R", result.entropy_change_over_r, None),
        ], decimals)
        p2, t2 = oblique_p1 * result.p2_over_p1, oblique_t1 * result.t2_over_t1
        st.markdown("#### Downstream state / Estado aguas abajo")
        metric_grid(state_metrics(p2, t2, gamma, r_gas, pressure_unit, temperature_unit, density_unit, speed_unit), decimals)
        st.markdown("#### θ–β–M chart / Diagrama θ–β–M")
        st.plotly_chart(theta_beta_m_figure(oblique_mach, gamma, selected_beta_deg=result.beta_deg, selected_theta_deg=theta), width="stretch")
        with st.expander("Key equations / Fórmulas clave"):
            equations_oblique()
        st.link_button(text["theory"], LECTURES["oblique"])
    except ValueError as exc:
        theta_limit, _ = theta_max(oblique_mach, gamma)
        st.error(str(exc))
        st.warning(f"θmax = {theta_limit:.{decimals}f}°")

# 06 Prandtl-Meyer
with tabs[5]:
    section_banner(
        "Prandtl–Meyer expansion" if language == "en" else "Expansión de Prandtl–Meyer",
        "Finite centered expansion, Mach angles before and after the fan, absolute properties, and explicit isentropic ratio construction."
        if language == "en" else
        "Expansión centrada finita, ángulos de Mach antes y después, propiedades absolutas y construcción explícita de las razones isoentrópicas.",
    )
    modes = ["M₁ and expansion turn", "Target ν"] if language == "en" else ["M₁ y giro de expansión", "ν objetivo"]
    pm_mode = st.radio("Input / Entrada", modes, horizontal=True)
    try:
        if pm_mode == modes[0]:
            c1, c2 = st.columns(2)
            pm_mach = c1.number_input("M₁", min_value=1.000001, value=2.0, step=0.001, format="%.6f", key="pm_mach_v3")
            pm_theta = c2.number_input("Expansion turn θ [deg]", min_value=0.000001, value=15.0, step=0.1, format="%.6f", key="pm_theta_v3")
            c3, c4 = st.columns(2)
            with c3:
                pm_p1 = pressure_input(f"p₁ [{pressure_unit}]", 120_000.0, "pm_p1_v3", pressure_unit)
            with c4:
                pm_t1 = temperature_input(f"T₁ [{temperature_unit}]", 300.0, "pm_t1_v3", temperature_unit)
            result = expansion(pm_mach, pm_theta, gamma)
            metric_grid([
                ("ν₁", result.nu1_deg, "°"), ("ν₂", result.nu2_deg, "°"),
                ("μ₁", result.mu1_deg, "°"), ("μ₂", result.mu2_deg, "°"),
                ("M₂", result.mach2, None), ("p₂/p₁", result.p2_over_p1, None),
                ("ρ₂/ρ₁", result.rho2_over_rho1, None), ("T₂/T₁", result.t2_over_t1, None),
                ("p₀₂/p₀₁", result.p02_over_p01, None), ("νmax", nu_max_deg(gamma), "°"),
            ], decimals)
            p2, t2 = pm_p1 * result.p2_over_p1, pm_t1 * result.t2_over_t1
            st.markdown("#### Downstream state / Estado aguas abajo")
            metric_grid(state_metrics(p2, t2, gamma, r_gas, pressure_unit, temperature_unit, density_unit, speed_unit), decimals)
            st.markdown('<div class="info-card">The property ratios are obtained from the same isentropic total–static relations evaluated at M₁ and M₂. Because T₀ and p₀ are constant through the fan, their quotient gives T₂/T₁, p₂/p₁, and ρ₂/ρ₁.</div>', unsafe_allow_html=True)
        else:
            nu_target = st.number_input("ν [deg]", min_value=0.0, value=26.3798, step=0.001, format="%.6f", key="nu_target_v3")
            mach = mach_from_nu(nu_target, gamma)
            mu = math.degrees(math.asin(1.0 / mach)) if mach >= 1.0 else math.nan
            metric_grid([("Mach", mach, None), ("Mach angle μ", mu, "°"), ("νmax", nu_max_deg(gamma), "°")], decimals)
        with st.expander("Key equations / Fórmulas clave"):
            equations_pm()
        st.link_button(text["theory"], LECTURES["pm"])
    except ValueError as exc:
        st.error(str(exc))

# 07 Geometry
with tabs[6]:
    section_banner(
        "Shock–expansion geometry" if language == "en" else "Geometría choque–expansión",
        "Piecewise-linear surface, clearer region-based Pitot placement, separate state/wave tables, and local Pitot diagnostic schematic."
        if language == "en" else
        "Superficie por paneles, ubicación del Pitot por región, tablas separadas de estados/ondas y esquema diagnóstico local.",
    )
    side_options = ["lower", "upper"] if language == "en" else ["inferior", "superior"]
    example_options = ["Diamond half-surface", "Compression–expansion ramp", "Custom"] if language == "en" else ["Medio perfil diamante", "Rampa compresión–expansión", "Personalizada"]
    example = st.selectbox("Preset / Configuración", example_options)
    idx = example_options.index(example)
    if idx == 0:
        default_df = pd.DataFrame({"length": [1.0, 1.0], "angle_deg": [5.0, -5.0]})
        default_mach, default_side = 3.0, side_options[0]
    elif idx == 1:
        default_df = pd.DataFrame({"length": [1.0, 1.2, 1.0], "angle_deg": [0.0, 8.0, -2.0]})
        default_mach, default_side = 2.5, side_options[0]
    else:
        default_df = pd.DataFrame({"length": [1.0, 1.0], "angle_deg": [0.0, 0.0]})
        default_mach, default_side = 2.5, side_options[0]

    st.caption("length = panel length; angle_deg = absolute panel inclination. The first panel is compared with the horizontal freestream.")
    geometry_df = st.data_editor(default_df, num_rows="dynamic", width="stretch", key=f"geometry_editor_v3_{language}_{idx}", column_config={
        "length": st.column_config.NumberColumn("Length / Longitud", min_value=0.001, format="%.6f"),
        "angle_deg": st.column_config.NumberColumn("Panel angle [deg] / Inclinación", format="%.6f"),
    })
    c1, c2, c3, c4 = st.columns(4)
    mach_inf = c1.number_input("M∞", min_value=1.000001, value=default_mach, step=0.001, format="%.6f", key=f"mach_inf_v3_{idx}")
    with c2:
        p_inf = pressure_input(f"p∞ [{pressure_unit}]", 101_325.0, f"p_inf_v3_{idx}", pressure_unit)
    with c3:
        t_inf = temperature_input(f"T∞ [{temperature_unit}]", 288.15, f"t_inf_v3_{idx}", temperature_unit)
    side = c4.selectbox("Surface / Superficie", side_options, index=side_options.index(default_side), key=f"side_v3_{language}_{idx}")

    c5, c6 = st.columns(2)
    geometry_branch = c5.selectbox("Shock branch / Rama", ["weak", "strong"], key="geometry_branch_v3")
    max_segment = max(len(geometry_df), 1)
    pitot_options = ["None / Ninguno"] + [f"R{i + 1} · panel {i}" for i in range(1, max_segment + 1)]
    pitot_choice = c6.selectbox("Pitot sampling region / Región muestreada", pitot_options, index=0, key=f"pitot_region_v3_{idx}")
    pitot_segment = None if pitot_choice == pitot_options[0] else pitot_options.index(pitot_choice)

    try:
        clean = geometry_df[["length", "angle_deg"]].dropna()
        segments = clean.to_dict(orient="records")
        surface_df, waves, vertices, pitot_data = solve_surface(
            segments, mach_inf=mach_inf, pressure_inf_pa=p_inf, temperature_inf_k=t_inf,
            gamma=gamma, side=side, branch=geometry_branch, pitot_segment=pitot_segment,
        )
        region_df = build_region_table(surface_df, mach_inf=mach_inf, pressure_inf_pa=p_inf, temperature_inf_k=t_inf, gamma=gamma, pitot_data=pitot_data)
        main_rows = region_df[region_df["path_code"] == "main_flow"]
        final = main_rows.iloc[-1]
        shock_count = int((surface_df["event_code"] == "oblique_shock").sum())
        expansion_count = int((surface_df["event_code"] == "prandtl_meyer_expansion").sum())
        metric_grid([
            ("Final main-flow M", float(final["mach"]), None),
            ("Final p/p∞", float(final["pressure_ratio_to_freestream"]), None),
            ("Final p₀/p₀∞", float(final["total_pressure_ratio_to_freestream"]), None),
            ("Waves", f"{shock_count} S · {expansion_count} E", None),
        ], decimals)

        st.markdown("#### Geometry and external waves / Geometría y ondas externas")
        st.plotly_chart(geometry_figure(vertices, waves, pitot_segment=pitot_segment, pitot_data=pitot_data), width="stretch")

        display_df = localized_region_table(region_df, language, geometry_branch, r_gas, pressure_unit, temperature_unit, density_unit)
        result_view = st.radio("Results view / Vista de resultados", ["State table / Estados", "Wave details / Ondas"], horizontal=True)
        if result_view.startswith("State"):
            display_state_table(display_df, decimals, pressure_unit, temperature_unit, density_unit)
        else:
            display_wave_table(display_df, decimals)

        if pitot_data:
            st.markdown('<div class="warning-card"><strong>Local diagnostic branch.</strong> The probe samples the selected region; its normal shock does not alter the external state propagated to later panels.</div>', unsafe_allow_html=True)
            st.markdown(f"#### Pitot diagnostic in R{pitot_data['source_region']}")
            st.plotly_chart(pitot_schematic_figure(pitot_data), width="stretch")
            pitot_metrics = [
                ("Local M₁", float(pitot_data["mach1"]), None),
                (f"Measured p [{pressure_unit}]", pressure_from_pa(float(pitot_data["p_measured_pa"]), pressure_unit), None),
                ("p₀₂/p₀∞", float(pitot_data["p02_ratio_to_freestream_total"]), None),
            ]
            if pitot_data["normal_shock_present"]:
                pitot_metrics.extend([
                    ("Post-shock M₂", float(pitot_data["mach2"]), None),
                    ("p₂/p₁", float(pitot_data["p2_over_p1"]), None),
                    ("T₂/T₁", float(pitot_data["t2_over_t1"]), None),
                    ("p₀₂/p₀₁", float(pitot_data["p02_over_p01"]), None),
                ])
            metric_grid(pitot_metrics, decimals)

        with st.expander("Method equations / Ecuaciones del método"):
            st.latex(r"\Delta\theta_i=\alpha_i-\alpha_{i-1}")
            st.markdown("Compression → oblique shock; expansion → Prandtl–Meyer fan.")
            st.latex(r"\mathbf{S}_i=(M_i,p_i,T_i,\rho_i,p_{0,i},\alpha_i)")
            st.latex(r"\frac{p_{0,f}}{p_{0,\infty}}=\prod_j\frac{p_{0,j+1}}{p_{0,j}}")

        full_csv = region_df.to_csv(index=False).encode("utf-8")
        raw_csv = surface_df.to_csv(index=False).encode("utf-8")
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button(text["download"], data=full_csv, file_name="shock_expansion_region_table.csv", mime="text/csv", width="stretch")
        d2.download_button("Raw surface states CSV", data=raw_csv, file_name="surface_states.csv", mime="text/csv", width="stretch")
        st.link_button(text["theory"], LECTURES["geometry"])

        if (surface_df["event_code"] == "detached_shock").any():
            st.error("Detached-shock condition detected. The straight attached-shock sequence was stopped.")
        st.info("Future specialized tools: complete diamond-airfoil performance versus angle of attack and a dedicated supersonic flat-plate calculator.")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        st.error(str(exc))
