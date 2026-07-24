"""Plotly figures styled to match the academic portal."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go

from core.oblique_shock import theta_from_beta, theta_max

NAVY = "#071d33"
NAVY_2 = "#0b3558"
BLUE = "#0d568d"
SKY = "#9fd6fa"
ORANGE = "#d67b28"
GREEN = "#176b47"
MUTED = "#607181"
LINE = "#d8e2ea"
SOFT = "#f5f8fa"
WHITE = "#ffffff"


def _base_layout(
    fig: go.Figure,
    *,
    height: int = 460,
    equal_axes: bool = False,
    legend_below: bool = False,
) -> None:
    legend = dict(
        orientation="h",
        xanchor="left",
        x=0.0,
        bgcolor="rgba(255,255,255,0.82)",
        bordercolor=LINE,
        borderwidth=0,
    )
    if legend_below:
        legend.update(yanchor="top", y=-0.16)
        bottom = 95
    else:
        legend.update(yanchor="bottom", y=1.02)
        bottom = 55

    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WHITE,
        font=dict(family="Inter, Arial, sans-serif", color=NAVY),
        margin=dict(l=58, r=28, t=55, b=bottom),
        legend=legend,
        hoverlabel=dict(bgcolor=WHITE, font_color=NAVY),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, automargin=True)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, automargin=True)
    if equal_axes:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)


def _curve_for_mach(mach: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    mu = math.asin(1.0 / mach)
    beta = np.linspace(mu + 1e-7, 0.5 * math.pi - 1e-7, 650)
    theta = np.array(
        [max(0.0, math.degrees(theta_from_beta(mach, value, gamma))) for value in beta]
    )
    return theta, np.degrees(beta)


def theta_beta_m_figure(
    mach1: float,
    gamma: float = 1.4,
    *,
    selected_beta_deg: float | None = None,
    selected_theta_deg: float | None = None,
    mach_curves: Iterable[float] | None = None,
) -> go.Figure:
    """Return an Anderson-style theta-beta-M chart.

    The horizontal axis is flow deflection theta, the vertical axis is shock
    angle beta, and each curve corresponds to a constant upstream Mach number.
    """
    if mach1 <= 1.0:
        raise ValueError("M1 must be greater than 1.")

    base_curves = list(mach_curves or [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0])
    values = sorted({round(float(value), 8) for value in base_curves if value > 1.0} | {round(float(mach1), 8)})

    fig = go.Figure()
    for mach in values:
        theta, beta = _curve_for_mach(mach, gamma)
        selected_curve = math.isclose(mach, mach1, rel_tol=0.0, abs_tol=1e-8)
        fig.add_trace(
            go.Scatter(
                x=theta,
                y=beta,
                mode="lines",
                line=dict(
                    color=BLUE if selected_curve else "#9eb4c5",
                    width=4.5 if selected_curve else 1.65,
                ),
                opacity=1.0 if selected_curve else 0.8,
                name=f"M₁ = {mach:g}" if selected_curve else f"M={mach:g}",
                showlegend=selected_curve,
                hovertemplate=(
                    f"M₁={mach:g}<br>θ=%{{x:.3f}}°<br>β=%{{y:.3f}}°<extra></extra>"
                ),
            )
        )

        # Label each reference curve near its maximum deflection without a legend wall.
        theta_peak, beta_peak = theta_max(mach, gamma)
        if not selected_curve:
            fig.add_annotation(
                x=theta_peak,
                y=beta_peak,
                text=f"{mach:g}",
                showarrow=False,
                xshift=8,
                font=dict(size=10, color=MUTED),
                bgcolor="rgba(255,255,255,0.62)",
            )

    theta_peak_selected, beta_peak_selected = theta_max(mach1, gamma)
    fig.add_trace(
        go.Scatter(
            x=[theta_peak_selected],
            y=[beta_peak_selected],
            mode="markers",
            marker=dict(size=11, color=NAVY, line=dict(color=WHITE, width=1.5)),
            name="Attachment limit",
            hovertemplate="θmax=%{x:.3f}°<br>β=%{y:.3f}°<extra></extra>",
        )
    )

    if selected_beta_deg is not None and selected_theta_deg is not None:
        fig.add_trace(
            go.Scatter(
                x=[selected_theta_deg],
                y=[selected_beta_deg],
                mode="markers",
                marker=dict(size=15, color=GREEN, symbol="diamond", line=dict(color=WHITE, width=2)),
                name="Selected solution",
                hovertemplate="Selected θ=%{x:.4f}°<br>β=%{y:.4f}°<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Flow deflection θ [deg]",
        yaxis_title="Shock angle β [deg]",
        xaxis=dict(range=[0.0, max(48.0, theta_peak_selected * 1.15)]),
        yaxis=dict(range=[0.0, 92.0]),
    )
    _base_layout(fig, height=590, legend_below=False)
    return fig


def geometry_figure(
    vertices: list[tuple[float, float]],
    waves: list[dict],
    *,
    pitot_segment: int | None = None,
    pitot_data: dict | None = None,
) -> go.Figure:
    """Plot the surface, external wave pattern, and a compact Pitot location marker."""
    x = [point[0] for point in vertices]
    y = [point[1] for point in vertices]
    total_length = max(max(x) - min(x), 1.0)
    wave_length = 0.44 * total_length

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(width=7, color=NAVY),
            marker=dict(size=8, color=NAVY, line=dict(color=WHITE, width=1.5)),
            name="Surface",
            hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<extra></extra>",
        )
    )

    shock_legend = True
    expansion_legend = True
    for wave in waves:
        vx, vy = vertices[wave["vertex_index"]]
        if wave["kind"] == "shock":
            angle = math.radians(wave["direction_deg"])
            target_x = vx + wave_length * math.cos(angle)
            target_y = vy + wave_length * math.sin(angle)
            fig.add_trace(
                go.Scatter(
                    x=[vx, target_x],
                    y=[vy, target_y],
                    mode="lines",
                    line=dict(color=ORANGE, dash="dash", width=4),
                    name="Oblique shock",
                    legendgroup="shock",
                    showlegend=shock_legend,
                    customdata=[[wave.get("beta_deg"), wave.get("turn_deg")]] * 2,
                    hovertemplate="Oblique shock<br>β=%{customdata[0]:.3f}°<br>turn=%{customdata[1]:.3f}°<extra></extra>",
                )
            )
            shock_legend = False
        else:
            a0 = wave["direction_start_deg"]
            a1 = wave["direction_end_deg"]
            for index, angle_deg in enumerate(np.linspace(a0, a1, 9)):
                angle = math.radians(angle_deg)
                fig.add_trace(
                    go.Scatter(
                        x=[vx, vx + 0.74 * wave_length * math.cos(angle)],
                        y=[vy, vy + 0.74 * wave_length * math.sin(angle)],
                        mode="lines",
                        line=dict(color=GREEN, width=1.8),
                        name="Prandtl–Meyer fan",
                        legendgroup="expansion",
                        showlegend=expansion_legend and index == 0,
                        hovertemplate="Expansion characteristic<extra></extra>",
                    )
                )
            expansion_legend = False

    for segment in range(1, len(vertices)):
        x_mid = 0.5 * (vertices[segment - 1][0] + vertices[segment][0])
        y_mid = 0.5 * (vertices[segment - 1][1] + vertices[segment][1])
        fig.add_annotation(
            x=x_mid,
            y=y_mid,
            text=f"R{segment + 1}",
            showarrow=False,
            yshift=-22,
            font=dict(color=NAVY_2, size=12),
            bgcolor="rgba(255,255,255,0.84)",
            bordercolor=LINE,
            borderwidth=1,
        )

    x0 = min(x) - 0.30 * total_length
    x1 = min(x) - 0.055 * total_length
    y0 = y[0]
    fig.add_annotation(
        x=x1,
        y=y0,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="M∞",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.15,
        arrowwidth=3,
        arrowcolor=BLUE,
        font=dict(color=BLUE, size=14),
    )

    if pitot_segment is not None and 1 <= pitot_segment <= len(vertices) - 1:
        p0 = vertices[pitot_segment - 1]
        p1 = vertices[pitot_segment]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        tx, ty = dx / seg_len, dy / seg_len
        sign = float(pitot_data.get("fluid_normal_sign", 1.0)) if pitot_data else 1.0
        nx, ny = sign * (-ty), sign * tx

        wall_x = 0.5 * (p0[0] + p1[0])
        wall_y = 0.5 * (p0[1] + p1[1])
        offset = 0.105 * total_length
        sample_x = wall_x + offset * nx
        sample_y = wall_y + offset * ny
        stem = 0.055 * total_length
        mouth_x = sample_x - stem * tx
        mouth_y = sample_y - stem * ty

        fig.add_trace(
            go.Scatter(
                x=[wall_x, sample_x, mouth_x],
                y=[wall_y, sample_y, mouth_y],
                mode="lines+markers",
                line=dict(width=5, color=BLUE),
                marker=dict(size=[0, 11, 0], color=BLUE, symbol="circle"),
                name="Pitot location",
                hovertemplate=f"Pitot samples region R{pitot_segment + 1}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=sample_x,
            y=sample_y,
            text=f"Pitot · R{pitot_segment + 1}",
            showarrow=True,
            arrowhead=2,
            ax=45,
            ay=-35,
            font=dict(color=BLUE, size=12),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=SKY,
            borderwidth=1,
        )

    fig.update_layout(xaxis_title="x", yaxis_title="y")
    _base_layout(fig, height=565, equal_axes=True, legend_below=False)
    return fig


def pitot_schematic_figure(pitot_data: dict) -> go.Figure:
    """Return a compact state-path schematic for the selected Pitot branch."""
    fig = go.Figure()
    normal = bool(pitot_data.get("normal_shock_present"))

    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 0.0],
            mode="lines",
            line=dict(color=BLUE, width=5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if normal:
        fig.add_shape(type="line", x0=0.36, x1=0.36, y0=-0.24, y1=0.24, line=dict(color=ORANGE, width=6, dash="dash"))
        fig.add_annotation(x=0.18, y=0.16, text=f"Local flow<br>M₁={pitot_data['mach1']:.4f}", showarrow=False, font=dict(color=NAVY))
        fig.add_annotation(x=0.36, y=0.33, text="Normal shock", showarrow=False, font=dict(color=ORANGE))
        fig.add_annotation(x=0.60, y=0.16, text=f"Post-shock<br>M₂={pitot_data['mach2']:.4f}", showarrow=False, font=dict(color=NAVY))
    else:
        fig.add_annotation(x=0.25, y=0.16, text=f"Local flow<br>M={pitot_data['mach1']:.4f}", showarrow=False, font=dict(color=NAVY))

    fig.add_shape(type="line", x0=0.86, x1=0.86, y0=-0.18, y1=0.18, line=dict(color=BLUE, width=7))
    fig.add_shape(type="line", x0=0.86, x1=0.98, y0=0.0, y1=0.0, line=dict(color=BLUE, width=7))
    fig.add_annotation(x=0.90, y=0.30, text="Stagnation<br>p measured = p₀₂", showarrow=False, font=dict(color=BLUE))

    fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    fig.update_yaxes(visible=False, range=[-0.45, 0.5])
    _base_layout(fig, height=250, legend_below=False)
    fig.update_layout(margin=dict(l=10, r=10, t=15, b=10))
    return fig
