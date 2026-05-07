from pathlib import Path

import numpy as np
import polars as pl
from dash import Dash, dcc, html
from plotly import express as px
from plotly import graph_objects as go

DATA_PATH = (
    "cache/cybersecurity-intrusion-detection-datatset/cybersecurity_intrusion_data.csv"
)

ATTACK_COLOR = "#D85A30"
SAFE_COLOR = "#1D9E75"

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #f0f2f5; color: #2c3e50; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
h1 { font-size: 20px; font-weight: 500; color: #1a1a2e; margin-bottom: 4px; }
.subtitle { font-size: 12px; color: #95a5a6; margin-bottom: 24px; }
.stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
.stat-card { background: white; padding: 16px 20px; border-radius: 12px; border: 0.5px solid rgba(0,0,0,0.07); }
.stat-card strong { display: block; font-size: 11px; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: #95a5a6; margin-bottom: 8px; }
.stat-card span { font-size: 26px; font-weight: 500; color: #2c3e50; }
.stat-card.danger span { color: #D85A30; }
.stat-card.safe span { color: #1D9E75; }
.stat-card .sub { display: block; font-size: 11px; color: #bdc3c7; margin-top: 4px; }
.section-label { font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #95a5a6; margin: 28px 0 12px; display: flex; align-items: center; gap: 10px; }
.section-label::after { content: ''; flex: 1; height: 0.5px; background: rgba(0,0,0,0.1); }
.charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }
.chart { background: white; padding: 16px 18px; border-radius: 12px; border: 0.5px solid rgba(0,0,0,0.07); }
.chart .js-plotly-plot { width: 100%; }
"""

INDEX_STRING = (
    """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Intrusion Detection Dashboard</title>
{%favicon%}
{%css%}
<style>"""
    + CSS
    + """</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""
)

CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        size=12,
        color="#2c3e50",
    ),
    margin=dict(l=48, r=24, t=48, b=48),
)

LEGEND_BOTTOM = dict(
    orientation="h",
    y=-0.2,
    x=0.5,
    xanchor="center",
    yanchor="top",
    bgcolor="rgba(0,0,0,0)",
    borderwidth=0,
)


def apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**{**CHART_LAYOUT, **kwargs})
    return fig


def build_figures(df: pl.DataFrame) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {}

    total = len(df)
    attacks = df.filter(pl.col("attack_detected") == 1).height

    fig = go.Figure(
        go.Pie(
            labels=["Attack", "Safe"],
            values=[attacks, total - attacks],
            hole=0.6,
            marker_colors=[ATTACK_COLOR, SAFE_COLOR],
            textinfo="percent",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{attacks / total * 100:.1f}%</b><br>attack rate",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#2c3e50"),
        align="center",
    )
    figures["attack_donut"] = apply_layout(
        fig, title="Overall attack rate", showlegend=True
    )

    def attack_rate_by(col: str) -> pl.DataFrame:
        return (
            df.group_by(col)
            .agg(
                pl.len().alias("total"),
                pl.col("attack_detected").sum().alias("attacks"),
            )
            .with_columns(
                (pl.col("attacks") / pl.col("total") * 100)
                .round(1)
                .alias("attack_rate")
            )
            .sort("attack_rate", descending=False)
        )

    for col, title in [
        ("protocol_type", "Attack rate by protocol (%)"),
        ("encryption_used", "Attack rate by encryption (%)"),
    ]:
        rates = attack_rate_by(col)
        fig = px.bar(
            rates,
            x="attack_rate",
            y=col,
            orientation="h",
            title=title,
            color="attack_rate",
            color_continuous_scale=[
                [0, SAFE_COLOR],
                [0.5, "#f39c12"],
                [1, ATTACK_COLOR],
            ],
            labels={"attack_rate": "Attack rate (%)"},
            text="attack_rate",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        figures[f"{col}_rate"] = apply_layout(fig, showlegend=False)

    browser_rates = attack_rate_by("browser_type")
    fig = go.Figure()
    for row in browser_rates.iter_rows(named=True):
        fig.add_shape(
            type="line",
            x0=0,
            x1=row["attack_rate"],
            y0=row["browser_type"],
            y1=row["browser_type"],
            line=dict(color="#e0e0e0", width=2),
        )
    fig.add_trace(
        go.Scatter(
            x=browser_rates["attack_rate"].to_list(),
            y=browser_rates["browser_type"].to_list(),
            mode="markers+text",
            marker=dict(
                size=14,
                color=browser_rates["attack_rate"].to_list(),
                colorscale=[[0, SAFE_COLOR], [0.5, "#f39c12"], [1, ATTACK_COLOR]],
                showscale=False,
            ),
            text=[f"{v}%" for v in browser_rates["attack_rate"].to_list()],
            textposition="middle right",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Attack rate (%)", yaxis_title="")
    figures["browser_type_rate"] = apply_layout(
        fig, title="Attack rate by browser (%)", showlegend=False
    )

    failed_agg = (
        df.with_columns(pl.col("failed_logins").clip(0, 5).alias("failed_capped"))
        .group_by(["failed_capped", "attack_detected"])
        .len()
        .sort("failed_capped")
    )
    totals = failed_agg.group_by("failed_capped").agg(
        pl.col("len").sum().alias("total")
    )
    failed_pct = (
        failed_agg.join(totals, on="failed_capped")
        .with_columns((pl.col("len") / pl.col("total") * 100).round(1).alias("pct"))
        .with_columns(pl.col("attack_detected").cast(pl.String))
    )
    fig = px.bar(
        failed_pct,
        x="failed_capped",
        y="pct",
        color="attack_detected",
        barmode="stack",
        title="Failed logins — attack share (%)",
        color_discrete_map={"0": SAFE_COLOR, "1": ATTACK_COLOR},
        labels={
            "failed_capped": "Failed logins",
            "pct": "Share (%)",
            "attack_detected": "",
        },
        text="pct",
    )
    fig.update_traces(
        texttemplate="%{text:.0f}%", textposition="inside", textfont_size=10
    )
    figures["failed_logins"] = apply_layout(fig, legend=LEGEND_BOTTOM)

    unusual_agg = (
        df.group_by(["unusual_time_access", "attack_detected"])
        .len()
        .sort("unusual_time_access")
        .with_columns(
            pl.col("attack_detected").cast(pl.String),
            pl.col("unusual_time_access").cast(pl.String),
        )
    )
    fig = px.bar(
        unusual_agg,
        x="unusual_time_access",
        y="len",
        color="attack_detected",
        barmode="group",
        title="Unusual time access vs attacks",
        color_discrete_map={"0": SAFE_COLOR, "1": ATTACK_COLOR},
        labels={
            "unusual_time_access": "Unusual time (0=no, 1=yes)",
            "len": "Sessions",
            "attack_detected": "",
        },
    )
    figures["unusual_time"] = apply_layout(fig, legend=LEGEND_BOTTOM)

    df_str = df.with_columns(pl.col("attack_detected").cast(pl.String))

    fig = px.box(
        df_str,
        x="attack_detected",
        y="login_attempts",
        color="attack_detected",
        title="Login attempts distribution",
        color_discrete_map={"0": SAFE_COLOR, "1": ATTACK_COLOR},
        labels={"attack_detected": "", "login_attempts": "Login attempts"},
        points="outliers",
    )
    fig.update_layout(xaxis=dict(tickvals=["0", "1"], ticktext=["Safe", "Attack"]))
    figures["login_attempts"] = apply_layout(fig, showlegend=False)

    fig = px.violin(
        df_str,
        x="attack_detected",
        y="ip_reputation_score",
        color="attack_detected",
        box=True,
        title="IP reputation score — safe vs attack",
        color_discrete_map={"0": SAFE_COLOR, "1": ATTACK_COLOR},
        labels={"attack_detected": "", "ip_reputation_score": "IP reputation score"},
    )
    fig.update_layout(xaxis=dict(tickvals=["0", "1"], ticktext=["Safe", "Attack"]))
    figures["ip_reputation"] = apply_layout(fig, showlegend=False)

    fig = px.histogram(
        df_str,
        x="session_duration",
        color="attack_detected",
        nbins=30,
        title="Session duration distribution",
        barmode="overlay",
        opacity=0.68,
        color_discrete_map={"0": SAFE_COLOR, "1": ATTACK_COLOR},
        labels={"session_duration": "Session duration (s)", "attack_detected": ""},
    )
    figures["session_duration"] = apply_layout(fig, legend=LEGEND_BOTTOM)

    fig = px.box(
        df_str,
        x="attack_detected",
        y="network_packet_size",
        color="attack_detected",
        title="Network packet size — safe vs attack",
        color_discrete_map={"0": SAFE_COLOR, "1": ATTACK_COLOR},
        labels={"attack_detected": "", "network_packet_size": "Packet size (bytes)"},
        points="outliers",
    )
    fig.update_layout(xaxis=dict(tickvals=["0", "1"], ticktext=["Safe", "Attack"]))
    figures["packet_size"] = apply_layout(fig, showlegend=False)

    numeric_cols = [
        "network_packet_size",
        "login_attempts",
        "session_duration",
        "ip_reputation_score",
        "failed_logins",
        "unusual_time_access",
        "attack_detected",
    ]
    corr_data = df.select(numeric_cols).to_numpy()
    z_values = np.corrcoef(corr_data, rowvar=False).tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=numeric_cols,
            y=numeric_cols,
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size": 9},
        )
    )
    figures["correlation"] = apply_layout(
        fig,
        title="Feature correlation heatmap",
        showlegend=False,
        margin=dict(l=120, r=24, t=48, b=120),
    )

    return figures


def build_stats(df: pl.DataFrame) -> list[tuple[str, str, str, str]]:
    total = len(df)
    attacks = df.filter(pl.col("attack_detected") == 1).height
    avg_ip_atk = df.filter(pl.col("attack_detected") == 1)["ip_reputation_score"].mean()
    avg_ip_saf = df.filter(pl.col("attack_detected") == 0)["ip_reputation_score"].mean()
    avg_sess = df["session_duration"].mean()
    return [
        ("Total sessions", f"{total:,}", "", ""),
        (
            "Attacks detected",
            f"{attacks:,}",
            f"{attacks / total * 100:.1f}% of sessions",
            "danger",
        ),
        (
            "Avg IP score — attack",
            f"{avg_ip_atk:.2f}",
            f"vs {avg_ip_saf:.2f} for safe",
            "danger",
        ),
        ("Avg session duration", f"{avg_sess:.0f}s", "", ""),
    ]


def section(label: str, charts: list) -> list:
    return [
        html.P(label, className="section-label"),
        html.Div(charts, className="charts-grid"),
    ]


def main() -> None:
    df = pl.read_csv(DATA_PATH)
    figures = build_figures(df)
    stats = build_stats(df)

    app = Dash(__name__, index_string=INDEX_STRING)

    app.layout = html.Div(
        [
            html.H1("Intrusion Detection: Overview"),
            html.P("Cybersecurity session analysis", className="subtitle"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(label),
                            html.Span(value),
                            html.Span(sub, className="sub"),
                        ],
                        className=f"stat-card {css_class}".strip(),
                    )
                    for label, value, sub, css_class in stats
                ],
                className="stats-row",
            ),
            *section(
                "Overview",
                [
                    html.Div(
                        [dcc.Graph(figure=figures["attack_donut"])], className="chart"
                    ),
                    html.Div(
                        [dcc.Graph(figure=figures["protocol_type_rate"])],
                        className="chart",
                    ),
                    html.Div(
                        [dcc.Graph(figure=figures["encryption_used_rate"])],
                        className="chart",
                    ),
                ],
            ),
            *section(
                "Attack rate by category",
                [
                    html.Div(
                        [dcc.Graph(figure=figures["browser_type_rate"])],
                        className="chart",
                    ),
                    html.Div(
                        [dcc.Graph(figure=figures["failed_logins"])], className="chart"
                    ),
                    html.Div(
                        [dcc.Graph(figure=figures["unusual_time"])], className="chart"
                    ),
                ],
            ),
            *section(
                "Distributions",
                [
                    html.Div(
                        [dcc.Graph(figure=figures["login_attempts"])], className="chart"
                    ),
                    html.Div(
                        [dcc.Graph(figure=figures["ip_reputation"])], className="chart"
                    ),
                    html.Div(
                        [dcc.Graph(figure=figures["session_duration"])],
                        className="chart",
                    ),
                    html.Div(
                        [dcc.Graph(figure=figures["packet_size"])], className="chart"
                    ),
                ],
            ),
            *section(
                "Feature correlation",
                [
                    html.Div(
                        [dcc.Graph(figure=figures["correlation"])], className="chart"
                    ),
                ],
            ),
        ],
        style={"padding": "24px", "maxWidth": "1400px", "margin": "0 auto"},
    )

    app.run(debug=True)
