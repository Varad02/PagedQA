"""
charts.py — PagedQA
Takes the output of BenchmarkResult.to_dict() and returns Plotly figures
ready to drop into Gradio.

Three charts:
    1. latency_chart()      — grouped bar, cold vs warm median + p95 latency
    2. throughput_chart()   — bar chart, cold vs warm tokens/sec
    3. cache_hit_chart()    — line chart, cache hit rate over warm requests
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Color palette (consistent across all charts)
# ---------------------------------------------------------------------------

COLD_COLOR  = "#E8593C"   # coral — cold run (no caching)
WARM_COLOR  = "#1D9E75"   # teal  — warm run (with caching)
GRID_COLOR  = "rgba(0,0,0,0.05)"
FONT_FAMILY = "Inter, system-ui, sans-serif"


# ---------------------------------------------------------------------------
# Chart 1 — Latency comparison
# ---------------------------------------------------------------------------

def latency_chart(result: dict) -> go.Figure:
    """
    Grouped bar chart comparing cold vs warm latency.
    Shows both median and p95 for each condition.
    """
    cold = [m["latency"] for m in result["cold"]]
    warm = [m["latency"] for m in result["warm"]]

    cold_ttft = [m["ttft"] for m in result["cold"]]
    warm_ttft = [m["ttft"] for m in result["warm"]]

    categories   = ["Median latency", "p95 latency", "Median TTFT", "p95 TTFT"]
    cold_values  = [
        _median(cold),
        _p95(cold),
        _median(cold_ttft),
        _p95(cold_ttft),
    ]
    warm_values  = [
        _median(warm),
        _p95(warm),
        _median(warm_ttft),
        _p95(warm_ttft),
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Cold (no cache)",
        x=categories,
        y=cold_values,
        marker=dict(color=COLD_COLOR, line=dict(width=0)),
        text=[f"{v:.2f}s" for v in cold_values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.3f}s<extra>Cold</extra>",
    ))

    fig.add_trace(go.Bar(
        name="Warm (prefix cached)",
        x=categories,
        y=warm_values,
        marker=dict(color=WARM_COLOR, line=dict(width=0)),
        text=[f"{v:.2f}s" for v in warm_values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.3f}s<extra>Warm</extra>",
    ))

    speedup = result["summary"]["speedup"]

    fig.update_layout(
        **_base_layout(),
        title=dict(
            text=f"Latency: Cold vs Warm  ·  {speedup}× speedup from prefix caching",
            font=dict(size=15),
        ),
        barmode="group",
        yaxis=dict(title="Seconds", gridcolor=GRID_COLOR),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ---------------------------------------------------------------------------
# Chart 2 — Throughput comparison
# ---------------------------------------------------------------------------

def throughput_chart(result: dict) -> go.Figure:
    """
    Bar chart comparing cold vs warm tokens per second.
    Each bar represents one request; a summary bar shows the median.
    """
    cold_tps = [m["tps"] for m in result["cold"]]
    warm_tps = [m["tps"] for m in result["warm"]]
    request_labels = [f"Q{i+1}" for i in range(len(cold_tps))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Cold (no cache)",
        x=request_labels,
        y=cold_tps,
        marker=dict(color=COLD_COLOR, opacity=0.85, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.1f} tok/s<extra>Cold</extra>",
    ))

    fig.add_trace(go.Bar(
        name="Warm (prefix cached)",
        x=request_labels,
        y=warm_tps,
        marker=dict(color=WARM_COLOR, opacity=0.85, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.1f} tok/s<extra>Warm</extra>",
    ))

    # Median reference lines
    fig.add_hline(
        y=_median(cold_tps),
        line_dash="dash",
        line_color=COLD_COLOR,
        annotation_text=f"Cold median: {_median(cold_tps):.1f} tok/s",
        annotation_position="top left",
    )
    fig.add_hline(
        y=_median(warm_tps),
        line_dash="dash",
        line_color=WARM_COLOR,
        annotation_text=f"Warm median: {_median(warm_tps):.1f} tok/s",
        annotation_position="top right",
    )

    fig.update_layout(
        **_base_layout(),
        title=dict(text="Throughput per Request (tokens / second)", font=dict(size=15)),
        barmode="group",
        yaxis=dict(title="Tokens / second", gridcolor=GRID_COLOR),
        xaxis=dict(title="Request"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Cache hit rate over requests
# ---------------------------------------------------------------------------

def cache_hit_chart(result: dict) -> go.Figure:
    """
    Line chart showing cache hit rate climbing over the warm run requests.
    Cold run stays flat at 0% for visual contrast.
    """
    warm_hit_rates = [m["cache_hit_rate"] * 100 for m in result["warm"]]
    cold_hit_rates = [0.0] * len(result["cold"])
    request_labels = [f"Q{i+1}" for i in range(len(warm_hit_rates))]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        name="Cold (no cache)",
        x=request_labels,
        y=cold_hit_rates,
        mode="lines+markers",
        line=dict(color=COLD_COLOR, width=2, dash="dash"),
        marker=dict(size=6),
    ))

    fig.add_trace(go.Scatter(
        name="Warm (prefix cached)",
        x=request_labels,
        y=warm_hit_rates,
        mode="lines+markers",
        line=dict(color=WARM_COLOR, width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor=f"rgba(29, 158, 117, 0.1)",
    ))

    final_hit_rate = warm_hit_rates[-1] if warm_hit_rates else 0
    fig.update_layout(
        **_base_layout(),
        title=dict(
            text=f"Prefix Cache Hit Rate over Requests  ·  Final: {final_hit_rate:.1f}%",
            font=dict(size=15),
        ),
        yaxis=dict(title="Cache hit rate (%)", range=[0, 105], gridcolor=GRID_COLOR),
        xaxis=dict(title="Request"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ---------------------------------------------------------------------------
# Convenience: return all three at once
# ---------------------------------------------------------------------------

def all_charts(result: dict) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Return (latency_fig, throughput_fig, cache_hit_fig) for a result dict."""
    return (
        latency_chart(result),
        throughput_chart(result),
        cache_hit_chart(result),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _base_layout() -> dict:
    """Shared Plotly layout applied to every chart."""
    return dict(
        font=dict(family=FONT_FAMILY, size=13, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=70, b=44, l=64, r=24),
        hoverlabel=dict(
            font_size=13,
            bgcolor="white",
            bordercolor="#e2e8f0",
        ),
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * 0.95)
    return s[min(idx, len(s) - 1)]