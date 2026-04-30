"""
LLM Cost Autopilot — Streamlit Dashboard

Run:
    cd cost-autopilot
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.queries import (
    get_daily_costs,
    get_escalation_trend,
    get_quality_distribution,
    get_recent_requests,
    get_routing_distribution,
    get_summary_stats,
    get_tier_breakdown,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .metric-box {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .headline {
        font-size: 3rem;
        font-weight: 800;
        color: #a6e3a1;
        line-height: 1.1;
    }
    .subheadline {
        font-size: 1.1rem;
        color: #cdd6f4;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #cdd6f4;
        margin: 24px 0 8px 0;
        border-bottom: 1px solid #313244;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ LLM Cost Autopilot")
    lookback = st.slider("Lookback window (days)", 1, 30, 14)
    st.divider()
    if st.button("Refresh data"):
        st.cache_data.clear()

# ---------------------------------------------------------------------------
# Data load (cached 30s)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_all(days):
    return {
        "stats":     get_summary_stats(),
        "daily":     get_daily_costs(days),
        "routing":   get_routing_distribution(),
        "quality":   get_quality_distribution(),
        "escalation":get_escalation_trend(days),
        "recent":    get_recent_requests(100),
        "tiers":     get_tier_breakdown(),
    }

data = load_all(lookback)
stats = data["stats"]

# ---------------------------------------------------------------------------
# Header — money shot metric
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="text-align:center; padding: 28px 0 12px 0;">
  <div class="headline">💰 ${stats['saved_usd']:.4f} saved</div>
  <div class="subheadline">
    <b style="color:#a6e3a1">{stats['pct_saved']}% cost reduction</b>
    &nbsp;vs. sending everything to GPT-4o
    &nbsp;·&nbsp; {stats['total_requests']:,} requests routed
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Top KPI row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Actual Cost",
    f"${stats['actual_cost']:.4f}",
    delta=f"-${stats['saved_usd']:.4f} vs GPT-4o",
    delta_color="inverse",
)
c2.metric("GPT-4o Baseline", f"${stats['baseline_cost']:.4f}")
c3.metric("Quality Pass Rate", f"{stats['quality_pass_rate']}%")
c4.metric("Escalation Rate", f"{stats['escalation_rate']}%")
c5.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f} ms")

st.divider()

# ---------------------------------------------------------------------------
# Row 1: Cost over time  |  Routing distribution
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-header">Daily Cost: Actual vs. GPT-4o Baseline</div>', unsafe_allow_html=True)

    daily_df = pd.DataFrame(data["daily"])
    if not daily_df.empty:
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=daily_df["day"], y=daily_df["baseline_cost"],
            name="GPT-4o Baseline", fill="tozeroy",
            line=dict(color="#f38ba8", width=2),
            fillcolor="rgba(243,139,168,0.15)",
        ))
        fig_cost.add_trace(go.Scatter(
            x=daily_df["day"], y=daily_df["actual_cost"],
            name="Actual Cost", fill="tozeroy",
            line=dict(color="#a6e3a1", width=2),
            fillcolor="rgba(166,227,161,0.2)",
        ))
        fig_cost.update_layout(
            height=300, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(tickprefix="$", gridcolor="#313244"),
            xaxis=dict(gridcolor="#313244"),
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.info("No data yet — run some requests first.")

with col_right:
    st.markdown('<div class="section-header">Routing Distribution</div>', unsafe_allow_html=True)

    routing_df = pd.DataFrame(data["routing"])
    if not routing_df.empty:
        fig_pie = px.pie(
            routing_df, values="count", names="model",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.45,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            height=300, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 2: Quality distribution  |  Escalation trend  |  Tier breakdown
# ---------------------------------------------------------------------------
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown('<div class="section-header">Quality Score Distribution</div>', unsafe_allow_html=True)

    q_df = pd.DataFrame(data["quality"])
    if not q_df.empty:
        fig_q = px.bar(
            q_df, x="score", y="count",
            color="score",
            color_continuous_scale=["#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#a6e3a1"],
        )
        fig_q.update_layout(
            height=260, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(title="Score (1–5)", gridcolor="#313244"),
            yaxis=dict(title="Requests", gridcolor="#313244"),
        )
        st.plotly_chart(fig_q, use_container_width=True)
    else:
        st.info("No quality scores yet.")

with col_b:
    st.markdown('<div class="section-header">Escalation Rate Over Time</div>', unsafe_allow_html=True)

    esc_df = pd.DataFrame(data["escalation"])
    if not esc_df.empty:
        fig_esc = go.Figure()
        fig_esc.add_trace(go.Bar(
            x=esc_df["day"], y=esc_df["total"],
            name="Total", marker_color="#89b4fa", opacity=0.4,
        ))
        fig_esc.add_trace(go.Bar(
            x=esc_df["day"], y=esc_df["escalated"],
            name="Escalated", marker_color="#f38ba8",
        ))
        fig_esc.update_layout(
            height=260, barmode="overlay",
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(gridcolor="#313244"),
            yaxis=dict(title="Requests", gridcolor="#313244"),
        )
        st.plotly_chart(fig_esc, use_container_width=True)
    else:
        st.info("No escalation data yet.")

with col_c:
    st.markdown('<div class="section-header">Savings by Complexity Tier</div>', unsafe_allow_html=True)

    tier_df = pd.DataFrame(data["tiers"])
    if not tier_df.empty:
        tier_df["tier_label"] = tier_df["tier"].map({1: "Tier 1 (Simple)", 2: "Tier 2 (Moderate)", 3: "Tier 3 (Complex)"})
        fig_tier = go.Figure()
        fig_tier.add_trace(go.Bar(
            name="Baseline Cost", x=tier_df["tier_label"],
            y=tier_df["baseline_cost"], marker_color="#f38ba8", opacity=0.7,
        ))
        fig_tier.add_trace(go.Bar(
            name="Actual Cost", x=tier_df["tier_label"],
            y=tier_df["total_cost"], marker_color="#a6e3a1",
        ))
        fig_tier.update_layout(
            height=260, barmode="group",
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(tickprefix="$", gridcolor="#313244"),
            xaxis=dict(gridcolor="#313244"),
        )
        st.plotly_chart(fig_tier, use_container_width=True)

# ---------------------------------------------------------------------------
# Tier stats table
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">Tier Breakdown</div>', unsafe_allow_html=True)

tier_df = pd.DataFrame(data["tiers"])
if not tier_df.empty:
    tier_df["tier_label"] = tier_df["tier"].map({1: "Tier 1 — Simple", 2: "Tier 2 — Moderate", 3: "Tier 3 — Complex"})
    tier_df["saved"] = tier_df["baseline_cost"] - tier_df["total_cost"]
    tier_df["pct_saved"] = ((tier_df["saved"] / tier_df["baseline_cost"].replace(0, 1)) * 100).round(1)
    tier_display = tier_df[["tier_label", "count", "avg_latency", "total_cost", "baseline_cost", "saved", "pct_saved", "avg_quality"]].copy()
    tier_display.columns = ["Tier", "Requests", "Avg Latency (ms)", "Actual Cost ($)", "Baseline Cost ($)", "Saved ($)", "Saved (%)", "Avg Quality"]
    st.dataframe(tier_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Recent requests log
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">Recent Requests</div>', unsafe_allow_html=True)

recent_df = pd.DataFrame(data["recent"])
if not recent_df.empty:
    recent_df["passed"] = recent_df["passed"].map({1: "✅", 0: "❌", None: "—"})
    recent_df["escalated"] = recent_df["escalated"].map({1: "⬆️", 0: "—", None: "—"})
    recent_df["tier"] = recent_df["tier"].map({1: "1 Simple", 2: "2 Moderate", 3: "3 Complex"})
    display_cols = ["timestamp", "prompt", "tier", "model", "latency_ms", "cost_usd", "baseline_cost", "quality_score", "passed", "escalated"]
    display_labels = ["Timestamp", "Prompt", "Tier", "Model", "Latency (ms)", "Cost ($)", "Baseline ($)", "Quality", "Pass", "Esc"]
    st.dataframe(
        recent_df[display_cols].rename(columns=dict(zip(display_cols, display_labels))),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"LLM Cost Autopilot · {stats['total_requests']:,} total requests · "
    f"${stats['saved_usd']:.4f} saved ({stats['pct_saved']}% reduction) · "
    f"Quality pass rate {stats['quality_pass_rate']}%"
)
