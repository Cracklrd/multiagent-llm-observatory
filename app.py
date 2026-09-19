"""Streamlit interface for the multi-agent LLM observatory."""

from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import load_settings
from tools import calculate_cost_breakdown, evaluate_quality_details, load_models, recommend_model


INK = "#171a1c"
TEAL = "#087f8c"
CORAL = "#e85d3f"
PAPER = "#f7f7f4"
WHITE = "#ffffff"
MUTED = "#697176"
GRID = "#d8d8d2"
MODEL_COLORS = [TEAL, CORAL, "#4c6e91", "#8a6f3d", "#6d597a", "#3f7d5c"]
DEFAULT_MISSION = (
    "Recommend the best simulated LLM for a workload with 120,000 input tokens and "
    "30,000 output tokens. Balance cost, latency, and quality."
)

st.set_page_config(page_title="LLM Observatory", page_icon="O", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {{ color-scheme: light; }}
    html, body, [class*="css"] {{ font-family: 'DM Sans', -apple-system, sans-serif; }}
    .stApp {{ background: {PAPER}; color: {INK}; }}
    .block-container {{ max-width: 1440px; padding: 2.1rem 3rem 4rem; }}
    header[data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu, footer {{ display: none !important; }}

    [data-testid="stSidebar"] {{
        background: #242b2e;
        border-right: 1px solid #394246;
        min-width: 270px;
    }}
    [data-testid="stSidebar"] .block-container {{ padding: 2rem 1.35rem; }}
    [data-testid="stSidebar"] h1 {{ color: #fff; font-size: 1.08rem; margin: 0; }}
    [data-testid="stSidebar"] p {{ color: #a9b0b3; }}
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0.35rem; }}
    [data-testid="stSidebar"] [data-baseweb="radio"] {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 0.7rem 0.75rem;
        transition: all 120ms ease;
    }}
    [data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {{
        background: #343e42;
        border-color: #4b575c;
    }}
    [data-testid="stSidebar"] [data-baseweb="radio"] label,
    [data-testid="stSidebar"] [data-baseweb="radio"] div {{ color: #edf0f1 !important; }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{ display: none; }}

    h1, h2, h3 {{ color: {INK}; letter-spacing: 0; }}
    h1 {{ font-size: clamp(2rem, 3vw, 3.15rem); line-height: 1.02; margin: 0.15rem 0 0.65rem; }}
    h2 {{ font-size: 1.32rem; margin-top: 1.8rem; }}
    h3 {{ font-size: 1rem; }}
    p {{ color: {MUTED}; }}

    .brand-lockup {{ border-bottom: 1px solid #353a3c; padding: 0 0.2rem 1.35rem; margin-bottom: 1.1rem; }}
    .brand-mark {{
        width: 34px; height: 34px; display: grid; place-items: center;
        border: 1px solid {CORAL}; color: {CORAL}; font: 600 0.82rem 'IBM Plex Mono';
        margin-bottom: 0.85rem;
    }}
    .brand-title {{ color: #fff; font-weight: 700; font-size: 1.05rem; }}
    .brand-subtitle {{ color: #858e92; font: 500 0.7rem 'IBM Plex Mono'; margin-top: 0.25rem; }}
    .sidebar-foot {{ position: fixed; bottom: 1.5rem; width: 215px; border-top: 1px solid #353a3c; padding-top: 1rem; }}
    .system-line {{ color: #dbe0e2; font: 500 0.72rem 'IBM Plex Mono'; }}
    .system-line::before {{
        content: ''; display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: #6fcf97; margin-right: 0.55rem; box-shadow: 0 0 0 3px rgba(111,207,151,.12);
    }}

    .eyebrow {{ color: {TEAL}; font: 600 0.72rem 'IBM Plex Mono'; text-transform: uppercase; margin-bottom: 0.65rem; }}
    .page-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 2rem; border-bottom: 1px solid {GRID}; padding-bottom: 1.55rem; margin-bottom: 1.6rem; }}
    .page-copy {{ max-width: 780px; }}
    .page-copy p {{ max-width: 690px; font-size: 1rem; line-height: 1.55; margin: 0; }}
    .phase-stamp {{
        border: 1px solid {GRID}; padding: 0.75rem 0.9rem; background: rgba(255,255,255,.45);
        font: 600 0.7rem 'IBM Plex Mono'; color: {MUTED}; white-space: nowrap;
    }}

    .metric-strip {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); border: 1px solid {GRID}; background: {WHITE}; margin: 1.25rem 0 1.7rem; }}
    .metric-cell {{ padding: 1rem 1.15rem; border-right: 1px solid {GRID}; min-width: 0; }}
    .metric-cell:last-child {{ border-right: 0; }}
    .metric-label {{ color: {MUTED}; font: 500 0.67rem 'IBM Plex Mono'; text-transform: uppercase; margin-bottom: 0.45rem; }}
    .metric-value {{ color: {INK}; font-size: 1.25rem; font-weight: 700; overflow-wrap: anywhere; }}
    .metric-accent {{ color: {TEAL}; }}
    .cost-summary {{ display: grid; grid-template-columns: 1fr 1fr; border: 1px solid {GRID}; background: {WHITE}; margin-top: 1rem; }}
    .cost-stat {{ padding: .9rem; min-width: 0; border-right: 1px solid {GRID}; border-bottom: 1px solid {GRID}; }}
    .cost-stat:nth-child(2n) {{ border-right: 0; }}
    .cost-stat:nth-child(n+3) {{ border-bottom: 0; }}
    .cost-stat strong {{ display: block; color: {INK}; font-size: 1.05rem; margin-top: .35rem; overflow-wrap: anywhere; }}
    .cost-stat:first-child strong {{ color: {TEAL}; }}
    .explain-band {{
        display: grid; grid-template-columns: 1.2fr repeat(3, .7fr); border-top: 1px solid {GRID};
        border-bottom: 1px solid {GRID}; margin: 1.35rem 0 1.6rem; background: rgba(255,255,255,.38);
    }}
    .explain-copy {{ padding: 1rem 1.1rem 1rem 0; }}
    .explain-copy strong {{ display: block; color: {INK}; font-size: .9rem; margin-bottom: .3rem; }}
    .explain-copy p {{ font-size: .78rem; line-height: 1.45; margin: 0; }}
    .explain-stat {{ padding: 1rem; border-left: 1px solid {GRID}; }}
    .explain-stat strong {{ display: block; color: {INK}; font-size: 1.05rem; margin-top: .35rem; }}
    .formula-panel {{ background: #eef6f5; border-left: 3px solid {TEAL}; padding: .9rem 1rem; margin-top: 1rem; }}
    .formula-panel code {{ display: inline; color: #25565b; font: 500 .68rem/1.65 'IBM Plex Mono'; white-space: normal; }}
    .formula-panel p {{ margin: .55rem 0 0; font-size: .76rem; line-height: 1.45; }}
    .dataset-note {{
        display: flex; gap: 1.5rem; align-items: baseline; border-left: 3px solid {TEAL};
        padding: .75rem 1rem; margin: 1rem 0 1.35rem; background: #eef6f5;
    }}
    .dataset-note strong {{ color: {INK}; white-space: nowrap; }}
    .dataset-note span {{ color: {MUTED}; font-size: .82rem; line-height: 1.45; }}

    .section-label {{ display: flex; align-items: center; gap: .6rem; color: {INK}; font: 600 0.75rem 'IBM Plex Mono'; text-transform: uppercase; margin: 1.55rem 0 .8rem; }}
    .section-label::after {{ content: ''; flex: 1; height: 1px; background: {GRID}; }}
    .agent-flow {{ display: grid; grid-template-columns: repeat(5, 1fr); margin: 1rem 0 1.8rem; border: 1px solid {GRID}; background: {WHITE}; }}
    .agent-node {{ position: relative; padding: 1rem; border-right: 1px solid {GRID}; min-height: 92px; }}
    .agent-node:last-child {{ border-right: 0; }}
    .agent-index {{ color: {CORAL}; font: 600 .66rem 'IBM Plex Mono'; }}
    .agent-name {{ color: {INK}; font-weight: 700; margin-top: .5rem; }}
    .agent-role {{ color: {MUTED}; font-size: .74rem; margin-top: .2rem; }}

    [data-testid="stForm"] {{
        border: 1px solid #dfe2e0; border-radius: 4px; background: {WHITE}; padding: 1.25rem;
        box-shadow: 0 8px 24px rgba(30, 39, 43, .045);
    }}
    [data-testid="stTextArea"] [data-baseweb="base-input"],
    [data-testid="stTextInput"] [data-baseweb="base-input"],
    [data-testid="stNumberInput"] [data-baseweb="base-input"],
    .stTextArea > div,
    .stTextInput > div > div,
    [data-baseweb="textarea"] textarea,
    [data-baseweb="input"] input {{
        background: #ffffff !important; color: {INK} !important; border-radius: 3px;
        -webkit-text-fill-color: {INK};
    }}
    [data-testid="stTextArea"] [data-baseweb="base-input"],
    [data-testid="stTextInput"] [data-baseweb="base-input"] {{ border: 1px solid #dfe3e5; }}
    [data-baseweb="select"] > div {{ background: {WHITE}; border-radius: 3px; border-color: #dfe3e5; }}
    input:disabled {{ background: #f5f6f6 !important; -webkit-text-fill-color: #899196 !important; }}
    .stButton > button, [data-testid="stFormSubmitButton"] > button {{
        background: {CORAL}; color: white; border: 1px solid {CORAL}; border-radius: 3px;
        min-height: 2.75rem; font-weight: 700; letter-spacing: 0;
    }}
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{ background: #c9472f; border-color: #c9472f; color: white; }}
    .stButton > button p, [data-testid="stFormSubmitButton"] > button p {{ color: white !important; }}
    [data-testid="stExpander"] {{ background: {WHITE}; border: 1px solid {GRID}; border-radius: 0; margin-bottom: .45rem; }}
    [data-testid="stMetric"] {{ background: {WHITE}; border: 1px solid {GRID}; border-radius: 0; padding: 1rem; }}
    [data-testid="stPlotlyChart"] {{ background: {WHITE}; border: 1px solid {GRID}; padding: .5rem; }}
    [data-testid="stDataFrame"] {{ border: 1px solid {GRID}; }}
    [data-testid="stAlert"] {{ border-radius: 3px; }}
    [data-baseweb="tab-list"] {{ gap: 1.3rem; border-bottom: 1px solid {GRID}; }}
    [data-baseweb="tab"] {{ padding-left: 0; padding-right: 0; }}

    .result-block {{
        background: {WHITE}; padding: 1.4rem 1.5rem; border: 1px solid {GRID};
        border-left: 4px solid {CORAL}; margin: 1rem 0 1.25rem;
    }}
    .result-block .result-kicker {{ color: {TEAL}; font: 600 .68rem 'IBM Plex Mono'; text-transform: uppercase; }}
    .result-block .result-copy {{ color: {INK}; font-size: 1rem; line-height: 1.65; margin-top: .7rem; }}
    .token {{ display: inline-block; border: 1px solid {GRID}; background: {WHITE}; padding: .32rem .5rem; margin: .2rem .2rem .2rem 0; font: 500 .74rem 'IBM Plex Mono'; }}
    .token.missing {{ border-color: #e8b5a9; color: #a33c28; background: #fff7f4; }}

    @media (max-width: 900px) {{
        .block-container {{ padding: 1.5rem 1rem 3rem; }}
        .page-head {{ display: block; }} .phase-stamp {{ display: inline-block; margin-top: 1rem; }}
        .metric-strip {{ grid-template-columns: repeat(2, 1fr); }}
        .metric-cell:nth-child(2) {{ border-right: 0; }}
        .metric-cell:nth-child(-n+2) {{ border-bottom: 1px solid {GRID}; }}
        .agent-flow {{ grid-template-columns: repeat(5, 130px); overflow-x: auto; overscroll-behavior-inline: contain; }}
        .agent-node {{ border-right: 1px solid {GRID}; border-bottom: 0; min-height: 92px; }}
        .agent-node:last-child {{ border-right: 0; }}
        .agent-role {{ white-space: nowrap; }}
        h1 {{ font-size: 2.35rem; }}
        .explain-band {{ grid-template-columns: repeat(3, 1fr); }}
        .explain-copy {{ grid-column: 1 / -1; padding: 1rem 0; }}
        .explain-stat:first-of-type {{ border-left: 0; }}
        .dataset-note {{ display: block; }}
        .dataset-note span {{ display: block; margin-top: .35rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def model_table() -> pd.DataFrame:
    return pd.DataFrame(model.to_dict() for model in load_models())


def page_header(index: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <div class="page-copy">
            <div class="eyebrow">Workspace {index} / LLM operations</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
          </div>
          <div class="phase-stamp">BUILD 03 · PHASE 3</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_strip(metrics: list[tuple[str, str, bool]]) -> None:
    cells = "".join(
        f'<div class="metric-cell"><div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value{" metric-accent" if accent else ""}">{escape(value)}</div></div>'
        for label, value, accent in metrics
    )
    st.markdown(f'<div class="metric-strip">{cells}</div>', unsafe_allow_html=True)


def style_figure(figure: go.Figure, height: int = 410) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=35, r=25, t=35, b=35),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="DM Sans", color=INK),
        legend_title_text="",
        hoverlabel=dict(bgcolor=INK, font_color=WHITE, bordercolor=INK),
    )
    figure.update_xaxes(gridcolor="#e8e7e2", zeroline=False)
    figure.update_yaxes(gridcolor="#e8e7e2", zeroline=False)
    return figure


models = load_models()
models_by_name = {model.model_name: model for model in models}
table = model_table()
settings = load_settings()

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">OBS</div>
          <div class="brand-title">LLM Observatory</div>
          <div class="brand-subtitle">MULTI-AGENT CONTROL SYSTEM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Workspace",
        ["Command center", "Model field", "Cost lab", "Quality gate"],
        label_visibility="collapsed",
    )
    mode_text = "OPENAI READY" if settings["openai_api_key_available"] else "LOCAL DEMO"
    st.markdown(
        f'<div class="sidebar-foot"><div class="system-line">SYSTEM ONLINE</div>'
        f'<div class="brand-subtitle" style="margin-top:.55rem">{mode_text} · SQLITE READY</div></div>',
        unsafe_allow_html=True,
    )

if page == "Command center":
    page_header(
        "01",
        "Multi-agent control room",
        "Dispatch one optimization mission through five specialized agents and inspect every observable handoff.",
    )
    st.markdown(
        """
        <div class="agent-flow">
          <div class="agent-node"><div class="agent-index">01</div><div class="agent-name">Coordinator</div><div class="agent-role">Scope and sequence</div></div>
          <div class="agent-node"><div class="agent-index">02</div><div class="agent-name">Researcher</div><div class="agent-role">Local evidence</div></div>
          <div class="agent-node"><div class="agent-index">03</div><div class="agent-name">Analyst</div><div class="agent-role">Cost and trade-offs</div></div>
          <div class="agent-node"><div class="agent-index">04</div><div class="agent-name">Verifier</div><div class="agent-role">Quality control</div></div>
          <div class="agent-node"><div class="agent-index">05</div><div class="agent-name">Recommender</div><div class="agent-role">Final decision</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_available = bool(settings["openai_api_key_available"])
    with st.form("mission_form"):
        form_left, form_right = st.columns([2.2, 1], gap="large")
        with form_left:
            mission = st.text_area("Mission brief", value=DEFAULT_MISSION, height=132)
        with form_right:
            use_openai = st.toggle(
                "Live OpenAI execution",
                value=live_available and not bool(settings["demo_mode"]),
                disabled=not live_available,
            )
            openai_model = st.text_input(
                "Execution model",
                value=str(settings["openai_model"]),
                disabled=not use_openai,
            )
            submitted = st.form_submit_button("Dispatch mission", width="stretch")

    if not live_available:
        st.caption("LOCAL DEMO ACTIVE · Add OPENAI_API_KEY to .env to unlock live execution.")

    st.markdown(
        """
        <div class="explain-band">
          <div class="explain-copy">
            <strong>What the token numbers mean</strong>
            <p>The default brief models a context-heavy application. These are comparison assumptions, not tokens consumed by the CSV dataset or by this page.</p>
          </div>
          <div class="explain-stat"><div class="metric-label">Local dataset</div><strong>0 tokens</strong></div>
          <div class="explain-stat"><div class="metric-label">Input assumption</div><strong>120,000 · 80%</strong></div>
          <div class="explain-stat"><div class="metric-label">Output assumption</div><strong>30,000 · 20%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if submitted:
        try:
            with st.spinner("Crew active · processing five sequential stages"):
                from core.crew import run_demo_crew, run_live_crew

                execution = run_live_crew(mission, openai_model) if use_openai else run_demo_crew(mission)
            st.session_state["crew_execution"] = execution
        except Exception as error:
            st.error(f"Execution failed: {error}")

    execution = st.session_state.get("crew_execution")
    if execution:
        metric_strip(
            [
                ("Run mode", execution.mode.upper(), True),
                ("Execution model", execution.model_name, False),
                ("Completed stages", f"{len(execution.task_outputs)} / 5", False),
                ("Run status", "SUCCESS", True),
            ]
        )
        st.markdown('<div class="section-label">Decision output</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-block"><div class="result-kicker">Verified recommendation</div>'
            f'<div class="result-copy">{escape(execution.final_output)}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Agent handoffs</div>', unsafe_allow_html=True)
        agent_names = ["01 · Coordinator", "02 · Researcher", "03 · Analyst", "04 · Verifier", "05 · Recommender"]
        for agent_name, output in zip(agent_names, execution.task_outputs, strict=True):
            with st.expander(agent_name, expanded=agent_name.startswith("05")):
                st.markdown(output)
        if execution.usage:
            with st.expander("API usage ledger"):
                st.json(execution.usage)

elif page == "Model field":
    page_header(
        "02",
        "Model decision field",
        "Read the catalog as a trade-off map. Bubble size represents combined token price; position reveals speed and quality.",
    )
    priority = st.segmented_control(
        "Optimization priority",
        options=["cost", "speed", "balanced", "quality"],
        default="balanced",
        format_func=str.title,
    )
    st.markdown(
        f"""
        <div class="dataset-note">
          <strong>Dataset · {len(models)} local profiles</strong>
          <span><code>data/models.csv</code> stores simulated price, latency, and quality values. Reading it is local and uses zero API tokens. Prices are normalized per one million tokens so different workloads remain directly comparable.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    recommendation = recommend_model(priority or "balanced")
    metric_strip(
        [
            ("Recommended", recommendation.model_name, True),
            ("Quality score", f"{recommendation.quality_score:.0f} / 100", False),
            ("Average latency", f"{recommendation.avg_latency:.1f} seconds", False),
            ("Combined price", f"${recommendation.input_price + recommendation.output_price:.2f} / 1M", False),
        ]
    )

    chart = px.scatter(
        table,
        x="avg_latency",
        y="quality_score",
        size=table["input_price"] + table["output_price"],
        color="model_name",
        text="model_name",
        hover_data={"input_price": ":.2f", "output_price": ":.2f"},
        labels={"avg_latency": "Average latency (seconds)", "quality_score": "Quality score", "model_name": "Model"},
        color_discrete_sequence=MODEL_COLORS,
        size_max=42,
    )
    chart.update_traces(textposition="top center", marker=dict(line=dict(color=WHITE, width=2)))
    style_figure(chart, 465)
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

    display_table = table.rename(
        columns={"model_name": "Model", "input_price": "Input $/1M", "output_price": "Output $/1M", "avg_latency": "Latency (s)", "quality_score": "Quality /100"}
    )
    st.dataframe(
        display_table.style.highlight_max(subset=["Quality /100"], color="#dff0e8").format(
            {"Input $/1M": "${:.2f}", "Output $/1M": "${:.2f}", "Latency (s)": "{:.1f}", "Quality /100": "{:.0f}"}
        ),
        width="stretch",
        hide_index=True,
    )

elif page == "Cost lab":
    page_header(
        "03",
        "Token cost laboratory",
        "Stress-test one workload against every simulated model price and expose the input/output cost split.",
    )
    control_col, chart_col = st.columns([1, 2], gap="large")
    with control_col:
        st.markdown('<div class="section-label">Workload</div>', unsafe_allow_html=True)
        model_name = st.selectbox("Reference model", list(models_by_name), index=3)
        input_tokens = st.number_input("Input tokens", min_value=0, value=120_000, step=5_000)
        output_tokens = st.number_input("Output tokens", min_value=0, value=30_000, step=1_000)
        breakdown = calculate_cost_breakdown(model_name, int(input_tokens), int(output_tokens))
        st.markdown(
            f"""
            <div class="cost-summary">
              <div class="cost-stat"><div class="metric-label">Total cost</div><strong>${breakdown.total_cost:,.6f}</strong></div>
              <div class="cost-stat"><div class="metric-label">Total tokens</div><strong>{breakdown.input_tokens + breakdown.output_tokens:,}</strong></div>
              <div class="cost-stat"><div class="metric-label">Input cost</div><strong>${breakdown.input_cost:,.6f}</strong></div>
              <div class="cost-stat"><div class="metric-label">Output cost</div><strong>${breakdown.output_cost:,.6f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        total_tokens = breakdown.input_tokens + breakdown.output_tokens
        input_share = breakdown.input_tokens / max(total_tokens, 1)
        if input_share >= 0.7:
            workload_shape = "context-heavy: most tokens are documents, prompts, or prior context sent to the model"
        elif input_share >= 0.4:
            workload_shape = "balanced: input context and generated output have similar weight"
        else:
            workload_shape = "generation-heavy: the model produces more text than it receives"
        selected_model = models_by_name[model_name]
        st.markdown(
            f"""
            <div class="formula-panel">
              <div class="metric-label">Why this amount</div>
              <code>{breakdown.input_tokens:,} / 1,000,000 × ${selected_model.input_price:.2f} = ${breakdown.input_cost:.6f} input</code><br>
              <code>{breakdown.output_tokens:,} / 1,000,000 × ${selected_model.output_price:.2f} = ${breakdown.output_cost:.6f} output</code>
              <p>This workload is {input_share:.0%} input and {1 - input_share:.0%} output, so it is {workload_shape}. The values are adjustable simulation inputs; actual OpenAI usage is reported separately after a live crew run.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    comparison_rows = []
    for model in models:
        result = calculate_cost_breakdown(model.model_name, int(input_tokens), int(output_tokens))
        comparison_rows.append({"Model": model.model_name, "Estimated cost": result.total_cost})
    comparison = pd.DataFrame(comparison_rows).sort_values("Estimated cost")
    with chart_col:
        st.markdown('<div class="section-label">Cross-model estimate</div>', unsafe_allow_html=True)
        cost_chart = px.bar(
            comparison,
            x="Estimated cost",
            y="Model",
            orientation="h",
            color="Model",
            text="Estimated cost",
            color_discrete_sequence=MODEL_COLORS,
        )
        cost_chart.update_traces(texttemplate="$%{text:.5f}", textposition="outside", marker_line_width=0)
        cost_chart.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        style_figure(cost_chart, 430)
        st.plotly_chart(cost_chart, width="stretch", config={"displayModeBar": False})

else:
    page_header(
        "04",
        "Deterministic quality gate",
        "Audit generated text with a repeatable coverage rule and make every matched or missing concept visible.",
    )
    input_col, score_col = st.columns([1.55, 1], gap="large")
    with input_col:
        keywords_text = st.text_input("Expected concepts", value="cost, latency, quality, tokens")
        generated_text = st.text_area(
            "Generated response",
            value="The recommended model reduces cost and latency while preserving quality.",
            height=205,
        )
    keywords = [keyword.strip() for keyword in keywords_text.split(",") if keyword.strip()]
    evaluation = evaluate_quality_details(keywords, generated_text)

    with score_col:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=evaluation.score,
                number={"suffix": " / 100", "font": {"size": 34, "color": INK}},
                gauge={
                    "axis": {"range": [0, 100], "visible": False},
                    "bar": {"color": TEAL},
                    "bgcolor": "#e7e6e1",
                    "borderwidth": 0,
                    "steps": [{"range": [0, 60], "color": "#f2d9d2"}, {"range": [60, 85], "color": "#efe5c8"}, {"range": [85, 100], "color": "#d9ebe2"}],
                },
            )
        )
        gauge.update_layout(height=235, margin=dict(l=30, r=30, t=25, b=15), paper_bgcolor=WHITE)
        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="section-label">Coverage evidence</div>', unsafe_allow_html=True)
    matched = "".join(f'<span class="token">FOUND · {escape(item)}</span>' for item in evaluation.matched_keywords)
    missing = "".join(f'<span class="token missing">MISSING · {escape(item)}</span>' for item in evaluation.missing_keywords)
    st.markdown(matched + missing or '<span class="token missing">NO EXPECTED CONCEPTS</span>', unsafe_allow_html=True)
