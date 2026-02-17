import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
import requests
import numpy as np

# Page Config
st.set_page_config(
    page_title="AI Financial Foresight",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }
    .highlight {
        color: #38bdf8;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(128, 128, 128, 0.1);
        color: #38bdf8;
    }
    .chart-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

# Import Local Engine
from engines.benchmarking_engine import BenchmarkingEngine
from engines.competitor_discovery import CompetitorDiscoveryEngine
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

@st.cache_data(show_spinner=False)
def get_data(ticker_list):
    """Fetch benchmark data using the local engine."""
    engine = BenchmarkingEngine()
    try:
        return engine.run_benchmark(ticker_list, detailed=True)
    except Exception as e:
        st.error(f"Engine Error: {e}")
        return {}

def get_financial_statement(company_data, stmt_type):
    """Extracts a specific financial statement as a DataFrame."""
    try:
        raw = company_data.get('raw_data', {}).get('financials', {}).get('annual', {})
        stmt_data = raw.get(stmt_type, {})
        
        if not stmt_data:
            return pd.DataFrame()
            
        # Structure: { "Line Item": { "Date": Value, ... } }
        df = pd.DataFrame(stmt_data)
        # Transpose so Dates are columns, Line Items are rows (Yahoo Style)
        # But first, let's sort columns by date descending
        df = df.T 
        # Sort columns (dates) descending
        df = df[sorted(df.columns, reverse=True)]
        return df
    except Exception as e:
        return pd.DataFrame()

def find_value_rec(data, target_key):
    """Recursively search for a key in a nested dict."""
    if target_key in data:
        return data[target_key]
    for k, v in data.items():
        if isinstance(v, dict):
            res = find_value_rec(v, target_key)
            if res is not None:
                return res
    return None

def get_value_for_ticker(companies, ticker, key):
    """Smart lookup for a value (searches metrics first, then raw data flat search)."""
    c = next((x for x in companies if x['ticker'] == ticker), None)
    if not c: return 0
    
    # 1. Check computed metrics
    if key in c.get('metrics', {}):
        return c['metrics'][key]
        
    # 2. Key might be a specific raw path? No, let's do a "Leaf Search"
    # We assume 'key' is the leaf name (e.g. "Total Revenue")
    # This is expensive but flexible
    val = find_value_rec(c.get('raw_data', {}), key)
    
    # Value might be a dict of {Date: Val}. If so, get the most recent.
    if isinstance(val, dict):
        try:
            dates = sorted(val.keys(), reverse=True)
            return val[dates[0]]
        except:
            return 0
            
    return val if val is not None else 0

def generate_analyst_note(focus_company, companies):
    """Generates a rule-based executive summary."""
    ticker = focus_company['ticker']
    m = focus_company['metrics']
    
    notes = []
    
    # cohort stats
    avg_rev_cagr = np.mean([c['metrics'].get('Revenue CAGR (3y)', 0) for c in companies])
    avg_gross_margin = np.mean([c['metrics'].get('Gross Margin %', 0) for c in companies])
    avg_roic = np.mean([c['metrics'].get('ROIC %', 0) for c in companies])
    
    # Growth
    cagr = m.get('Revenue CAGR (3y)', 0)
    if cagr > avg_rev_cagr + 5:
        notes.append(f"**High Growth**: Revenue CAGR of {cagr}% significantly outperforms the cohort average ({avg_rev_cagr:.1f}%).")
    elif cagr < avg_rev_cagr - 5:
        notes.append(f"**Lagging Growth**: Revenue growth ({cagr}%) is below the cohort average.")

    # Profitability
    gm = m.get('Gross Margin %', 0)
    if gm > avg_gross_margin + 5:
        notes.append(f"**Strong Pricing Power**: Gross Margins ({gm}%) are superior to peers.")
    
    # Efficiency
    roic = m.get('ROIC %', 0)
    if roic > avg_roic + 2:
        notes.append(f"**Capital Efficient**: ROIC of {roic}% indicates superior capital allocation.")
    elif roic < avg_roic - 2:
        notes.append(f"**Capital Inefficient**: ROIC ({roic}%) lags the cohort.")

    # Leverage
    lev = m.get('Net Debt / EBITDA', 0)
    if lev > 3.0:
        notes.append(f"**Highly Levered**: Net Debt/EBITDA is {lev}x, suggesting potential solvency risks.")
    elif lev < 0:
        notes.append(f"**Fortress Balance Sheet**: Company holds more cash than debt.")

    if not notes:
        return f"{ticker} performs in line with the cohort across major metrics."
        
    return " ".join(notes)

# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
st.sidebar.title("Configuration")

# API Key Config (Sidebar)
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Required for Competitor Discovery")
if not api_key:
    # Try getting from env
    api_key = os.getenv("OPENAI_API_KEY")

# Ticker Management (with Session State)
if 'ticker_input_str' not in st.session_state:
    st.session_state['ticker_input_str'] = "MSFT, AAPL, AMZN, GOOGL"

tickers_text = st.sidebar.text_area("Tickers", key="ticker_input_str")

# --- Competitor Discovery (AI Powered) ---
with st.sidebar.expander("🤖 Find Competitors (AI)", expanded=False):
    target_ticker = st.text_input("Target Company Ticker", placeholder="e.g. TSLA")
    
    if st.button("Find Competitors"):
        if not api_key:
            st.error("Please provide an OpenAI API Key.")
        elif not target_ticker:
            st.error("Enter a target ticker.")
        else:
            with st.spinner(f"Analyzing competitors for {target_ticker}..."):
                discovery_engine = CompetitorDiscoveryEngine(api_key=api_key)
                result = discovery_engine.find_competitors(target_ticker)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.session_state['competitor_results'] = result
                    st.success("Analysis Complete!")

    if 'competitor_results' in st.session_state:
        res = st.session_state['competitor_results']
        
        st.markdown("**Direct Competitors**")
        selected_direct = []
        for comp in res.get("direct", []):
            if st.checkbox(f"{comp['ticker']} ({comp['name']})", key=f"chk_{comp['ticker']}"):
                selected_direct.append(comp['ticker'])
        
        st.markdown("**Broad Competitors**")
        selected_broad = []
        for comp in res.get("broad", []):
            if st.checkbox(f"{comp['ticker']} ({comp['name']})", key=f"chk_{comp['ticker']}"):
                selected_broad.append(comp['ticker'])
                
        def add_selected_competitors():
            """Callback to update ticker list from selected checkboxes."""
            res = st.session_state.get('competitor_results', {})
            new_tickers = []
            
            # Collect from Direct
            for comp in res.get("direct", []):
                if st.session_state.get(f"chk_{comp['ticker']}", False):
                    new_tickers.append(comp['ticker'])
            
            # Collect from Broad
            for comp in res.get("broad", []):
                if st.session_state.get(f"chk_{comp['ticker']}", False):
                    new_tickers.append(comp['ticker'])
            
            # Merge with existing
            current_str = st.session_state.get('ticker_input_str', "")
            current_list = [t.strip() for t in current_str.split(",") if t.strip()]
            
            for t in new_tickers:
                if t not in current_list:
                    current_list.append(t)
            
            st.session_state['ticker_input_str'] = ", ".join(current_list)

        st.button("Add Selected to Analysis", on_click=add_selected_competitors)

if st.sidebar.button("Run Analysis", type="primary"):
    st.session_state['run_requested'] = True

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear Cache & Reload"):
    st.cache_data.clear()
st.sidebar.caption("v3.3 - Sandbox Fixed")


# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------
if st.session_state.get('run_requested', False):
    if not tickers_text:
        st.error("Enter tickers.")
    else:
        t_list = [t.strip() for t in tickers_text.split(",") if t.strip()]
        with st.spinner("Requesting analysis from backend..."):
            data = get_data(t_list)
            if data:
                st.session_state['data'] = data
                st.session_state['run_requested'] = False

if 'data' in st.session_state:
    data = st.session_state['data']
    companies = data.get("companies", [])
    if not companies:
        st.warning("No company data returned.")
    else:
        ticker_list = [c['ticker'] for c in companies]
        
        # Focus Selector
        focus_ticker = st.sidebar.selectbox("Main Focus", options=ticker_list)
        focus_company = next(c for c in companies if c['ticker'] == focus_ticker)
        
        # Header
        st.title("AI Financial Foresight")
        
        # Analyst Note
        note = generate_analyst_note(focus_company, companies)
        st.info(f"🤖 **Analyst Insight**: {note}")
        
        # --- Tabs ---
    tabs = st.tabs(["🏆 Strategic Comparison", "💎 DuPont Analysis", "📊 Financial Statements", "🧪 Advanced Sandbox", "📑 Deep Dive"])
    
    # 1. Strategic Comparison (Heatmap + Charts)
    with tabs[0]:
        st.subheader("Comparative Analysis")
        
        # Focus Selector — dynamically add Custom KPIs if any have been saved
        focus_options = ["Overview", "Liquidity", "Solvency", "Efficiency", "Returns"]
        
        # Define Metrics per Focus
        metric_groups = {
            "Overview": ["Revenue ($B)", "Revenue CAGR (3y)", "Gross Margin %", "EBITDA Margin %", "Net Margin %", "ROIC %", "CCC (Days)"],
            "Liquidity": ["Current Ratio", "Quick Ratio", "Cash Ratio", "CCC (Days)", "DIO (Days)", "DSO (Days)", "DPO (Days)"],
            "Solvency": ["Net Debt / EBITDA", "Interest Coverage", "Debt / Equity", "Financial Leverage", "Quick Ratio"],
            "Efficiency": ["Asset Turnover", "ROIC %", "ROE %", "Fixed Asset Turnover", "CCC (Days)"],
            "Returns": ["ROE %", "ROIC %", "Shareholder Yield %", "Net Margin %", "Dupont ROE"]
        }
        
        # Add Custom KPIs focus mode if any saved
        saved_kpis = st.session_state.get('saved_sandbox_kpis', {})
        if saved_kpis:
            metric_groups["📌 Custom KPIs"] = list(saved_kpis.keys())
            focus_options.append("📌 Custom KPIs")
        
        focus_mode = st.radio("Benchmark Focus", focus_options, horizontal=True)
        
        # Fallback to Overview if key missing
        metrics_to_show = metric_groups.get(focus_mode, metric_groups["Overview"])
        
        # A. Heatmap Matrix
        with st.expander(f"Heatmap View ({focus_mode})", expanded=False):
            matrix = []
            for c in companies:
                row = {"Ticker": c['ticker']}
                for m in metrics_to_show:
                    # Handle missing keys gracefully
                    row[m] = c['metrics'].get(m, 0)
                matrix.append(row)
            
            df_heat = pd.DataFrame(matrix).set_index("Ticker")
            st.dataframe(
                df_heat.style.background_gradient(cmap="RdYlGn", axis=0), 
                use_container_width=True
            )
            
        # B. Individual KPI Graphics
        st.markdown(f"### Key {focus_mode} Indicators")
        
        view_mode = st.radio("View", ["📊 Bar Charts", "🎯 Gauge View"], horizontal=True, key="chart_view_mode")
        
        if view_mode == "📊 Bar Charts":
            # --- Bar Chart View (original) ---
            cols = st.columns(3)
            for i, metric in enumerate(metrics_to_show):
                col = cols[i % 3]
                with col:
                    chart_data = []
                    for c in companies:
                        chart_data.append({
                            "Ticker": c['ticker'],
                            "Value": c['metrics'].get(metric, 0),
                            "Color": '#38bdf8' if c['ticker'] == focus_ticker else '#334155'
                        })
                    df_chart = pd.DataFrame(chart_data)
                    
                    fig = px.bar(
                        df_chart, x="Ticker", y="Value", color="Ticker",
                        color_discrete_map={row['Ticker']: row['Color'] for _, row in df_chart.iterrows()},
                        title=metric
                    )
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=250,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(showgrid=True, gridcolor='#334155'),
                        xaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            # --- Gauge View (Industry Benchmark style) ---
            # Metrics where LOWER is better (inverted color logic)
            lower_is_better = {"CCC (Days)", "DIO (Days)", "DSO (Days)", "DPO (Days)", 
                               "Net Debt / EBITDA", "Debt / Equity", "Financial Leverage"}
            
            cols = st.columns(3)
            for i, metric in enumerate(metrics_to_show):
                col = cols[i % 3]
                with col:
                    # Get focus company value
                    focus_val = focus_company['metrics'].get(metric, 0)
                    if focus_val is None: focus_val = 0
                    
                    # Get cohort values for range and average
                    cohort_vals = [c['metrics'].get(metric, 0) for c in companies]
                    cohort_vals = [v for v in cohort_vals if v is not None]
                    if not cohort_vals:
                        cohort_vals = [0]
                    
                    cohort_avg = np.mean(cohort_vals)
                    cohort_min = min(cohort_vals)
                    cohort_max = max(cohort_vals)
                    
                    # Build dynamic range with some padding
                    range_pad = max(abs(cohort_max - cohort_min) * 0.3, abs(cohort_avg) * 0.2, 0.5)
                    gauge_min = min(cohort_min, focus_val) - range_pad
                    gauge_max = max(cohort_max, focus_val) + range_pad
                    
                    # For ratio metrics that should start at 0
                    if "Ratio" in metric or "Turnover" in metric or "Coverage" in metric:
                        gauge_min = max(0, gauge_min)
                    
                    delta_val = focus_val - cohort_avg
                    
                    # Color steps based on whether higher or lower is better
                    if metric in lower_is_better:
                        bar_color = "#4ade80" if focus_val <= cohort_avg else "#f87171"
                        steps = [
                            {"range": [gauge_min, gauge_min + (gauge_max - gauge_min) * 0.33], "color": "rgba(74, 222, 128, 0.15)"},
                            {"range": [gauge_min + (gauge_max - gauge_min) * 0.33, gauge_min + (gauge_max - gauge_min) * 0.66], "color": "rgba(250, 204, 21, 0.15)"},
                            {"range": [gauge_min + (gauge_max - gauge_min) * 0.66, gauge_max], "color": "rgba(248, 113, 113, 0.15)"},
                        ]
                    else:
                        bar_color = "#4ade80" if focus_val >= cohort_avg else "#f87171"
                        steps = [
                            {"range": [gauge_min, gauge_min + (gauge_max - gauge_min) * 0.33], "color": "rgba(248, 113, 113, 0.15)"},
                            {"range": [gauge_min + (gauge_max - gauge_min) * 0.33, gauge_min + (gauge_max - gauge_min) * 0.66], "color": "rgba(250, 204, 21, 0.15)"},
                            {"range": [gauge_min + (gauge_max - gauge_min) * 0.66, gauge_max], "color": "rgba(74, 222, 128, 0.15)"},
                        ]
                    
                    # Format number for display
                    if abs(focus_val) >= 100:
                        num_fmt = f"{focus_val:,.0f}"
                    elif abs(focus_val) >= 1:
                        num_fmt = f"{focus_val:.2f}"
                    else:
                        num_fmt = f"{focus_val:.3f}"
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=focus_val,
                        title={"text": f"<b>{metric}</b><br><span style='font-size:11px;color:gray'>{focus_ticker}</span>", "font": {"size": 14}},
                        number={"font": {"size": 28}},
                        delta={
                            "reference": cohort_avg,
                            "relative": False,
                            "valueformat": ".2f",
                            "increasing": {"color": "#4ade80" if metric not in lower_is_better else "#f87171"},
                            "decreasing": {"color": "#f87171" if metric not in lower_is_better else "#4ade80"},
                            "suffix": " vs avg",
                            "font": {"size": 12}
                        },
                        gauge={
                            "axis": {"range": [gauge_min, gauge_max], "tickfont": {"size": 10}},
                            "bar": {"color": bar_color, "thickness": 0.3},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 0,
                            "steps": steps,
                            "threshold": {
                                "line": {"color": "#f8fafc", "width": 2},
                                "thickness": 0.8,
                                "value": cohort_avg
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=200,
                        margin=dict(l=20, r=20, t=50, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#e2e8f0"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # 2. DuPont Analysis (Redesigned)
    with tabs[1]:
        st.subheader(f"💎 DuPont Analysis: {focus_ticker}")
        st.caption("Decomposing ROE into its three fundamental drivers: Profitability, Efficiency, and Leverage.")
        
        # --- Data Prep ---
        m = focus_company['metrics']
        net_margin = m.get('Net Margin %', 0)
        asset_turnover = m.get('Asset Turnover', 0)
        leverage = m.get('Financial Leverage', 1)
        roe = m.get('ROE %', 0)
        dupont_roe = m.get('DuPont ROE', 0)
        
        # Cohort averages for delta comparison
        avg_nm = np.mean([c['metrics'].get('Net Margin %', 0) for c in companies])
        avg_at = np.mean([c['metrics'].get('Asset Turnover', 0) for c in companies])
        avg_lev = np.mean([c['metrics'].get('Financial Leverage', 1) for c in companies])
        avg_roe = np.mean([c['metrics'].get('ROE %', 0) for c in companies])
        
        # --- A. Hero Metric Cards with Deltas ---
        st.markdown("### 📊 Driver Decomposition")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("ROE %", f"{roe:.1f}%", delta=f"{roe - avg_roe:+.1f}% vs avg", delta_color="normal")
        mc2.metric("DuPont ROE", f"{dupont_roe:.1f}%", help="Net Margin × Asset Turnover × Leverage")
        mc3.metric("Net Margin %", f"{net_margin:.1f}%", delta=f"{net_margin - avg_nm:+.1f}% vs avg", delta_color="normal", help="Profitability: Net Income / Revenue")
        mc4.metric("Asset Turnover", f"{asset_turnover:.2f}x", delta=f"{asset_turnover - avg_at:+.2f}x vs avg", delta_color="normal", help="Efficiency: Revenue / Total Assets")
        mc5.metric("Fin. Leverage", f"{leverage:.2f}x", delta=f"{leverage - avg_lev:+.2f}x vs avg", delta_color="inverse", help="Leverage: Total Assets / Equity (lower = safer)")
        
        

        # --- D. Strategic Positioning Scatter (Enhanced) ---
        st.markdown("### 🎯 Strategic Positioning Map")
        st.caption("High Margin (Premium/IP-driven) vs High Turnover (Volume/Retail). Bubble size = Revenue.")
        
        scatter_data = []
        for c in companies:
            scatter_data.append({
                "Ticker": c['ticker'],
                "Net Margin %": c['metrics'].get('Net Margin %', 0),
                "Asset Turnover": c['metrics'].get('Asset Turnover', 0),
                "ROE %": c['metrics'].get('ROE %', 0),
                "Revenue ($B)": max(c['metrics'].get('Revenue ($B)', 0.1), 0.1),
            })
        
        df_scatter = pd.DataFrame(scatter_data)
        
        fig_dupont = px.scatter(
            df_scatter,
            x="Asset Turnover",
            y="Net Margin %",
            size="Revenue ($B)",
            color="ROE %",
            text="Ticker",
            hover_data=["ROE %", "Revenue ($B)"],
            title="",
            color_continuous_scale="RdYlGn",
            size_max=60
        )
        
        # Add quadrant lines (cohort averages)
        fig_dupont.add_hline(y=avg_nm, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text=f"Avg Margin: {avg_nm:.1f}%")
        fig_dupont.add_vline(x=avg_at, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text=f"Avg Turnover: {avg_at:.2f}x")
        
        fig_dupont.update_traces(textposition='top center', textfont=dict(size=12))
        fig_dupont.update_layout(
            height=550,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title="Asset Turnover (Efficiency) →", zeroline=True),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title="← Net Margin % (Profitability)"),
            coloraxis_colorbar=dict(title="ROE %")
        )
        st.plotly_chart(fig_dupont, use_container_width=True)

    # 3. Financial Statements (Yahoo Style)
    with tabs[2]:
        st.subheader(f"Financials: {focus_ticker}")
        stmt_type = st.radio("Statement Type", ["income_statement", "balance_sheet", "cash_flow"], horizontal=True, format_func=lambda x: x.replace("_", " ").title())
        
        df_stmt = get_financial_statement(focus_company, stmt_type)
        
        if not df_stmt.empty:
            st.dataframe(df_stmt.style.format("{:,.0f}"), height=600, use_container_width=True)
        else:
            st.warning("No data available for this statement.")

    # 4. Advanced Sandbox 3.0
    with tabs[3]:
        st.subheader("🧪 Custom KPI Lab")
        
        import re
        
        # --- Collect available variables ---
        sample_metrics = [k for k in focus_company['metrics'].keys() if k != "Profile"]
        raw_keys_inc, raw_keys_bs, raw_keys_cf = [], [], []
        try:
            raw_inc = focus_company['raw_data']['financials']['annual']['income_statement']
            raw_keys_inc = sorted(raw_inc.keys())
        except: pass
        try:
            raw_bs = focus_company['raw_data']['financials']['annual']['balance_sheet']
            raw_keys_bs = sorted(raw_bs.keys())
        except: pass
        try:
            raw_cf = focus_company['raw_data']['financials']['annual']['cash_flow']
            raw_keys_cf = sorted(raw_cf.keys())
        except: pass
        
        all_options = sorted(list(set(sample_metrics + raw_keys_inc + raw_keys_bs + raw_keys_cf)))
        
        # --- A. Variable Explorer (Multi-select with comparison table) ---
        st.markdown("### 🔍 Variable Explorer")
        st.caption("Search and select multiple variables to compare them side by side across all companies, then use their names in a formula below.")
        
        selected_vars = st.multiselect(
            "Search & select variables",
            options=all_options,
            default=[],
            placeholder="Type to search... (e.g. Revenue, EBITDA, Current Assets)",
            key="sandbox_var_explorer"
        )
        
        if selected_vars:
            # Build side-by-side comparison table
            comparison_rows = []
            for c_item in companies:
                row = {"Ticker": c_item['ticker']}
                for var_name in selected_vars:
                    val = get_value_for_ticker(companies, c_item['ticker'], var_name)
                    row[var_name] = val if val is not None else 0
                comparison_rows.append(row)
            
            df_comparison = pd.DataFrame(comparison_rows).set_index("Ticker")
            st.dataframe(
                df_comparison.style.format("{:,.2f}").background_gradient(cmap="Blues", axis=0),
                use_container_width=True
            )
            
            # Show formatted names for copy-paste
            st.caption("**Copy these into your formula** (single-quoted variable names):")
            formula_parts = "  ".join([f"`'{v}'`" for v in selected_vars])
            st.markdown(formula_parts)
        
        st.divider()
        
        # --- B. Formula Builder ---
        st.markdown("### ⚡ Formula Builder")
        st.caption("Use single quotes around variable names. Operators: `+` `-` `*` `/` `abs()` `round()`")
        
        f_c1, f_c2 = st.columns([3, 1])
        with f_c1:
            formula = st.text_input(
                "Formula", 
                placeholder="e.g. ('Total Revenue' - 'Cost Of Revenue') / 'fullTimeEmployees'",
                key="formula_input"
            )
        with f_c2:
            metric_name = st.text_input(
                "Metric Name", 
                value="My Custom KPI",
                key="metric_name_input"
            )
        
        st.divider()

        # --- C. Calculate & Visualize ---
        calc_col1, calc_col2 = st.columns([1, 1])
        with calc_col1:
            run_calc = st.button("🚀 Calculate", type="primary", use_container_width=True)
        with calc_col2:
            save_to_comparison = st.button("💾 Save to Strategic Comparison", use_container_width=True)
        
        if run_calc or save_to_comparison:
            if not formula:
                st.error("Enter a formula first.")
            else:
                results = []
                input_var_data = {}
                
                needed_vars = re.findall(r"'(.*?)'", formula)
                
                for c_item in companies:
                    safe_eval_dict = {}
                    
                    for v_name in needed_vars:
                        val = get_value_for_ticker(companies, c_item['ticker'], v_name)
                        safe_eval_dict[v_name] = val if val is not None else 0
                        
                        if v_name not in input_var_data:
                            input_var_data[v_name] = []
                        input_var_data[v_name].append({"Ticker": c_item['ticker'], "Value": val if val is not None else 0})
                    
                    eval_formula = formula
                    eval_context = {"abs": abs, "round": round, "min": min, "max": max}
                    
                    for idx_v, v_name in enumerate(needed_vars):
                        placeholder = f"__v{idx_v}__"
                        eval_formula = eval_formula.replace(f"'{v_name}'", placeholder)
                        eval_context[placeholder] = safe_eval_dict[v_name]
                    
                    try:
                        final_val = eval(eval_formula, {"__builtins__": None}, eval_context)
                        results.append({
                            "Ticker": c_item['ticker'],
                            metric_name: round(float(final_val), 2) if final_val is not None else 0.0,
                        })
                    except Exception as e:
                        results.append({
                            "Ticker": c_item['ticker'],
                            metric_name: 0.0,
                        })
                        st.warning(f"⚠️ Error for {c_item['ticker']}: {str(e)}")
                
                df_res = pd.DataFrame(results).set_index("Ticker")
                
                # --- Result Chart ---
                st.markdown(f"### 📊 Result: {metric_name}")
                
                res_c1, res_c2 = st.columns([1, 2])
                with res_c1:
                    st.dataframe(df_res[[metric_name]], use_container_width=True)
                with res_c2:
                    fig = px.bar(
                        df_res, x=df_res.index, y=metric_name, 
                        color=df_res.index, title=metric_name,
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
                        xaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- D. Input Variable Breakdown ---
                if len(needed_vars) > 0:
                    st.markdown("### 🔬 Input Variable Breakdown")
                    st.caption("Individual values for each variable used in your formula.")
                    
                    n_cols = min(len(needed_vars), 3)
                    var_viz_cols = st.columns(n_cols)
                    for v_idx, v_name in enumerate(needed_vars):
                        with var_viz_cols[v_idx % n_cols]:
                            v_df = pd.DataFrame(input_var_data[v_name])
                            fig_v = px.bar(
                                v_df, x="Ticker", y="Value", title=v_name,
                                color="Ticker",
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig_v.update_layout(
                                showlegend=False,
                                height=250,
                                margin=dict(l=0, r=0, t=30, b=0),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
                                xaxis=dict(showgrid=False)
                            )
                            st.plotly_chart(fig_v, use_container_width=True)
                
                # --- E. Save to Strategic Comparison ---
                if save_to_comparison:
                    # Write computed values directly into the companies list (same object reference)
                    for r in results:
                        ticker = r['Ticker']
                        for comp in companies:
                            if comp['ticker'] == ticker:
                                comp['metrics'][metric_name] = r[metric_name]
                    
                    # Track saved KPIs so Strategic Comparison can show them
                    if 'saved_sandbox_kpis' not in st.session_state:
                        st.session_state['saved_sandbox_kpis'] = {}
                    st.session_state['saved_sandbox_kpis'][metric_name] = formula
                    
                    st.success(f"✅ **{metric_name}** saved! Switch to Strategic Comparison → '📌 Custom KPIs' to view it.")
                    st.balloons()
        
        # --- F. Saved Custom KPIs ---
        saved_kpis = st.session_state.get('saved_sandbox_kpis', {})
        if saved_kpis:
            st.divider()
            st.markdown("### 💾 Saved Custom KPIs")
            for kpi_name, kpi_formula in saved_kpis.items():
                st.markdown(f"**{kpi_name}**: `{kpi_formula}`")

    # 5. Deep Dive (Expert Suite)
    with tabs[4]:
        st.subheader("📑 Expert Deep Dive")
        
        # A. Company Profile
        prof = focus_company['metrics'].get('Profile', {})
        summary_snippet = prof.get('Summary', '')[:120]
        st.markdown(f"### 🏢 {focus_ticker} — {prof.get('Sector', '')} / {prof.get('Industry', '')}")
        
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            st.markdown(f"**Sector**")
            st.caption(prof.get('Sector', 'N/A'))
        with p_c2:
            st.markdown(f"**Industry**")
            st.caption(prof.get('Industry', 'N/A'))
        with p_c3:
            emp = prof.get('Employees', 'N/A')
            st.markdown(f"**Employees**")
            st.caption(f"{emp:,}" if isinstance(emp, (int, float)) else str(emp))
        with p_c4:
            st.markdown(f"**HQ**")
            st.caption(f"{prof.get('City', '')}, {prof.get('Country', '')}")
        
        web = prof.get('Website', '')
        if web:
            st.markdown(f"🌐 [{web}]({web})")
            
        with st.expander("📄 Full Business Summary", expanded=False):
            st.write(prof.get('Summary', 'No summary available.'))
            
        st.divider()
        
        # B. Governance & Risk Scorecard (Enhanced)
        st.markdown("### 🏛️ Governance & Risk Scorecard")
        st.caption("Risk scores range from **1 (low risk)** to **10 (high risk)**. Scores are sourced from Yahoo Finance's ISS governance analytics. Lower is better.")
        
        m = focus_company['metrics']
        
        def risk_color(val):
            """Return emoji indicator based on risk score (1-10)."""
            try:
                v = float(val)
                if v <= 3: return "🟢"
                elif v <= 6: return "🟡"
                else: return "🔴"
            except: return "⚪"
        
        def risk_label(val):
            """Return text label for risk score."""
            try:
                v = float(val)
                if v <= 3: return "Low"
                elif v <= 6: return "Moderate"
                else: return "High"
            except: return "N/A"
        
        # Risk scores in a clear grid
        risk_items = [
            ("Audit Risk", m.get('Audit Risk', 'N/A'), "Risk that financial reporting may be inaccurate or non-compliant"),
            ("Board Risk", m.get('Board Risk', 'N/A'), "Risk related to board composition, independence, and oversight"),
            ("Compensation Risk", m.get('Compensation Risk', 'N/A'), "Risk from executive pay misalignment with shareholder interests"),
            ("Shareholder Rights Risk", m.get('Shareholder Rights Risk', 'N/A'), "Risk that shareholder voting rights and protections are weak"),
        ]
        
        gov_cols = st.columns(4)
        for idx, (label, val, tooltip) in enumerate(risk_items):
            with gov_cols[idx]:
                indicator = risk_color(val)
                st.markdown(f"**{indicator} {label}**")
                st.markdown(f"### {val} / 10")
                st.caption(f"_{risk_label(val)} risk_ — {tooltip}")
        
        # Overall governance score
        try:
            gov_score = m.get('Overall Governance Risk', None)
            if gov_score is not None:
                st.markdown(f"**Overall Governance Risk**: {risk_color(gov_score)} **{gov_score}** / 10 — _{risk_label(gov_score)} overall governance risk_")
        except: pass
        
        st.markdown("")
        
        # Ownership structure
        own_c1, own_c2, own_c3 = st.columns(3)
        with own_c1:
            inst = m.get('Inst. Ownership %', 0)
            st.metric("Institutional Ownership", f"{inst}%", help="% of shares held by large institutions (mutual funds, pension funds, etc.)")
        with own_c2:
            insider = m.get('Insider Ownership %', 0)
            st.metric("Insider Ownership", f"{insider}%", help="% of shares held by executives and board members — high insider ownership can signal alignment with shareholders")
        with own_c3:
            beta = m.get('Beta', 0)
            beta_label = "Higher than market" if beta > 1 else "Lower than market" if beta < 1 else "Market-level"
            st.metric("Beta (Volatility)", f"{beta}", help=f"Measures stock volatility vs. market. Beta=1 means market-average. {beta_label} volatility.")
             
        st.divider()

        # C. Capital Allocation Bridge (Cash Flow) — Fixed for negative OCF
        st.markdown(f"### 🌉 Capital Allocation Bridge: {focus_ticker}")
        st.caption("How the company allocates its operating cash flow across investments, distributions, and M&A.")
        
        try:
            cf = focus_company['raw_data']['financials']['annual']['cash_flow']
            
            def get_latest_val(key):
                """Get latest year value from cash flow dict."""
                if key in cf and cf[key]:
                    val = list(cf[key].values())[0]
                    return val if val is not None else 0
                return 0

            ocf = get_latest_val("Operating Cash Flow")
            capex = get_latest_val("Capital Expenditure")
            div = get_latest_val("Cash Dividends Paid")
            buyback = get_latest_val("Repurchase Of Capital Stock")
            acq = get_latest_val("Net Business Purchase And Sale")
            
            # Compute remaining
            remaining = ocf + capex + div + buyback + acq  # capex/div/buyback are usually negative
            
            if ocf < 0:
                # When OCF is negative, show a simple bar chart instead of waterfall
                # because waterfall doesn't make intuitive sense with a negative starting point
                st.warning(f"⚠️ Operating Cash Flow is **negative** (${ocf/1e9:.2f}B). The company burned cash from operations this period.")
                
                cf_items = {
                    "Operating CF": ocf,
                    "Capex": capex,
                    "Dividends": div,
                    "Buybacks": buyback,
                    "M&A": acq,
                }
                # Filter out zero items
                cf_items = {k: v for k, v in cf_items.items() if v != 0}
                
                fig_cf = px.bar(
                    x=list(cf_items.keys()), y=[v/1e9 for v in cf_items.values()],
                    title="Cash Flow Components ($B)",
                    color=[v > 0 for v in cf_items.values()],
                    color_discrete_map={True: "#4ade80", False: "#f87171"},
                    labels={"x": "", "y": "$ Billions", "color": ""},
                )
                fig_cf.update_layout(
                    showlegend=False, height=400,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
                )
                fig_cf.update_traces(text=[f"${v/1e9:.1f}B" for v in cf_items.values()], textposition="outside")
                st.plotly_chart(fig_cf, use_container_width=True)
            else:
                # Normal waterfall for positive OCF
                labels = ["Operating Cash Flow"]
                values = [ocf / 1e9]
                measures = ["absolute"]
                
                if capex != 0:
                    labels.append("Capex")
                    values.append(capex / 1e9)
                    measures.append("relative")
                if div != 0:
                    labels.append("Dividends")
                    values.append(div / 1e9)
                    measures.append("relative")
                if buyback != 0:
                    labels.append("Buybacks")
                    values.append(buyback / 1e9)
                    measures.append("relative")
                if acq != 0:
                    labels.append("M&A")
                    values.append(acq / 1e9)
                    measures.append("relative")
                    
                labels.append("Remaining")
                values.append(0)
                measures.append("total")
                
                fig_bridge = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=measures,
                    x=labels,
                    y=values,
                    textposition="outside",
                    text=[f"${v:.1f}B" for v in values[:-1]] + [f"${remaining/1e9:.1f}B"],
                    connector={"line": {"color": "rgba(255,255,255,0.2)"}},
                    increasing={"marker": {"color": "#4ade80"}},
                    decreasing={"marker": {"color": "#f87171"}},
                    totals={"marker": {"color": "#60a5fa"}},
                ))
                
                fig_bridge.update_layout(
                    title="Cash Flow Allocation Waterfall ($B)",
                    showlegend=False,
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title="$ Billions"),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_bridge, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not build Capital Allocation Bridge: {e}")

        st.divider()

        # D. Multi-Year Trends (Proper Line Charts)
        st.markdown("### 📈 Multi-Year Financial Trends")
        st.caption("Revenue, EBITDA, and Net Income trajectories for all companies in the cohort.")
        
        def extract_annual_series(company, statement, key_options):
            """Extract a time series from a company's financial statements."""
            raw = company.get('raw_data', {}).get('financials', {}).get('annual', {})
            stmt = raw.get(statement, {})
            key = next((k for k in key_options if k in stmt), None)
            if not key or not stmt[key]:
                return {}
            return stmt[key]
        
        trend_metrics = [
            ("Revenue", "income_statement", ["Total Revenue", "TotalRevenue", "Operating Revenue", "Revenue"]),
            ("EBITDA", "income_statement", ["EBITDA", "Normalized EBITDA"]),
            ("Net Income", "income_statement", ["Net Income", "NetIncome"]),
        ]
        
        trend_cols = st.columns(len(trend_metrics))
        
        for t_idx, (metric_label, stmt_name, key_opts) in enumerate(trend_metrics):
            with trend_cols[t_idx]:
                fig_trend = go.Figure()
                
                for c in companies:
                    series = extract_annual_series(c, stmt_name, key_opts)
                    if series:
                        sorted_dates = sorted(series.keys())
                        dates = sorted_dates[-5:]  # last 5 years
                        vals = [(series[d] / 1e9 if series[d] is not None else 0) for d in dates]  # Convert to billions
                        # Use year labels
                        years = [d[:4] if len(d) >= 4 else d for d in dates]
                        
                        fig_trend.add_trace(go.Scatter(
                            x=years, y=vals,
                            mode='lines+markers',
                            name=c['ticker'],
                            line=dict(width=3 if c['ticker'] == focus_ticker else 1.5),
                            opacity=1.0 if c['ticker'] == focus_ticker else 0.5,
                        ))
                
                fig_trend.update_layout(
                    title=f"{metric_label} ($B)",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
                    xaxis=dict(showgrid=False),
                    legend=dict(font=dict(size=10), orientation="h", y=-0.2),
                    showlegend=(t_idx == 0),  # Only show legend on first chart
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        st.divider()
        
        # E. Detailed Metrics (Categorized Cards)
        st.markdown(f"### 📋 All Metrics: {focus_ticker}")
        st.caption("Complete breakdown of all computed metrics, organized by category.")
        
        m = focus_company['metrics']
        
        # Categorize metrics
        metric_categories = {
            "📈 Growth": ["Revenue ($B)", "Revenue CAGR (3y)", "EBITDA CAGR (3y)"],
            "💰 Profitability": ["Gross Margin %", "EBITDA Margin %", "Net Margin %", "Operating Margin %"],
            "💧 Liquidity": ["Current Ratio", "Quick Ratio", "Cash Ratio", "CCC (Days)", "DIO (Days)", "DSO (Days)", "DPO (Days)"],
            "🏗️ Solvency": ["Net Debt / EBITDA", "Interest Coverage", "Debt / Equity", "Financial Leverage"],
            "⚙️ Efficiency": ["Asset Turnover", "Fixed Asset Turnover", "ROIC %", "ROE %", "ROA %"],
            "🎯 Returns & Valuation": ["Dupont ROE", "Shareholder Yield %", "EV / EBITDA", "P/E", "P/B", "Dividend Yield %"],
        }
        
        # Compute cohort averages for comparison
        cohort_avgs = {}
        for metric_list in metric_categories.values():
            for met in metric_list:
                vals = [c['metrics'].get(met, None) for c in companies if c['metrics'].get(met) is not None]
                if vals:
                    cohort_avgs[met] = np.mean(vals)
        
        for cat_name, cat_metrics in metric_categories.items():
            with st.expander(cat_name, expanded=True):
                n_cols = min(len(cat_metrics), 4)
                cols = st.columns(n_cols)
                for m_idx, met in enumerate(cat_metrics):
                    with cols[m_idx % n_cols]:
                        val = m.get(met, None)
                        avg = cohort_avgs.get(met, None)
                        
                        if val is not None and avg is not None:
                            diff = val - avg
                            # Format delta string
                            if abs(val) > 100:
                                delta_str = f"{diff:+,.0f} vs avg"
                            else:
                                delta_str = f"{diff:+.2f} vs avg"
                            st.metric(met, f"{val:,.2f}" if isinstance(val, float) else str(val), delta=delta_str, delta_color="normal")
                        elif val is not None:
                            st.metric(met, f"{val:,.2f}" if isinstance(val, float) else str(val))
                        else:
                            st.metric(met, "—")




else:
    st.write("👈 Select tickers to begin.")
