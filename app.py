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
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #334155;
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
        color: white;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f172a;
        color: #38bdf8;
    }
    .chart-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #334155;
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
st.sidebar.caption("v3.2 - AI Enhanced")


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
        
        # --- Tabs ---
    tabs = st.tabs(["🏆 Strategic Comparison", "💎 DuPont Analysis", "📊 Financial Statements", "🧪 Advanced Sandbox", "📑 Deep Dive"])
    
    # 1. Strategic Comparison (Heatmap + Charts)
    with tabs[0]:
        st.subheader("Comparative Analysis")
        
        # A. Heatmap Matrix
        with st.expander("Heatmap View", expanded=False):
            metrics_to_show = ["Revenue ($B)", "Revenue CAGR (3y)", "Gross Margin %", "EBITDA Margin %", "Net Margin %", "ROIC %", "CCC (Days)", "Net Debt / EBITDA"]
            matrix = []
            for c in companies:
                row = {"Ticker": c['ticker']}
                for m in metrics_to_show:
                    row[m] = c['metrics'].get(m, 0)
                matrix.append(row)
            
            df_heat = pd.DataFrame(matrix).set_index("Ticker")
            st.dataframe(
                df_heat.style.background_gradient(cmap="RdYlGn", axis=0), 
                use_container_width=True
            )
            
        # B. Individual KPI Graphics
        st.markdown("### Key Performance Indicators")
        
        # Define layout (3 columns)
        cols = st.columns(3)
        metrics_grid = [
            ("Growth", ["Revenue ($B)", "Revenue CAGR (3y)"]),
            ("Profitability", ["Gross Margin %", "EBITDA Margin %", "Net Margin %"]),
            ("Efficiency", ["ROIC %", "CCC (Days)", "Net Debt / EBITDA"])
        ]
        
        # Flatten the list for grid iteration
        all_metrics = [m for cat, ms in metrics_grid for m in ms]
        
        for i, metric in enumerate(all_metrics):
            col = cols[i % 3]
            with col:
                # Prepare Data for Chart
                chart_data = []
                for c in companies:
                    chart_data.append({
                        "Ticker": c['ticker'],
                        "Value": c['metrics'].get(metric, 0),
                        "Color": '#38bdf8' if c['ticker'] == focus_ticker else '#334155'
                    })
                df_chart = pd.DataFrame(chart_data)
                
                # Plotly Chart
                fig = px.bar(
                    df_chart, 
                    x="Ticker", 
                    y="Value", 
                    color="Ticker",
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

    # 2. DuPont Analysis (NEW)
    with tabs[1]:
        st.subheader(f"💎 DuPont Analysis: {focus_ticker}")
        st.caption("Decomposing ROE into Profitability (Net Margin), Efficiency (Asset Turnover), and Leverage.")
        
        # A. Driver Tree (Focus Company)
        m = focus_company['metrics']
        
        # Check if new metrics are available (might need to handle empty if not re-run yet)
        net_margin = m.get('Net Margin %', 0)
        asset_turnover = m.get('Asset Turnover', 0)
        leverage = m.get('Financial Leverage', 1)
        roe = m.get('ROE %', 0)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROE %", f"{roe}%", delta=None)
        c2.metric("Net Margin %", f"{net_margin}%", help="Net Income / Revenue")
        c3.metric("Asset Turnover", f"{asset_turnover}x", help="Revenue / Total Assets")
        c4.metric("Fin. Leverage", f"{leverage}x", help="Assets / Equity")
        
        st.divider()
        
        # B. Strategic Positioning (Scatter)
        st.markdown("### 🎯 Strategic Positioning (Margin vs Turnover)")
        st.caption("Where do competitors play? High Margin (Luxury) vs High Velocity (Retail).")
        
        scatter_data = []
        for c in companies:
            scatter_data.append({
                "Ticker": c['ticker'],
                "Net Margin %": c['metrics'].get('Net Margin %', 0),
                "Asset Turnover": c['metrics'].get('Asset Turnover', 0),
                "ROE %": c['metrics'].get('ROE %', 0),
                "Revenue ($B)": c['metrics'].get('Revenue ($B)', 0),
            })
        
        df_scatter = pd.DataFrame(scatter_data)
        
        fig_dupont = px.scatter(
            df_scatter,
            x="Asset Turnover",
            y="Net Margin %",
            size="Revenue ($B)",
            color="ROE %",
            text="Ticker",
            hover_data=["ROE %"],
            title="DuPont Map: Efficiency vs Profitability",
            color_continuous_scale="RdYlGn"
        )
        # Add efficient frontier lines?
        # Iso-ROE curves: y * x * lev = ROE
        # Simplify visual for now
        
        fig_dupont.update_traces(textposition='top center')
        fig_dupont.update_layout(
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#334155', title="Asset Turnover (Efficiency)"),
            yaxis=dict(showgrid=True, gridcolor='#334155', title="Net Margin % (Profitability)")
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

    # 4. Advanced Sandbox
    with tabs[3]:
        st.subheader("🧪 Formula Sandbox")
        st.info("Combine ANY metric or raw data point. Try 'Total Revenue' / 'Full Time Employees'.")
        
        sample_metrics = list(focus_company['metrics'].keys())
        
        # Get raw keys
        raw_keys = []
        try:
            inc = focus_company['raw_data']['financials']['annual']['income_statement']
            raw_keys.extend(list(inc.keys()))
            bs = focus_company['raw_data']['financials']['annual']['balance_sheet']
            raw_keys.extend(list(bs.keys()))
        except: pass
        
        available_fields = sorted(list(set(sample_metrics + raw_keys)))
        
        c1, c2, c3, c4 = st.columns([3, 1, 3, 2])
        var_a = c1.selectbox("Variable A", options=available_fields, index=0)
        op = c2.selectbox("Op", ["/", "*", "+", "-"])
        var_b = c3.selectbox("Variable B", options=available_fields, index=min(1, len(available_fields)-1))
        
        new_name = c4.text_input("Name Result", value="Custom Metric")
        
        if st.button("Calculate Custom Metric"):
            res_data = []
            for c in companies:
                val_a = get_value_for_ticker(companies, c['ticker'], var_a)
                val_b = get_value_for_ticker(companies, c['ticker'], var_b)
                
                res = None
                try:
                    if op == "/": res = val_a / val_b if val_b else 0
                    elif op == "*": res = val_a * val_b
                    elif op == "+": res = val_a + val_b
                    elif op == "-": res = val_a - val_b
                except: res = 0
                
                res_data.append({"Ticker": c['ticker'], new_name: res, f"{var_a}": val_a, f"{var_b}": val_b})
            
            df_res = pd.DataFrame(res_data).set_index("Ticker")
            
            s1, s2 = st.columns([1, 2])
            with s1:
                st.dataframe(df_res)
            with s2:
                fig = px.bar(df_res, x=df_res.index, y=new_name, color=df_res.index, title=new_name)
                st.plotly_chart(fig, use_container_width=True)

    # 5. Deep Dive Table
    with tabs[4]:
        full_df = pd.json_normalize([{'Ticker': c['ticker'], **c['metrics']} for c in companies]).set_index("Ticker")
        st.dataframe(full_df.style.background_gradient(cmap="viridis"), use_container_width=True)

else:
    st.write("👈 Select tickers to begin.")
