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
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

# Import Local Engine
from engines.benchmarking_engine import BenchmarkingEngine

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
tickers_input = st.sidebar.text_area("Tickers", value="MSFT, AAPL, AMZN, GOOGL")

if st.sidebar.button("Run Analysis", type="primary"):
    st.session_state['run_requested'] = True

st.sidebar.markdown("---")
st.sidebar.caption("v3.1 - Self-Contained")


# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------
if st.session_state.get('run_requested', False):
    if not tickers_input:
        st.error("Enter tickers.")
    else:
        t_list = [t.strip() for t in tickers_input.split(",") if t.strip()]
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
        
        # --- Performance Heatmap (Top View) ---
        with st.expander("🏆 Comparative Heatmap (Click to Expand)", expanded=True):
            # Build Matrix
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

        # --- Tabs ---
        tabs = st.tabs(["📊 Financial Statements", "🧪 Advanced Sandbox", "📈 Visual Analysis", "📑 Deep Dive"])

        # 1. Financial Statements (Yahoo Style)
        with tabs[0]:
            st.subheader(f"Financials: {focus_ticker}")
            stmt_type = st.radio("Statement Type", ["income_statement", "balance_sheet", "cash_flow"], horizontal=True, format_func=lambda x: x.replace("_", " ").title())
            
            df_stmt = get_financial_statement(focus_company, stmt_type)
            
            if not df_stmt.empty:
                st.dataframe(df_stmt.style.format("{:,.0f}"), height=600, use_container_width=True)
            else:
                st.warning("No data available for this statement.")

        # 2. Advanced Sandbox
        with tabs[1]:
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

        # 3. Visual Analysis (Existing Charts)
        with tabs[2]:
            c1, c2 = st.columns(2)
            # Highlight Focus
            df_chart = pd.DataFrame([{
                "Ticker": c['ticker'],
                "ROIC": c['metrics']['ROIC %'],
                "Net Margin": c['metrics']['Net Margin %'],
                "Revenue": c['metrics']['Revenue ($B)'],
                "Color": '#38bdf8' if c['ticker'] == focus_ticker else '#334155'
            } for c in companies])
            
            with c1:
                fig = px.scatter(df_chart, x="ROIC", y="Net Margin", size="Revenue", color="Ticker", title="Efficiency Frontier")
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                fig = px.bar(df_chart, x="Ticker", y="Revenue", color="Ticker", title="Revenue Scale")
                st.plotly_chart(fig, use_container_width=True)

        # 4. Deep Dive Table
        with tabs[3]:
            full_df = pd.json_normalize([{'Ticker': c['ticker'], **c['metrics']} for c in companies]).set_index("Ticker")
            st.dataframe(full_df.style.background_gradient(cmap="viridis"), use_container_width=True)

else:
    st.write("👈 Select tickers to begin.")
