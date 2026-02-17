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
        
        # Analyst Note
        note = generate_analyst_note(focus_company, companies)
        st.info(f"🤖 **Analyst Insight**: {note}")
        
        # --- Tabs ---
    tabs = st.tabs(["🏆 Strategic Comparison", "💎 DuPont Analysis", "📊 Financial Statements", "🧪 Advanced Sandbox", "📑 Deep Dive"])
    
    # 1. Strategic Comparison (Heatmap + Charts)
    with tabs[0]:
        st.subheader("Comparative Analysis")
        
        # Focus Selector
        focus_mode = st.radio("Benchmark Focus", ["Overview", "Liquidity", "Solvency", "Efficiency", "Returns"], horizontal=True)
        
        # Define Metrics per Focus
        metric_groups = {
            "Overview": ["Revenue ($B)", "Revenue CAGR (3y)", "Gross Margin %", "EBITDA Margin %", "Net Margin %", "ROIC %", "CCC (Days)"],
            "Liquidity": ["Current Ratio", "Quick Ratio", "Cash Ratio", "CCC (Days)", "DIO (Days)", "DSO (Days)", "DPO (Days)"],
            "Solvency": ["Net Debt / EBITDA", "Interest Coverage", "Debt / Equity", "Financial Leverage", "Quick Ratio"],
            "Efficiency": ["Asset Turnover", "ROIC %", "ROE %", "Fixed Asset Turnover", "CCC (Days)"],
            "Returns": ["ROE %", "ROIC %", "Shareholder Yield %", "Net Margin %", "Dupont ROE"]
        }
        
        # Fallback to Overview if key missing
        metrics_to_show = metric_groups.get(focus_mode, metric_groups["Overview"])
        
        # A. Heatmap Matrix
        with st.expander(f"Heatmap View ({focus_mode})", expanded=True):
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
        
        # Dynamic Grid based on Selection
        # Use 3 cols
        cols = st.columns(3)
        
        for i, metric in enumerate(metrics_to_show):
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
        c3.metric("Asset Turnover", f"{asset_turnover:.2f}x", help="Revenue / Total Assets")
        c4.metric("Fin. Leverage", f"{leverage:.2f}x", help="Assets / Equity")
        
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

    # 4. Advanced Sandbox 2.0
    with tabs[3]:
        st.subheader("🧪 Sandbox 2.0: Formula Engine")
        st.caption("Build custom KPIs using free-form arithmetic. Search variables below to see their names.")

        # A. Variable Search (Helper)
        # Collect all available keys from the focus company for autocomplete/reference
        sample_metrics = list(focus_company['metrics'].keys())
        raw_keys = []
        try:
            inc = focus_company['raw_data']['financials']['annual']['income_statement']
            raw_keys.extend(list(inc.keys()))
            bs = focus_company['raw_data']['financials']['annual']['balance_sheet']
            raw_keys.extend(list(bs.keys()))
        except: pass
        
        all_options = sorted(list(set(sample_metrics + raw_keys)))
        
        selected_vars = st.multiselect("🔍 Search Variables (Copy-Paste names below)", options=all_options)
        
        if selected_vars:
            st.code("Assignments:\n" + "\n".join([f"'{v}'" for v in selected_vars]), language="python")

        # B. Formula Input
        c1, c2 = st.columns([3, 1])
        with c1:
            formula = st.text_input("Formula", placeholder="e.g. ('Total Revenue' - 'Cost Of Revenue') / 'Full Time Employees'", help="Use single quotes for variable names.")
        with c2:
            metric_name = st.text_input("Metric Name", value="My Custom KPI")
            
        st.info("💡 Tip: Use `+`, `-`, `*`, `/`, and `()` for logic. wrap variable names in single quotes `'`.")

        if st.button("Calculate", type="primary"):
            if not formula:
                st.error("Enter a formula.")
            else:
                results = []
                
                # Evaluation Logic
                for c in companies:
                    # Create a safe local dictionary for evaluations
                    # We need to flattening the company data so that keys are direct variables
                    local_vars = {}
                    
                    # Add computed metrics
                    for k, v in c['metrics'].items():
                        local_vars[k] = v
                        
                    # Add raw data (deep search for keys in formula)
                    # Optimization: Only fetch keys present in formula text
                    # Simple regex or string check
                    # For now, let's just dynamic lookup inside eval using a custom class or pre-calc?
                    # Pre-calc is safer.
                    
                    # We will support the variable names provided in 'all_options'
                    # But we can't pre-load 500 variables.
                    # Strategy: Regex parse the formula for quoted strings?
                    import re
                    # Find all strings inside single quotes
                    needed_vars = re.findall(r"'(.*?)'", formula)
                    
                    safe_eval_dict = {}
                    missing = []
                    
                    for v_name in needed_vars:
                        val = get_value_for_ticker(companies, c['ticker'], v_name)
                        safe_eval_dict[v_name] = val
                    
                    # Replace user's 'Var Name' with Python valid identifiers or dictionary lookups?
                    # Python `eval` with a dict works if names are valid IDs. but 'Total Revenue' is not.
                    # We can leave them as strings if we use pandas eval? No.
                    # Simplest: Replace 'Var Name' in the string with `locals['Var Name']`?
                    
                    # Let's try: Replace 'Var Name' with a temp placeholder `_var_0`, `_var_1`...
                    
                    eval_formula = formula
                    eval_context = {}
                    
                    for idx, v_name in enumerate(needed_vars):
                        placeholder = f"__v{idx}__"
                        eval_formula = eval_formula.replace(f"'{v_name}'", placeholder)
                        eval_context[placeholder] = safe_eval_dict[v_name]
                    
                    try:
                        # Safe arithmetic eval
                        # Allow only basic math?
                        final_val = eval(eval_formula, {"__builtins__": None}, eval_context)
                        results.append({
                            "Ticker": c['ticker'],
                            metric_name: float(final_val),
                            "Debug": str(eval_context)
                        })
                    except Exception as e:
                        results.append({
                            "Ticker": c['ticker'],
                            metric_name: 0.0,
                            "Debug": f"Error: {str(e)}"
                        })
                
                df_res = pd.DataFrame(results).set_index("Ticker")
                
                s1, s2 = st.columns([1, 2])
                with s1:
                    st.dataframe(df_res[[metric_name]])
                with s2:
                    fig = px.bar(
                        df_res, x=df_res.index, y=metric_name, 
                        color=df_res.index, title=metric_name,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # 5. Deep Dive Table
    with tabs[4]:
        st.subheader("📑 Deep Dive & Trends")
        
        # A. Sparklines (Trends)
        st.markdown("### 📈 3-Year Trends")
        
        trend_data = []
        for c in companies:
            ticker = c['ticker']
            raw = c.get('raw_data', {}).get('financials', {}).get('annual', {})
            inc = raw.get('income_statement', {})
            
            def get_trend(data_dict, key):
                if not data_dict or key not in data_dict: return [0,0,0]
                series = data_dict[key]
                # Sort by date ascending to show trend left-to-right
                sorted_dates = sorted(series.keys())
                # Take last 5 years max
                vals = [series[d] for d in sorted_dates][-5:]
                return vals
            
            # Try to find revenue key
            rev_key = next((k for k in ["Total Revenue", "TotalRevenue", "Operating Revenue", "Revenue"] if k in inc), None)
            ebitda_key = next((k for k in ["EBITDA", "Normalized EBITDA"] if k in inc), None)
            ni_key = next((k for k in ["Net Income", "NetIncome"] if k in inc), None)
            
            trend_data.append({
                "Ticker": ticker,
                "Revenue Trend": get_trend(inc, rev_key),
                "EBITDA Trend": get_trend(inc, ebitda_key),
                "Net Income Trend": get_trend(inc, ni_key)
            })
            
        df_trends = pd.DataFrame(trend_data)
        
        st.dataframe(
            df_trends,
            column_config={
                "Revenue Trend": st.column_config.LineChartColumn("Revenue (5y)", y_min=0, width="medium"),
                "EBITDA Trend": st.column_config.LineChartColumn("EBITDA (5y)", width="medium"),
                "Net Income Trend": st.column_config.LineChartColumn("Net Income (5y)", width="medium")
            },
            use_container_width=True,
            hide_index=True
        )

        # B. Full Metrics Table
        st.markdown("### 📋 Detailed Metrics")
        full_df = pd.json_normalize([{'Ticker': c['ticker'], **c['metrics']} for c in companies]).set_index("Ticker")
        st.dataframe(full_df.style.background_gradient(cmap="viridis"), use_container_width=True)




else:
    st.write("👈 Select tickers to begin.")
