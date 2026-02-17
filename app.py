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
        st.subheader("🧪 Sandbox 3.0: Custom KPI Lab")
        
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
        
        # --- A. Template Gallery ---
        st.markdown("### 📋 Quick Templates")
        st.caption("Select a template to auto-fill the formula, or build your own below.")
        
        template_options = {
            "(none)": {"formula": "", "name": "My Custom KPI"},
            "EBITDA per Employee": {"formula": "'EBITDA' / 'fullTimeEmployees'", "name": "EBITDA / Employee"},
            "Revenue per Employee": {"formula": "'Total Revenue' / 'fullTimeEmployees'", "name": "Revenue / Employee"},
            "R&D Efficiency": {"formula": "'Total Revenue' / 'Research And Development'", "name": "R&D Efficiency (Rev/R&D)"},
            "Capex / OCF %": {"formula": "abs('Capital Expenditure') / 'Operating Cash Flow' * 100", "name": "Capex / OCF %"},
            "Earnings Quality": {"formula": "'Net Income' / 'Operating Cash Flow'", "name": "Earnings Quality"},
            "Working Capital / Rev %": {"formula": "('Current Assets' - 'Current Liabilities') / 'Total Revenue' * 100", "name": "Working Capital / Revenue %"},
        }
        
        selected_template = st.selectbox("Choose template", list(template_options.keys()), key="sandbox_template")
        
        # Set defaults from template
        tpl = template_options[selected_template]
        default_formula = tpl["formula"]
        default_name = tpl["name"]
        
        st.divider()
        
        # --- B. Variable Browser (Dropdown-based) ---
        st.markdown("### 🔍 Variable Browser")
        st.caption("Select a variable to see its formatted name — copy and paste it into the formula.")
        
        var_category = st.selectbox("Category", ["Computed Metrics", "Income Statement", "Balance Sheet", "Cash Flow"], key="var_cat")
        
        if var_category == "Computed Metrics":
            var_list = sorted(sample_metrics)
        elif var_category == "Income Statement":
            var_list = raw_keys_inc
        elif var_category == "Balance Sheet":
            var_list = raw_keys_bs
        else:
            var_list = raw_keys_cf
        
        if var_list:
            selected_var = st.selectbox("Select variable", ["(browse)"] + var_list, key="var_select")
            if selected_var != "(browse)":
                st.code(f"'{selected_var}'", language="python")
                st.caption("👆 Copy the text above and paste it into the formula field below.")
                
                # Show current values across companies
                var_preview = []
                for c_item in companies:
                    val = get_value_for_ticker(companies, c_item['ticker'], selected_var)
                    var_preview.append({"Ticker": c_item['ticker'], "Value": val if val is not None else 0})
                st.dataframe(pd.DataFrame(var_preview).set_index("Ticker"), use_container_width=True, height=150)
        else:
            st.info("No variables available in this category.")
        
        st.divider()
        
        # --- C. Formula Builder ---
        st.markdown("### ⚡ Formula Builder")
        st.caption("Use single quotes around variable names. Operators: `+` `-` `*` `/` `abs()` `round()`")
        
        f_c1, f_c2 = st.columns([3, 1])
        with f_c1:
            formula = st.text_input(
                "Formula", 
                value=default_formula,
                placeholder="e.g. ('Total Revenue' - 'Cost Of Revenue') / 'fullTimeEmployees'",
                key="formula_input"
            )
        with f_c2:
            metric_name = st.text_input(
                "Metric Name", 
                value=default_name,
                key="metric_name_input"
            )
        
        st.divider()

        # --- D. Calculate & Visualize ---
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
                
                # --- E. Input Variable Breakdown ---
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
                
                # --- F. Save to Strategic Comparison ---
                if save_to_comparison:
                    if 'custom_kpis' not in st.session_state:
                        st.session_state['custom_kpis'] = {}
                    
                    for r in results:
                        ticker = r['Ticker']
                        for comp in st.session_state.get('data', {}).get('companies', []):
                            if comp['ticker'] == ticker:
                                comp['metrics'][metric_name] = r[metric_name]
                    
                    st.session_state['custom_kpis'][metric_name] = formula
                    st.success(f"✅ **{metric_name}** saved! It now appears in the Strategic Comparison tab.")
                    st.balloons()

    # 5. Deep Dive (Expert Suite)
    with tabs[4]:
        st.subheader("📑 Expert Deep Dive")
        
        # A. Company Profile
        st.markdown(f"### 🏢 Company Profile: {focus_company['metrics'].get('Profile', {}).get('Summary', '')[:100]}...")
        
        prof = focus_company['metrics'].get('Profile', {})
        
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            st.markdown(f"**Sector**: {prof.get('Sector', 'N/A')}")
            st.markdown(f"**Industry**: {prof.get('Industry', 'N/A')}")
        with p_c2:
            st.markdown(f"**Employees**: {prof.get('Employees', 'N/A')}")
            st.markdown(f"**HQ**: {prof.get('City', '')}, {prof.get('Country', '')}")
        with p_c3:
            web = prof.get('Website', '#')
            st.markdown(f"**Website**: [{web}]({web})")
            
        with st.expander("Full Business Summary", expanded=False):
            st.write(prof.get('Summary', 'No summary available.'))
            
        st.divider()
        
        # B. Governance & Risk Scorecard
        st.markdown("### 🏛️ Governance & Risk")
        gov_cols = st.columns(4)
        
        # Focus Company Risk Profile
        m = focus_company['metrics']
        
        with gov_cols[0]:
            st.metric("Audit Risk", m.get('Audit Risk', 'N/A'))
            st.metric("Board Risk", m.get('Board Risk', 'N/A'))
        with gov_cols[1]:
            st.metric("Comp. Risk", m.get('Compensation Risk', 'N/A'))
            st.metric("Shareholder Risk", m.get('Shareholder Rights Risk', 'N/A'))
        with gov_cols[2]:
            st.metric("Inst. Ownership", f"{m.get('Inst. Ownership %', 0)}%")
            st.metric("Insider Ownership", f"{m.get('Insider Ownership %', 0)}%")
        with gov_cols[3]:
             st.metric("Beta", m.get('Beta', 0))
             
        st.divider()

        # C. Capital Allocation Bridge (Cash Flow)
        st.markdown(f"### 🌉 Capital Allocation Bridge (Last Fiscal Year): {focus_ticker}")
        
        # Get raw CF data for bridge
        try:
            cf = focus_company['raw_data']['financials']['annual']['cash_flow']
            
            # Helper to get latest absolute value
            def get_latest_abs(key):
                val = 0
                if key in cf:
                    # Get first value
                     val = list(cf[key].values())[0] if cf[key] else 0
                return val

            ocf = get_latest_abs("Operating Cash Flow")
            capex = get_latest_abs("Capital Expenditure") # Usually negative
            div = get_latest_abs("Cash Dividends Paid") # Usually negative
            buyback = get_latest_abs("Repurchase Of Capital Stock") # Usually negative
            acq = get_latest_abs("Net Business Purchase And Sale") # usually negative
            
            # Waterfall
            fig_bridge = go.Figure(go.Waterfall(
                name = "20", orientation = "v",
                measure = ["absolute", "relative", "relative", "relative", "relative", "total"],
                x = ["Operating Cash Flow", "Capex", "Dividends", "Buybacks", "M&A", "Remaining Cash Gen"],
                textposition = "outside",
                # Note: Capex/Divs/Buybacks are usually negative in Yahoo data.
                # If they are positive, we invert them for the bridge logic (Uses of Cash)
                y = [ocf, capex, div, buyback, acq, 0],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            
            fig_bridge.update_layout(title = "Cash Flow Allocation Waterfall", showlegend = True, height=500)
            st.plotly_chart(fig_bridge, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not build Capital Allocation Bridge: {e}")

        st.divider()

        # D. Sparklines (Trends)
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

        # E. Full Metrics Table
        st.markdown("### 📋 Detailed Metrics")
        full_df = pd.json_normalize([{'Ticker': c['ticker'], **c['metrics']} for c in companies]).set_index("Ticker")
        st.dataframe(full_df.style.background_gradient(cmap="viridis"), use_container_width=True)




else:
    st.write("👈 Select tickers to begin.")
