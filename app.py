import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
import requests
import numpy as np
import re

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
from engines.currency import (
    convert_value, get_rate, currency_symbol, money_suffix, normalize_currency,
    metric_is_absolute, metric_currency_field, make_signature, clear_fx_cache,
    COMMON_CURRENCIES, DEFAULT_PROVIDER_ORDER,
)
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

@st.cache_data(show_spinner=False)
def get_data(ticker_list, fx_signature):
    """Fetch benchmark data using the local engine.

    ``fx_signature`` (provider order + manual overrides) is part of the cache key
    because it drives the intra-company (ADR) currency correction baked into the
    data. The *display* currency is applied at render time, so switching currency
    does NOT re-fetch.
    """
    engine = BenchmarkingEngine()
    try:
        return engine.run_benchmark(ticker_list, detailed=True, fx_signature=fx_signature)
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
st.sidebar.caption("v3.4 - Multi-currency")


# -------------------------------------------------------------------------
# Currency & FX configuration (sidebar)
# -------------------------------------------------------------------------
with st.sidebar.expander("💱 Currency & FX", expanded=False):
    # Option list = common currencies + any currency present in the loaded
    # cohort + an "Other" escape hatch for ANY ISO 4217 code. The FX providers
    # resolve arbitrary pairs (Yahoo any pair + USD triangulation; open.er-api
    # 160+ codes), so the target is not limited to this list.
    _cohort_ccys = []
    _loaded = st.session_state.get("data")
    if isinstance(_loaded, dict):
        _cinfo = _loaded.get("currency", {}) or {}
        _cohort_ccys = list((_cinfo.get("distinct_currencies", {}) or {}).keys())
    _base = list(COMMON_CURRENCIES)
    for _c in _cohort_ccys:
        if _c and _c not in _base:
            _base.append(_c)
    _ccy_options = ["NATIVE"] + _base + ["OTHER"]

    def _ccy_label(c):
        if c == "NATIVE":
            return "Native (no conversion)"
        if c == "OTHER":
            return "✏️ Other (type any code)…"
        return c

    _choice = st.selectbox(
        "Display currency",
        options=_ccy_options,
        index=_ccy_options.index("EUR"),
        format_func=_ccy_label,
        help="Convert all absolute figures to this currency (ratios/percentages are currency-neutral and never converted). Pick 'Other' to enter ANY ISO 4217 code — e.g. PHP, IDR, CLP.",
        key="display_currency_choice",
    )
    if _choice == "OTHER":
        _custom = st.text_input(
            "ISO currency code", value=st.session_state.get("display_currency_custom", ""),
            max_chars=3, placeholder="e.g. PHP",
        ).strip().upper()
        st.session_state["display_currency_custom"] = _custom
        target_currency = _custom if _custom else "NATIVE"
        if _custom:
            st.caption(f"Converting to **{_custom}** (symbol: {currency_symbol(_custom)}). Falls back to native if no FX rate is found.")
    else:
        target_currency = _choice

    _provider_labels = {
        "yahoo": "Yahoo Finance",
        "frankfurter": "Frankfurter (ECB)",
        "open.er-api": "open.er-api",
        "manual": "Manual override",
    }
    provider_order = st.multiselect(
        "FX source priority (in order)",
        options=list(_provider_labels.keys()),
        default=list(DEFAULT_PROVIDER_ORDER),
        format_func=lambda k: _provider_labels[k],
        help="Tried top-to-bottom until one returns a rate. Yahoo reuses your existing data source; the rest are free, no-key fallbacks.",
        key="fx_provider_order",
    )
    if not provider_order:
        provider_order = list(DEFAULT_PROVIDER_ORDER)

    st.caption("Manual rate override (optional)")
    _mo1, _mo2, _mo3 = st.columns(3)
    with _mo1:
        _ov_from = st.text_input("From", key="ov_from", placeholder="USD")
    with _mo2:
        _ov_to = st.text_input("To", key="ov_to", placeholder="EUR")
    with _mo3:
        _ov_rate = st.number_input("Rate", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="ov_rate")
    manual_priority = st.checkbox(
        "Pin manual rates as top priority", value=False,
        help="On = your override beats live sources. Off = it's the final fallback.",
        key="fx_manual_priority",
    )
    if st.button("Add / update override"):
        if _ov_from and _ov_to and _ov_rate > 0:
            _ovs = dict(st.session_state.get("fx_overrides", {}))
            _ovs[f"{_ov_from.strip().upper()}{_ov_to.strip().upper()}"] = float(_ov_rate)
            st.session_state["fx_overrides"] = _ovs
            st.success(f"Saved {_ov_from.strip().upper()}→{_ov_to.strip().upper()} = {float(_ov_rate):.4f}")

    fx_overrides = dict(st.session_state.get("fx_overrides", {}))
    if fx_overrides:
        st.caption("Active overrides:")
        for _k, _v in list(fx_overrides.items()):
            _oc1, _oc2 = st.columns([3, 1])
            _oc1.markdown(f"`{_k[:3]}→{_k[3:]}` = **{_v:.4f}**")
            if _oc2.button("✕", key=f"rm_ov_{_k}"):
                fx_overrides.pop(_k, None)
                st.session_state["fx_overrides"] = fx_overrides
                st.rerun()

    if st.button("🔄 Refresh FX rates"):
        clear_fx_cache()
        # Also invalidate the benchmark cache so the engine's ADR correction
        # (EV/EBITDA, Shareholder Yield) is recomputed with fresh rates too.
        try:
            get_data.clear()
        except Exception:
            st.cache_data.clear()
        st.rerun()

# Resolve the active FX configuration for this run.
# (target_currency was set inside the Currency & FX expander above.)
try:
    target_currency
except NameError:
    target_currency = "EUR"
st.session_state["display_currency"] = target_currency
fx_overrides = dict(st.session_state.get("fx_overrides", {}))
fx_signature = make_signature(provider_order, fx_overrides, {}, manual_priority)


# -------------------------------------------------------------------------
# Currency display helpers (depend on the active target_currency / fx_signature)
# -------------------------------------------------------------------------
def company_currency(company, field="financial"):
    cur = company.get("currencies", {}) or {}
    return cur.get("financial") if field == "financial" else cur.get("trading")


def convert_to_display(value, from_ccy):
    """Convert a native value to the display currency.

    Returns (value, shown_currency). Falls back to the native value + native
    currency when conversion is unavailable or disabled (Native mode)."""
    if value is None:
        return None, from_ccy
    if target_currency == "NATIVE" or not from_ccy:
        return value, from_ccy
    conv, rate = convert_value(value, from_ccy, target_currency, fx_signature)
    if conv is None:
        return value, from_ccy
    return conv, target_currency


def display_factor(from_ccy):
    """Multiplicative factor to convert a native amount to the display currency."""
    if target_currency == "NATIVE" or not from_ccy:
        return 1.0, from_ccy
    r = get_rate(from_ccy, target_currency, fx_signature)
    if not r:
        return 1.0, from_ccy
    return r["rate"], target_currency


def metric_value(company, metric):
    """Metric value for display — converted iff it's an absolute money metric."""
    val = company.get("metrics", {}).get(metric, 0)
    if not metric_is_absolute(metric):
        return val
    from_ccy = company_currency(company, metric_currency_field(metric))
    conv, _ = convert_to_display(val, from_ccy)
    return conv if conv is not None else val


def metric_label(metric):
    """Chart/card title — swaps '($B)' for the display-currency suffix on absolutes."""
    if metric_is_absolute(metric) and target_currency != "NATIVE":
        base = metric.replace(" ($B)", "")
        return f"{base} {money_suffix(target_currency)}"
    return metric


def currency_note():
    """Short caption clarifying which figures are converted on a given chart."""
    if target_currency == "NATIVE":
        return "Absolutes in each company's native currency (not cross-comparable)."
    return f"Absolute figures in {target_currency} (FX-normalized); ratios are currency-neutral."


def compute_display_degraded(data):
    """Tickers whose native currency cannot be converted to the display currency.

    Their absolute figures fall back to native, so they must not be silently
    averaged/plotted alongside converted peers. Uses the cached get_rate, so it's
    consistent with what the charts will resolve on this same render."""
    if target_currency == "NATIVE":
        return []
    failed = []
    for c in data.get("companies", []):
        cur = c.get("currencies", {}) or {}
        for ccy in {cur.get("financial"), cur.get("trading")}:
            if ccy and normalize_currency(ccy) != target_currency:
                if get_rate(ccy, target_currency, fx_signature) is None:
                    failed.append(c["ticker"])
                    break
    return failed


def render_currency_banner(data):
    """Cohort-level currency flags + provenance, driven by the engine's report."""
    cinfo = data.get("currency", {}) or {}
    if target_currency != "NATIVE":
        st.caption(f"💱 Display currency: **{target_currency}** — absolute figures converted at latest FX; ratios are currency-neutral.")
    else:
        st.caption("💱 Display currency: **Native** — each company shown in its own reporting currency.")

    dc = cinfo.get("distinct_currencies", {}) or {}
    if cinfo.get("mixed_cohort"):
        parts = ", ".join(f"{k} ({', '.join(v)})" for k, v in dc.items())
        if target_currency == "NATIVE":
            st.warning(f"⚠️ Cohort mixes currencies — {parts}. Absolute figures are **not comparable** in Native mode; pick a display currency to normalize.")
        else:
            st.info(f"🌐 Cohort spans {len(dc)} reporting currencies — {parts}. Absolutes normalized to **{target_currency}**.")

    mism = cinfo.get("intra_company_mismatch", []) or []
    if mism:
        st.warning(f"🏷️ ADR / dual-listing: **{', '.join(mism)}** trade and report in different currencies — EV/EBITDA & Shareholder Yield were FX-corrected to the reporting currency.")

    degraded = cinfo.get("degraded", []) or []
    if degraded:
        st.error(f"❗ FX unavailable for **{', '.join(degraded)}** — their absolute figures are shown un-normalized.")

    # Display-layer check: per company, can we actually convert its native
    # currency to the chosen target? If not, its absolutes fall back to native
    # and would be silently mixed into the cohort — so flag it explicitly.
    disp_degraded = compute_display_degraded(data)
    if disp_degraded:
        st.error(
            f"❗ Could not convert to **{target_currency}** for "
            f"**{', '.join(disp_degraded)}** (no FX rate found) — their absolute "
            "figures are shown in their native currency and are **not comparable** "
            "with the rest of the cohort. Pick a different display currency or add a "
            "manual rate in the Currency & FX panel."
        )

    rates = cinfo.get("rates_used", []) or []
    if rates:
        with st.expander("ℹ️ FX rates used (intra-company corrections)", expanded=False):
            for r in rates:
                st.caption(f"{r['from']}→{r['to']} = {r['rate']:.4f} · source: {r.get('source','?')} · as-of {r.get('as_of','?')}")


# -------------------------------------------------------------------------
# Time-series helpers (shared by Over Time view and the Data Export tab)
# -------------------------------------------------------------------------

# Metric keys that can be computed as an annual time series from the statements.
OVERTIME_METRICS = {
    "Revenue ($B)", "Net Income ($B)", "EBITDA ($B)",
    "Gross Margin %", "EBITDA Margin %", "Net Margin %", "Operating Margin %",
    "Current Ratio", "Quick Ratio", "Debt / Equity", "ROE %", "ROA %",
    "Asset Turnover", "CCC (Days)", "Net Debt / EBITDA", "Interest Coverage",
    "ROIC %", "DIO (Days)", "DSO (Days)", "DPO (Days)",
}


def company_color_map(tickers):
    """Stable, high-contrast color per ticker, consistent across all charts."""
    palette = (px.colors.qualitative.Bold + px.colors.qualitative.Safe +
               px.colors.qualitative.Vivid + px.colors.qualitative.Pastel)
    return {t: palette[i % len(palette)] for i, t in enumerate(tickers)}


class _SeriesComputer:
    """Per-company time series over the company's OWN statement dates. Missing
    values are None (a gap), never a fabricated 0."""

    def __init__(self, dates, fn):
        self.dates = dates          # this company's actual YYYY-MM-DD period-ends
        self._fn = fn

    def series(self, metric):
        return self._fn(metric)     # -> {date: value or None}


def make_series_computer(company, n_periods=5, saved_kpis=None):
    """Build a per-company series computer keyed on the company's ACTUAL reporting
    dates (fiscal year-ends differ between companies/countries). Handles built-in
    KPIs, saved custom-KPI formulas and raw statement line items. Any missing input
    yields None (gap) rather than 0. Values are NATIVE currency."""
    saved_kpis = saved_kpis or {}
    annual = (company.get('raw_data', {}) or {}).get('financials', {}).get('annual', {}) or {}
    inc = annual.get('income_statement', {}) or {}
    bs = annual.get('balance_sheet', {}) or {}
    cf = annual.get('cash_flow', {}) or {}

    dset = set()
    for stmt in (inc, bs, cf):
        for series in stmt.values():
            if isinstance(series, dict):
                for d in series.keys():
                    ds = str(d)
                    if re.match(r'^\d{4}-\d{2}-\d{2}', ds):
                        dset.add(ds)
    dates = sorted(dset)[-n_periods:]

    def gv(stmt, keys, dk):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if k in stmt and stmt[k] and dk in stmt[k]:
                val = stmt[k][dk]
                if val is not None:
                    return val
        norm = {k.lower().replace(" ", ""): k for k in stmt.keys()}
        for k in keys:
            kk = k.lower().replace(" ", "")
            if kk in norm:
                rk = norm[kk]
                if rk in stmt and stmt[rk] and dk in stmt[rk]:
                    val = stmt[rk][dk]
                    if val is not None:
                        return val
        return None

    def rt(a, b, scale=1.0):
        """Ratio that returns None (not 0) when either side is missing or denom==0."""
        if a is None or b is None or b == 0:
            return None
        return (a / b) * scale

    REV = ["Total Revenue", "TotalRevenue", "Operating Revenue", "Revenue"]
    EQ = ["Stockholders Equity", "Total Equity Gross Minority Interest"]
    CASH = ["Cash And Cash Equivalents", "CashAndCashEquivalents"]
    CL = ["Current Liabilities", "Total Current Liabilities"]
    CA = ["Current Assets", "Total Current Assets"]
    NI = ["Net Income", "NetIncome"]
    EBITDA = ["EBITDA", "Normalized EBITDA"]
    EBIT = ["EBIT", "Operating Income"]
    TA = ["Total Assets", "TotalAssets"]

    def abs_b(field):
        def fn(dk):
            v = gv(inc, field, dk)
            return v / 1e9 if v is not None else None
        return fn

    def gross_margin(dk):
        rev = gv(inc, REV, dk)
        gp = gv(inc, ["Gross Profit", "GrossProfit"], dk)
        cogs = gv(inc, "Cost Of Revenue", dk)
        if gp is not None and rev is not None:
            return rt(gp, rev, 100)
        if rev is not None and cogs is not None:
            return rt(rev - cogs, rev, 100)
        return None

    def quick_ratio(dk):
        ca = gv(bs, CA, dk)
        cl = gv(bs, CL, dk)
        if ca is None or cl is None:
            return None
        inv = gv(bs, "Inventory", dk) or 0
        return rt(ca - inv, cl)

    def ccc(dk):
        cogs = gv(inc, "Cost Of Revenue", dk)
        rev = gv(inc, REV, dk)
        if cogs is None or rev is None:
            return None
        inv = gv(bs, "Inventory", dk) or 0
        rec = gv(bs, ["Receivables", "Accounts Receivable"], dk) or 0
        ap = gv(bs, ["Accounts Payable", "Payables"], dk) or 0
        return (rt(inv, cogs, 365) or 0) + (rt(rec, rev, 365) or 0) - (rt(ap, cogs, 365) or 0)

    def nd_ebitda(dk):
        e = gv(inc, EBITDA, dk)
        if e is None:
            return None
        nd = gv(bs, "Net Debt", dk)
        if nd is None:
            debt = gv(bs, "Total Debt", dk)
            if debt is None:
                return None
            nd = debt - (gv(bs, CASH, dk) or 0)
        return rt(nd, e)

    def roic(dk):
        ebit = gv(inc, EBIT, dk)
        eq = gv(bs, EQ, dk)
        debt = gv(bs, "Total Debt", dk)
        if ebit is None or (eq is None and debt is None):
            return None
        ic = (eq or 0) + (debt or 0) - (gv(bs, CASH, dk) or 0)
        return rt(ebit * 0.79, ic, 100)

    mc = {
        "Revenue ($B)": abs_b(REV),
        "Net Income ($B)": abs_b(NI),
        "EBITDA ($B)": abs_b(EBITDA),
        "Gross Margin %": gross_margin,
        "EBITDA Margin %": lambda dk: rt(gv(inc, EBITDA, dk), gv(inc, REV, dk), 100),
        "Net Margin %": lambda dk: rt(gv(inc, NI, dk), gv(inc, REV, dk), 100),
        "Operating Margin %": lambda dk: rt(gv(inc, EBIT, dk), gv(inc, REV, dk), 100),
        "Current Ratio": lambda dk: rt(gv(bs, CA, dk), gv(bs, CL, dk)),
        "Quick Ratio": quick_ratio,
        "Debt / Equity": lambda dk: rt(gv(bs, "Total Debt", dk), gv(bs, EQ, dk)),
        "ROE %": lambda dk: rt(gv(inc, NI, dk), gv(bs, EQ, dk), 100),
        "ROA %": lambda dk: rt(gv(inc, NI, dk), gv(bs, TA, dk), 100),
        "Asset Turnover": lambda dk: rt(gv(inc, REV, dk), gv(bs, TA, dk)),
        "CCC (Days)": ccc,
        "Net Debt / EBITDA": nd_ebitda,
        "Interest Coverage": lambda dk: rt(gv(inc, EBIT, dk), gv(inc, "Interest Expense", dk)),
        "ROIC %": roic,
        "DIO (Days)": lambda dk: rt(gv(bs, "Inventory", dk), gv(inc, "Cost Of Revenue", dk), 365),
        "DSO (Days)": lambda dk: rt(gv(bs, ["Receivables", "Accounts Receivable"], dk), gv(inc, REV, dk), 365),
        "DPO (Days)": lambda dk: rt(gv(bs, ["Accounts Payable", "Payables"], dk), gv(inc, "Cost Of Revenue", dk), 365),
    }

    def var_at(name, dk, stack):
        if name in stack:
            return None
        if name in mc:
            try:
                return mc[name](dk)
            except Exception:
                return None
        if name in saved_kpis:
            return formula_at(saved_kpis[name], dk, stack | {name})
        for stmt in (inc, bs, cf):
            v = gv(stmt, [name], dk)
            if v is not None:
                return v
        return None

    def formula_at(formula, dk, stack):
        names = []
        for n in re.findall(r"'(.*?)'", formula):
            if n not in names:
                names.append(n)
        expr = formula
        ctx = {"abs": abs, "round": round, "min": min, "max": max}
        for i, n in enumerate(names):
            ph = f"__v{i}__"
            expr = expr.replace(f"'{n}'", ph)
            val = var_at(n, dk, stack)
            if val is None:
                return None  # missing input -> gap, not 0
            ctx[ph] = val
        try:
            r = eval(expr, {"__builtins__": None}, ctx)
            return float(r) if r is not None else None
        except Exception:
            return None

    def compute(metric, dk):
        if metric in mc:
            try:
                v = mc[metric](dk)
            except Exception:
                v = None
        elif metric in saved_kpis:
            v = formula_at(saved_kpis[metric], dk, {metric})
        else:
            v = None
            for stmt in (inc, bs, cf):
                vv = gv(stmt, [metric], dk)
                if vv is not None:
                    v = vv
                    break
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return round(v, 4)
        return None if isinstance(v, bool) else v

    cache = {}

    def series(metric):
        if metric in cache:
            return cache[metric]
        out = {dk: compute(metric, dk) for dk in dates}
        cache[metric] = out
        return out

    return _SeriesComputer(dates, series)


def cohort_periods(companies_subset, computers):
    """Sorted union of the actual reporting dates across the selected companies."""
    alld = set()
    for c in companies_subset:
        alld.update(computers[c['ticker']].dates)
    return sorted(alld)


def metric_timeseries_frame(companies_subset, metric, computers, convert=True):
    """Wide DataFrame index=actual period-end dates (union), columns=tickers, for
    `metric`. Missing cells are NaN (gaps), never 0. Absolute money metrics are
    converted to the display currency per company (each from its own currency)."""
    is_abs = metric_is_absolute(metric)
    periods = cohort_periods(companies_subset, computers)
    data = {}
    for c in companies_subset:
        comp = computers[c['ticker']]
        s = comp.series(metric)
        factor = 1.0
        if is_abs and convert:
            factor, _ = display_factor(company_currency(c, 'financial'))
        col = []
        for d in periods:
            v = s.get(d)
            if v is not None and is_abs and convert:
                v = v * factor
            col.append(v)
        data[c['ticker']] = col
    df = pd.DataFrame(data, index=periods)
    df.index.name = "Period"
    return df


def timeseries_long(companies_subset, metric, computers, convert=True):
    """Long/tidy frame for charting: one row per (company, its-own period). Missing
    values stay None so lines break (gap) and bars are omitted — no fake 0s."""
    is_abs = metric_is_absolute(metric)
    rows = []
    for c in companies_subset:
        comp = computers[c['ticker']]
        s = comp.series(metric)
        factor = 1.0
        if is_abs and convert:
            factor, _ = display_factor(company_currency(c, 'financial'))
        for d in comp.dates:
            v = s.get(d)
            if v is not None and is_abs and convert:
                v = v * factor
            rows.append({"Period": d, "Company": c['ticker'], "Value": v})
    df = pd.DataFrame(rows, columns=["Period", "Company", "Value"])
    return df


def style_timeseries_fig(fig, single, is_abs):
    """Shared styling: full-precision hover, single-company labels, clean layout."""
    xfmt = "%{x}" if is_abs else "%{x|%Y-%m-%d}"
    fig.update_traces(hovertemplate="%{fullData.name} · " + xfmt + "<br>%{y:,.4f}<extra></extra>")
    if single:
        fig.update_traces(texttemplate="%{y:,.1f}",
                          textposition=("outside" if is_abs else "top center"),
                          textfont=dict(size=11))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=46, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', title=None),
        xaxis=dict(showgrid=False, title=None),
        showlegend=(not single),
        legend=dict(orientation='h', yanchor='top', y=-0.18, x=0, font=dict(size=10), title=None),
        bargap=0.2, bargroupgap=0.06,
    )
    return fig


def build_timeseries_fig(companies_subset, metric, computers, cmap, focus_ticker=None):
    """One chart for `metric`: grouped BARS for absolutes (categorical date axis,
    uniform bars), LINES for ratios/margins (true date axis, gaps where missing)."""
    is_abs = metric_is_absolute(metric)
    single = len(companies_subset) == 1
    title = metric_label(metric)
    if is_abs:
        # Bars: categorical axis of the actual period-end dates -> uniform bars,
        # same-date companies grouped, no fabricated bars for missing periods.
        df_wide = metric_timeseries_frame(companies_subset, metric, computers)
        long_df = df_wide.reset_index().melt(id_vars="Period", var_name="Company", value_name="Value")
        fig = px.bar(long_df, x="Period", y="Value", color="Company",
                     barmode="group", color_discrete_map=cmap, title=title)
        fig.update_xaxes(type="category")
    else:
        # Lines: true date axis at each company's own reporting dates; None -> gap.
        long_df = timeseries_long(companies_subset, metric, computers)
        long_df = long_df.copy()
        long_df["Period"] = pd.to_datetime(long_df["Period"], errors="coerce")
        fig = px.line(long_df, x="Period", y="Value", color="Company",
                      markers=True, color_discrete_map=cmap, title=title)
        fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
        if not single and focus_ticker is not None and focus_ticker in set(long_df["Company"]):
            fig.update_traces(selector=dict(name=focus_ticker), line=dict(width=4.5))
    return style_timeseries_fig(fig, single, is_abs)


# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------
if st.session_state.get('run_requested', False):
    if not tickers_text:
        st.error("Enter tickers.")
    else:
        t_list = [t.strip() for t in tickers_text.split(",") if t.strip()]
        with st.spinner("Requesting analysis from backend..."):
            data = get_data(t_list, fx_signature)
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

        # Currency flags + provenance
        render_currency_banner(data)
        
        # --- Tabs ---
    tabs = st.tabs(["🏆 Strategic Comparison", "💎 DuPont Analysis", "📊 Financial Statements", "🧪 Advanced Sandbox", "📑 Deep Dive", "📤 Data Export"])
    
    # 1. Strategic Comparison (Heatmap + Charts)
    with tabs[0]:
        st.subheader("Comparative Analysis")
        
        # Focus Selector — dynamically add Custom KPIs if any have been saved
        focus_options = ["Overview", "Liquidity", "Solvency", "Efficiency", "Returns", "Valuation"]

        # Define Metrics per Focus
        metric_groups = {
            "Overview": ["Revenue ($B)", "Revenue CAGR (3y)", "Gross Margin %", "EBITDA Margin %", "Net Margin %", "ROIC %", "CCC (Days)"],
            "Liquidity": ["Current Ratio", "Quick Ratio", "Cash Ratio", "CCC (Days)", "DIO (Days)", "DSO (Days)", "DPO (Days)"],
            "Solvency": ["Net Debt / EBITDA", "Interest Coverage", "Debt / Equity", "Financial Leverage", "Quick Ratio"],
            "Efficiency": ["Asset Turnover", "ROIC %", "ROE %", "Fixed Asset Turnover", "CCC (Days)"],
            "Returns": ["ROE %", "ROIC %", "Shareholder Yield %", "Net Margin %", "Dupont ROE"],
            "Valuation": ["Market Cap ($B)", "Enterprise Value ($B)", "Revenue ($B)", "Net Income ($B)", "EBITDA ($B)", "EV/EBITDA", "P/E Ratio"]
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
                    # Handle missing keys gracefully; convert absolutes to display ccy
                    row[metric_label(m)] = metric_value(c, m)
                matrix.append(row)

            df_heat = pd.DataFrame(matrix).set_index("Ticker")
            st.dataframe(
                df_heat.style.background_gradient(cmap="RdYlGn", axis=0),
                use_container_width=True
            )
            st.caption(currency_note())
            
        # B. Individual KPI Graphics
        st.markdown(f"### Key {focus_mode} Indicators")
        
        # Formula definitions for clickable help
        kpi_formulas = {
            "Revenue ($B)": {"formula": "Total Revenue / 1,000,000,000", "meaning": "Top-line sales in billions.", "source": "Income Statement"},
            "Revenue CAGR (3y)": {"formula": "(Revenue_latest / Revenue_3y_ago)^(1/3) − 1", "meaning": "Compound annual growth rate of revenue over 3 years.", "source": "Income Statement"},
            "EBITDA CAGR (3y)": {"formula": "(EBITDA_latest / EBITDA_3y_ago)^(1/3) − 1", "meaning": "Compound annual growth rate of EBITDA over 3 years.", "source": "Income Statement"},
            "Gross Margin %": {"formula": "(Revenue − COGS) / Revenue × 100", "meaning": "How much of each dollar of revenue is retained after direct costs.", "source": "Income Statement"},
            "EBITDA Margin %": {"formula": "EBITDA / Revenue × 100", "meaning": "Profitability before interest, taxes, depreciation & amortization.", "source": "Income Statement"},
            "Net Margin %": {"formula": "Net Income / Revenue × 100", "meaning": "Bottom-line profitability — what % of revenue becomes profit.", "source": "Income Statement"},
            "Operating Margin %": {"formula": "Operating Income / Revenue × 100", "meaning": "Profitability from core operations before interest and taxes.", "source": "Income Statement"},
            "Current Ratio": {"formula": "Current Assets / Current Liabilities", "meaning": "Can the company cover short-term obligations? >1 = healthy.", "source": "Balance Sheet (or Yahoo metadata)"},
            "Quick Ratio": {"formula": "(Current Assets − Inventory) / Current Liabilities", "meaning": "Like Current Ratio but excludes inventory (harder to liquidate).", "source": "Balance Sheet (or Yahoo metadata)"},
            "Cash Ratio": {"formula": "Cash & Equivalents / Current Liabilities", "meaning": "Strictest liquidity test — only cash counted.", "source": "Balance Sheet"},
            "CCC (Days)": {"formula": "DIO + DSO − DPO", "meaning": "Cash Conversion Cycle: days to convert inventory investment into cash. Lower = better.", "source": "Computed from IS + BS"},
            "DIO (Days)": {"formula": "(Inventory / COGS) × 365", "meaning": "Days to sell inventory. Lower = faster turnover.", "source": "Balance Sheet + Income Statement"},
            "DSO (Days)": {"formula": "(Accounts Receivable / Revenue) × 365", "meaning": "Days to collect payment from customers. Lower = faster collection.", "source": "Balance Sheet + Income Statement"},
            "DPO (Days)": {"formula": "(Accounts Payable / COGS) × 365", "meaning": "Days to pay suppliers. Higher = better cash preservation.", "source": "Balance Sheet + Income Statement"},
            "Net Debt / EBITDA": {"formula": "(Total Debt − Cash) / EBITDA", "meaning": "Leverage: how many years of EBITDA to repay net debt. <3 is healthy.", "source": "Balance Sheet + Income Statement"},
            "Interest Coverage": {"formula": "EBITDA / Interest Expense", "meaning": "Can the company afford its interest payments? >3 is comfortable.", "source": "Income Statement"},
            "Debt / Equity": {"formula": "Total Debt / Shareholders' Equity", "meaning": "Financial leverage. Higher = more debt-financed.", "source": "Balance Sheet"},
            "Financial Leverage": {"formula": "Total Assets / Shareholders' Equity", "meaning": "Asset-to-equity multiplier. Higher = more leverage.", "source": "Balance Sheet"},
            "Asset Turnover": {"formula": "Revenue / Total Assets", "meaning": "Efficiency: how much revenue per dollar of assets.", "source": "Income Statement + Balance Sheet"},
            "Fixed Asset Turnover": {"formula": "Revenue / Net PP&E", "meaning": "Revenue generated per dollar of fixed assets (property, plant, equipment).", "source": "Income Statement + Balance Sheet"},
            "ROIC %": {"formula": "NOPAT / Invested Capital × 100", "meaning": "Return on Invested Capital — core profitability measure. NOPAT = EBIT × (1 - tax rate).", "source": "Income Statement + Balance Sheet"},
            "ROE %": {"formula": "Net Income / Shareholders' Equity × 100", "meaning": "Return on Equity — profit per dollar of shareholder investment.", "source": "Income Statement + Balance Sheet"},
            "ROA %": {"formula": "Net Income / Total Assets × 100", "meaning": "Return on Assets — profit per dollar of total assets.", "source": "Income Statement + Balance Sheet"},
            "Dupont ROE": {"formula": "Net Margin × Asset Turnover × Financial Leverage", "meaning": "Decomposes ROE into three drivers: profitability, efficiency, leverage.", "source": "Computed"},
            "Shareholder Yield %": {"formula": "(Dividends + Buybacks) / Market Cap × 100", "meaning": "Total cash returned to shareholders as % of market cap.", "source": "Cash Flow + Market Data"},
            "EV / EBITDA": {"formula": "Enterprise Value / EBITDA", "meaning": "Valuation multiple. Lower = cheaper relative to earnings.", "source": "Market Data + Income Statement"},
            "P/E": {"formula": "Market Price / Earnings Per Share", "meaning": "Price-to-Earnings ratio. How much investors pay per dollar of earnings.", "source": "Market Data + Income Statement"},
            "P/B": {"formula": "Market Price / Book Value Per Share", "meaning": "Price-to-Book ratio. >1 means market values company above its book value.", "source": "Market Data + Balance Sheet"},
            "Dividend Yield %": {"formula": "Annual Dividends Per Share / Share Price × 100", "meaning": "Annual dividend income as % of share price.", "source": "Market Data"},
            "Net Income ($B)": {"formula": "Net Income / 1,000,000,000", "meaning": "Bottom-line profit in billions (converted to the display currency).", "source": "Income Statement"},
            "EBITDA ($B)": {"formula": "EBITDA / 1,000,000,000", "meaning": "Earnings before interest, taxes, depreciation & amortization, in billions.", "source": "Income Statement"},
            "Market Cap ($B)": {"formula": "Share Price × Shares Outstanding / 1e9", "meaning": "Total equity market value in billions (trading currency, converted to display).", "source": "Market Data"},
            "Enterprise Value ($B)": {"formula": "(Market Cap + Net Debt) / 1e9", "meaning": "Total firm value (equity + net debt) in billions. Market cap is FX-aligned to the reporting currency before summing — fixing the ADR mismatch.", "source": "Market Data + Balance Sheet"},
        }
        
        view_mode = st.radio("View", ["📊 Bar Charts", "🎯 Gauge View", "📈 Over Time"], horizontal=True, key="chart_view_mode")
        
        # Helper: get formula caption string
        def formula_caption(metric):
            if metric in kpi_formulas:
                f = kpi_formulas[metric]
                return f"*{f['formula']}* — {f['meaning']}"
            return ""
        
        if view_mode == "📊 Bar Charts":
            # --- Bar Chart View ---
            cols = st.columns(3)
            for i, metric in enumerate(metrics_to_show):
                col = cols[i % 3]
                with col:
                    chart_data = []
                    for c in companies:
                        chart_data.append({
                            "Ticker": c['ticker'],
                            "Value": metric_value(c, metric),
                            "Color": '#38bdf8' if c['ticker'] == focus_ticker else '#334155'
                        })
                    df_chart = pd.DataFrame(chart_data)

                    fig = px.bar(
                        df_chart, x="Ticker", y="Value", color="Ticker",
                        color_discrete_map={row['Ticker']: row['Color'] for _, row in df_chart.iterrows()},
                        title=metric_label(metric)
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
                    fc = formula_caption(metric)
                    if fc:
                        with st.expander("ℹ️", expanded=False):
                            st.caption(fc)
        
        elif view_mode == "🎯 Gauge View":
            # --- Gauge View (Industry Benchmark style — Premium) ---
            lower_is_better = {"CCC (Days)", "DIO (Days)", "DSO (Days)", 
                               "Net Debt / EBITDA", "Debt / Equity", "Financial Leverage"}
            
            for row_start in range(0, len(metrics_to_show), 2):
                row_metrics = metrics_to_show[row_start:row_start + 2]
                gauge_cols = st.columns(len(row_metrics))
                
                for g_idx, metric in enumerate(row_metrics):
                    with gauge_cols[g_idx]:
                        focus_val = metric_value(focus_company, metric)
                        if focus_val is None: focus_val = 0

                        cohort_vals = [metric_value(c, metric) for c in companies]
                        cohort_vals = [v for v in cohort_vals if v is not None]
                        if not cohort_vals: cohort_vals = [0]
                        
                        cohort_avg = np.mean(cohort_vals)
                        cohort_min = min(cohort_vals)
                        cohort_max = max(cohort_vals)
                        
                        spread = abs(cohort_max - cohort_min)
                        range_pad = max(spread * 0.3, abs(cohort_avg) * 0.2, 0.5)
                        gauge_min = min(cohort_min, focus_val) - range_pad
                        gauge_max = max(cohort_max, focus_val) + range_pad
                        
                        if "Ratio" in metric or "Turnover" in metric or "Coverage" in metric:
                            gauge_min = max(0, gauge_min)
                        
                        rng = gauge_max - gauge_min
                        if metric in lower_is_better:
                            bar_color = "#22c55e" if focus_val <= cohort_avg else "#ef4444"
                            zone_colors = [
                                "rgba(34, 197, 94, 0.3)", "rgba(132, 204, 22, 0.2)",
                                "rgba(234, 179, 8, 0.2)", "rgba(249, 115, 22, 0.2)",
                                "rgba(239, 68, 68, 0.3)",
                            ]
                        else:
                            bar_color = "#22c55e" if focus_val >= cohort_avg else "#ef4444"
                            zone_colors = [
                                "rgba(239, 68, 68, 0.3)", "rgba(249, 115, 22, 0.2)",
                                "rgba(234, 179, 8, 0.2)", "rgba(132, 204, 22, 0.2)",
                                "rgba(34, 197, 94, 0.3)",
                            ]
                        
                        steps = [{"range": [gauge_min + rng * (i/5), gauge_min + rng * ((i+1)/5)], "color": zone_colors[i]} for i in range(5)]
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=focus_val,
                            title={"text": f"<b style='font-size:15px'>{metric_label(metric)}</b><br><span style='font-size:12px;color:#888'>{focus_ticker} vs cohort avg</span>", "font": {"size": 16}},
                            number={"font": {"size": 36, "color": bar_color}, "valueformat": ",.2f" if abs(focus_val) < 100 else ",.0f"},
                            delta={
                                "reference": cohort_avg, "relative": False, "valueformat": ".2f",
                                "increasing": {"color": "#22c55e" if metric not in lower_is_better else "#ef4444"},
                                "decreasing": {"color": "#ef4444" if metric not in lower_is_better else "#22c55e"},
                                "suffix": " vs avg", "font": {"size": 13}
                            },
                            gauge={
                                "shape": "angular",
                                "axis": {"range": [gauge_min, gauge_max], "tickfont": {"size": 11}, "tickcolor": "#888"},
                                "bar": {"color": bar_color, "thickness": 0.4},
                                "bgcolor": "rgba(128,128,128,0.08)",
                                "borderwidth": 1, "bordercolor": "rgba(128,128,128,0.15)",
                                "steps": steps,
                                "threshold": {"line": {"color": "#f97316", "width": 4}, "thickness": 0.85, "value": cohort_avg}
                            }
                        ))
                        
                        fig.update_layout(height=280, margin=dict(l=30, r=30, t=80, b=20), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Clean info line: cohort avg + formula in one caption
                        fc = formula_caption(metric)
                        caption_text = f"🟠 Cohort avg: **{cohort_avg:.2f}**"
                        st.caption(caption_text)
                        if fc:
                            with st.expander("ℹ️", expanded=False):
                                st.caption(fc)
                        st.markdown("---")
        
        else:
            # --- Over Time View (multi-company) ---
            st.caption(
                "📈 KPIs plotted at each company's **actual fiscal period-end dates** "
                "(reporting dates differ across companies/countries); missing values are shown "
                "as **gaps, not zeros**. "
                + (f"Absolute figures (Revenue, Net Income, EBITDA) converted to **{target_currency}** per company at the latest FX rate; ratios are currency-neutral."
                   if target_currency != "NATIVE"
                   else "Absolute figures shown in each company's native currency.")
            )

            ot_selected = st.multiselect(
                "Companies to plot",
                options=ticker_list,
                default=ticker_list,
                key="overtime_companies",
            )
            if not ot_selected:
                st.info("Select at least one company to plot.")
                ot_selected = [focus_ticker]

            sel_companies = [c for c in companies if c['ticker'] in ot_selected]
            saved_kpis = st.session_state.get('saved_sandbox_kpis', {})
            computers = {c['ticker']: make_series_computer(c, 5, saved_kpis) for c in sel_companies}
            periods = cohort_periods(sel_companies, computers)

            if not periods:
                st.warning("No annual statement data available for time series.")
            else:
                cmap = company_color_map(ticker_list)

                computable = [m for m in metrics_to_show if (m in OVERTIME_METRICS or m in saved_kpis)]
                skipped = [m for m in metrics_to_show if m not in computable]
                if skipped:
                    st.caption("⏭️ No annual series for: " + ", ".join(skipped)
                               + " (point-in-time metrics like Market Cap / EV / P/E).")
                if not computable:
                    st.info("No time-series metrics in this focus group — try Overview, Returns, or Efficiency.")

                for row_start in range(0, len(computable), 2):
                    row_metrics = computable[row_start:row_start + 2]
                    ts_cols = st.columns(len(row_metrics))
                    for t_idx, metric in enumerate(row_metrics):
                        with ts_cols[t_idx]:
                            df_m = metric_timeseries_frame(sel_companies, metric, computers)
                            if df_m.dropna(how="all").empty:
                                st.caption(f"No data available for {metric_label(metric)}.")
                                continue

                            fig_ts = build_timeseries_fig(sel_companies, metric, computers, cmap, focus_ticker)
                            st.plotly_chart(fig_ts, use_container_width=True)

                            with st.expander("📋 Data table (copy / download)", expanded=False):
                                st.dataframe(df_m, use_container_width=True)
                                st.download_button(
                                    "⬇️ CSV", df_m.to_csv(),
                                    file_name=f"{metric}_over_time.csv", mime="text/csv",
                                    key=f"ot_dl_{row_start}_{t_idx}",
                                )

                            fc = formula_caption(metric)
                            if fc:
                                with st.expander("ℹ️ Formula", expanded=False):
                                    st.caption(fc)

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
        st.caption("High Margin (Premium/IP-driven) vs High Turnover (Volume/Retail). Bubble size = Revenue. " + currency_note())
        
        _rev_lbl = metric_label('Revenue ($B)')  # currency-aware label for the bubble dim
        scatter_data = []
        for c in companies:
            scatter_data.append({
                "Ticker": c['ticker'],
                "Net Margin %": c['metrics'].get('Net Margin %', 0),
                "Asset Turnover": c['metrics'].get('Asset Turnover', 0),
                "ROE %": c['metrics'].get('ROE %', 0),
                _rev_lbl: max(metric_value(c, 'Revenue ($B)') or 0.1, 0.1),
            })

        df_scatter = pd.DataFrame(scatter_data)

        fig_dupont = px.scatter(
            df_scatter,
            x="Asset Turnover",
            y="Net Margin %",
            size=_rev_lbl,
            color="ROE %",
            text="Ticker",
            hover_data=["ROE %", _rev_lbl],
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

        _stmt_fin = company_currency(focus_company, "financial")
        df_stmt = get_financial_statement(focus_company, stmt_type)

        _convert_stmt = False
        if target_currency != "NATIVE" and _stmt_fin and normalize_currency(_stmt_fin) != target_currency:
            _convert_stmt = st.checkbox(
                f"Convert to {target_currency}", value=False,
                help="Statements are shown in the reporting currency by default for auditability.",
            )
        if _convert_stmt:
            _f, _sc = display_factor(_stmt_fin)
            if _sc == target_currency:
                # Conversion actually resolved (the checkbox guard rules out identity).
                df_stmt = df_stmt * _f
                st.caption(f"Converted to **{target_currency}** at the latest FX rate (×{_f:.4f}).")
            else:
                st.warning(f"FX rate {_stmt_fin}→{target_currency} unavailable — showing **{_stmt_fin}** (not converted).")
        else:
            st.caption(f"Reported in **{_stmt_fin or 'native currency'}**.")

        if not df_stmt.empty:
            st.dataframe(df_stmt.style.format("{:,.0f}"), height=600, use_container_width=True)
        else:
            st.warning("No data available for this statement.")

    # 4. Advanced Sandbox 3.0
    with tabs[3]:
        st.subheader("🧪 Custom KPI Lab")

        if data.get('currency', {}).get('mixed_cohort'):
            st.warning("⚠️ This cohort mixes currencies. Raw statement values below are in each company's native reporting currency and are **not** FX-converted here — cross-company comparisons and currency-mixing formulas can be misleading.")

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
            remove_selection = st.multiselect(
                "Select KPIs to remove",
                options=list(saved_kpis.keys()),
                default=[],
                key="remove_saved_kpis"
            )
            if remove_selection and st.button("Remove selected KPIs", key="remove_saved_kpis_button"):
                for kpi_name in remove_selection:
                    st.session_state['saved_sandbox_kpis'].pop(kpi_name, None)
                for comp in companies:
                    metrics = comp.get('metrics')
                    if isinstance(metrics, dict):
                        for kpi_name in remove_selection:
                            metrics.pop(kpi_name, None)
                st.session_state["remove_saved_kpis"] = []
                st.success(f"Removed: {', '.join(remove_selection)}")
                saved_kpis = st.session_state.get('saved_sandbox_kpis', {})
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
        st.subheader("⚖️ Governance & Risk Profile")
        st.caption("Assessing board, audit, and compensation risks. Scores supplied by Institutional Shareholder Services (ISS).")
        
        with st.expander("ℹ️ Methodology & Sources"):
            st.markdown("""
            **Governance QualityScores** are provided by **Institutional Shareholder Services (ISS)**.
            *   **Scale**: 1 to 10.
            *   **1** = Low Risk (Best) 🟢
            *   **10** = High Risk (Worst) 🔴
            *   **Data Source**: Yahoo Finance / ISS Governance.
            """)
        
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

            # Cash flow is in the reporting currency; convert to display currency.
            _cf_fin = company_currency(focus_company, "financial")
            cf_factor, cf_ccy = display_factor(_cf_fin)
            sym = currency_symbol(cf_ccy)
            unit = money_suffix(cf_ccy)  # e.g. "(B EUR)"

            def b(v):
                """Native value -> display currency, in billions."""
                return (v * cf_factor) / 1e9

            def get_latest_val(key):
                """Get latest year value from cash flow dict (native currency)."""
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
                st.warning(f"⚠️ Operating Cash Flow is **negative** ({sym}{b(ocf):.2f}B). The company burned cash from operations this period.")

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
                    x=list(cf_items.keys()), y=[b(v) for v in cf_items.values()],
                    title=f"Cash Flow Components {unit}",
                    color=[v > 0 for v in cf_items.values()],
                    color_discrete_map={True: "#4ade80", False: "#f87171"},
                    labels={"x": "", "y": f"{cf_ccy} Billions", "color": ""},
                )
                fig_cf.update_layout(
                    showlegend=False, height=400,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
                )
                fig_cf.update_traces(text=[f"{sym}{b(v):.1f}B" for v in cf_items.values()], textposition="outside")
                st.plotly_chart(fig_cf, use_container_width=True)
            else:
                # Normal waterfall for positive OCF
                labels = ["Operating Cash Flow"]
                values = [b(ocf)]
                measures = ["absolute"]

                if capex != 0:
                    labels.append("Capex")
                    values.append(b(capex))
                    measures.append("relative")
                if div != 0:
                    labels.append("Dividends")
                    values.append(b(div))
                    measures.append("relative")
                if buyback != 0:
                    labels.append("Buybacks")
                    values.append(b(buyback))
                    measures.append("relative")
                if acq != 0:
                    labels.append("M&A")
                    values.append(b(acq))
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
                    text=[f"{sym}{v:.1f}B" for v in values[:-1]] + [f"{sym}{b(remaining):.1f}B"],
                    connector={"line": {"color": "rgba(255,255,255,0.2)"}},
                    increasing={"marker": {"color": "#4ade80"}},
                    decreasing={"marker": {"color": "#f87171"}},
                    totals={"marker": {"color": "#60a5fa"}},
                ))

                fig_bridge.update_layout(
                    title=f"Cash Flow Allocation Waterfall {unit}",
                    showlegend=False,
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title=f"{cf_ccy} Billions"),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_bridge, use_container_width=True)
            st.caption(currency_note())

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
        
        _trend_suffix = money_suffix(target_currency) if target_currency != "NATIVE" else "($B)"
        for t_idx, (trend_label, stmt_name, key_opts) in enumerate(trend_metrics):
            with trend_cols[t_idx]:
                fig_trend = go.Figure()

                for c in companies:
                    series = extract_annual_series(c, stmt_name, key_opts)
                    if series:
                        sorted_dates = sorted(series.keys())
                        dates = sorted_dates[-5:]  # last 5 years
                        # Each company converted from its own reporting currency.
                        _ccy_c = company_currency(c, "financial")
                        _cf, _shown_c = display_factor(_ccy_c)
                        vals = [((series[d] * _cf / 1e9) if series[d] is not None else None) for d in dates]
                        # Use year labels
                        years = [d[:4] if len(d) >= 4 else d for d in dates]

                        # If this company's FX conversion failed, its line is in its
                        # native currency, not the axis label — mark it.
                        _trace_name = c['ticker']
                        if target_currency != "NATIVE" and _ccy_c and _shown_c != target_currency:
                            _trace_name = f"{c['ticker']} (native {_ccy_c})"

                        fig_trend.add_trace(go.Scatter(
                            x=years, y=vals,
                            mode='lines+markers',
                            name=_trace_name,
                            line=dict(width=3 if c['ticker'] == focus_ticker else 1.5),
                            opacity=1.0 if c['ticker'] == focus_ticker else 0.5,
                        ))

                fig_trend.update_layout(
                    title=f"{trend_label} {_trend_suffix}",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
                    xaxis=dict(type='category', showgrid=False),  # categorical -> no 2022.5 ticks
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
            "💵 Size & Value": ["Market Cap ($B)", "Enterprise Value ($B)", "Net Income ($B)", "EBITDA ($B)"],
        }

        # Compute cohort averages for comparison (absolutes converted to display ccy)
        cohort_avgs = {}
        for metric_list in metric_categories.values():
            for met in metric_list:
                vals = [metric_value(c, met) for c in companies if c['metrics'].get(met) is not None]
                vals = [v for v in vals if v is not None]
                if vals:
                    cohort_avgs[met] = np.mean(vals)

        for cat_name, cat_metrics in metric_categories.items():
            with st.expander(cat_name, expanded=True):
                n_cols = min(len(cat_metrics), 4)
                cols = st.columns(n_cols)
                for m_idx, met in enumerate(cat_metrics):
                    with cols[m_idx % n_cols]:
                        present = m.get(met, None) is not None
                        val = metric_value(focus_company, met) if present else None
                        avg = cohort_avgs.get(met, None)

                        if val is not None and avg is not None:
                            diff = val - avg
                            # Format delta string
                            if abs(val) > 100:
                                delta_str = f"{diff:+,.0f} vs avg"
                            else:
                                delta_str = f"{diff:+.2f} vs avg"
                            st.metric(metric_label(met), f"{val:,.2f}" if isinstance(val, float) else str(val), delta=delta_str, delta_color="normal")
                        elif val is not None:
                            st.metric(metric_label(met), f"{val:,.2f}" if isinstance(val, float) else str(val))
                        else:
                            st.metric(metric_label(met), "—")




    # 6. Data Export
    with tabs[5]:
        st.subheader("📤 Data Export")
        st.caption(
            "Pick companies (rows) and fields (columns), then copy or download the table. "
            "Computed money KPIs are converted to the display currency; raw statement line "
            "items are in each company's native reporting currency (see the Currency column)."
        )

        export_mode = st.radio(
            "Table type",
            ["Snapshot (companies × fields)", "Time series (one field × years)"],
            horizontal=True, key="export_mode",
        )

        # Available fields: computed metrics + raw statement line items
        metric_fields = sorted({k for c in companies for k in c.get('metrics', {}).keys() if k != 'Profile'})
        raw_fields_set = set()
        for c in companies:
            annual = (c.get('raw_data', {}) or {}).get('financials', {}).get('annual', {}) or {}
            for stmt in annual.values():
                if isinstance(stmt, dict):
                    raw_fields_set.update(stmt.keys())
        all_export_fields = list(dict.fromkeys(metric_fields + sorted(raw_fields_set)))

        if export_mode.startswith("Snapshot"):
            ex_rows = st.multiselect("Companies (rows)", ticker_list, default=ticker_list, key="export_rows")
            _default_cols = [c for c in ["Revenue ($B)", "Net Income ($B)", "EBITDA ($B)", "Net Margin %", "ROE %", "EV/EBITDA"] if c in metric_fields]
            ex_cols = st.multiselect("Fields (columns)", all_export_fields, default=_default_cols, key="export_cols")
            if ex_rows and ex_cols:
                rows = []
                for t in ex_rows:
                    c = next((x for x in companies if x['ticker'] == t), None)
                    if not c:
                        continue
                    row = {"Ticker": t, "Currency": company_currency(c, 'financial') or '?'}
                    for f in ex_cols:
                        if metric_is_absolute(f):
                            row[metric_label(f)] = metric_value(c, f)
                        elif f in c.get('metrics', {}):
                            row[f] = c['metrics'].get(f)
                        else:
                            row[f] = get_value_for_ticker(companies, t, f)
                    rows.append(row)
                df_ex = pd.DataFrame(rows).set_index("Ticker")
                st.dataframe(df_ex, use_container_width=True)
                st.download_button("⬇️ Download CSV", df_ex.to_csv(), "export_snapshot.csv", "text/csv")
                with st.expander("📋 Copy values (TSV — paste straight into Excel / Sheets)"):
                    st.code(df_ex.to_csv(sep='\t'), language=None)
            else:
                st.info("Select at least one company and one field.")
        else:
            ts_options = [m for m in metric_fields if m in OVERTIME_METRICS] + sorted(st.session_state.get('saved_sandbox_kpis', {}).keys())
            if not ts_options:
                st.info("No time-series fields available for this cohort.")
            else:
                ts_field = st.selectbox("Field (one annual KPI)", ts_options, key="export_ts_field")
                ex_rows = st.multiselect("Companies", ticker_list, default=ticker_list, key="export_ts_rows")
                sel = [c for c in companies if c['ticker'] in ex_rows]
                computers = {c['ticker']: make_series_computer(c, 5, st.session_state.get('saved_sandbox_kpis', {})) for c in sel}
                periods = cohort_periods(sel, computers) if sel else []
                if ts_field and sel and periods:
                    df_ts = metric_timeseries_frame(sel, ts_field, computers)
                    st.dataframe(df_ts, use_container_width=True)
                    st.download_button("⬇️ Download CSV", df_ts.to_csv(), "export_timeseries.csv", "text/csv")
                    with st.expander("📋 Copy values (TSV — paste straight into Excel / Sheets)"):
                        st.code(df_ts.to_csv(sep='\t'), language=None)

                    # Quick chart of the exported series (gaps for missing periods)
                    figx = build_timeseries_fig(sel, ts_field, computers, company_color_map(ticker_list))
                    figx.update_layout(height=400)
                    st.plotly_chart(figx, use_container_width=True)
                else:
                    st.info("Pick a field and at least one company.")


else:
    st.write("👈 Select tickers to begin.")
