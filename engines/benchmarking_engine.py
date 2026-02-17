import pandas as pd
import numpy as np
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime
import sys
import os

# Import sibling module (relative import for package compatibility)
from .financial_analytics import FinancialAnalyticsEngine

class BenchmarkingEngine:
    """
    Calculates standardized KPIs for a cohort of companies and computes aggregate statistics.
    Pure Python/Pandas implementation - No LLM.
    """
    
    def __init__(self):
        self.financial_engine = FinancialAnalyticsEngine()

    def run_benchmark(self, tickers: List[str], detailed: bool = False) -> Dict[str, Any]:
        """
        Main entry point: Fetches data, calculates KPIs, and aggregates stats.
        If detailed=True, includes the full Universal Extraction data for each company.
        """
        print(f"\n[BENCHMARK ENGINE] Starting analysis for cohort: {tickers}")
        
        company_results = []
        
        for ticker in tickers:
            try:
                # 1. Fetch Full Data
                raw_data = self.financial_engine.fetch_full_analysis(ticker)
                if not raw_data:
                    print(f"Skipping {ticker} (Data fetch failed)")
                    continue
                
                # 2. Calculate KPIs
                kpis = self._calculate_company_kpis(ticker, raw_data)
                
                # 3. Append to Results
                result_entry = kpis # kpis is { "ticker": ..., "metrics": ... }
                if detailed:
                    result_entry["raw_data"] = raw_data
                
                company_results.append(result_entry)
                
            except Exception as e:
                print(f"Error benchmarking {ticker}: {e}")
                import traceback
                traceback.print_exc()

        # 3. Compute Cohort Stats
        cohort_stats = self._compute_cohort_stats(company_results)
        
        final_output = {
            "meta": {
                "cohort": tickers,
                "generated_at": str(datetime.now().date()),
                "count": len(company_results)
            },
            "cohort_stats": cohort_stats,
            "companies": company_results
        }
        
        return final_output

    def _calculate_company_kpis(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derives standard financial metrics from the raw Universal Extraction data.
        """
        print(f"  > Calculating KPIs for {ticker}...")
        
        # Shortcut to annual statements (most relevant for benchmarking)
        # Note: Universal Extraction puts them in output['financials']['annual']
        financials = data.get("financials", {}).get("annual", {})
        inc = pd.DataFrame(financials.get("income_statement", {}))
        bs = pd.DataFrame(financials.get("balance_sheet", {}))
        cf = pd.DataFrame(financials.get("cash_flow", {}))
        
        # Helper to safely get the latest year value
        def get_latest(df, field):
            if field not in df.columns: return 0.0
            try:
                # Filter out None/NaN
                series = df[field].dropna().sort_index(ascending=False)
                if series.empty: return 0.0
                return float(series.iloc[0])
            except:
                return 0.0

        def get_latest_any(df, fields: List[str]):
            """Tries multiple fields in order and returns the first non-zero found."""
            for f in fields:
                val = get_latest(df, f)
                if val != 0.0: return val
            return 0.0

        def get_cagr(df, field, years=3):
            if field not in df.columns: return 0.0
            series = df[field].dropna().sort_index(ascending=True) # Oldest to Newest
            if len(series) < 2: return 0.0
            
            start_val = series.iloc[0]
            end_val = series.iloc[-1]
            period = len(series) - 1
            
            if start_val <= 0 or period == 0: return 0.0 # Avoid complex numbers
            
            return ( (end_val / start_val) ** (1/period) ) - 1

        # --- KPI DEFINITIONS ---
        
        # Helper for safer division
        def safe_div(n, d):
            return (n / d) if d else 0.0

        # 1. Growth
        revenue_cagr = get_cagr(inc, "Total Revenue")
        
        # 2. Profitability (Margins)
        rev = get_latest_any(inc, ["Total Revenue", "TotalRevenue", "Operating Revenue", "Revenue"])
        gp = get_latest_any(inc, ["Gross Profit", "GrossProfit"])
        ebitda = get_latest_any(inc, ["EBITDA", "Normalized EBITDA"])
        ebit = get_latest_any(inc, ["EBIT", "Operating Income", "OperatingIncome"])
        net_income = get_latest_any(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
        
        gross_margin = safe_div(gp, rev)
        ebitda_margin = safe_div(ebitda, rev)
        net_margin = safe_div(net_income, rev)
        
        # 3. Efficiency / Ops (CCC)
        inventory = get_latest(bs, "Inventory")
        receivables = get_latest_any(bs, ["Receivables", "Accounts Receivable"])
        payables = get_latest_any(bs, ["Accounts Payable", "Payables"])
        cogs = get_latest(inc, "Cost Of Revenue")
        
        dio = safe_div(inventory, cogs) * 365
        dso = safe_div(receivables, rev) * 365
        dpo = safe_div(payables, cogs) * 365
        ccc = dio + dso - dpo

        # 4. Returns (ROIC Granular)
        # NOPAT = EBIT * (1 - Tax Rate)
        tax_provision = get_latest(inc, "Tax Provision")
        pretax_inc = get_latest(inc, "Pretax Income")
        effective_tax_rate = safe_div(tax_provision, pretax_inc)
        # Fallback to 21% if weird
        if effective_tax_rate <= 0 or effective_tax_rate > 0.5: effective_tax_rate = 0.21
        
        nopat = ebit * (1 - effective_tax_rate)
        
        # Invested Capital
        equity = get_latest_any(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "TotalEquity"])
        debt = get_latest(bs, "Total Debt")
        cash = get_latest_any(bs, ["Cash And Cash Equivalents", "CashAndCashEquivalents"])
        invested_capital = get_latest_any(bs, ["Invested Capital", "InvestedCapital"])
        
        if not invested_capital:
            invested_capital = (equity + debt - cash)

        roic = safe_div(nopat, invested_capital)
        roe = safe_div(net_income, equity)
        
        # 5. Solvency & Credit
        net_debt = get_latest(bs, "Net Debt")
        if not net_debt: net_debt = debt - cash
        interest_expense = get_latest(inc, "Interest Expense")
        
        net_debt_ebitda = safe_div(net_debt, ebitda)
        interest_coverage = safe_div(ebit, interest_expense)

        # 6. Cash Flow & Quality
        ocf = get_latest(cf, "Operating Cash Flow")
        fcf = get_latest(cf, "Free Cash Flow")
        
        fcf_conversion = safe_div(fcf, ebitda)
        quality_of_earnings = safe_div(ocf, net_income)

        # 7. Capital Allocation
        capex = abs(get_latest(cf, "Capital Expenditure"))
        rnd = get_latest(inc, "Research And Development")
        dividends = abs(get_latest(cf, "Cash Dividends Paid"))
        buybacks = abs(get_latest(cf, "Repurchase Of Capital Stock"))
        
        # Market Data
        # Note: Universal Extraction puts stock.info into "meta"
        meta = data.get("meta", {})
        
        # Safe Metadata Extraction (Handle None values explicitly)
        mkt_cap = meta.get("marketCap") or meta.get("market_cap") or 0
        
        # Enterprise Value
        # Net Debt = Debt - Cash. If None, assume 0.
        net_debt_val = net_debt if net_debt is not None else 0
        ev = mkt_cap + net_debt_val
        
        # P/E Ratio
        pe_ratio = meta.get("trailingPE") or meta.get("forwardPE") or 0.0

        ev_ebitda = safe_div(ev, ebitda)

        shareholder_yield = safe_div((dividends + buybacks), mkt_cap)

        return {
            "ticker": ticker,
            "metrics": {
                "Revenue ($B)": round(rev / 1e9, 2),
                "Revenue CAGR (3y)": round(revenue_cagr * 100, 1),
                
                # Margins
                "Gross Margin %": round(gross_margin * 100, 1),
                "EBITDA Margin %": round(ebitda_margin * 100, 1),
                "Net Margin %": round(net_margin * 100, 1),
                
                # Efficiency
                "CCC (Days)": round(ccc, 1),
                "DIO (Days)": round(dio, 1),
                "DSO (Days)": round(dso, 1),
                "DPO (Days)": round(dpo, 1),
                
                # Liquidity (Yahoo Finance uses 'Current Assets' / 'Current Liabilities')
                "Current Ratio": round(safe_div(get_latest_any(bs, ["Current Assets", "Total Current Assets"]), get_latest_any(bs, ["Current Liabilities", "Total Current Liabilities"])), 2),
                "Quick Ratio": round(safe_div(
                    (get_latest_any(bs, ["Current Assets", "Total Current Assets"]) - get_latest(bs, "Inventory")), 
                    get_latest_any(bs, ["Current Liabilities", "Total Current Liabilities"])
                ), 2),
                "Cash Ratio": round(safe_div(cash, get_latest_any(bs, ["Current Liabilities", "Total Current Liabilities"])), 2),

                # Returns
                "ROIC %": round(roic * 100, 1),
                "ROE %": round(roe * 100, 1),
                
                # Solvency
                "Net Debt / EBITDA": round(net_debt_ebitda, 2),
                "Interest Coverage": round(interest_coverage, 2),
                "Debt / Equity": round(safe_div(debt, equity), 2),
                
                # Quality & Allocation
                "FCF Conversion %": round(fcf_conversion * 100, 1),
                "Quality of Earnings": round(quality_of_earnings, 2),
                "Capex Intensity %": round(safe_div(capex, rev) * 100, 1),
                "R&D Intensity %": round(safe_div(rnd, rev) * 100, 1),
                "Shareholder Yield %": round(shareholder_yield * 100, 1),
                
                # Expert: Governance & Risks (from Meta)
                "Audit Risk": meta.get('auditRisk') or 0,
                "Board Risk": meta.get('boardRisk') or 0,
                "Compensation Risk": meta.get('compensationRisk') or 0,
                "Shareholder Rights Risk": meta.get('shareholderRightsRisk') or 0,
                "Overall Risk": meta.get('overallRisk') or 0,
                "Beta": round(meta.get('beta') if meta.get('beta') else 0, 2),
                "Inst. Ownership %": round((meta.get('heldPercentInstitutions') or 0) * 100, 1),
                "Insider Ownership %": round((meta.get('heldPercentInsiders') or 0) * 100, 1),
                
                # Company Profile (Hidden from Heatmap, used in Deep Dive)
                "Profile": {
                    "Summary": meta.get('longBusinessSummary') or meta.get('description', 'No description available.'),
                    "Sector": meta.get('sector', 'N/A'),
                    "Industry": meta.get('industry', 'N/A'),
                    "Employees": meta.get('fullTimeEmployees', 'N/A'),
                    "City": meta.get('city', 'N/A'),
                    "Country": meta.get('country', 'N/A'),
                    "Website": meta.get('website', 'N/A')
                },

                # DuPont Drivers
                "Asset Turnover": round(safe_div(rev, get_latest_any(bs, ["Total Assets", "TotalAssets"])), 2),
                "Financial Leverage": round(safe_div(get_latest_any(bs, ["Total Assets", "TotalAssets"]), equity), 2),
                "DuPont ROE": round(net_margin * safe_div(rev, get_latest_any(bs, ["Total Assets", "TotalAssets"])) * safe_div(get_latest_any(bs, ["Total Assets", "TotalAssets"]), equity) * 100, 1),
                
                # Efficiency (additional)
                "Fixed Asset Turnover": round(safe_div(rev, get_latest_any(bs, ["Net PPE", "Property Plant And Equipment", "Gross PPE"])), 2),
                
                # Valuations
                "EV/EBITDA": round(ev_ebitda, 1),
                "P/E Ratio": round(pe_ratio if pe_ratio else 0.0, 1)
            }
        }

    def _compute_cohort_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates individual company KPIs into cohort statistics (Avg, Median, High, Low).
        """
        if not results: return {}
        
        # Pivot to: { "Gross Margin %": [50, 60, 40], ... }
        metrics_lists = {}
        first_metrics = results[0]["metrics"]
        
        for k in first_metrics.keys():
            # Skip non-numeric fields like Profile
            if k == "Profile": continue
            
            # Robust get: Handle missing key AND None value
            # If value is string (e.g. "Low"), this might fail later in np.mean, caught by try/except
            metrics_lists[k] = [(r["metrics"].get(k) or 0.0) for r in results]

        stats = {}
        for k, values in metrics_lists.items():
            try:
                # Filter out pure zeros if they seem like errors? No, keep them for now.
                # Ensure values are numeric
                numeric_values = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                
                if not numeric_values:
                    stats[k] = {"avg": 0, "median": 0, "min": 0, "max": 0}
                    continue
                    
                arr = np.array(numeric_values)
                stats[k] = {
                    "avg": round(float(np.mean(arr)), 2),
                    "median": round(float(np.median(arr)), 2),
                    "min": round(float(np.min(arr)), 2),
                    "max": round(float(np.max(arr)), 2)
                }
            except Exception as e:
                # If metric is text-based or fails
                # print(f"Skipping stats for {k}: {e}")
                stats[k] = {"avg": 0, "median": 0, "min": 0, "max": 0}
            
        return stats

if __name__ == "__main__":
    pass
