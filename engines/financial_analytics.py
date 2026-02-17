import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialAnalyticsEngine:
    """
    The 'Senior Analyst' Engine (V2 - Universal Extraction).
    Fetches DEEP historical financials and market statistics.
    Now extracts ALL available fields from Yahoo Finance instead of a selected subset.
    """
    
    def __init__(self):
        # We don't need a log file for the Streamlit app
        pass

    def fetch_full_analysis(self, ticker: str, history_years: int = 5) -> Dict[str, Any]:
        """
        Orchestrates the full acquisition of Hard Data.
        """
        print(f"\n[FINANCIAL ENGINE] Starting Deep Analysis for {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            
            # 1. Financial Statements (Annual & Quarterly)
            financials = self._fetch_comprehensive_financials(stock)
            
            # 2. Market Statistics (Price & Volatility)
            market_stats = self._fetch_market_stats(stock, years=history_years)
            
            # 3. Enhanced Metadata (Sector, Risk, Audit, etc.)
            metadata = self._extract_metadata(stock)
            
            # 4. Construct Final Bundle
            analysis = {
                "meta": {
                    "ticker": ticker,
                    "generated_at": str(datetime.now().date()),
                    **metadata
                },
                "financials": financials, # Contains both annual and quarterly
                "market_intelligence": market_stats,
            }
            
            return analysis

        except Exception as e:
            print(f"CRITICAL ENGINE FAILURE: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _extract_metadata(self, stock) -> Dict[str, Any]:
        """
        Extracts full metadata from stock.info, sanitizing for JSON.
        """
        try:
            info = stock.info
            # List of high-value keys we definitely want, but we'll take everything that's serializable
            return {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))}
        except:
            return {}

    def _fetch_comprehensive_financials(self, stock) -> Dict[str, Any]:
        """
        Fetches full available history for statements (Annual AND Quarterly).
        Dumps the ENTIRE dataframe content to JSON.
        """
        print("  > Fetching Universal Financial Statements...")
        
        def process_statement_df(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
            """
            Converts a dataframe where Columns=Dates, Index=Fields into a clean JSON structure.
            Output format: { "Total Revenue": { "2023-12-31": 100, ... }, ... }
            """
            if df.empty: return {}
            
            # Ensure index is string-based (field names)
            df.index = df.index.map(str)
            
            result = {}
            for field in df.index:
                # Get the time series for this field
                series = df.loc[field]
                
                # Convert timestamps to string dates and handle NaNs
                clean_series = {}
                for date_val, value in series.items():
                    date_str = str(date_val.date()) if hasattr(date_val, 'date') else str(date_val)
                    
                    # Store only valid numbers
                    if pd.notnull(value):
                        clean_series[date_str] = float(value)
                    else:
                        clean_series[date_str] = None
                        
                result[field] = clean_series
            return result

        # Fetch Annual
        annual = {
            "income_statement": process_statement_df(stock.financials),
            "balance_sheet": process_statement_df(stock.balance_sheet),
            "cash_flow": process_statement_df(stock.cashflow)
        }
        
        # Fetch Quarterly
        quarterly = {
            "income_statement": process_statement_df(stock.quarterly_financials),
            "balance_sheet": process_statement_df(stock.quarterly_balance_sheet),
            "cash_flow": process_statement_df(stock.quarterly_cashflow)
        }
        
        return {
            "annual": annual,
            "quarterly": quarterly
        }

    def _fetch_market_stats(self, stock, years: int) -> Dict[str, Any]:
        """
        Calculates volatility, beta, and price trends.
        """
        print("  > Analyzing Market Data...")
        
        try:
            # We already get static stats from .info, so here we focus on time-series derived stats
            
            # Historical Calculation
            hist = stock.history(period=f"{years}y")
            if hist.empty:
                return {"note": "No history available"}
            
            if 'Close' not in hist.columns:
                return {}

            # Calculate Daily Returns
            hist['Returns'] = hist['Close'].pct_change()
            
            # Volatility (Annualized)
            volatility = hist['Returns'].std() * np.sqrt(252)
            
            # CAGRs or Price Change
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            total_return = (end_price - start_price) / start_price
            
            # Max Drawdown
            rolling_max = hist['Close'].cummax()
            drawdown = (hist['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            stats = {
                "volatility_annualized": round(volatility, 4),
                "total_return_period": round(total_return, 4),
                "max_drawdown": round(max_drawdown, 4),
                "price_history_summary": {
                    "start_date": str(hist.index[0].date()),
                    "end_date": str(hist.index[-1].date()),
                    "start_price": round(start_price, 2),
                    "end_price": round(end_price, 2)
                }
            }
            
            return stats

        except Exception as e:
            print(f"  Warning: Market Stats failed: {e}")
            return {}

if __name__ == "__main__":
    pass
