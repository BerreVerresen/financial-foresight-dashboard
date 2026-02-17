import yfinance as yf
import json
import pandas as pd

def discover_gem_data(ticker="MSFT"):
    print(f"--- Discovering Hidden Gems for {ticker} ---")
    stock = yf.Ticker(ticker)
    
    # 1. Info Keys (Metadata)
    info = stock.info
    print(f"\n[INFO] keys found: {len(info)}")
    # Print interesting keys that might be relevant for consulting
    for k in sorted(info.keys()):
        if any(x in k.lower() for x in ['audit', 'risk', 'officer', 'holder', 'short', 'beta', 'margin', 'growth', 'employee', 'segment']):
            print(f"  - {k}: {info[k]}")

    # 2. Institutional Holders
    try:
        inst = stock.institutional_holders
        if inst is not None and not inst.empty:
            print(f"\n[INSTITUTIONAL] Found data (Top 5):")
            print(inst.head())
    except: print("[INSTITUTIONAL] Not available")

    # 3. Insider Roster
    try:
        ins = stock.insider_roster_holders
        if ins is not None and not ins.empty:
            print(f"\n[INSIDER] Found data (Top 5):")
            print(ins.head())
    except: print("[INSIDER] Not available")
    
    # 4. Cash Flow Items (for Bridge)
    cf = stock.cashflow
    if not cf.empty:
        print(f"\n[CASH FLOW] Items for Bridge:")
        print(cf.index.tolist())

if __name__ == "__main__":
    discover_gem_data()
