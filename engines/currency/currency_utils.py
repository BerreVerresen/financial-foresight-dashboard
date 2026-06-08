"""
Pure currency helpers — no Streamlit, no network. Safe to import anywhere
(engine, UI, tests). Handles currency-code normalization, pence/cents
pseudo-currencies, symbol/label formatting, and resolving a company's two
relevant currencies from yfinance metadata.
"""
from typing import Optional, Tuple, Dict, Any

# Display symbols for the majors we care about. Anything missing falls back to
# the ISO code so we never render a wrong symbol.
CURRENCY_SYMBOLS: Dict[str, str] = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "CHF ",
    "CAD": "C$", "AUD": "A$", "NZD": "NZ$", "CNY": "CN¥", "HKD": "HK$",
    "INR": "₹", "KRW": "₩", "SGD": "S$", "SEK": "kr ", "NOK": "kr ",
    "DKK": "kr ", "ZAR": "R ", "BRL": "R$", "MXN": "MX$", "TWD": "NT$",
    "RUB": "₽", "TRY": "₺", "PLN": "zł ", "ILS": "₪", "AED": "AED ",
    "SAR": "SAR ", "THB": "฿",
}

# Pseudo-currencies: a sub-unit Yahoo sometimes quotes in. Maps the pseudo code
# to (real_major_code, divisor_to_get_major). E.g. GBp (pence) -> GBP / 100.
# Detection is CASE-SENSITIVE (the lowercase 'p' in 'GBp' is the signal), so we
# check this map before any uppercasing.
PSEUDO_CURRENCIES: Dict[str, Tuple[str, float]] = {
    "GBp": ("GBP", 100.0),
    "GBX": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ZAX": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
}

# A reasonable picklist of target currencies for the UI. Any company's own
# reporting currency is also unioned into the selector at runtime, and the FX
# providers can resolve far more pairs than this (Yahoo: any pair; open.er-api:
# 160+ currencies), so this list is just the convenient defaults.
COMMON_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY", "HKD",
    "INR", "KRW", "SGD", "TWD", "SEK", "NOK", "DKK", "PLN", "ZAR", "MXN",
    "BRL", "TRY", "AED", "SAR", "ILS", "THB",
]


def pseudo_scale(code: Optional[str]) -> Tuple[Optional[str], float]:
    """
    Resolve a pseudo-currency to its major and the divisor needed to convert a
    value quoted in the pseudo into the major. For normal currencies returns
    (code, 1.0). Returns (None, 1.0) for falsy input.
    """
    if not code:
        return None, 1.0
    raw = str(code).strip()
    if raw in PSEUDO_CURRENCIES:
        major, divisor = PSEUDO_CURRENCIES[raw]
        return major, divisor
    return raw.upper(), 1.0


def normalize_currency(code: Optional[str]) -> Optional[str]:
    """
    Normalize a currency code to an uppercase ISO major code, folding pence/cents
    pseudo-currencies into their major (GBp -> GBP). Returns None if undeterminable.
    """
    major, _ = pseudo_scale(code)
    return major


def currency_symbol(code: Optional[str]) -> str:
    """Return a display symbol for a currency, falling back to the ISO code."""
    norm = normalize_currency(code)
    if not norm:
        return ""
    return CURRENCY_SYMBOLS.get(norm, f"{norm} ")


def format_money(value: Optional[float], code: Optional[str], scale: str = "B") -> str:
    """
    Format a monetary value with the right symbol. ``scale='B'`` means the value
    is already expressed in billions and we append a 'B' suffix; ``scale='raw'``
    formats the absolute number with thousands separators.
    """
    if value is None:
        return "—"
    sym = currency_symbol(code)
    if scale == "B":
        return f"{sym}{value:,.2f}B"
    return f"{sym}{value:,.0f}"


def money_suffix(code: Optional[str], scale: str = "B") -> str:
    """Render a label suffix like '(B EUR)' / '(EUR)' for chart titles/axes."""
    norm = normalize_currency(code) or "?"
    if scale == "B":
        return f"(B {norm})"
    return f"({norm})"


def resolve_company_currencies(meta: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    From a company's yfinance ``meta`` (stock.info) return
    ``(financial_currency, trading_currency)`` normalized to majors.

    * ``financialCurrency`` = reporting currency of the statements (revenue, debt,
      EBITDA, cash flow).
    * ``currency`` = trading/quote currency (marketCap, price, trailingPE).

    Each falls back to the other when missing; both may be None if neither exists.
    """
    if not isinstance(meta, dict):
        return None, None
    financial = normalize_currency(meta.get("financialCurrency"))
    trading = normalize_currency(meta.get("currency"))
    if not financial:
        financial = trading
    if not trading:
        trading = financial
    return financial, trading
