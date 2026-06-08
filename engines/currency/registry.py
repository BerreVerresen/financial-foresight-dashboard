"""
Single source of truth for "which metrics are money, and in which currency".

The display layer converts only ``absolute`` metrics; every ratio / percentage /
days / score is currency-neutral and must be passed through untouched (converting
a ratio would actively introduce errors). ``currency_field`` says which of a
company's two currencies the native value is denominated in:

* ``financial`` -> meta['financialCurrency'] (statement-derived: revenue, EBITDA,
  net income, net debt, and the ADR-corrected enterprise value).
* ``trading``   -> meta['currency'] (price-derived: market cap).
"""
from typing import Dict, Optional

# metric key (must match the engine's emitted keys EXACTLY) -> currency field
ABSOLUTE_METRICS: Dict[str, str] = {
    "Revenue ($B)": "financial",
    "Net Income ($B)": "financial",
    "EBITDA ($B)": "financial",
    "Enterprise Value ($B)": "financial",
    "Market Cap ($B)": "trading",
}


def metric_is_absolute(metric: str) -> bool:
    """True if the metric is an absolute monetary amount that should be converted."""
    return metric in ABSOLUTE_METRICS


def metric_currency_field(metric: str) -> Optional[str]:
    """Return 'financial' | 'trading' | None for a metric."""
    return ABSOLUTE_METRICS.get(metric)


def raw_field_currency(statement: str) -> str:
    """
    Currency field for a raw statement line item. Income statement, balance sheet
    and cash flow are all reported in ``financialCurrency``; only price/market-cap
    derived figures are in the trading ``currency``.
    """
    return "financial"
