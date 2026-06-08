"""
Currency subsystem for the financial dashboard.

* currency_utils — pure helpers (normalization, pseudo-currencies, symbols, labels)
* registry       — which metrics are money, and in which currency
* providers      — pluggable FX providers (Yahoo, Frankfurter, open.er-api, manual)
* service        — provider-chain orchestration (pure)
* cache          — Streamlit-aware caching boundary + convenience converters
"""
from .currency_utils import (
    CURRENCY_SYMBOLS,
    COMMON_CURRENCIES,
    normalize_currency,
    pseudo_scale,
    currency_symbol,
    format_money,
    money_suffix,
    resolve_company_currencies,
)
from .registry import (
    ABSOLUTE_METRICS,
    metric_is_absolute,
    metric_currency_field,
    raw_field_currency,
)
from .providers import Rate, DEFAULT_PROVIDER_ORDER, PROVIDER_REGISTRY
from .service import FXService
from .cache import (
    get_rate,
    convert_value,
    make_signature,
    clear_fx_cache,
    DEFAULT_TTL,
)

__all__ = [
    "CURRENCY_SYMBOLS",
    "COMMON_CURRENCIES",
    "normalize_currency",
    "pseudo_scale",
    "currency_symbol",
    "format_money",
    "money_suffix",
    "resolve_company_currencies",
    "ABSOLUTE_METRICS",
    "metric_is_absolute",
    "metric_currency_field",
    "raw_field_currency",
    "Rate",
    "DEFAULT_PROVIDER_ORDER",
    "PROVIDER_REGISTRY",
    "FXService",
    "get_rate",
    "convert_value",
    "make_signature",
    "clear_fx_cache",
    "DEFAULT_TTL",
]
