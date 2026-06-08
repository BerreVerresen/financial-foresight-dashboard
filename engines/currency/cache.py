"""
Streamlit-aware caching boundary for FX rates — the ONLY module that touches
st.cache_data, so providers/service stay framework-agnostic and unit-testable.

Rates are cached per (from, to, chain-signature) with a TTL and returned together
with their as-of/source provenance. Failures are NOT memoized as successes: the
cached function raises on a miss, which Streamlit does not cache, so a transient
outage is retried on the next render rather than pinned for the whole TTL.

When Streamlit isn't importable (CLI / tests) it degrades to a process-local TTL
dict, so the engine works headless.
"""
import time
from typing import Optional, Dict, List, Tuple, Any

from .service import FXService
from .currency_utils import pseudo_scale

DEFAULT_TTL = 21600  # 6h — FX is fine to cache for hours for an analytics dashboard


class FXUnavailable(Exception):
    """Raised when no provider can supply a rate (so the result is not cached)."""


def _build_service(signature: Tuple) -> FXService:
    provider_order, overrides_items, as_of_items, manual_priority = signature
    return FXService(
        provider_order=list(provider_order),
        overrides=dict(overrides_items),
        override_as_of=dict(as_of_items),
        manual_priority=bool(manual_priority),
    )


def make_signature(
    provider_order: Optional[List[str]] = None,
    overrides: Optional[Dict[str, float]] = None,
    override_as_of: Optional[Dict[str, str]] = None,
    manual_priority: bool = False,
) -> Tuple:
    """Build a hashable signature for the FX config (used as a cache key)."""
    from .providers import DEFAULT_PROVIDER_ORDER
    order = tuple(provider_order) if provider_order else tuple(DEFAULT_PROVIDER_ORDER)
    ov = tuple(sorted((overrides or {}).items()))
    ao = tuple(sorted((override_as_of or {}).items()))
    return (order, ov, ao, bool(manual_priority))


def _fetch_uncached(frm: str, to: str, signature: Tuple) -> Optional[Dict[str, Any]]:
    service = _build_service(signature)
    rate = service.get_rate(frm, to)
    if rate is None:
        return None
    return {"rate": rate.rate, "as_of": rate.as_of, "source": rate.source}


# --- caching layer: Streamlit if available, else a process-local TTL dict -------
try:
    import streamlit as st

    @st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
    def _cached_rate(frm: str, to: str, signature: Tuple) -> Dict[str, Any]:
        result = _fetch_uncached(frm, to, signature)
        if result is None:
            raise FXUnavailable(f"No FX rate for {frm}->{to}")
        return result

    def clear_fx_cache() -> None:
        _cached_rate.clear()

    _HEADLESS = False

except Exception:  # pragma: no cover - exercised only outside Streamlit
    _LOCAL_CACHE: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}

    def _cached_rate(frm: str, to: str, signature: Tuple) -> Dict[str, Any]:
        key = (frm, to, signature)
        hit = _LOCAL_CACHE.get(key)
        if hit and (time.time() - hit[0]) < DEFAULT_TTL:
            return hit[1]
        result = _fetch_uncached(frm, to, signature)
        if result is None:
            raise FXUnavailable(f"No FX rate for {frm}->{to}")
        _LOCAL_CACHE[key] = (time.time(), result)
        return result

    def clear_fx_cache() -> None:
        _LOCAL_CACHE.clear()

    _HEADLESS = True


def get_rate(frm: Optional[str], to: Optional[str], signature: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
    """
    Cached rate lookup. Returns ``{rate, as_of, source}`` or ``None`` if unavailable.
    Pseudo-currencies are normalized to majors before the lookup.
    """
    frm_major, _ = pseudo_scale(frm)
    to_major, _ = pseudo_scale(to)
    if not frm_major or not to_major:
        return None
    if frm_major == to_major:
        return {"rate": 1.0, "as_of": "", "source": "identity"}
    if signature is None:
        signature = make_signature()
    try:
        return _cached_rate(frm_major, to_major, signature)
    except FXUnavailable:
        return None
    except Exception:
        return None


def convert_value(
    value: Optional[float],
    frm: Optional[str],
    to: Optional[str],
    signature: Optional[Tuple] = None,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """
    Convert ``value`` from ``frm`` into ``to`` using the cached rate, accounting for
    pseudo-currency sub-units (pence/cents). Returns ``(converted, rate_dict)`` or
    ``(None, None)`` when no rate is available.
    """
    if value is None:
        return None, None
    _, divisor = pseudo_scale(frm)
    major_value = value / divisor if divisor else value
    rate = get_rate(frm, to, signature)
    if rate is None:
        return None, None
    return major_value * rate["rate"], rate
