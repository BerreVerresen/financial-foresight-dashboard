"""
FX rate providers — a pluggable chain. Each provider returns a ``Rate`` only when
it has a finite, positive number; otherwise ``None`` so the service advances to the
next provider. No Streamlit imports here (pure + unit-testable). Network calls are
wrapped with explicit timeouts and try/except so one slow/down source never hangs.
"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List

import requests

HTTP_TIMEOUT = 5  # seconds — keep the UI responsive even if a provider is down

# open.er-api.com requires visible attribution.
ER_API_ATTRIBUTION = "Rates by https://www.exchangerate-api.com"


@dataclass
class Rate:
    rate: float
    as_of: str          # ISO date/datetime string of the quote
    source: str         # provider name, for provenance display


def _valid(value) -> bool:
    """A rate is usable only if it is a finite, strictly-positive number."""
    try:
        return value is not None and math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _today() -> str:
    try:
        return datetime.now().date().isoformat()
    except Exception:
        return ""


class FXProvider:
    """Abstract base. Subclasses implement get_rate(frm, to) -> Rate | None."""

    name = "base"

    def get_rate(self, frm: str, to: str) -> Optional[Rate]:  # pragma: no cover
        raise NotImplementedError


class YahooFXProvider(FXProvider):
    """
    Yahoo Finance via yfinance using the 'FROMTO=X' symbol convention. Tries
    fast_info first (cheap), then a 5-day history close (so weekends/holidays still
    return the last close), then the inverse pair, then USD triangulation.
    """

    name = "yahoo"

    def _spot(self, symbol: str):
        """Return (rate, as_of) for a Yahoo FX symbol, or (None, None)."""
        try:
            import yfinance as yf
        except Exception:
            return None, None
        try:
            t = yf.Ticker(symbol)
            # 1. fast_info (lightweight, avoids the slow/fragile .info dict)
            try:
                fi = t.fast_info
                last = fi.get("last_price") if hasattr(fi, "get") else getattr(fi, "last_price", None)
                if _valid(last):
                    return float(last), _today()
            except Exception:
                pass
            # 2. history fallback — period='5d' so Fri/holiday requests aren't empty
            hist = t.history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if not closes.empty and _valid(closes.iloc[-1]):
                    as_of = ""
                    try:
                        as_of = str(closes.index[-1].date())
                    except Exception:
                        as_of = _today()
                    return float(closes.iloc[-1]), as_of
        except Exception:
            return None, None
        return None, None

    def get_rate(self, frm: str, to: str) -> Optional[Rate]:
        # direct pair
        rate, as_of = self._spot(f"{frm}{to}=X")
        if _valid(rate):
            return Rate(rate, as_of, self.name)
        # inverse pair
        inv, as_of = self._spot(f"{to}{frm}=X")
        if _valid(inv):
            return Rate(1.0 / float(inv), as_of, self.name)
        # triangulate via USD for non-USD crosses
        if frm != "USD" and to != "USD":
            a, as_of_a = self._spot(f"{frm}USD=X")
            b, as_of_b = self._spot(f"USD{to}=X")
            if _valid(a) and _valid(b):
                return Rate(float(a) * float(b), as_of_b or as_of_a, self.name)
        return None


class FrankfurterProvider(FXProvider):
    """ECB reference rates via frankfurter.app — free, no key, ~30 majors only."""

    name = "frankfurter"

    def get_rate(self, frm: str, to: str) -> Optional[Rate]:
        try:
            resp = requests.get(
                "https://api.frankfurter.app/latest",
                params={"from": frm, "to": to},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            rate = (data.get("rates") or {}).get(to)
            if _valid(rate):
                return Rate(float(rate), data.get("date", _today()), self.name)
        except Exception:
            return None
        return None


class ErApiProvider(FXProvider):
    """open.er-api.com — free, no key, 160+ currencies incl. exotics. Wider backstop."""

    name = "open.er-api"

    def get_rate(self, frm: str, to: str) -> Optional[Rate]:
        try:
            resp = requests.get(
                f"https://open.er-api.com/v6/latest/{frm}",
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("result") != "success":
                return None
            rate = (data.get("rates") or {}).get(to)
            if _valid(rate):
                as_of = data.get("time_last_update_utc") or _today()
                return Rate(float(rate), str(as_of), self.name)
        except Exception:
            return None
        return None


class ManualProvider(FXProvider):
    """
    User-pinned overrides. ``overrides`` maps 'FROMTO' (e.g. 'USDEUR') -> rate.
    Tagged source='manual' so stale pins are visually distinct from live data.
    """

    name = "manual"

    def __init__(self, overrides: Optional[Dict[str, float]] = None, as_of_map: Optional[Dict[str, str]] = None):
        self.overrides = overrides or {}
        self.as_of_map = as_of_map or {}

    def get_rate(self, frm: str, to: str) -> Optional[Rate]:
        key = f"{frm}{to}"
        rate = self.overrides.get(key)
        if _valid(rate):
            return Rate(float(rate), self.as_of_map.get(key, _today()), self.name)
        # allow an inverse manual entry
        inv = self.overrides.get(f"{to}{frm}")
        if _valid(inv):
            return Rate(1.0 / float(inv), self.as_of_map.get(f"{to}{frm}", _today()), self.name)
        return None


# Registry of provider keys -> constructor (manual handled separately by the service).
PROVIDER_REGISTRY = {
    "yahoo": YahooFXProvider,
    "frankfurter": FrankfurterProvider,
    "open.er-api": ErApiProvider,
}

DEFAULT_PROVIDER_ORDER: List[str] = ["yahoo", "frankfurter", "open.er-api"]
