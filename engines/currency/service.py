"""
FXService — orchestrates the provider chain into a single get_rate / convert.

Pure logic: same-currency short-circuit, pseudo-currency normalization, optional
top-priority manual override, then walk the configured providers until one yields
a valid rate. Returns ``None`` for an unknown pair (NEVER silently 1.0), so callers
can degrade explicitly.
"""
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from .providers import (
    Rate, FXProvider, ManualProvider, PROVIDER_REGISTRY, DEFAULT_PROVIDER_ORDER,
)
from .currency_utils import pseudo_scale


def _now() -> str:
    try:
        return datetime.now().isoformat(timespec="seconds")
    except Exception:
        return ""


class FXService:
    def __init__(
        self,
        provider_order: Optional[List[str]] = None,
        overrides: Optional[Dict[str, float]] = None,
        override_as_of: Optional[Dict[str, str]] = None,
        manual_priority: bool = False,
    ):
        self.provider_order = list(provider_order) if provider_order else list(DEFAULT_PROVIDER_ORDER)
        self.manual = ManualProvider(overrides, override_as_of)
        self.manual_priority = manual_priority
        self._providers: List[FXProvider] = self._build_chain()

    def _build_chain(self) -> List[FXProvider]:
        chain: List[FXProvider] = []
        if self.manual_priority:
            chain.append(self.manual)
        for key in self.provider_order:
            if key == "manual":
                chain.append(self.manual)
            else:
                cls = PROVIDER_REGISTRY.get(key)
                if cls is not None:
                    chain.append(cls())
        if not self.manual_priority and self.manual not in chain:
            chain.append(self.manual)  # manual as final fallback
        return chain

    def get_rate(self, frm: Optional[str], to: Optional[str]) -> Optional[Rate]:
        """Resolve the rate to convert 1 unit of ``frm`` into ``to``."""
        # Normalize pseudo-currencies down to their major codes for lookup.
        frm_major, _ = pseudo_scale(frm)
        to_major, _ = pseudo_scale(to)
        if not frm_major or not to_major:
            return None
        # identity short-circuit — the only case that returns 1.0
        if frm_major == to_major:
            return Rate(1.0, _now(), "identity")
        for provider in self._providers:
            try:
                rate = provider.get_rate(frm_major, to_major)
            except Exception:
                rate = None
            if rate is not None:
                return rate
        return None

    def convert(self, amount: Optional[float], frm: Optional[str], to: Optional[str]) -> Tuple[Optional[float], Optional[Rate]]:
        """
        Convert ``amount`` from ``frm`` into ``to``, accounting for pseudo-currency
        sub-units. Returns ``(converted_amount, Rate)`` or ``(None, None)`` when no
        rate is available.
        """
        if amount is None:
            return None, None
        # Scale a value quoted in a pseudo sub-unit (e.g. pence) into its major.
        _, frm_div = pseudo_scale(frm)
        major_amount = amount / frm_div if frm_div else amount
        rate = self.get_rate(frm, to)
        if rate is None:
            return None, None
        return major_amount * rate.rate, rate
