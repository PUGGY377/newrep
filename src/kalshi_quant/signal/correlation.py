"""Related/correlated market detection.

Kalshi frequently lists multiple markets tied to the same underlying
driver -- e.g. several strike-price markets under one event (event_ticker
shared), or separate markets whose outcomes are logically linked (a game
winner market and that game's total-points market). Treating these as
independent signals double-counts the same underlying bet and understates
true portfolio risk.

Two clustering mechanisms are provided:
  1. Structural: same event_ticker -> always clustered (Kalshi guarantees
     these share an underlying event).
  2. Declared: an explicit correlation graph for cross-event links (e.g.
     two different games both depending on the same star player's
     availability) that structural clustering can't see. This starts
     empty and is meant to be populated by category-specific knowledge as
     it's discovered -- it is deliberately not inferred automatically,
     since a wrong inferred correlation is worse than no correlation
     assumption at all for a system this size.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CorrelationGraph:
    """Union-find over market tickers. `event_ticker` clustering is applied
    automatically; call `declare_correlated` to link tickers across events."""

    _parent: dict[str, str] = field(default_factory=dict)

    def _find(self, ticker: str) -> str:
        self._parent.setdefault(ticker, ticker)
        root = ticker
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[ticker] != root:
            self._parent[ticker], ticker = root, self._parent[ticker]
        return root

    def declare_correlated(self, ticker_a: str, ticker_b: str) -> None:
        root_a, root_b = self._find(ticker_a), self._find(ticker_b)
        if root_a != root_b:
            self._parent[root_a] = root_b

    def register_event_group(self, event_ticker: str, market_tickers: list[str]) -> None:
        if not market_tickers:
            return
        first = market_tickers[0]
        for other in market_tickers[1:]:
            self.declare_correlated(first, other)

    def cluster_of(self, ticker: str) -> str:
        """Stable cluster id for a ticker (the union-find root). Two tickers
        with the same cluster id must be treated as one risk unit by the
        portfolio layer."""
        return self._find(ticker)

    def clusters(self, tickers: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for t in tickers:
            groups.setdefault(self.cluster_of(t), []).append(t)
        return groups
