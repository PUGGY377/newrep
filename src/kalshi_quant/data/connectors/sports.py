"""Sports data connector.

Two data sources are combined for the starting sports Forecaster:

1. Kalshi's OWN settlement history. Once a Kalshi sports market settles,
   `Market.result` tells us which side won -- for a market structured as
   "does {TEAM} win?", that IS the game outcome. This makes the system
   self-contained for the Elo baseline: no external API key is required
   to train ratings, only to bootstrap the team/game structure from
   Kalshi's own market metadata.

2. An OPTIONAL external odds API (configured via ODDS_API_KEY) for a
   reference market probability / injury-style features to enrich the
   feature snapshot later. This is stubbed as a hook, not wired to a
   specific vendor, because no odds API key was provided when this was
   built -- see `fetch_reference_odds` below.

IMPORTANT -- VERIFY BEFORE RELYING ON THIS: `parse_matchup` below assumes
Kalshi sports market/event titles follow a "TEAM_A vs TEAM_B" or
"Will TEAM_A beat TEAM_B" style pattern, which matches Kalshi's publicly
visible sports markets at the time this was written but is NOT something
this system has validated against a live API pull yet (no API calls were
made while building this scaffold). Before training the Elo model on real
data, pull a sample of real event/market titles for the leagues you care
about (`KalshiClient.get_events(series_ticker=...)`) and confirm this
parser extracts the right teams -- adjust the regexes if the format
differs. Treat this function as the one place most likely to need a fix
once real data shows up.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

from kalshi_quant.data.connectors.base import DataNotAvailableError, ExternalDataConnector
from kalshi_quant.data.models import FeatureSnapshot, Market

_VS_PATTERN = re.compile(r"^\s*(?P<away>.+?)\s+(?:vs\.?|@|at)\s+(?P<home>.+?)\s*$", re.IGNORECASE)
_WILL_BEAT_PATTERN = re.compile(
    r"will\s+(?:the\s+)?(?P<subject>.+?)\s+(?:beat|defeat)\s+(?:the\s+)?(?P<opponent>.+?)\??\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Matchup:
    home_team: str
    away_team: str
    yes_team: str  # which team's win the YES side of this specific market corresponds to


def parse_matchup(event_title: str, market_title: str, market_subtitle: str) -> Matchup:
    """Best-effort parse of team names out of Kalshi sports metadata. See
    module docstring -- this needs validation against real API responses."""
    m = _VS_PATTERN.match(event_title)
    if m:
        away, home = m.group("away").strip(), m.group("home").strip()
        yes_team = market_subtitle.strip() or home
        return Matchup(home_team=home, away_team=away, yes_team=yes_team)

    m = _WILL_BEAT_PATTERN.search(market_title)
    if m:
        subject, opponent = m.group("subject").strip(), m.group("opponent").strip()
        return Matchup(home_team=opponent, away_team=subject, yes_team=subject)

    raise DataNotAvailableError(
        f"Could not parse matchup from event_title={event_title!r} "
        f"market_title={market_title!r} subtitle={market_subtitle!r}; "
        "update parse_matchup() for this title format before trading this market."
    )


class SportsDataConnector(ExternalDataConnector):
    category = "sports"

    def __init__(self, league_series_prefixes: tuple[str, ...] = ("KXNFL", "KXNBA", "KXCFB", "KXCBB")):
        self._league_series_prefixes = league_series_prefixes

    def relevant_to(self, market: Market) -> bool:
        return any(market.event_ticker.startswith(p) for p in self._league_series_prefixes)

    def fetch_point_in_time(self, market: Market, as_of: datetime) -> FeatureSnapshot:
        """Team identity is knowable as soon as the event/market is listed
        (well before as_of, for any pre-game forecast), so there is no
        point-in-time risk in this part of the snapshot. This method does
        NOT include game scores/results -- pulling those would leak the
        outcome we're trying to forecast."""
        # NOTE: event_title is not on the Market model directly; callers
        # (the scripts that assemble snapshots) are expected to have joined
        # Market with its parent Event to get the title. This connector
        # takes the already-joined strings via market.title/market.subtitle
        # as a pragmatic stand-in until that join is wired up in
        # scripts/ingest_historical.py.
        matchup = parse_matchup(market.title, market.title, market.subtitle)
        values: dict[str, float | str | None] = {
            "home_team": matchup.home_team,
            "away_team": matchup.away_team,
            "yes_team": matchup.yes_team,
        }
        sources = {k: "kalshi_market_metadata" for k in values}

        reference_prob = self._try_fetch_reference_odds(matchup, as_of)
        if reference_prob is not None:
            values["reference_market_probability"] = reference_prob
            sources["reference_market_probability"] = "external_odds_api"

        return FeatureSnapshot(market_ticker=market.ticker, as_of=as_of, values=values, sources=sources)

    def _try_fetch_reference_odds(self, matchup: Matchup, as_of: datetime) -> float | None:
        """Returns None (not an error) when no odds API key is configured --
        this is a nice-to-have enrichment, not a required feature, so its
        absence must never break the pipeline."""
        if not os.environ.get("ODDS_API_KEY"):
            return None
        # Deliberately not implemented against a specific vendor: no odds
        # API key was available when this was built. Wire up a real
        # provider here (e.g. The Odds API) when one is chosen, making sure
        # whatever "line" is fetched is the one that was live AT `as_of`
        # (a closing-line/historical-odds endpoint), never the current line.
        raise DataNotAvailableError(
            "ODDS_API_KEY is set but no odds provider integration has been implemented yet."
        )
