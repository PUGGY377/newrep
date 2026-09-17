from datetime import datetime

import pytest

from kalshi_quant.data.connectors.base import DataNotAvailableError
from kalshi_quant.data.connectors.sports import SportsDataConnector, parse_matchup
from kalshi_quant.data.models import Market, MarketStatus

T0 = datetime(2024, 1, 1)


def test_parse_matchup_vs_pattern():
    matchup = parse_matchup("Chiefs vs Bills", "Will the Chiefs win?", "Chiefs")
    assert matchup.away_team == "Chiefs"
    assert matchup.home_team == "Bills"
    assert matchup.yes_team == "Chiefs"


def test_parse_matchup_will_beat_pattern():
    matchup = parse_matchup(
        "NFL Week 1 Matchup", "Will the Chiefs beat the Bills?", "",
    )
    assert matchup.away_team == "Chiefs"
    assert matchup.home_team == "Bills"
    assert matchup.yes_team == "Chiefs"


def test_parse_matchup_raises_when_unparseable():
    with pytest.raises(DataNotAvailableError):
        parse_matchup("garbage title", "also garbage", "")


def make_market(title, subtitle, event_ticker="KXNFLGAME-A") -> Market:
    return Market(
        ticker="KXNFLGAME-A-X", event_ticker=event_ticker, title=title, subtitle=subtitle,
        status=MarketStatus.OPEN, open_time=T0, close_time=T0, expiration_time=T0,
        settlement_criteria="x", fetched_at=T0,
    )


def test_connector_relevant_to_checks_series_prefix():
    connector = SportsDataConnector(league_series_prefixes=("KXNFL",))
    nfl_market = make_market("Chiefs vs Bills", "Chiefs", event_ticker="KXNFLGAME-A")
    other_market = make_market("Chiefs vs Bills", "Chiefs", event_ticker="KXWEATHER-A")
    assert connector.relevant_to(nfl_market)
    assert not connector.relevant_to(other_market)


def test_connector_fetch_point_in_time_returns_team_features():
    connector = SportsDataConnector()
    market = make_market("Chiefs vs Bills", "Chiefs")
    snapshot = connector.fetch_point_in_time(market, T0)
    assert snapshot.values["away_team"] == "Chiefs"
    assert snapshot.values["home_team"] == "Bills"
    assert snapshot.values["yes_team"] == "Chiefs"
    assert "reference_market_probability" not in snapshot.values  # no ODDS_API_KEY in test env
