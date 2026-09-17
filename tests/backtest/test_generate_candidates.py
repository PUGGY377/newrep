from datetime import datetime, timedelta

from kalshi_quant.backtest.engine import generate_candidates
from kalshi_quant.data.models import FeatureSnapshot, Market, MarketStatus, OrderbookLevel, OrderbookSnapshot, Side
from kalshi_quant.data.store import DataStore
from kalshi_quant.forecasting.base import Forecaster, ProbabilityEstimate
from kalshi_quant.signal.correlation import CorrelationGraph

T0 = datetime(2024, 1, 1)


class RecordingForecaster(Forecaster):
    name = "recording"
    version = "1"

    def __init__(self):
        self.fit_calls: list[list] = []

    def fit(self, training_examples):
        # The critical point-in-time assertion: every training example's
        # snapshot.as_of must be BEFORE its market's settlement time.
        for market, snapshot, _outcome in training_examples:
            if market.settled_at is not None:
                assert snapshot.as_of < market.settled_at, "lookahead leak in training example!"
        self.fit_calls.append(training_examples)

    def predict(self, market, snapshot):
        return ProbabilityEstimate(
            market_ticker=market.ticker, probability=0.6, variance=0.05,
            model_name=self.name, model_version=self.version,
        )


def make_market(ticker, close_time, settled=False) -> Market:
    return Market(
        ticker=ticker, event_ticker=f"{ticker}-EVT", title="t", subtitle="s",
        status=MarketStatus.SETTLED if settled else MarketStatus.OPEN,
        open_time=close_time - timedelta(days=1), close_time=close_time, expiration_time=close_time,
        settlement_criteria="x", fetched_at=T0, result=Side.YES if settled else None,
        settled_at=close_time if settled else None,
    )


def make_ob(ticker, as_of) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        market_ticker=ticker, as_of=as_of,
        yes_bids=(OrderbookLevel(54, 100),), yes_asks=(OrderbookLevel(56, 100),),
        no_bids=(OrderbookLevel(44, 100),), no_asks=(OrderbookLevel(46, 100),),
    )


def test_generate_candidates_never_leaks_future_training_data(tmp_path):
    store = DataStore(tmp_path / "test.duckdb")

    # 5 historical settled markets, one every 10 days, plus one future
    # market we're generating a candidate for.
    markets = []
    for i in range(5):
        m = make_market(f"HIST-{i}", T0 + timedelta(days=10 * i), settled=True)
        store.write_market(m)
        store.write_orderbook_snapshot(make_ob(m.ticker, m.close_time - timedelta(hours=1)))
        markets.append(m)

    future_market = make_market("FUTURE", T0 + timedelta(days=200), settled=False)
    store.write_market(future_market)
    store.write_orderbook_snapshot(make_ob(future_market.ticker, future_market.close_time - timedelta(hours=1)))
    markets.append(future_market)

    def feature_fetcher(market, as_of):
        return FeatureSnapshot(market_ticker=market.ticker, as_of=as_of, values={"x": 1.0})

    forecaster = RecordingForecaster()
    graph = CorrelationGraph()

    candidates = generate_candidates(
        store, forecaster, markets, feature_fetcher, graph, category="sports",
        decision_offset_before_close=timedelta(hours=1), train_window_days=1000, step_days=1,
    )

    assert forecaster.fit_calls  # retraining happened at least once
    assert any(c.market.ticker == "FUTURE" for c in candidates)
    store.close()
