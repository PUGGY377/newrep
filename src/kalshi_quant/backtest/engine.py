"""Event-driven, point-in-time backtest engine.

Two-phase design so sensitivity analysis (varying edge threshold, Kelly
cap, or fee assumptions) never has to redo forecasting or data joins:

  Phase 1: `generate_candidates` walks markets chronologically by DECISION
  TIME, calling the (walk-forward-retrained) Forecaster and computing an
  Edge for every market considered -- whether or not it would pass the
  current signal filter. Nothing here depends on threshold/sizing config.

  Phase 2: `simulate` replays a list of TradeCandidates chronologically,
  applying the signal filter, position sizing, portfolio/correlation
  limits, and the drawdown circuit breaker, producing a full trade log and
  equity curve. This is a cheap, pure function of (candidates, SimParams)
  -- backtest/sensitivity.py calls it repeatedly across a parameter grid.

Walk-forward retraining: `generate_candidates` retrains the forecaster on
a schedule (config.backtest.walk_forward) using only markets that had
ALREADY SETTLED before the current training cutoff -- it never fits on a
window that overlaps the period it's about to generate predictions for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from kalshi_quant.data.models import Market, MarketStatus, OrderbookSnapshot, Side
from kalshi_quant.data.store import DataStore
from kalshi_quant.forecasting.base import Forecaster, ProbabilityEstimate
from kalshi_quant.risk.circuit_breaker import should_halt_new_positions
from kalshi_quant.risk.portfolio import BankrollState, Portfolio, PortfolioConstraints, Position
from kalshi_quant.risk.sizer import size_position
from kalshi_quant.signal.correlation import CorrelationGraph
from kalshi_quant.signal.edge import Edge, NoLiquidityError, compute_edge, passes_signal_filter
from kalshi_quant.signal.fees import fee_schedule_as_of

from kalshi_quant.backtest.fills import simulate_fill


@dataclass(frozen=True)
class TradeCandidate:
    market: Market
    decision_time: datetime
    estimate: ProbabilityEstimate
    orderbook: OrderbookSnapshot
    edge: Edge
    cluster_id: str
    category: str
    series_ticker: str


@dataclass(frozen=True)
class TradeRecord:
    market_ticker: str
    opened_at: datetime
    settled_at: datetime
    side: Side
    contracts: int
    entry_price: float
    fee_paid: float
    model_probability: float
    outcome_yes: bool
    pnl_dollars: float
    binding_constraint: str


@dataclass
class SimParams:
    min_edge_after_fees: float
    min_confidence: float
    kelly_shrinkage_k: float
    max_kelly_fraction: float
    portfolio_constraints: PortfolioConstraints
    max_orderbook_depth_fraction: float
    max_acceptable_slippage: float
    circuit_breaker_lookback_trades: int
    risk_of_ruin_halt_threshold: float
    max_drawdown_hard_stop: float
    starting_capital: float
    fee_multiplier_scale: float = 1.0  # sensitivity knob: scales each candidate's precomputed fee


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    halted_at: datetime | None = None
    halt_reason: str = ""

    @property
    def final_bankroll(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else 0.0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_dollars > 0)
        return wins / len(self.trades)


def generate_candidates(
    store: DataStore,
    forecaster: Forecaster,
    markets: list[Market],
    feature_fetcher,  # Callable[[Market, datetime], FeatureSnapshot] -- typically a connector
    correlation_graph: CorrelationGraph,
    category: str,
    decision_offset_before_close: timedelta,
    train_window_days: int,
    step_days: int,
) -> list[TradeCandidate]:
    """`feature_fetcher` is injected rather than hardcoded to a specific
    connector so this function works for any category. `markets` must
    already be sorted or will be sorted here by close_time."""
    sorted_markets = sorted(markets, key=lambda m: m.close_time)
    if not sorted_markets:
        return []

    candidates: list[TradeCandidate] = []
    last_train_cutoff: datetime | None = None

    for market in sorted_markets:
        decision_time = market.close_time - decision_offset_before_close
        if decision_time <= market.open_time:
            decision_time = market.open_time

        # Walk-forward retrain on schedule: refit whenever we've moved
        # `step_days` past the last training cutoff, using only markets
        # settled strictly before (decision_time - train window start).
        if last_train_cutoff is None or (decision_time - last_train_cutoff).days >= step_days:
            train_cutoff = decision_time
            train_start = train_cutoff - timedelta(days=train_window_days)
            training_examples = []
            for m in sorted_markets:
                if m.status != MarketStatus.SETTLED or m.settled_at is None:
                    continue
                if not (train_start <= m.settled_at < train_cutoff):
                    continue
                try:
                    snap = feature_fetcher(m, m.settled_at - timedelta(minutes=1))
                except Exception:
                    continue
                outcome_yes = m.result == Side.YES
                training_examples.append((m, snap, outcome_yes))
            if training_examples:
                forecaster.fit(training_examples)
            last_train_cutoff = train_cutoff

        market_asof = store.get_market_as_of(market.ticker, decision_time) or market
        orderbook = store.get_orderbook_as_of(market.ticker, decision_time)
        if orderbook is None:
            continue

        try:
            snapshot = feature_fetcher(market_asof, decision_time)
            estimate = forecaster.predict(market_asof, snapshot)
        except Exception:
            continue

        fee_schedule = fee_schedule_as_of(decision_time)
        for side in (Side.YES, Side.NO):
            try:
                edge = compute_edge(
                    estimate, orderbook, side, market.event_ticker, fee_schedule,
                    expected_slippage_probability=0.0,
                )
            except NoLiquidityError:
                continue
            candidates.append(
                TradeCandidate(
                    market=market_asof, decision_time=decision_time, estimate=estimate,
                    orderbook=orderbook, edge=edge,
                    cluster_id=correlation_graph.cluster_of(market.ticker),
                    category=category, series_ticker=market.event_ticker,
                )
            )

    return candidates


def _scaled_edge(edge: Edge, fee_multiplier_scale: float) -> Edge:
    if fee_multiplier_scale == 1.0:
        return edge
    new_fee = edge.fee_per_contract_dollars * fee_multiplier_scale
    delta = edge.fee_per_contract_dollars - new_fee
    return Edge(
        market_ticker=edge.market_ticker, side=edge.side, model_probability=edge.model_probability,
        model_confidence=edge.model_confidence, execution_price_probability=edge.execution_price_probability,
        expected_slippage_probability=edge.expected_slippage_probability,
        fee_per_contract_dollars=new_fee, raw_edge=edge.raw_edge,
        net_edge_per_contract=edge.net_edge_per_contract + delta,
    )


def simulate(candidates: list[TradeCandidate], params: SimParams) -> BacktestResult:
    """Pure function: no I/O, deterministic given (candidates, params).

    Event-driven over a chronological merge of OPEN events (one per
    market, at decision_time) and SETTLE events (scheduled dynamically the
    moment a position is actually opened, at the market's settlement
    time). This -- not settling a trade immediately after opening it --
    is what lets the portfolio/correlation layer see truly concurrent open
    positions (e.g. several same-day games, or several strike markets
    under one event) competing for the same variance budget, which is the
    entire point of risk/portfolio.py.
    """
    import heapq

    # One candidate per market: whichever side has the best net edge after
    # fee scaling. (Both sides of the same market are rarely both
    # profitable simultaneously, and only one side can be taken anyway.)
    best_by_market: dict[str, TradeCandidate] = {}
    for c in candidates:
        scaled = _scaled_edge(c.edge, params.fee_multiplier_scale)
        current = best_by_market.get(c.market.ticker)
        if current is None or scaled.net_edge_per_contract > _scaled_edge(
            current.edge, params.fee_multiplier_scale
        ).net_edge_per_contract:
            best_by_market[c.market.ticker] = c

    bankroll = BankrollState(starting_capital=params.starting_capital, cash=params.starting_capital)
    portfolio = Portfolio()
    result = BacktestResult()

    if not best_by_market:
        return result
    result.equity_curve.append(
        (min(c.decision_time for c in best_by_market.values()), bankroll.current_bankroll)
    )

    peak_bankroll = bankroll.current_bankroll
    recent_bankroll_before: list[float] = []
    recent_bankroll_after: list[float] = []
    halted = False

    # Heap entries: (timestamp, seq, kind, data). kind is "open" or "settle".
    heap: list[tuple[datetime, int, str, object]] = []
    seq = 0
    for c in best_by_market.values():
        heapq.heappush(heap, (c.decision_time, seq, "open", c))
        seq += 1

    open_positions: dict[str, tuple[Position, TradeCandidate, float, str]] = {}
    # ticker -> (pos, candidate, fee_paid, binding_constraint)

    while heap:
        _, _, kind, payload = heapq.heappop(heap)

        if kind == "open":
            candidate: TradeCandidate = payload  # type: ignore[assignment]
            if halted:
                continue
            edge = _scaled_edge(candidate.edge, params.fee_multiplier_scale)
            if not passes_signal_filter(edge, params.min_edge_after_fees, params.min_confidence):
                continue

            sized = size_position(
                edge, candidate.estimate.variance, candidate.orderbook, bankroll, portfolio,
                candidate.cluster_id, candidate.category, params.kelly_shrinkage_k,
                params.max_kelly_fraction, params.portfolio_constraints,
                params.max_orderbook_depth_fraction, params.max_acceptable_slippage,
            )
            if sized.contracts <= 0:
                continue

            fill = simulate_fill(candidate.orderbook, edge.side, sized.contracts)
            if fill.contracts_filled <= 0:
                continue

            if candidate.market.result is None:
                continue  # market never settled in our data; can't backtest it

            fee_schedule = fee_schedule_as_of(candidate.decision_time)
            fee_paid = fee_schedule.fee_dollars(
                candidate.series_ticker, fill.contracts_filled, fill.average_price_probability
            ) * params.fee_multiplier_scale

            position = Position(
                market_ticker=candidate.market.ticker, cluster_id=candidate.cluster_id,
                category=candidate.category, side=edge.side, contracts=fill.contracts_filled,
                entry_price_probability=fill.average_price_probability,
                model_probability_at_entry=edge.model_probability,
                stake_dollars=fill.total_cost_dollars,
            )
            portfolio.positions.append(position)
            bankroll.cash -= fill.total_cost_dollars + fee_paid
            open_positions[candidate.market.ticker] = (position, candidate, fee_paid, sized.binding_constraint)

            settle_time = candidate.market.settled_at or candidate.decision_time
            heapq.heappush(heap, (settle_time, seq, "settle", candidate.market.ticker))
            seq += 1

        else:  # settle
            ticker: str = payload  # type: ignore[assignment]
            entry = open_positions.pop(ticker, None)
            if entry is None:
                continue
            position, candidate, fee_paid, binding_constraint = entry
            portfolio.positions.remove(position)

            market = candidate.market
            outcome_yes = market.result == Side.YES
            won = outcome_yes if position.side == Side.YES else not outcome_yes
            payoff = float(position.contracts) if won else 0.0
            pnl = payoff - position.stake_dollars - fee_paid

            bankroll_before_this_trade = bankroll.current_bankroll
            bankroll.cash += payoff
            bankroll.realized_pnl += pnl
            bankroll_after_this_trade = bankroll.current_bankroll

            recent_bankroll_before.append(bankroll_before_this_trade)
            recent_bankroll_after.append(bankroll_after_this_trade)
            if len(recent_bankroll_before) > params.circuit_breaker_lookback_trades:
                recent_bankroll_before.pop(0)
                recent_bankroll_after.pop(0)

            result.trades.append(
                TradeRecord(
                    market_ticker=market.ticker, opened_at=candidate.decision_time,
                    settled_at=market.settled_at or candidate.decision_time, side=position.side,
                    contracts=position.contracts, entry_price=position.entry_price_probability,
                    fee_paid=fee_paid, model_probability=candidate.edge.model_probability,
                    outcome_yes=outcome_yes, pnl_dollars=pnl, binding_constraint=binding_constraint,
                )
            )
            result.equity_curve.append((market.settled_at or candidate.decision_time, bankroll.current_bankroll))

            peak_bankroll = max(peak_bankroll, bankroll.current_bankroll)
            current_drawdown = (
                0.0 if peak_bankroll <= 0 else max(0.0, 1 - bankroll.current_bankroll / peak_bankroll)
            )
            halt, reason = should_halt_new_positions(
                recent_bankroll_before, recent_bankroll_after, current_drawdown,
                params.risk_of_ruin_halt_threshold, params.max_drawdown_hard_stop,
            )
            if halt and not halted:
                halted = True
                result.halted_at = market.settled_at or candidate.decision_time
                result.halt_reason = reason

    return result
