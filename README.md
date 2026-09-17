# Kalshi Quant Trading System

A quantitative research and trading system for Kalshi event-contract markets:
data ingestion → calibrated probability forecasting → fee/slippage-adjusted
edge detection → correlation-aware Kelly position sizing → point-in-time
backtesting → paper trading (live trading gated behind explicit
confirmation). Built for personal trading research, not distributed advice.

**Paper trading is the default and only wired-up execution mode.** Live
trading requires three independent things to all be true at once (see
`execution/live.py`) and is not something a config edit alone can turn on.

## Status: starting category is Sports

The first end-to-end `Forecaster` is an Elo-rating model
(`forecasting/sports_elo.py`) trained on Kalshi's own settlement history,
paired with a `SportsDataConnector` (`data/connectors/sports.py`). See
"Adding a new forecaster / data source" below for how a second category
plugs in.

## Architecture

```
src/kalshi_quant/
├── config.py               # YAML config loading + the live-trading double-gate
├── data/
│   ├── kalshi_client.py    # REST client: RSA-PSS auth, retry/backoff, pagination
│   ├── models.py           # Market/Event/Series/Orderbook/Trade/FeatureSnapshot
│   ├── store.py            # DuckDB point-in-time storage (every read takes an as_of cutoff)
│   └── connectors/         # Pluggable external data sources, one per category
│       ├── base.py         # ExternalDataConnector ABC + point-in-time contract
│       └── sports.py       # team/matchup parsing from Kalshi market metadata
├── forecasting/
│   ├── base.py             # Forecaster ABC: predict() / fit()
│   ├── calibration.py      # Platt/isotonic calibration, Brier score, log loss
│   ├── ensemble.py         # confidence-weighted ensembling
│   └── sports_elo.py       # Elo rating engine + calibration adapter
├── signal/
│   ├── fees.py             # versioned Kalshi fee schedule
│   ├── edge.py             # fee/slippage-adjusted edge computation
│   └── correlation.py      # correlated/related-market clustering
├── risk/
│   ├── kelly.py            # fractional Kelly + confidence shrinkage
│   ├── portfolio.py        # correlation-aware variance budgeting + hard ceilings
│   ├── liquidity.py        # order-book-depth-aware slippage capping
│   ├── circuit_breaker.py  # risk-of-ruin-derived drawdown halt
│   └── sizer.py            # combines all of the above into one final stake
├── backtest/
│   ├── engine.py           # event-driven, point-in-time, walk-forward simulation
│   ├── fills.py            # order-book-walking fill simulation
│   ├── report.py           # equity curve, drawdown, Calmar ratio, calibration
│   └── sensitivity.py      # parameter-grid fragility analysis
├── execution/
│   ├── base.py             # ExecutionEngine ABC
│   ├── paper.py            # DEFAULT: simulated fills against real live order books
│   ├── live.py             # real orders; triple-gated, built/tested last
│   ├── reconcile.py        # local vs. Kalshi account state reconciliation
│   └── audit.py            # full JSONL decision/order/fill audit log
└── monitoring/
    ├── dashboard.py        # Streamlit: P&L, exposure, calibration drift
    ├── alerts.py           # model drift / API failure / drawdown alerts
    └── reports.py          # periodic (weekly) performance report from the audit log

scripts/
├── ingest_historical.py            # pull real Kalshi data into the local DataStore
├── generate_synthetic_demo_data.py # SYNTHETIC data for exercising the pipeline (see below)
└── run_backtest.py                 # run the backtest + sensitivity grid, print a report

tests/            # mirrors src/ layout, one test file per module
config/default.yaml   # all tunable parameters, with inline rationale comments
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"
cp .env.example .env   # fill in KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH
pytest tests/ -q       # 107 tests, all passing as of this build
```

Kalshi auth uses an RSA key pair (not a static token): generate a keypair,
upload the **public** key in the Kalshi UI, point `KALSHI_PRIVATE_KEY_PATH`
at the private key file locally, and never commit it (`.gitignore` already
excludes `*.pem`/`secrets/`).

## Running a backtest

```bash
# Real data (requires .env credentials configured):
python scripts/ingest_historical.py --series-ticker KXNFLGAME --days-back 180
python scripts/run_backtest.py --db-path kalshi_quant.duckdb --event-prefix KXNFLGAME

# Synthetic demo data (no credentials needed, proves the pipeline runs):
python scripts/generate_synthetic_demo_data.py /tmp/demo.duckdb
python scripts/run_backtest.py --db-path /tmp/demo.duckdb --event-prefix KXNFLGAME
```

### Important limitation on real historical backtests

Kalshi's REST API does **not** expose historical order-book depth for past
timestamps — `get_orderbook()` only returns the book as it looks *right
now*. For a market that already settled, `ingest_historical.py` can only
record today's (irrelevant) book state, not the spread that was actually
live at decision time historically. **Genuine point-in-time order-book
history can only be built going forward**, by running ingestion (or a
WebSocket subscriber) on a schedule starting today. Trade history
(`get_trades`) *is* available historically and is a reasonable stand-in for
realistic execution prices in the meantime. `generate_synthetic_demo_data.py`
exists specifically to let the backtest engine, sizing math, and reporting
be exercised and validated end-to-end before real collected history has
accumulated — **its output is not evidence of a profitable strategy.**

### What the synthetic-data run actually showed (read this before trusting any backtest number)

Running the default config against the synthetic dataset produced a
misleadingly good headline number (+23% return) that turned out to be an
artifact of the risk-of-ruin circuit breaker halting the strategy after a
lucky streak of exactly 5 winning trades — the minimum sample size needed
before the breaker's ruin estimate activates at all. Disabling that halt
to see the full picture showed the pipeline only found ~8 qualifying edges
out of 800 candidates over ~400 synthetic games, and that fuller sample was
roughly breakeven-to-negative. Neither number means anything on synthetic
data with only a few hundred games, but the lesson is real: **always check
what the circuit breaker and small-sample effects are doing to a headline
return number before believing it, and use `backtest/sensitivity.py`'s
grid, not a single run, as the actual evidence.** This is precisely the
kind of fragility the sensitivity analysis and this README section exist
to surface rather than hide.

## The risk/sizing math ("theoretically profitable" design)

Position size is computed mathematically per trade, not from a fixed
percentage:

1. **Base Kelly fraction**: `f* = (p_hat - price) / (1 - price)`, derived
   from maximizing expected log-wealth for a binary contract (see
   `risk/kelly.py`).
2. **Confidence shrinkage**: `f_shrunk = f* / (1 + k * variance(p_hat))` —
   a forecaster's own reported uncertainty directly reduces the stake, so
   an uncertain prediction never gets sized as if it were certain.
3. **Correlation-aware variance budgeting**: positions in the same market
   cluster (`signal/correlation.py`) are treated as perfectly correlated
   (worst case), positions across clusters as independent, and a new
   position is capped so total portfolio variance stays under a configured
   budget (`risk/portfolio.py`).
4. **Liquidity capping**: stake is capped by how many contracts the order
   book can actually absorb within an acceptable slippage budget
   (`risk/liquidity.py`).
5. **Hard fractional ceilings**: fixed backstop percentages (per-market,
   per-category, total exposure) that clip whatever the math above
   produces — safety nets, not the sizing mechanism itself.
6. **Drawdown circuit breaker**: derived from a Brownian-motion
   risk-of-ruin estimate over the strategy's own recent bankroll-relative
   log-returns (`risk/circuit_breaker.py`), not an arbitrary percentage —
   plus a hard backstop drawdown limit in case that model itself is wrong.

Final stake = `min(kelly, variance_budget, hard_ceiling, liquidity)` — see
`risk/sizer.py`.

## Compliance flags (check before relying on any of this with real money)

- **Kalshi fee schedule** (`signal/fees.py`): the fee formula used is
  Kalshi's publicly documented general-market formula at the time this was
  built. Kalshi has changed fee structures before and does not expose a
  stable "current fees" API endpoint. **Re-verify the current schedule in
  the Kalshi UI before trading any specific series**, and add a new
  versioned `FeeSchedule` entry rather than editing the existing one.
- **Kalshi API auth/base URLs** (`data/kalshi_client.py`): confirmed
  against Kalshi's official `kalshi-starter-code-python` reference client
  at build time. Kalshi has changed its production hostname before
  (`api.elections.kalshi.com` despite covering non-election categories) —
  re-verify if auth starts failing.
- **Sports market title parsing** (`data/connectors/sports.py`): the
  regex-based team/matchup parser was written from general knowledge of
  Kalshi's sports market conventions, **not validated against a live API
  pull** (no API calls were made while building this, since no session
  credentials were available). Pull real event/market titles for your
  target leagues and confirm this parses correctly before training on
  real data.
- **Automated trading rules**: Kalshi's account-tier restrictions and
  rate limits on programmatic trading should be checked against your
  specific account before running any live loop — this system rate-limits
  itself conservatively (`data/kalshi_client.py`) but does not know your
  account's specific limits.
- **Orderbook schema in `execution/paper.py`**: derives ask-side prices
  from Kalshi's documented bid-only orderbook representation
  (`yes`/`no` resting-bid arrays; ask = 100 - opposite side's bid). Re-verify
  field names against a live response if this throws a `KeyError`.

## Adding a new forecaster

1. Subclass `forecasting.base.Forecaster`, implement `predict()` and `fit()`.
   `predict()` must only use data present in the `FeatureSnapshot` it's
   given — never reach out to a live data source inside `predict()`, or
   point-in-time correctness breaks silently in backtests.
2. Report a genuine `variance` in your `ProbabilityEstimate` — it directly
   drives Kelly shrinkage (`risk/kelly.py`); a model that always claims
   near-zero variance will get sized as if perfectly calibrated.
3. Wire calibration via `forecasting.calibration.Calibrator`, fit ONLY on a
   held-out set your model didn't train on.
4. To combine with other forecasters, use `forecasting.ensemble.ensemble_estimates`.

## Adding a new data source / category

1. Subclass `data.connectors.base.ExternalDataConnector`.
2. Implement `relevant_to(market)` and `fetch_point_in_time(market, as_of)`.
3. The point-in-time contract is the single most important thing to get
   right: never return a feature whose real-world publish time is after
   `as_of`. Raise `DataNotAvailableError` rather than returning a stale or
   partial snapshot silently.
4. Add the category to `config/default.yaml` under `categories.enabled`.

## Running paper trading / the dashboard

Paper trading loop wiring (`execution/paper.py` + `execution/audit.py`) is
built and tested; a scheduled/looping `scripts/run_paper_trading.py` driver
that ties it to live Kalshi demo-environment polling is the natural next
piece to build once real ingested history exists to retrain the forecaster
against.

```bash
streamlit run src/kalshi_quant/monitoring/dashboard.py
```

reads directly from the audit log JSONL files (`audit_logs/` by default) —
there's exactly one record of what happened, and the dashboard is a view
over it, not a second copy that could drift out of sync.

## Testing

```bash
pytest tests/ -q            # unit tests, 107 passing
pytest tests/ --cov=kalshi_quant --cov-report=term-missing
```

Point-in-time correctness has an explicit regression test:
`tests/backtest/test_generate_candidates.py` asserts that training data
handed to a forecaster during walk-forward retraining never has an
`as_of` timestamp at or after the market's own settlement time.
