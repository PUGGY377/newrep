import pytest

from kalshi_quant.risk.circuit_breaker import estimate_risk_of_ruin, should_halt_new_positions


def test_insufficient_history_forces_maximal_caution():
    estimate = estimate_risk_of_ruin(
        bankroll_before=[100.0, 105.0], bankroll_after=[105.0, 95.0],
        current_drawdown_fraction=0.05, hard_stop_drawdown_fraction=0.4,
    )
    assert estimate.n_trades == 2
    assert estimate.probability_of_ruin_at_current_drawdown == 1.0


def test_positive_drift_low_variance_gives_low_ruin_probability():
    # consistent small wins: bankroll grows 2% each trade
    before = [100.0 * (1.02 ** i) for i in range(30)]
    after = [b * 1.02 for b in before]
    estimate = estimate_risk_of_ruin(
        before, after, current_drawdown_fraction=0.05, hard_stop_drawdown_fraction=0.4,
    )
    assert estimate.mu > 0
    assert estimate.probability_of_ruin_at_current_drawdown < 0.5


def test_negative_drift_gives_certain_ruin():
    before = [100.0 * (0.98 ** i) for i in range(30)]
    after = [b * 0.98 for b in before]
    estimate = estimate_risk_of_ruin(
        before, after, current_drawdown_fraction=0.05, hard_stop_drawdown_fraction=0.4,
    )
    assert estimate.mu <= 0
    assert estimate.probability_of_ruin_at_current_drawdown == 1.0


def test_full_stake_loss_on_small_fraction_is_not_treated_as_ruin():
    # A trade that loses its ENTIRE stake, but the stake was only 5% of
    # bankroll, should register as a mild negative bankroll return (-5%),
    # not "wealth went to zero" -- this is the bug this module's design
    # exists to avoid (see module docstring).
    before = [100.0] * 10
    after = [95.0] * 10  # lost the full 5-unit stake every time
    estimate = estimate_risk_of_ruin(
        before, after, current_drawdown_fraction=0.05, hard_stop_drawdown_fraction=0.4,
    )
    assert estimate.mu == pytest.approx(-0.05129, abs=1e-3)  # ln(0.95), not ln(~0)


def test_should_halt_on_hard_stop_regardless_of_model():
    before = [100.0 * (1.02 ** i) for i in range(30)]
    after = [b * 1.02 for b in before]
    halt, reason = should_halt_new_positions(
        before, after, current_drawdown_fraction=0.45,
        risk_of_ruin_halt_threshold=0.5, max_drawdown_hard_stop=0.4,
    )
    assert halt
    assert "Hard-stop" in reason


def test_should_not_halt_when_healthy():
    before = [100.0 * (1.02 ** i) for i in range(30)]
    after = [b * 1.02 for b in before]
    halt, reason = should_halt_new_positions(
        before, after, current_drawdown_fraction=0.05,
        risk_of_ruin_halt_threshold=0.05, max_drawdown_hard_stop=0.4,
    )
    assert not halt


def test_should_halt_on_computed_risk_of_ruin_before_hard_stop():
    before = [100.0 * (0.98 ** i) for i in range(10)]
    after = [b * 0.98 for b in before]
    halt, reason = should_halt_new_positions(
        before, after, current_drawdown_fraction=0.10,
        risk_of_ruin_halt_threshold=0.05, max_drawdown_hard_stop=0.4,
    )
    assert halt
    assert "risk-of-ruin" in reason


def test_insufficient_history_does_not_halt_only_hard_stop_does():
    # Regression test: with only a couple of trades, should_halt_new_positions
    # must NOT act on the "maximal caution" P=1 reading from
    # estimate_risk_of_ruin, or the breaker would trip permanently after
    # trade #1 with no way to ever accumulate enough history to recover.
    halt, reason = should_halt_new_positions(
        bankroll_before=[100.0], bankroll_after=[95.0],
        current_drawdown_fraction=0.05, risk_of_ruin_halt_threshold=0.05,
        max_drawdown_hard_stop=0.4,
    )
    assert not halt
