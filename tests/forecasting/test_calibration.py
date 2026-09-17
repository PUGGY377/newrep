import numpy as np
import pytest

from kalshi_quant.forecasting.calibration import Calibrator, evaluate_calibration


def test_calibrator_requires_minimum_samples():
    c = Calibrator(method="isotonic")
    with pytest.raises(ValueError):
        c.fit(np.array([0.1, 0.2]), np.array([0, 1]))


def test_calibrator_isotonic_improves_miscalibrated_scores():
    rng = np.random.default_rng(42)
    n = 500
    true_p = rng.uniform(0, 1, n)
    outcomes = (rng.uniform(0, 1, n) < true_p).astype(float)
    # miscalibrated raw score: compressed toward 0.5
    raw_scores = 0.5 + (true_p - 0.5) * 0.3

    before = evaluate_calibration(raw_scores, outcomes)

    c = Calibrator(method="isotonic")
    c.fit(raw_scores, outcomes)
    calibrated = c.transform(raw_scores)
    after = evaluate_calibration(calibrated, outcomes)

    assert after.brier_score < before.brier_score


def test_calibrator_transform_before_fit_raises():
    c = Calibrator()
    with pytest.raises(RuntimeError):
        c.transform(np.array([0.5]))


def test_evaluate_calibration_perfect_predictions_have_zero_brier():
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    preds = np.array([1.0, 0.0, 1.0, 0.0])
    report = evaluate_calibration(preds, outcomes)
    assert report.brier_score == pytest.approx(0.0, abs=1e-4)


def test_evaluate_calibration_bin_shapes():
    outcomes = np.random.default_rng(0).integers(0, 2, 100).astype(float)
    preds = np.random.default_rng(1).uniform(0, 1, 100)
    report = evaluate_calibration(preds, outcomes, n_bins=5)
    assert len(report.bin_edges) == 6
    assert len(report.bin_mean_predicted) == 5
    assert report.bin_counts.sum() == 100
