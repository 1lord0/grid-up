from __future__ import annotations

import numpy as np

from build_v22_lb_calibrated_ensemble import (
    error_covariance,
    optimize_nonnegative_weights,
    pairwise_log_distance,
)


def test_pairwise_distance_is_symmetric() -> None:
    predictions = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    distance = pairwise_log_distance(predictions)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)


def test_covariance_recovers_error_cross_product() -> None:
    target = np.array([1.5, 2.5, 4.0])
    predictions = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    errors = predictions - target[:, None]
    losses = np.mean(np.square(errors), axis=0)
    covariance = error_covariance(
        losses, pairwise_log_distance(predictions)
    )
    assert np.allclose(covariance, errors.T @ errors / len(target))


def test_optimizer_returns_convex_weights() -> None:
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]])
    weights, score = optimize_nonnegative_weights(covariance)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0.0).all()
    assert score < 1.0
