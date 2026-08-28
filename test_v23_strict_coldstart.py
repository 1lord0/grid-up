import numpy as np
import pandas as pd

from train_v23_strict_coldstart import (
    HierarchicalResidualPrior,
    _commissioned_rows,
    _log_blend,
    calculate_rmsle,
    prepare_panel,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tanim": ["left", "left", "new", "new", "new"],
            "guc": [100, 100, 250, 250, 250],
            "tarih": pd.to_datetime(
                ["2025-01-01", "2025-01-10", "2025-02-01", "2025-02-02", "2025-02-03"]
            ),
            "lokasyon": ["IZMIR>METROPOL>KONAK"] * 5,
            "tuketim": [10.0, 11.0, 20.0, 21.0, 22.0],
        }
    )


def test_commissioned_rows_excludes_left_truncated_and_future_targets() -> None:
    panel = prepare_panel(_panel())
    selected = _commissioned_rows(panel, pd.Timestamp("2025-02-02"))
    assert set(selected["tanim"]) == {"new"}
    assert selected["tarih"].max() == pd.Timestamp("2025-02-02")


def test_calendar_and_cohort_features_are_label_free() -> None:
    original = _panel()
    changed = original.copy()
    changed["tuketim"] = changed["tuketim"] * 1000
    left = prepare_panel(original).drop(columns="tuketim")
    right = prepare_panel(changed).drop(columns="tuketim")
    pd.testing.assert_frame_equal(left, right)


def test_holiday_features_cover_test_period_rules() -> None:
    frame = pd.DataFrame(
        {
            "tanim": ["x", "x", "x"],
            "guc": [100, 100, 100],
            "tarih": pd.to_datetime(["2026-05-26", "2026-05-27", "2026-06-27"]),
            "lokasyon": ["MANISA>SARIGOL"] * 3,
        }
    )
    panel = prepare_panel(frame)
    assert panel["event_type"].tolist()[:2] == ["religious_eve", "religious_holiday"]
    assert panel["event_window"].tolist()[:2] == ["0", "0"]
    assert panel["school_closed"].tolist() == [0, 0, 1]


def test_log_blend_endpoints_and_metric_domain() -> None:
    static_log = np.log1p(np.array([0.0, 10.0, 100.0]))
    cohort_log = np.log1p(np.array([2.0, 20.0, 200.0]))
    np.testing.assert_allclose(_log_blend(static_log, cohort_log, 0.0), [0.0, 10.0, 100.0])
    np.testing.assert_allclose(_log_blend(static_log, cohort_log, 1.0), [2.0, 20.0, 200.0])
    assert calculate_rmsle(np.array([0.0]), np.array([-5.0])) == 0.0


def test_hierarchical_prior_prediction_does_not_read_prediction_labels() -> None:
    fit_panel = prepare_panel(_panel())
    model = HierarchicalResidualPrior(alpha_scale=4.0).fit(fit_panel)
    prediction_panel = prepare_panel(_panel())
    first = model.predict(prediction_panel, shrink=0.5)
    prediction_panel["tuketim"] = 999999.0
    second = model.predict(prediction_panel, shrink=0.5)
    np.testing.assert_allclose(first, second)
