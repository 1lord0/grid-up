from __future__ import annotations

import numpy as np
import pandas as pd

from train_v21_cohort_router import (
    CAT_FEATURES,
    FEATURES,
    apply_log_correction,
    divergence_gate,
    prepare_panel,
)


def _tiny_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tanim": ["a", "a", "b", "b"],
            "guc": [250, 250, 630, 630],
            "tarih": pd.to_datetime(
                ["2026-05-11", "2026-05-12", "2026-05-11", "2026-05-13"]
            ),
            "lokasyon": [
                "IZMIR>METROPOL>BORNOVA",
                "IZMIR>METROPOL>BORNOVA",
                "MANISA>MERKEZ>YUNUSEMRE",
                "MANISA>MERKEZ>YUNUSEMRE",
            ],
        }
    )


def test_prepare_panel_uses_only_observable_structure() -> None:
    prepared = prepare_panel(_tiny_panel())
    assert set(FEATURES).issubset(prepared.columns)
    assert prepared["age_days"].tolist() == [0, 1, 0, 2]
    assert prepared["cohort_size"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert all(str(prepared[column].dtype) == "object" for column in CAT_FEATURES)


def test_divergence_gate_is_symmetric_and_shrinks_extremes() -> None:
    delta = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    weights = divergence_gate(delta, scale=0.75)
    assert np.allclose(weights, weights[::-1])
    assert weights[2] == 0.5
    assert weights[0] < weights[1] < weights[2]


def test_log_correction_preserves_nonnegative_domain() -> None:
    base = np.array([0.0, 100.0, 1000.0])
    delta = np.array([-10.0, -0.5, 0.5])
    corrected, weights = apply_log_correction(base, delta)
    assert np.isfinite(corrected).all()
    assert (corrected >= 0.0).all()
    assert ((weights >= 0.0) & (weights <= 1.0)).all()
