from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_service.predictors.base import PredictionContext
from ml_service.predictors.rl import ACTION_NAMES, RLPredictor


def test_recurrent_action_order_is_wait_enter_hold_exit(tmp_path: Path) -> None:
    predictor = RLPredictor("agent", tmp_path, "cpu")

    assert ACTION_NAMES == ("wait", "enter", "hold", "exit")
    flat = PredictionContext(
        features={"execution_valid_numeric": 1.0},
        position_state=[0.0] * 8,
    )
    assert predictor._action_mask(flat, gate_active=True).tolist() == [
        True,
        True,
        False,
        False,
    ]

    open_too_early = PredictionContext(
        features={"execution_valid_numeric": 1.0},
        position_state=[1.0, 0.0, 0.0, 0.0, 9.0 / 3000.0, 0.0, 0.0, 0.0],
    )
    assert predictor._action_mask(
        open_too_early, gate_active=True
    ).tolist() == [False, False, True, False]

    open_exit_allowed = PredictionContext(
        features={"execution_valid_numeric": 1.0},
        position_state=[
            1.0,
            0.0,
            0.0,
            0.0,
            10.0 / 3000.0,
            0.0,
            0.0,
            0.0,
        ],
    )
    assert predictor._action_mask(
        open_exit_allowed, gate_active=True
    ).tolist() == [False, False, True, True]


def test_rl_missing_feature_is_zero_after_normalization(tmp_path: Path) -> None:
    predictor = RLPredictor("agent", tmp_path, "cpu")
    predictor.feature_columns = ["missing_value", "finite_value"]
    predictor.feature_mean = np.asarray([10.0, 10.0], dtype=np.float32)
    predictor.feature_std = np.asarray([2.0, 2.0], dtype=np.float32)

    row = predictor._feature_row(
        {"missing_value": np.nan, "finite_value": 12.0},
        q35_bps=30.0,
        time_from_gate_sec=0.0,
        active=True,
    )

    np.testing.assert_array_equal(
        row,
        np.asarray([0.0, 1.0], dtype=np.float32),
    )
