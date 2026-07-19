from __future__ import annotations

from pathlib import Path

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
