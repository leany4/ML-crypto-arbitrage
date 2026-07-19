from __future__ import annotations

from ml_service.monitor import _rl_runtime_view


def test_rl_runtime_view_explains_gate_and_inference_errors() -> None:
    waiting = _rl_runtime_view(
        strategy_active=True,
        model_ready=True,
        q35_bps=25.0,
        gate_threshold_bps=30.0,
        execution_valid=True,
        outputs={"action": 0, "gate_active": 0},
    )
    assert waiting["rl_status"] == "WAIT Q35"
    assert waiting["rl_gate_margin_bps"] == -5.0

    failed = _rl_runtime_view(
        strategy_active=True,
        model_ready=True,
        q35_bps=80.0,
        gate_threshold_bps=30.0,
        execution_valid=True,
        outputs={"_error": "missing feature"},
    )
    assert failed["rl_status"] == "ERROR"
    assert failed["rl_error"] == "missing feature"

    active = _rl_runtime_view(
        strategy_active=True,
        model_ready=True,
        q35_bps=80.0,
        gate_threshold_bps=30.0,
        execution_valid=True,
        outputs={
            "action": 0,
            "gate_active": 1,
            "frozen_q35_bps": 80.0,
        },
    )
    assert active["rl_status"] == "WAIT"
    assert active["rl_gate_active"] is True
