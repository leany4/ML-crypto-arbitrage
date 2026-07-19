from ml_service.device import resolve_device


def test_cpu_is_always_available() -> None:
    choice = resolve_device("cpu")
    assert choice.resolved == "cpu"
    assert choice.fallback_reason is None

