"""Выбор устройства для PyTorch-моделей с безопасным fallback на CPU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceChoice:
    """Результат выбора устройства и причина возможного fallback."""

    requested: str
    resolved: str
    fallback_reason: str | None = None


def resolve_device(requested: str) -> DeviceChoice:
    """Разрешить `auto/cpu/cuda` без падения при недоступной CUDA."""

    requested = requested.lower().strip()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cpu":
        return DeviceChoice(requested=requested, resolved="cpu")

    try:
        import torch
    except ImportError:
        return DeviceChoice(
            requested=requested,
            resolved="cpu",
            fallback_reason="torch is not installed",
        )

    if torch.cuda.is_available():
        return DeviceChoice(requested=requested, resolved="cuda")
    return DeviceChoice(
        requested=requested,
        resolved="cpu",
        fallback_reason="CUDA is not available to PyTorch",
    )
