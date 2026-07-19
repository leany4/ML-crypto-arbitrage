"""HTTP API управления историческим replay."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from market_simulator import __version__
from market_simulator.config import SimulatorSettings
from market_simulator.replay import ReplayController


settings = SimulatorSettings.load()
controller = ReplayController(settings)
app = FastAPI(title="Arbitrage Market Simulator", version=__version__)


class StartRequest(BaseModel):
    """Параметры нового виртуального прогона."""

    speed: float = Field(default=settings.default_speed, gt=0, le=100)
    start_ts: int | None = None
    end_ts: int | None = None
    duration_seconds: int | None = Field(
        default=settings.default_duration_seconds,
        ge=1,
    )


def get_controller() -> ReplayController:
    """Вернуть singleton-контроллер текущего процесса."""

    return controller


Controller = Annotated[ReplayController, Depends(get_controller)]


@app.get("/")
def root() -> dict[str, str]:
    """Вернуть имя и версию сервиса."""

    return {"service": "arb-market-simulator", "version": __version__}


@app.get("/health/live")
def live() -> dict[str, str]:
    """Подтвердить доступность HTTP-процесса."""

    return {"status": "alive"}


@app.get("/v1/replay/status")
def replay_status(replay: Controller) -> dict[str, object]:
    """Вернуть прогресс и счётчики текущего replay."""

    return replay.status()


@app.get("/v1/replay/pairs")
def replay_pairs(replay: Controller) -> list[dict[str, object]]:
    """Вернуть направления пар из подготовленного набора."""

    return replay.pairs()


@app.post("/v1/replay/start")
def replay_start(request: StartRequest, replay: Controller) -> dict[str, object]:
    """Запустить replay в отдельном фоновом потоке."""

    try:
        return replay.start(
            speed=request.speed,
            start_ts=request.start_ts,
            end_ts=request.end_ts,
            duration_seconds=request.duration_seconds,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/replay/pause")
def replay_pause(replay: Controller) -> dict[str, object]:
    """Приостановить виртуальное время без потери позиции чтения."""

    try:
        return replay.pause()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/replay/resume")
def replay_resume(replay: Controller) -> dict[str, object]:
    """Продолжить ранее приостановленный replay."""

    try:
        return replay.resume()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/replay/stop")
def replay_stop(replay: Controller) -> dict[str, object]:
    """Остановить replay и дождаться фонового потока."""

    return replay.stop()
