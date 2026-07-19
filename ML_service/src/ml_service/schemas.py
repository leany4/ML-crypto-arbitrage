"""Pydantic-контракты входного рыночного потока и управляющего API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PriceLevel(BaseModel):
    """Один ценовой уровень стакана."""

    price: float = Field(gt=0)
    volume: float = Field(ge=0)


class L2Snapshot(BaseModel):
    """Пять лучших bid/ask уровней одного рынка."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    exchange: str
    exchange_ts: int
    machine_ts: int | None = None
    machine_ts_final: int
    is_perp: bool
    bids: list[PriceLevel] = Field(min_length=1, max_length=5)
    asks: list[PriceLevel] = Field(min_length=1, max_length=5)

    @field_validator("bids")
    @classmethod
    def bids_must_descend(cls, levels: list[PriceLevel]) -> list[PriceLevel]:
        """Проверить убывание bid-цен от лучшей к худшей."""

        if any(left.price < right.price for left, right in zip(levels, levels[1:])):
            raise ValueError("bid prices must be descending")
        return levels

    @field_validator("asks")
    @classmethod
    def asks_must_ascend(cls, levels: list[PriceLevel]) -> list[PriceLevel]:
        """Проверить возрастание ask-цен от лучшей к худшей."""

        if any(left.price > right.price for left, right in zip(levels, levels[1:])):
            raise ValueError("ask prices must be ascending")
        return levels

    @model_validator(mode="after")
    def book_must_not_cross(self) -> "L2Snapshot":
        """Отклонить пересечённый или неконсистентный стакан."""

        if self.bids[0].price > self.asks[0].price:
            raise ValueError("crossed L2 book")
        return self


class L2BatchRequest(BaseModel):
    """Batch снимков одной близкой временной точки."""

    model_config = ConfigDict(extra="forbid")

    snapshots: list[L2Snapshot] = Field(min_length=1, max_length=5_000)


class OHLCVCandle(BaseModel):
    """Закрытая свеча, причинно доступная после конца timeframe."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    exchange: str
    tf: Literal["1m", "5m"]
    ts: int
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    is_closed: bool = True

    @model_validator(mode="after")
    def prices_are_consistent(self) -> "OHLCVCandle":
        """Проверить OHLC-ограничения high/low."""

        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is below another candle price")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is above another candle price")
        return self


class OHLCVBatchRequest(BaseModel):
    """Batch новых закрытых свечей."""

    model_config = ConfigDict(extra="forbid")

    candles: list[OHLCVCandle] = Field(min_length=1, max_length=5_000)


class MarketRef(BaseModel):
    """Ссылка на конкретный рынок одной ноги."""

    exchange: str
    ticker: str
    is_perp: bool


class PairDefinition(BaseModel):
    """Направление сделки между двумя совместимыми рынками."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    base_ticker: str
    pair_type: Literal["perp_perp_cross_exchange", "spot_perp_same_exchange"]
    leg1: MarketRef
    leg2: MarketRef
    direction_code: Literal[0, 1] = 0
    enabled: bool = True

    @model_validator(mode="after")
    def pair_structure_matches_type(self) -> "PairDefinition":
        """Проверить тип рынков, биржи и допустимое направление пары."""

        if self.pair_type == "perp_perp_cross_exchange":
            if not self.leg1.is_perp or not self.leg2.is_perp:
                raise ValueError("perp-perp pair requires two perpetual legs")
            if self.leg1.exchange.lower() == self.leg2.exchange.lower():
                raise ValueError("cross-exchange pair requires different exchanges")
            return self

        if self.leg1.exchange.lower() != self.leg2.exchange.lower():
            raise ValueError("spot-perp pair requires the same exchange")
        if self.leg1.is_perp == self.leg2.is_perp:
            raise ValueError("spot-perp pair requires one spot and one perpetual")
        long_leg = self.leg1 if self.direction_code == 0 else self.leg2
        short_leg = self.leg2 if self.direction_code == 0 else self.leg1
        if long_leg.is_perp or not short_leg.is_perp:
            raise ValueError(
                "old spot-perp q35/RL contract requires long spot and short perpetual"
            )
        return self


class DirectPredictRequest(BaseModel):
    """Диагностический stateless-запрос к одному predictor."""

    features: dict[str, Any] = Field(default_factory=dict)
    transformer_input: dict[str, Any] | None = None
    heads: list[str] | None = None


class ModelActionResponse(BaseModel):
    """Результат load/reload/unload операции над моделью."""

    name: str
    status: str
    detail: str | None = None
