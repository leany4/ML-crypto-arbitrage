from __future__ import annotations

import pytest
from pydantic import ValidationError

from ml_service.schemas import MarketRef, PairDefinition


def test_spot_perp_accepts_long_spot_short_perp() -> None:
    pair = PairDefinition(
        pair_id="BTC-spot-perp",
        base_ticker="BTC",
        pair_type="spot_perp_same_exchange",
        leg1=MarketRef(exchange="gate", ticker="BTC/USDT", is_perp=False),
        leg2=MarketRef(
            exchange="gate", ticker="BTC/USDT:USDT", is_perp=True
        ),
        direction_code=0,
    )

    assert pair.leg1.is_perp is False
    assert pair.leg2.is_perp is True


def test_spot_perp_rejects_direction_outside_old_q35_domain() -> None:
    with pytest.raises(ValidationError, match="long spot and short perpetual"):
        PairDefinition(
            pair_id="BTC-perp-spot",
            base_ticker="BTC",
            pair_type="spot_perp_same_exchange",
            leg1=MarketRef(
                exchange="gate", ticker="BTC/USDT:USDT", is_perp=True
            ),
            leg2=MarketRef(
                exchange="gate", ticker="BTC/USDT", is_perp=False
            ),
            direction_code=0,
        )
