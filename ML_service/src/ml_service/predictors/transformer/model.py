"""PyTorch-архитектура multiscale decision Transformer."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBranch(nn.Module):
    """Кодирует временную последовательность через обучаемый CLS-токен."""

    def __init__(
        self,
        input_dim: int,
        token_count: int,
        d_model: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, token_count + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Вернуть агрегированное представление всей ветки."""

        tokens = self.input_projection(values)
        cls = self.cls_token.expand(values.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        return self.norm(self.encoder(tokens)[:, 0])


class StaticEncoder(nn.Module):
    """Кодирует тип пары, направление, биржи и устойчивый hash пары."""

    def __init__(self, exchange_count: int, pair_hash_buckets: int):
        super().__init__()
        self.pair_type = nn.Embedding(3, 4)
        self.direction = nn.Embedding(3, 4)
        self.leg1_exchange = nn.Embedding(exchange_count, 8)
        self.leg2_exchange = nn.Embedding(exchange_count, 8)
        self.pair_hash = nn.Embedding(pair_hash_buckets, 8)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Объединить категориальные embeddings в один вектор."""

        return torch.cat(
            [
                self.pair_type(batch["pair_type_id"]),
                self.direction(batch["direction_id"]),
                self.leg1_exchange(batch["leg1_exchange_id"]),
                self.leg2_exchange(batch["leg2_exchange_id"]),
                self.pair_hash(batch["pair_hash_id"]),
            ],
            dim=-1,
        )


class DecisionHeads(nn.Module):
    """Набор классификационных и квантильных торговых голов."""

    def __init__(self, hidden_dim: int, quantile_count: int = 3):
        super().__init__()

        def scalar_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, 96),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(96, 1),
            )

        def quantile_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, 96),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(96, quantile_count),
            )

        self.watch = scalar_head()
        self.enter = scalar_head()
        self.entry_executable = scalar_head()
        self.enter_now_quantiles = quantile_head()
        self.wait_executable = scalar_head()
        self.wait_best_quantiles = quantile_head()
        self.enter_advantage_quantiles = quantile_head()
        self.exit = scalar_head()
        self.exit_advantage = scalar_head()

    @staticmethod
    def ordered_quantiles(raw: torch.Tensor) -> torch.Tensor:
        """Гарантировать монотонный порядок q20 <= q35 <= q50."""

        q20 = raw[:, 0]
        q35 = q20 + nn.functional.softplus(raw[:, 1] - 3.0)
        q50 = q35 + nn.functional.softplus(raw[:, 2] - 3.0)
        return torch.stack([q20, q35, q50], dim=-1)

    def forward(
        self, context_state: torch.Tensor, entry_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Рассчитать классификационные logits и упорядоченные quantiles."""

        enter_now = self.ordered_quantiles(self.enter_now_quantiles(entry_state))
        wait_best = self.ordered_quantiles(self.wait_best_quantiles(entry_state))
        enter_advantage = self.ordered_quantiles(
            self.enter_advantage_quantiles(entry_state)
        )
        return {
            "watch": self.watch(context_state).squeeze(-1),
            "enter": self.enter(entry_state).squeeze(-1),
            "entry_executable": self.entry_executable(entry_state).squeeze(-1),
            "enter_now_quantiles": enter_now,
            "wait_executable": self.wait_executable(entry_state).squeeze(-1),
            "wait_best_quantiles": wait_best,
            "enter_advantage_quantiles": enter_advantage,
            "exit": self.exit(context_state).squeeze(-1),
            "exit_advantage": self.exit_advantage(context_state).squeeze(-1),
        }


class GatedMultiscaleDecisionTransformer(nn.Module):
    """Объединяет short/long L2, OHLCV и состояние paper-позиции."""

    def __init__(
        self,
        l2_input_dim: int,
        ohlcv_input_dim: int,
        local_steps: int,
        long_tokens: int,
        exchange_count: int,
        pair_hash_buckets: int,
        d_model: int = 96,
        heads: int = 4,
        layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.local_branch = TemporalBranch(
            l2_input_dim, local_steps, d_model, heads, layers, ff_dim, dropout
        )
        self.long_branch = TemporalBranch(
            l2_input_dim, long_tokens, d_model, heads, layers, ff_dim, dropout
        )
        self.static = StaticEncoder(exchange_count, pair_hash_buckets)
        self.position = nn.Sequential(
            nn.Linear(6, 32), nn.GELU(), nn.LayerNorm(32)
        )
        self.current_l2_token = nn.Sequential(
            nn.Linear(l2_input_dim, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.entry_snapshot = nn.Sequential(
            nn.Linear(6, 32), nn.GELU(), nn.LayerNorm(32)
        )
        self.l2_fusion = nn.Sequential(
            nn.Linear(d_model * 2 + 32 + 32, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(192),
        )
        self.ohlcv_encoder = nn.Sequential(
            nn.Linear(ohlcv_input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.LayerNorm(128),
        )
        self.ohlcv_residual = nn.Linear(128, 192)
        self.ohlcv_gate = nn.Sequential(
            nn.Linear(192 + 128, 192),
            nn.Sigmoid(),
        )
        self.context_norm = nn.LayerNorm(192)
        self.entry_fusion = nn.Sequential(
            nn.Linear(192 + 64 + 32, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(192),
        )
        self.heads = DecisionHeads(192)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Выполнить общий encoder и вернуть сырые выходы всех голов."""

        l2_context = self.l2_fusion(
            torch.cat(
                [
                    self.local_branch(batch["local"]),
                    self.long_branch(batch["long"]),
                    self.static(batch),
                    self.position(batch["position_state"]),
                ],
                dim=-1,
            )
        )
        ohlcv_context = self.ohlcv_encoder(batch["ohlcv_state"])
        gate = self.ohlcv_gate(torch.cat([l2_context, ohlcv_context], dim=-1))
        context_state = self.context_norm(
            l2_context + gate * self.ohlcv_residual(ohlcv_context)
        )
        entry_state = self.entry_fusion(
            torch.cat(
                [
                    context_state,
                    self.current_l2_token(batch["local"][:, -1]),
                    self.entry_snapshot(batch["entry_snapshot"]),
                ],
                dim=-1,
            )
        )
        return self.heads(context_state, entry_state)
