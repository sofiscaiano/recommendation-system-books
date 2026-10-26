"""Red única al estilo del Colab, con autor y género.

Lector, libro, autor y género son embeddings. Se concatenan y una sola red
predice el rating con un sigmoide. El ranking ordena por esa predicción.
No hay Optuna: los hiperparámetros salen de ``RatingMlpConfig``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import PipelineConfig
from .neural import InteractionCatalog, NeuralRanker, TrainSchedule, lookup_rows


class _RatingMlp(nn.Module):
    """Embeddings concatenados y la red densa del Colab."""

    def __init__(
        self,
        catalog: InteractionCatalog,
        *,
        embedding_dim: int,
        hidden_1: int,
        hidden_2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.user = nn.Embedding(len(catalog.user_ids), embedding_dim)
        self.book = nn.Embedding(len(catalog.item_ids), embedding_dim)
        self.author = nn.Embedding(catalog.n_authors, embedding_dim)
        self.genre = nn.Embedding(catalog.n_genres, embedding_dim)
        self.register_buffer(
            "item_author",
            torch.as_tensor(catalog.item_author, dtype=torch.long),
        )
        self.register_buffer(
            "item_genre",
            torch.as_tensor(catalog.item_genre, dtype=torch.long),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Rating predicho en 0–1 para cada par."""

        features = torch.cat(
            [
                self.user(users),
                self.book(items),
                self.author(lookup_rows(self.item_author, items)),
                self.genre(lookup_rows(self.item_genre, items)),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.mlp(features).squeeze(-1))


class RatingMlpModel(NeuralRanker):
    """Recomendador de una etapa. No usa el retrieval ni el LambdaRank."""

    name = "rating_mlp"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def build_net(self, catalog: InteractionCatalog) -> nn.Module:
        block = self.config.rating_mlp
        return _RatingMlp(
            catalog,
            embedding_dim=block.embedding_dim,
            hidden_1=block.hidden_1,
            hidden_2=block.hidden_2,
            dropout=block.dropout,
        )

    def schedule(self) -> TrainSchedule:
        block = self.config.rating_mlp
        return TrainSchedule(
            learning_rate=block.learning_rate,
            weight_decay=block.weight_decay,
            batch_size=block.batch_size,
            max_epochs=block.max_epochs,
            finetune_epochs=block.finetune_epochs,
            patience=block.patience,
        )

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Puntúa el catálogo por lotes. La cabeza no admite un solo producto."""

        assert isinstance(self._net, _RatingMlp)
        net = self._net
        device = next(net.parameters()).device
        n_items = net.book.num_embeddings
        user_chunk = 8
        item_chunk = 1024
        blocks: list[np.ndarray] = []
        for user_start in range(0, len(user_indices), user_chunk):
            user_block = user_indices[user_start : user_start + user_chunk]
            users = torch.as_tensor(user_block, dtype=torch.long, device=device)
            columns: list[torch.Tensor] = []
            for item_start in range(0, n_items, item_chunk):
                item_stop = min(item_start + item_chunk, n_items)
                items = torch.arange(item_start, item_stop, device=device)
                pairs_user = users.repeat_interleave(items.shape[0])
                pairs_item = items.repeat(users.shape[0])
                prediction = net(pairs_user, pairs_item).reshape(users.shape[0], items.shape[0])
                columns.append(prediction.detach().cpu())
            blocks.append(torch.cat(columns, dim=1).numpy())
        return np.concatenate(blocks, axis=0) if blocks else np.empty((0, n_items))

    def snapshot(self, catalog: InteractionCatalog) -> dict[str, object]:
        block = self.config.rating_mlp
        return {
            "model": self.name,
            "embedding_dim": block.embedding_dim,
            "hidden_1": block.hidden_1,
            "hidden_2": block.hidden_2,
            "dropout": block.dropout,
            "learning_rate": block.learning_rate,
            "weight_decay": block.weight_decay,
            "patience": block.patience,
            "batch_size": block.batch_size,
            "max_epochs": block.max_epochs,
            "finetune_epochs": block.finetune_epochs,
            "n_users": len(catalog.user_ids),
            "n_items": len(catalog.item_ids),
            "n_authors": catalog.n_authors,
            "n_genres": catalog.n_genres,
        }
