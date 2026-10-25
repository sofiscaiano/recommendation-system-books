"""Dos torres colaborativas. El score es el producto de los dos vectores.

La torre de usuario transforma el embedding del lector. La de ítem concatena
libro, autor y género y los transforma con otra red. No hay una tercera red
ni Optuna: los hiperparámetros salen de ``TwoTowerConfig``. El sigmoide solo
entra en la pérdida, para comparar el producto con el rating escalado.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import PipelineConfig
from .neural import InteractionCatalog, NeuralRanker, TrainSchedule, lookup_rows


class _TwoTower(nn.Module):
    """Una red por lado. La salida de 256 es lineal para que el producto tenga signo.

    Con ReLU en esa salida el producto no baja de cero y el sigmoide no puede
    representar un rating por debajo del punto medio.
    """

    def __init__(
        self,
        catalog: InteractionCatalog,
        *,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(len(catalog.user_ids), embedding_dim)
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
        self.user_hidden = nn.Linear(embedding_dim, hidden_dim)
        self.user_output = nn.Linear(hidden_dim, output_dim)
        self.item_hidden = nn.Linear(embedding_dim * 3, hidden_dim)
        self.item_output = nn.Linear(hidden_dim, output_dim)

    def user_vector(self, users: torch.Tensor) -> torch.Tensor:
        """Vector del lector. La capa oculta tiene ReLU y la salida no."""

        hidden = torch.relu(self.user_hidden(self.user_embedding(users)))
        return self.user_output(hidden)

    def item_vector(self, items: torch.Tensor) -> torch.Tensor:
        """Vector del libro a partir de libro, autor y género."""

        features = torch.cat(
            [
                self.book(items),
                self.author(lookup_rows(self.item_author, items)),
                self.genre(lookup_rows(self.item_genre, items)),
            ],
            dim=-1,
        )
        hidden = torch.relu(self.item_hidden(features))
        return self.item_output(hidden)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Sigmoide del producto, en el mismo rango que el rating escalado."""

        product = (self.user_vector(users) * self.item_vector(items)).sum(dim=-1)
        return torch.sigmoid(product)


class TwoTowerModel(NeuralRanker):
    """Recomendador de una etapa. No usa el retrieval ni el LambdaRank."""

    name = "two_tower"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def build_net(self, catalog: InteractionCatalog) -> nn.Module:
        block = self.config.two_tower
        return _TwoTower(
            catalog,
            embedding_dim=block.embedding_dim,
            hidden_dim=block.hidden_dim,
            output_dim=block.output_dim,
        )

    def schedule(self) -> TrainSchedule:
        block = self.config.two_tower
        return TrainSchedule(
            learning_rate=block.learning_rate,
            batch_size=block.batch_size,
            max_epochs=block.max_epochs,
            finetune_epochs=block.finetune_epochs,
        )

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Producto contra todos los libros. Los vectores de ítem se arman una vez."""

        assert isinstance(self._net, _TwoTower)
        net = self._net
        device = next(net.parameters()).device
        items = torch.arange(net.book.num_embeddings, device=device)
        item_vectors = net.item_vector(items)
        blocks: list[np.ndarray] = []
        user_chunk = 256
        for start in range(0, len(user_indices), user_chunk):
            users = torch.as_tensor(
                user_indices[start : start + user_chunk],
                dtype=torch.long,
                device=device,
            )
            dots = net.user_vector(users) @ item_vectors.T
            blocks.append(dots.detach().cpu().numpy())
        if not blocks:
            return np.empty((0, net.book.num_embeddings))
        return np.concatenate(blocks, axis=0)

    def snapshot(self, catalog: InteractionCatalog) -> dict[str, object]:
        block = self.config.two_tower
        return {
            "model": self.name,
            "embedding_dim": block.embedding_dim,
            "hidden_dim": block.hidden_dim,
            "output_dim": block.output_dim,
            "learning_rate": block.learning_rate,
            "batch_size": block.batch_size,
            "max_epochs": block.max_epochs,
            "finetune_epochs": block.finetune_epochs,
            "n_users": len(catalog.user_ids),
            "n_items": len(catalog.item_ids),
            "n_authors": catalog.n_authors,
            "n_genres": catalog.n_genres,
        }
