# Instrucciones para agentes

Recomendador de ranking (top-20 libros por `id_lector`). Contexto: [README.md](README.md).

## Stack y archivos

- Python ≥ 3.13, `uv sync`. Código de producción: `src/quelibroleo/` y `main.py`.
- No modificar `test_inicial.ipynb` salvo que el usuario lo pida.
- Datos: `data.db` (`libros`, `lectores`, `interacciones`). No modificar la base.
- Entrega: CSV `id_lector,id_libro`, 20 filas por lector, mismos usuarios que `ejemplo.csv`.
- Lectores sin historial: rellenar con `ejemplo.csv`.

## Modelado

- Ranking: NDCG@20 con `sklearn.metrics.ndcg_score` (unión top-k ∪ test; no recomendados con score menor al mínimo predicho).
- Diagnóstico del regresor: MSE y RMSE sobre ratings del hold-out.
- Features de usuario/ítem solo con train en eval; split temporal leave-last-20.
- Inferencia: `recommend_top_k` (catálogo completo menos vistos, por lotes). No duplicar loops ni filtrar por géneros vistos.

## Cómo cambiar el repo

- Preferir ediciones mínimas en el paquete. No crear commits ni PRs salvo pedido.
- Commits cortos, enfocados en el porqué.
