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
- Inferencia: `recommend_top_k` por lotes. Candidatos: `Settings.only_interacted_items` (default True) descarta libros sin ninguna fila en `interacciones`. `filter_seen_genres` / `filter_seen_authors` (default True) restringen a géneros y/o autores del historial; si ambos están activos, se unen (OR). Aplica a eval y submission.
- Eval escribe el top-k en `eval_recs.csv` (`--eval-output`) para análisis; no corre con `--skip-eval`.

## Cómo cambiar el repo

- Preferir ediciones mínimas en el paquete. No modificar `test_inicial.ipynb` salvo pedido.
- Cuando el usuario **apruebe** un cambio (ok explícito, confirmación de plan, o pedido de implementar), **commitear** esos cambios en git al terminar, sin esperar un segundo "commit" salvo que pida no hacerlo.
- No pushear a remoto salvo pedido.
- Commits cortos, en español o inglés, enfocados en el porqué. No incluir `data.db`, secretos ni artefactos grandes.
