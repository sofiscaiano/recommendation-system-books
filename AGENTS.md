# Instrucciones para agentes

Recomendador de ranking (top-20 libros por `id_lector`). Contexto: [README.md](README.md).

## Stack y archivos

- Python ≥ 3.13, `uv sync`. Código de producción: `src/quelibroleo/` y `main.py`.
- No modificar `test_inicial.ipynb` salvo que el usuario lo pida.
- Datos: `data.db` (`libros`, `lectores`, `interacciones`). No modificar la base.
- Entrega: CSV `id_lector,id_libro`, 20 filas por lector, mismos usuarios que `ejemplo.csv`.

## Modelado

- Ranking: NDCG@20 con `sklearn.metrics.ndcg_score`
- Split temporal para evaluacion: leave-last-20.

## Cómo cambiar el repo

- Preferir ediciones mínimas en el paquete. No modificar `test_inicial.ipynb` salvo pedido.
- Cada cambio aprobado se **commitea al terminar**, en su propio commit. No dejar el trabajo solo en el working tree y no esperar un pedido aparte de "commit". Aprobado incluye un ok explícito, la confirmación de un plan o un pedido de implementar. Omitir el commit solo si el usuario pide no hacerlo.
- No pushear a remoto salvo pedido.
- Commits cortos, en español, enfocados en el porqué. No incluir `data.db`, secretos ni artefactos grandes.
