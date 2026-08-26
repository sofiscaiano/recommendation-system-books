# Quelibroleo — sistema de recomendación

Recomendador de libros para usuarios de [Qué Libro Leo](https://www.quelibroleo.com/) a partir de interacciones históricas y de atributos de usuarios e ítems.

## Objetivo

Para cada `id_lector` que aparece en [`ejemplo.csv`](ejemplo.csv), generar **20 recomendaciones** de `id_libro` (ordenadas de más a menos relevantes).

La métrica a maximizar es **NDCG** (en la práctica, NDCG@20 sobre el ranking entregado). El problema es de **ranking**, no de predecir el rating con error mínimo.

## Datos

Tres tablas en `data.db` (SQLite):

| Tabla | Clave | Contenido |
| --- | --- | --- |
| `lectores` | `id_lector` | Perfil: nombre, género/sexo, `vive_en`, año de nacimiento |
| `libros` | `id_libro` | Catálogo: título, autor, género, editorial, año de edición, ISBN, resumen, imagen |
| `interacciones` | `id_lector` + `id_libro` + `fecha` | Historial: fecha y `rating` |

## Entrega

Formato igual que [`ejemplo.csv`](ejemplo.csv):

```text
id_lector,id_libro
```

- Una fila por posición del ranking.
- 20 libros por lector, en orden (la primera fila de cada usuario es el top-1).
- Salida actual de exploración: [`submission.csv`](submission.csv).

## Cómo trabajar

Python ≥ 3.13, dependencias con [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python main.py
```

Opciones: `--skip-eval`, `--eval-user-frac 0.1` (muestra aleatoria de lectores en eval, p. ej. 10%), `--eval-seed 42`, `--k 20`, `--batch-size 8`, `--output submission.csv`, `--eval-output eval_recs.csv` (top-k del hold-out, con `score_pred` y `ranking`).

Candidatos: por defecto solo libros con al menos una interacción (`Settings.only_interacted_items`) y, por lector, solo géneros u autores ya vistos (`filter_seen_genres` / `filter_seen_authors`; unión si ambos están activos). Flags en [`src/quelibroleo/config.py`](src/quelibroleo/config.py) y CLI: `--no-only-interacted-items`, `--no-filter-seen-genres`, `--no-filter-seen-authors`.

La pipeline está en [`src/quelibroleo/`](src/quelibroleo/) (`data`, `split`, `features`, `model`, `recommend`, `metrics`). [`test_inicial.ipynb`](test_inicial.ipynb) es exploración previa, no se usa para generar la entrega.

## Evaluación interna (propuesta)

Hasta tener el protocolo oficial cerrado, el hold-out interno es:

1. Ordenar interacciones por `id_lector` y `fecha`.
2. Reservar las **últimas 20** interacciones de cada lector como test.
3. Entrenar solo con el resto.
4. Reportar MSE/RMSE del rating en el hold-out y NDCG@20 del top-20 (`sklearn.metrics.ndcg_score`). El ranking de eval se escribe en [`eval_recs.csv`](eval_recs.csv) (mismo orden que la submission, más score).

## Alcance

**Dentro:** modelos sobre el catálogo y los lectores de `data.db`; features de usuario e ítem; ranking top-20; submission en el formato de ejemplo.
