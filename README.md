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
| `interacciones` | `id_lector` + `id_libro` | Historial: fecha y `rating` |

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
uv run python main.py evaluate --note "qué se está probando"
uv run python main.py submit --note "qué se está entregando"
```

La búsqueda completa de hiperparámetros es opcional y considerablemente más
costosa:

```bash
uv run python main.py evaluate --tune --note "búsqueda de hiperparámetros"
```

`--note` es obligatorio. Ese texto queda como descripción de la corrida.

Cada ejecución abre una corrida en el experimento MLflow `quelibroleo`
(`sqlite:///mlflow.db`, o `MLFLOW_TRACKING_URI` si está definido) con un nombre
único (`fecha-comando-modo-sufijo`). Ese nombre queda en el tag `run_name`, en
`artifacts/run.json` y en los JSON y CSV de diagnóstico. `submission.csv` no
cambia de columnas: el vínculo está en `submission.csv.run.json`. Con `--tune`, el study de Optuna abre una corrida hija `{run_name}__pipeline`
con la serie de trials en el step.
Para comparar:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Evaluación interna

Hasta tener el protocolo oficial cerrado, el hold-out interno es:

1. Ordenar interacciones por `id_lector` y `fecha`.
2. Reservar las **últimas 20** interacciones de cada lector como test.
3. Entrenar solo con el resto.

Solo se evalúan lectores con más de 20 interacciones. Los empates de fecha se
resuelven de forma estable por `id_libro`; la base no contiene hora. Dentro del
train se crean folds temporales rolling para ajustar retrieval y LambdaRank sin
usar el hold-out final.

## Arquitectura

El detalle de cada método, pesos y librerías está en
[`docs/metodologias.md`](docs/metodologias.md). El sistema tiene dos etapas:

1. **Retrieval**: popularidad, dos variantes item-item coseno, SVD explícito,
   ALS/BPR ponderados y perfil TF-IDF. Las fuentes se combinan con cuotas y
   Reciprocal Rank Fusion para producir entre 100 y 1.000 candidatos.
2. **Ranking**: LightGBM LambdaRank ordena los candidatos con relevancia lineal
   0–10 y señales colaborativas, de contenido y secuenciales.

Retrieval se controla con Recall@M, recall ponderado por rating y el techo
oracle de NDCG@20. La evaluación final usa NDCG@20 macro, Recall@20, cobertura,
exposición long-tail y bootstrap pareado contra popularidad.

Los artefactos de evaluación se escriben en `artifacts/`; la entrega final se
genera en `submission.csv`. Cada corrida también deja el log en
`artifacts/run.log`. Con `--tune`, los hiperparámetros elegidos por Optuna
quedan en `artifacts/tuned_hyperparameters.json`. La definición de cada archivo
y métrica está en [`docs/evaluacion.md`](docs/evaluacion.md). `data.db` siempre
se abre en modo de solo lectura.

## Pruebas

```bash
uv run pytest
uv run ruff check src tests main.py
```
