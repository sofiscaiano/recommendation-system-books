# Especificación — Quelibroleo

Recomendador de ranking de libros para usuarios de Qué Libro Leo. El problema es ordenar, no minimizar el error de rating.

Detalle de métodos: [`docs/metodologias.md`](docs/metodologias.md). Diagnósticos y artefactos: [`docs/evaluacion.md`](docs/evaluacion.md). Uso: [`README.md`](README.md).

## Objetivo

Para cada `id_lector` de [`ejemplo.csv`](ejemplo.csv), producir **20** `id_libro` ordenados de más a menos relevante.

La métrica a maximizar es **NDCG@20**, calculada con `sklearn.metrics.ndcg_score`. La relevancia de un libro del hold-out es su `rating` (1–10). Un libro recomendado que no está en esas interacciones vale 0.

El número de referencia es el NDCG@20 del modelo sobre los lectores de `ejemplo.csv` que entran al hold-out (`submit_ndcg_at_20`). Ese es el grupo que evalúa la entrega, no el promedio de todos los lectores.

## Datos

Fuente única: `data.db` (SQLite), abierta siempre en solo lectura. No se modifica.

| Tabla | Clave | Uso |
| --- | --- | --- |
| `lectores` | `id_lector` | Perfil: nombre, género/sexo, `vive_en`, año de nacimiento |
| `libros` | `id_libro` | Catálogo: título, autor, género, editorial, año de edición, ISBN, resumen, imagen |
| `interacciones` | `id_lector` + `id_libro` | Historial: `fecha` y `rating` |

Reglas de carga:

- Una fecha inválida no entra a entrenamiento ni a evaluación. Sí se usa para no volver a recomendar ese libro.
- Un libro ausente del catálogo no puede ser candidato.
- Los empates de fecha se resuelven de forma estable por `id_libro`. La base no tiene hora.

## Entrega

CSV con las mismas columnas que `ejemplo.csv`:

```text
id_lector,id_libro
```

- 20 filas por lector, en orden. La primera fila de cada usuario es el top-1.
- Los mismos usuarios que `ejemplo.csv`.
- Archivo de subida: `submission.csv` en la raíz.
- Cada `submit` archiva una copia en `submissions/{run_name}/` (`submission.csv` + `manifest.json`) y no pisa entregas anteriores.
- Los libros ya leídos se excluyen del ranking.

`submit` reentrena con todo el historial. El CSV del hold-out (`evaluation_recommendations.csv`) no es la entrega.

## Evaluación

Hold-out interno (leave-last-20), hasta que exista un protocolo oficial distinto:

1. Ordenar interacciones por `id_lector` y `fecha`.
2. Reservar las últimas 20 interacciones de cada lector como test.
3. Entrenar solo con el resto.

Solo entran al promedio los lectores con más de 20 interacciones.

Dentro del train hay folds temporales rolling (3 ventanas de 20 interacciones). Sirven para ajustar retrieval y LambdaRank. El hold-out final no entra a esa búsqueda. El objetivo de Optuna es el NDCG@20 del top-20 en el último fold.

Retrieval se juzga por el recall de los 20 libros de test dentro de los candidatos y por el techo oracle de NDCG@20. El baseline de comparación es popularidad, sobre los mismos grupos de lectores.

## Pipeline

Dos etapas. El entrenamiento usa solo el historial anterior al corte temporal.

### Retrieval

Fuentes, fusionadas con Reciprocal Rank Fusion ponderado (`k = 60`, fijo):

| Fuente | Señal |
| --- | --- |
| Popularidad | Conteo de interacciones en train; el mismo orden para todos, saltando lo ya leído |
| Item-item (coseno de ocurrencia) | Perfil del lector por producto interno, con peso `rating/10` y recencia |
| ALS | Factorización implícita de la confianza `(1 + rating/10) × recencia` |
| BPR | Mismo matriz; optimiza orden, no el valor de la confianza |
| TF-IDF | Coseno entre el perfil de texto del lector y cada libro |

No hay pesos ni corte fijos: salen de un `evaluate --tune` con la misma config y quedan en `artifacts/tuning_history.json`. Sin una entrada que coincida, `two_stage` pide correr `--tune` y no arranca.

No hay cupos por fuente. El corte (`rrf_cutoff`) se aplica al ranking ya fusionado. Cada fuente se lee más profundo que ese corte. La recencia usa semivida de 365 días.

### Ranking

LightGBM LambdaRank ordena los candidatos. Cada lector es un grupo. La etiqueta es el rating futuro si el candidato está en el objetivo del corte, y 0 si no.

`label_gain` no entra a Optuna:

- `linear` (defecto): identidad 0..10.
- `exponential`: `2^rating - 1`, la misma ganancia que `ndcg_score`.

Cambiar `label_gain` exige un `evaluate --tune` nuevo: el archivo guardado se eligió con la ganancia anterior. `lambdarank_truncation_level` y `eval_at` quedan en 20. El desempate del score es estable por `id_libro`.

Las columnas de `two_stage` (`two_stage/columns.py`) son numéricas: scores y rangos de cada fuente, estadísticas de lector e ítem, afinidad de autor y género, y señales de secuencia. Autor y género no entran como categóricas de alta cardinalidad; la editorial queda en el texto de TF-IDF. Un modelo de una sola etapa no usa esas columnas.

## Hiperparámetros

| Modo | Cuándo | Qué entrena |
| --- | --- | --- |
| `fixed` | `ease` sin su JSON, o `two_tower` | Valores fijos de ese modelo. `two_stage` no entra acá: sin un tuneo de esa config no arranca |
| `tuned` | Hay un tuneo de esa config y no se pasa `--tune` | Esa entrada del historial |
| `tune` | `evaluate --tune` o `submit --tune` | Un study de Optuna (`optuna_trials`, defecto 30) que agrega una entrada al historial de ese modelo |

`two_stage` agrega cada búsqueda a `artifacts/tuning_history.json` y copia la última en `artifacts/tuned_hyperparameters.json`. Sin `--tune` usa la entrada cuya config coincide, no la última si esa config es otra. Optuna recorre juntos el corte de RRF (100–1.000, log), un peso por fuente (0,1–3, log) y el bloque de LambdaRank en `RankerSearch`. El objetivo es el NDCG@20 de todos los lectores del último fold. Con `tune_on_example_users` o `--tune-on-example-users`, ese objetivo y el early stopping miran solo los de `ejemplo.csv`; el entrenamiento del ranker no cambia. Quedan fijos `n_estimators`, `early_stopping_rounds`, `rrf_k` y `label_gain`. Un trial se descarta si `num_leaves` supera `2^max_depth`, o si `min_child_samples * num_leaves` supera las filas de entrenamiento. Un JSON generado con cuotas por fuente no se reutiliza.

`ease` guarda `artifacts/ease_hyperparameters.json` y no lee ni escribe el archivo de `two_stage`. Optuna recorre `reg_lambda` (1–100.000, log), `min_interactions` (10–100, log) y el peso de la matriz (`binary`, `rating` o `confidence`). El objetivo es el mismo NDCG@20 del último fold.

`two_tower` no tiene study ni archivo: usa `TwoTowerConfig`. La torre de lector concatena el promedio de los embeddings de toda su historia (Covington et al., 2016), el de sus últimas `recent_reads` lecturas (defecto 20) y, para cada uno, el promedio de los autores de esos libros (`user_authors`). La de libro usa id, autor y género, y comparte con la de lector las tablas de id y de autor. La pérdida es softmax con negativos del batch y corrección logQ (Yi et al., 2019). Cada ejemplo es la historia anterior a un corte y una de las 20 lecturas siguientes, ponderada por su rating. Solo los libros con al menos `min_item_reads` lecturas son candidatos. La cantidad de épocas sale del NDCG@20 del último fold (patience 3); después se reentrena de cero con toda la historia recibida durante esas épocas.

Con `--two-tower-source`, `two_stage` suma el two-tower como sexta fuente de retrieval. En cada fold entrena con la historia de ese fold, con la misma validación interna y el mismo reentrenamiento, una sola vez antes del study. Optuna agrega `weight__two_tower` y LambdaRank recibe `score_two_tower`, `rank_two_tower` y `present_two_tower`. El two-tower corre en un proceso hijo porque torch y LightGBM cargan copias distintas de OpenMP y, en macOS, juntos en un proceso se cuelgan o abortan (LightGBM #6595). Si una fuente activa no tiene peso guardado, la corrida falla en vez de ignorarla: la primera vez hay que usar `--tune`.

## Interfaz

Python ≥ 3.13. Dependencias con `uv sync`. Código de producción: `src/quelibroleo/` y `main.py`.

```bash
uv run python main.py evaluate --model two_stage --note "qué se está probando"
uv run python main.py evaluate --model ease --tune --note "EASE contra el mismo hold-out"
uv run python main.py evaluate --model two_tower --note "two-tower contra el mismo hold-out"
uv run python main.py submit --note "qué se está entregando"
uv run python main.py evaluate --tune --label-gain exponential --note "..."
uv run python main.py evaluate --tune --label-gain exponential --two-tower-source --note "..."
uv run python main.py export-serving
```

`PipelineConfig.model` elige el algoritmo (`two_stage`, `popularity`, `ease` o `two_tower`). `--model` lo pisa en esa corrida. `PipelineConfig.random_seed` es la semilla; `--seed` la pisa. Train, folds de
validación y test salen siempre de `split_data.py`. La entrega sigue siendo
`submission.csv` con los lectores de `ejemplo.csv`. Un modelo nuevo solo
implementa `fit` y `recommend`; no tiene que separar retrieval y ranking.
`two_stage` sí lo hace, y sus columnas numéricas están en `two_stage/columns.py`.
`ease` es un único modelo ítem-ítem. `two_tower` es de una etapa: un vector por
lector y otro por libro, y el ranking es su producto.

`--note` es obligatorio en `evaluate`, `submit` y `all`. Sin subcomando, el CLI evalúa.

Cada corrida abre un run en el experimento MLflow `quelibroleo` (`sqlite:///mlflow.db`, o `MLFLOW_TRACKING_URI`). El nombre (`fecha-modelo-comando-modo-sufijo`) queda en el tag `run_name` y el algoritmo en el tag `model`. En `artifacts/run.json` y en los diagnósticos también queda el nombre. Con `--tune`, el study de `two_stage` corre en una hija `{run_name}__pipeline` y el de `ease` en `{run_name}__ease`.

`export-serving` ajusta las fuentes sobre toda la historia y escribe `artifacts/serving.joblib`. La API (`serving/app.py`) carga ese artefacto una vez y, por lector, solo rankea. Las valoraciones de la demo viven en el navegador y no se escriben en `data.db`.

## Fuera de alcance de un cambio habitual

- No modificar `data.db`.
- No modificar `test_inicial.ipynb` salvo pedido explícito.
- No versionar `data.db`, secretos, `artifacts/`, `mlflow.db`, `mlartifacts/` ni otros artefactos grandes.
- No pushear a remoto salvo pedido.

Pruebas:

```bash
uv run pytest
uv run ruff check src tests main.py serving/app.py
```
