# Guía de los diagnósticos

`uv run python main.py evaluate --note "..."` escribe estos archivos en [`artifacts/`](../artifacts/)
y los registra, junto con las métricas, en el experimento MLflow `quelibroleo`
(`mlflow.db`). Ese directorio, la base de MLflow y `mlartifacts/` no se versionan. Los números de esta ejecución corresponden al
holdout de las últimas 20 interacciones de 3.805 lectores, salvo donde se indica
que el diagnóstico usa un fold interno.

La relevancia de un libro de test es su rating, de 1 a 10. Un libro recomendado
que no está en esas 20 interacciones vale 0. Todos los promedios son la media
simple entre usuarios.

## `evaluation.json`

Resultado oficial del sistema completo contra el baseline de popularidad.

| Campo | Qué mide |
| --- | --- |
| `submit_ndcg_at_20` | NDCG@20 del modelo en los lectores de `ejemplo.csv` que entran al hold-out. Es el número a mirar: Kaggle evalúa a esa población, no al promedio de todos. |
| `submit_baseline_ndcg_at_20` | NDCG@20 de popularidad en esos mismos lectores. |
| `submit_users` | Cuántos lectores de la entrega tienen más de 20 interacciones y por eso entran al promedio. El resto no tiene últimas 20 que reservar. |
| `model_ndcg_at_20` | NDCG@20 del modelo sobre todos los lectores del hold-out. |
| `baseline_ndcg_at_20` | NDCG@20 de popularidad sobre todos los lectores del hold-out. |
| `retrieval_recall` | Proporción de los 20 libros de test que sobreviven dentro de los candidatos, antes de rankear. |
| `retrieval_oracle_ndcg_at_20` | NDCG@20 máximo posible con esos candidatos: ordena primero los libros de test recuperados, del rating más alto al más bajo. Es el techo del ranker. |
| `candidate_budget` | Corte del ranking ya fusionado: cuántos candidatos entran al ranker. Cada fuente se lee más profundo que este corte. |
| `evaluation_users` | Lectores incluidos en el holdout. |

Esas mismas claves se publican en MLflow. Los hiperparámetros de cada trial de
Optuna no se registran como métricas: la corrida hija solo grafica `objective`.

## `retrieval_folds.csv`

Diagnóstico del retrieval en los tres cortes internos. Estos cortes se toman
del historial anterior al holdout y no deben compararse directamente con
`evaluation.json`.

Cada fold reserva una ventana de 20 interacciones como objetivo y entrena solo
con lo anterior de ese lector. El fold 0 es el más antiguo y el fold 2 el más
cercano al holdout.

| Columna | Qué mide |
| --- | --- |
| `fold` | Índice del corte interno. |
| `recall` | Recall de los 20 objetivos dentro de los candidatos fusionados. |
| `weighted_recall` | Ese recall ponderado por rating. |
| `oracle_ndcg_at_20` | Techo de NDCG@20 disponible para el ranker en ese corte. |
| `mean_candidates` | Promedio de candidatos distintos por usuario. |

## `feature_ablations.csv`

Mide cuánto cambia el resultado al retirar una familia de señales. Hay dos
tipos de fila y ambos usan el último fold interno, no el holdout oficial.

| `stage` | `ablation` | Interpretación |
| --- | --- | --- |
| cualquiera | `full` | Referencia de ese bloque, en el mismo fold. Los deltas de las otras filas se restan contra esta. |
| `ranker` | `without_retrieval` | Reentrena LambdaRank sin scores, ranks ni señales de las fuentes. `ndcg_at_20` es el ranking resultante y `features` cuenta las variables conservadas. |
| `ranker` | `without_sequence` | Reentrena sin recencia, transiciones, coincidencia con el último autor o género, ni antigüedad de publicación. |
| `retrieval` | `without_<fuente>` | Quita esa fuente antes de fusionar candidatos. No reentrena el ranker: `recall` y `ndcg_at_20` describen el conjunto candidato y su techo oracle. |

`full` de retrieval es el conjunto con todas las fuentes. Es el mismo número que
el último fold de `retrieval_folds.csv`. `delta_ndcg_at_20` y `delta_recall` son
la fila menos esa referencia: negativo quiere decir que sacar la fuente empeoró
el conjunto. `full` de ranker reentrena con todas las features y los mismos
hiperparámetros fijos que las otras filas de ranker; no es el NDCG del hold-out.

`without_als` con delta de recall negativo indica que ALS aporta positivos que
las otras fuentes no recuperan. Una ablación no se compara con las otras filas
`without_*`: se compara con `full`.

## `feature_importance.csv`

Importancia por ganancia del LambdaRank entrenado para la evaluación. Una
variable con más ganancia participó en divisiones que mejoraron más el
objetivo. No indica causalidad ni permite comparar escalas entre variables muy
correlacionadas.

Los prefijos agrupan el origen:

- `score_`, `rank_`, `present_` y `reciprocal_`: salida de una fuente.
- `rrf_score` y `retrieval_rank`: resultado de la fusión.
- `recent_`, `transition_`, `same_last_` y `publication_age`: señales secuenciales.
- `user_` e `item_`: estadísticas históricas calculadas solo con train.

## `evaluation_recommendations.csv`

Top-20 producido para cada usuario del holdout, con `id_lector`, `id_libro`,
`score` y `rank`. El score de LambdaRank solo sirve para ordenar: no es un
rating entre 1 y 10. Este archivo no es la entrega; `submission.csv` se genera
después de reentrenar con todo el historial.

## `data_quality.json`

Auditoría de la carga. La fecha inválida queda fuera del entrenamiento y de la
evaluación, pero sí se usa para no volver a recomendar ese libro. Los usuarios
o libros sin fila en su catálogo se informan aquí; los libros ausentes no pueden
ser candidatos.

## `run.json`

Nombre único de la corrida MLflow (`run_name`). El mismo valor encabeza los JSON
y los CSV de diagnóstico de esa ejecución, y es el prefijo del study de
Optuna (`{run_name}__pipeline`). El CSV de entrega sigue siendo solo
`id_lector,id_libro`.

## Entregas archivadas

`submit` sigue escribiendo `submission.csv` en la raíz, que es el archivo para
subir. Además copia esa entrega a `submissions/{run_name}/submission.csv` y
deja `manifest.json` en la misma carpeta. Una corrida nueva crea otra carpeta;
no reemplaza las anteriores. `submission.csv.run.json` repite el manifiesto de
la última. La carpeta también queda como artefacto de esa corrida en MLflow.

| Campo | Qué identifica |
| --- | --- |
| `run_name` | Esta corrida de `submit`. Es el nombre de la carpeta. |
| `mlflow_run_id` | La corrida en MLflow, para abrir métricas y bajar la copia. |
| `git_commit` | SHA de `HEAD` al correr. |
| `git_dirty` | Si había cambios sin commitear. En ese caso el SHA no es el código exacto. |
| `hyperparameters_origin` | `tune` si este submit buscó, `saved` si reutilizó el JSON, `default` si usó la config fija. |
| `hyperparameters_run_name` | Corrida que eligió los hiperparámetros. El study de Optuna es `{ese nombre}__pipeline`. Vacío si el origen es `default`. |
| `hyperparameters` | Corte de RRF, pesos y LambdaRank con los que se entrenó esta entrega. |
| `note`, `seed` | Descripción obligatoria y semilla. |
| `csv_sha256` | Hash del CSV, para comprobar que una copia recuperada es la misma. |

## `config.json`

Parámetros con los que arrancó la ejecución: semilla, corte de RRF,
pesos y configuración de LightGBM. Son los valores de `PipelineConfig`.
Con `--tune`, Optuna no los pisa acá: el resultado de la búsqueda queda en
`tuned_hyperparameters.json`.

## `tuned_hyperparameters.json`

Solo aparece si la corrida usó `--tune`. Guarda el corte del ranking de RRF,
los pesos de cada fuente (normalizados para sumar 1), `label_gain` y los
hiperparámetros de LambdaRank elegidos por Optuna. `label_gain` no se busca:
queda el valor de esa corrida (`linear` si el archivo es anterior a este campo).
`evaluate` y `submit` sin `--tune` leen este archivo y entrenan con esos
valores. Si el `label_gain` de la corrida no coincide con el del archivo, el
log avisa que hay que volver a buscar. Si no existe, usan la config fija de
`PipelineConfig`. Un archivo generado con cuotas por fuente no sirve para esta
fusión. Otra corrida con `--tune` vuelve a buscar y reemplaza el archivo.

## `run.log`

Log de la corrida (retrieval, entrenamiento y, si aplica, trials de Optuna),
además de la salida en consola.

## Archivos que no son diagnósticos

`evaluated_pipeline.joblib` y `final_pipeline.joblib` guardan los modelos de la
evaluación y de la entrega.
