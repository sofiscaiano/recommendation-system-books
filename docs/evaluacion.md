# Guía de los diagnósticos

`uv run python main.py evaluate` escribe estos archivos en [`artifacts/`](../artifacts/)
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
| `baseline_ndcg_at_20` | NDCG@20 del ranking global por cantidad de interacciones, excluyendo libros ya leídos. |
| `model_ndcg_at_20` | NDCG@20 del top-20 final de LambdaRank. Es la métrica principal. |
| `model_recall_at_20` | Proporción de los 20 libros de test que aparecen en el top-20 final. Un 0,136 equivale a unos 2,7 aciertos por usuario. |
| `retrieval_recall` | Proporción de los 20 libros de test que sobreviven dentro de los candidatos, antes de rankear. Con M = 200, un 0,307 equivale a unos 6,1 aciertos disponibles por usuario. |
| `retrieval_weighted_recall` | Igual que el recall, pero cada libro pesa según su rating. Mide si se recuperan los libros mejor calificados y no solo cualquier interacción futura. |
| `retrieval_oracle_ndcg_at_20` | NDCG@20 máximo posible con esos candidatos: ordena primero los libros de test recuperados, del rating más alto al más bajo. Es el techo del ranker. |
| `catalog_coverage_at_20` | Cantidad de libros distintos que aparecen en algún top-20, dividida por los libros elegibles del entrenamiento. |
| `long_tail_exposure_at_20` | Proporción de posiciones del top-20 ocupadas por libros de cola larga. La cola incluye los libros cuyo conteo está en el 80 % inferior de popularidad. |
| `paired_delta` | Diferencia de NDCG@20 entre el modelo y la popularidad para el mismo usuario. `mean` es el promedio; `lower` y `upper` son el intervalo bootstrap del 95 % con 2.000 remuestreos. |
| `segments` | Métricas por grupo de usuarios. Cada grupo incluye NDCG@20, Recall@20 del ranking, recall de retrieval, cobertura y exposición long-tail. |
| `candidate_budget` | Cantidad máxima de candidatos entregada al ranker. |
| `evaluation_users` | Lectores incluidos en el holdout. |

Los segmentos `activity_q1` a `activity_q4` dividen a los usuarios evaluados en
cuatro grupos de igual tamaño según cuántas interacciones tienen en train. `q1`
es el grupo con menos historial y `q4` el de mayor actividad.
`single_date_history` reúne a quienes tienen todas sus interacciones de train
en una sola fecha; para ellos el orden temporal no es informativo.

La cobertura por segmento usa el mismo catálogo elegible que la métrica global:
cuenta cuántos libros distintos aparecen en el top-20 de ese grupo. Como cada
cuartil tiene el mismo número de usuarios, los valores son comparables entre sí.

## `evaluation_segments.csv`

Misma información que `segments` en `evaluation.json`, una fila por grupo.

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
| `ranker` | `without_retrieval` | Reentrena LambdaRank sin scores, ranks ni señales de las fuentes. `ndcg_at_20` es el ranking resultante y `features` cuenta las variables conservadas. |
| `ranker` | `without_sequence` | Reentrena sin recencia, transiciones, coincidencia con el último autor o género, ni antigüedad de publicación. |
| `retrieval` | `without_<fuente>` | Quita esa fuente antes de fusionar candidatos. No reentrena el ranker: `recall` y `ndcg_at_20` describen el conjunto candidato y su techo oracle. |

`without_als` con recall menor que las demás filas indica que ALS aporta
positivos que las otras fuentes no recuperan. Una ablación con métrica mayor no
significa que la fuente perjudique el ranking final; solo indica que, en ese
fold y con las cuotas fijas, la fusión sin ella dejó un conjunto candidato con
mejor cobertura.

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

## `config.json`

Parámetros con los que arrancó la ejecución: semilla, presupuesto de candidatos,
cuotas, pesos y configuración de LightGBM. Son los valores de `PipelineConfig`.
Con `--tune`, Optuna no los pisa acá: el resultado de la búsqueda queda en
`tuned_hyperparameters.json`.

## `tuned_hyperparameters.json`

Solo aparece si la corrida usó `--tune`. Guarda el presupuesto de candidatos,
los pesos y las cuotas de fusión, y los hiperparámetros de LambdaRank elegidos
por Optuna. `submit` no relee este archivo: vuelve a usar la config fija, salvo
que también se le pase `--tune`.

## `run.log`

Log de la corrida (retrieval, entrenamiento y, si aplica, trials de Optuna),
además de la salida en consola.

## Archivos que no son diagnósticos

`artifacts/cache/` guarda candidatos intermedios para no repetir el retrieval.
`evaluated_pipeline.joblib` y `final_pipeline.joblib` guardan los modelos de la
evaluación y de la entrega.
