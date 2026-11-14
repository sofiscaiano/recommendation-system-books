# Quelibroleo

Recomendador de libros para los lectores de [Qué Libro Leo](https://www.quelibroleo.com/). Para cada `id_lector` de `ejemplo.csv` produce 20 `id_libro`, del más relevante al menos relevante.

La métrica es **NDCG@20** (`sklearn.metrics.ndcg_score`). La relevancia de un libro del hold-out es su `rating` (1–10); un libro recomendado que no está en ese hold-out vale 0. El número de referencia es el NDCG@20 sobre los lectores de `ejemplo.csv` que entran al hold-out.

## Datos

`data.db` y `ejemplo.csv` van en la raíz del repositorio. No están versionados y la base se abre solo para lectura.

| Tabla | Clave | Contenido |
| --- | --- | --- |
| `lectores` | `id_lector` | Nombre, género, `vive_en`, año de nacimiento |
| `libros` | `id_libro` | Título, autor, género, editorial, año, ISBN, resumen |
| `interacciones` | `id_lector` + `id_libro` | `fecha` y `rating` |

Una fecha inválida no entra a entrenamiento ni a evaluación, pero ese libro no se vuelve a recomendar. Un libro que no está en el catálogo no es candidato. Si dos interacciones tienen la misma fecha, el orden estable es por `id_libro`.

## Modelo

El modelo por defecto es `two_stage`:

1. **Retrieval.** Popularidad, coseno ítem-ítem, ALS, BPR, TF-IDF, two-tower y EASE. Las listas se fusionan con Reciprocal Rank Fusion ponderado (`rrf_k = 60`). El corte y los pesos salen de Optuna.
2. **Ranking.** LightGBM LambdaRank ordena esos candidatos y deja el top-20. Los libros ya leídos se excluyen.

El hold-out es leave-last-20: las últimas 20 interacciones de cada lector son test y el resto es train. Solo se evalúan lectores con más de 20 interacciones. Dentro del train hay 3 folds temporales de 20 interacciones. Optuna maximiza el NDCG@20 del último fold. Por defecto ese objetivo, y el early stopping, usan solo los lectores de `ejemplo.csv`; el ranker se entrena con todos.

Cada fuente (EASE, coseno, ALS, BPR, TF-IDF, two-tower) puede tunearse sola con `--model`. `two_stage` no vuelve a buscar esos hiperparámetros: lee el JSON de la fuente o, si no existe, los valores de `src/quelibroleo/config.py`.

## Cómo ejecutarlo

Python ≥ 3.13 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`--note` es obligatorio en los tres comandos. El texto queda como descripción de la corrida.

### 1. Tuning

Busca con Optuna (30 trials) el corte de la fusión, el peso de cada fuente y los hiperparámetros de LambdaRank. Agrega el resultado a `artifacts/tuning_history.json` y deja la entrada usada en `artifacts/tuned_hyperparameters.json`.

```bash
uv run python main.py evaluate --tune --note "búsqueda de hiperparámetros"
```

La primera vez que se prende una fuente, o si se cambia `label_gain`, hace falta este paso: un tuneo de otra config no se reutiliza. `label_gain` no entra a Optuna. El valor de `RankerConfig` es `exponential` (`2^rating - 1`). `linear` es la identidad 0–10.

```bash
uv run python main.py evaluate --tune --label-gain linear --note "tuneo con ganancia lineal"
```

Para tunear una fuente por separado:

```bash
uv run python main.py evaluate --model ease --tune --note "tuneo de EASE"
```

Eso escribe `artifacts/ease_hyperparameters.json` (el mismo patrón vale para `item_cosine`, `als`, `bpr`, `tfidf` y `two_tower`).

### 2. Evaluate

Reutiliza el tuneo cuya config coincide y mide el leave-last-20. No escribe la entrega. Imprime el reporte y lo guarda en `artifacts/evaluation.json`.

```bash
uv run python main.py evaluate --note "evaluación con el tuneo guardado"
```

Si no hay un tuneo de esa config, `two_stage` se detiene y hay que correr el paso 1.

### 3. Submit

Vuelve a entrenar con todo el historial, con los mismos hiperparámetros, y escribe `submission.csv`: las mismas columnas que `ejemplo.csv`, 20 filas por lector, en orden.

```bash
uv run python main.py submit --note "entrega"
```

`evaluate` y `submit` se pueden encadenar:

```bash
uv run python main.py all --note "evaluación y entrega"
```

## Salidas y opciones

Cada corrida deja el log en `artifacts/run.log` y una corrida de MLflow en `sqlite:///mlflow.db` (o en `MLFLOW_TRACKING_URI`). Para verlas:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`submit` además copia el CSV a `submissions/<run_name>/` junto con un `manifest.json`.

| Flag | Efecto |
| --- | --- |
| `--db`, `--example`, `--output`, `--artifacts` | Rutas. Por defecto `data.db`, `ejemplo.csv`, `submission.csv`, `artifacts/` |
| `--model` | `two_stage`, `popularity`, `item_cosine`, `als`, `bpr`, `tfidf`, `ease` o `two_tower` |
| `--seed` | Semilla. Por defecto `42` |
| `--tune-on-example-users` / `--no-tune-on-example-users` | A quién puntúa Optuna. Por defecto, solo `ejemplo.csv` |
| `--two-tower-source` / `--no-two-tower-source` | Prende o apaga esa fuente. La primera vez exige `--tune` |
| `--ease-source` / `--no-ease-source` | Igual para EASE |

Un baseline sin búsqueda:

```bash
uv run python main.py evaluate --model popularity --note "baseline de popularidad"
```
