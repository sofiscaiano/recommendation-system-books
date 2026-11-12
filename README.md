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

El algoritmo es `PipelineConfig.model`. `--model` lo pisa en esa corrida. El corte no: train, los tres folds de
validación y el test de las últimas 20 salen de `split_data.py` para
cualquier modelo. `submit` reentrena con toda la historia y escribe el mismo
`submission.csv`.

```bash
uv run python main.py evaluate --model two_stage --note "pipeline actual"
uv run python main.py evaluate --model popularity --note "baseline"
uv run python main.py evaluate --model ease --note "EASE como modelo único"
```

Un modelo nuevo implementa `fit(split)` y `recommend(users)` y se registra
en `build_model` (`src/quelibroleo/models.py`). Puede ser un solo modelo:
no tiene que buscar candidatos y después ordenarlos. `split.history` es el
train, `split.test` el hold-out y `validation_folds` los cortes internos;
el último fold es la validación. `two_stage` separa retrieval y ranking.
`ease` puntúa el catálogo con una matriz ítem-ítem y devuelve el top-20.
La tabla numérica con la que LambdaRank ordena está en
`two_stage/columns.py` y no la usa un modelo único.

La búsqueda completa de hiperparámetros es opcional y considerablemente más
costosa:

```bash
uv run python main.py evaluate --tune --note "búsqueda de hiperparámetros"
uv run python main.py submit --note "entrega con el tuneo guardado"
```

`--tune-on-example-users` hace que ese study y el early stopping del último fold puntúen solo los lectores de `ejemplo.csv`. El ranker sigue entrenando con todos. Sin el flag se usa `PipelineConfig.tune_on_example_users`, prendido por defecto. Otro `--artifacts` evita pisar el JSON del tuneo sobre el fold completo.

`evaluate --tune` de `two_stage` agrega una entrada a
`artifacts/tuning_history.json` y copia la última en
`artifacts/tuned_hyperparameters.json`. Sin `--tune` reutiliza la entrada
cuya config coincide (fuentes, `label_gain`, semilla, split, `rrf_k` y el
resto de lo que el study no elige, incluidos los hiperparámetros ya resueltos
de cada fuente). Si no hay ninguna, hay que correr
`evaluate --tune`. Cada fuente con hiperparámetros propios escribe el suyo
(`ease_hyperparameters.json`, `item_cosine_hyperparameters.json`,
`als_hyperparameters.json`, `bpr_hyperparameters.json`,
`tfidf_hyperparameters.json`, `two_tower_hyperparameters.json`) y, si no tiene
archivo, usa los valores de su config. `two_stage` no pisa esos JSON. Si la
fuente está prendida, los lee para armar la lista.

`RankerConfig.label_gain` elige la ganancia de LambdaRank: `linear` (defecto,
identidad 0..10) o `exponential` (`2^rel-1`, igual que el NDCG de evaluación).
No es un sí/no y no entra a Optuna. La corrida usa ese campo de `config.py`.
`--label-gain` solo lo pisa si se pasa. Cambiarlo exige buscar de nuevo: el
tuneo guardado se eligió con la ganancia anterior, y otra config no lo reutiliza:

```bash
uv run python main.py evaluate --tune --note "tuneo con el label_gain de config.py"
```

`--note` es obligatorio. Ese texto queda como descripción de la corrida.

Cada ejecución abre una corrida en el experimento MLflow `quelibroleo`
(`sqlite:///mlflow.db`, o `MLFLOW_TRACKING_URI` si está definido) con un nombre
único (`fecha-modelo-comando-modo-sufijo`). El tag `model` es el algoritmo
(`two_stage`, `popularity`, `ease`, `item_cosine`, `als`, `bpr`, `tfidf` o `two_tower`). El modo es `tune` si busca, `tuned` si
reutiliza un tuneo de la misma config y `fixed` si ese modelo usa valores fijos
(una fuente sin su JSON). `two_stage` sin un tuneo de esa config no arranca. Ese
nombre queda en el tag `run_name`, en
`artifacts/run.json` y en los JSON y CSV de diagnóstico. `submission.csv` no
cambia de columnas. Cada `submit` copia esa entrega a `submissions/{run_name}/`
sin pisar las anteriores: ahí quedan el CSV y un `manifest.json` con el commit,
si el árbol tenía cambios sin commitear, el id de la corrida que tuneó y los
hiperparámetros usados. `submission.csv.run.json` repite el manifiesto de la
última. Esos archivos también se adjuntan a la corrida en MLflow. Con `--tune`,
el study de Optuna abre una corrida hija `{run_name}__pipeline` con la serie de
trials en el step.
Para comparar:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

El diagnóstico del hold-out no reentrena. Después de un `evaluate`:

```bash
uv run python diagnostico.py
```

Lee el top-20 de `artifacts/evaluation_recommendations.csv`. Si esa corrida fue
`two_stage` y dejó `artifacts/evaluation_retrieval_hits.csv` del mismo
`run_name`, separa la pérdida de NDCG en retrieval y ranking. Parte el hold-out
entre los lectores de `ejemplo.csv` y el resto, y dentro de cada grupo por
cuartil de historial y por si el test repite autor o género. Sin el CSV de hits
solo informa el NDCG y cuántos libros del test quedaron en el top-20.

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

1. **Retrieval**: popularidad, item-item por coseno, ALS, BPR, perfil TF-IDF, two-tower y EASE.
   Las fuentes se combinan con Reciprocal Rank Fusion ponderado. El corte y
   los pesos son los del último `--tune` (Optuna recorre el corte entre 100 y
   1.000).
2. **Ranking**: LightGBM LambdaRank ordena los candidatos con relevancia lineal
   0–10 y señales colaborativas, de contenido y secuenciales.

Retrieval se controla con el recall de candidatos y el techo oracle de
NDCG@20. La evaluación final publica el NDCG@20 de los lectores de la entrega,
el del hold-out completo y el de popularidad en esos mismos grupos.

Los artefactos de evaluación se escriben en `artifacts/`; la entrega final se
genera en `submission.csv`. Cada corrida también deja el log en
`artifacts/run.log`. Con `--tune`, los hiperparámetros elegidos por Optuna
quedan en `artifacts/tuning_history.json`. Una entrega posterior reutiliza
la entrada de esa misma config, no necesariamente la última búsqueda. La definición de cada archivo
y métrica está en [`docs/evaluacion.md`](docs/evaluacion.md). `data.db` siempre
se abre en modo de solo lectura.

## Prototipo en vivo

La página vive en [`web/`](web/) y se publica en Cloudflare Pages. La API corre
en esta máquina: carga una vez `artifacts/serving.joblib` y, por cada lector,
solo rankea. `evaluate` y `submit` no cambian. Las valoraciones de la demo
quedan en el navegador y no se escriben en `data.db`.

Después de un `submit`:

```bash
uv run python main.py export-serving
```

Eso ajusta las fuentes sobre toda la historia y escribe
`artifacts/serving.joblib`. Tarda cerca de un minuto y no se repite en cada
visita.

Para probar la API en local, un solo worker:

```bash
uv sync --extra serve
uv run uvicorn serving.app:app --host 127.0.0.1 --port 8000 --workers 1
```

`GET /health` responde `{"ready": true}` cuando el artefacto terminó de cargar.
La página, en otra terminal:

```bash
cd web
npm install
npm run dev
```

Sin `VITE_API_URL`, el front llama a `http://localhost:8000`.

En la demo, Docker deja el modelo prendido y el túnel lo publica en HTTPS.
`serving/.env` (a partir de `serving/.env.example`) lleva `TUNNEL_TOKEN` y
`CORS_ORIGINS`. El puerto de la API queda en `127.0.0.1`; a internet sale solo
el túnel. Docker Desktop necesita unos 6 GB para el VM.

```bash
caffeinate -dims docker compose -f serving/docker-compose.yml up --build api
caffeinate -dims docker compose -f serving/docker-compose.yml --profile tunnel up --build
```

En Cloudflare, el hostname público del túnel apunta a `http://api:8000`.
Pages usa la raíz `web`, el comando `npm run build` y la salida `dist`.
`VITE_API_URL` es la URL del túnel. `web/public/_redirects` deja que
`/recomendar/...` recargue directo. Con la tapa abierta el proceso sigue en
memoria; al bajar el compose la página queda publicada y la API no.

## Pruebas

```bash
uv run pytest
uv run ruff check src tests main.py serving/app.py
```
