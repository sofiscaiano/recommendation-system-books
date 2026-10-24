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
uv run python main.py submit --note "entrega con el tuneo guardado"
```

`evaluate --tune` escribe `artifacts/tuned_hyperparameters.json`. Las corridas
siguientes de `evaluate` y `submit` sin `--tune` entrenan con ese archivo. Si
no existe, usan los hiperparámetros fijos.

`--label-gain` elige la ganancia de LambdaRank: `linear` (defecto, identidad
0..10) o `exponential` (`2^rel-1`, igual que el NDCG de evaluación). No entra
a Optuna. Cambiarla exige buscar de nuevo, porque el archivo guardado se eligió
con la ganancia anterior:

```bash
uv run python main.py evaluate --tune --label-gain exponential --note "label_gain exponencial alineado con el NDCG"
```

`--note` es obligatorio. Ese texto queda como descripción de la corrida.

Cada ejecución abre una corrida en el experimento MLflow `quelibroleo`
(`sqlite:///mlflow.db`, o `MLFLOW_TRACKING_URI` si está definido) con un nombre
único (`fecha-comando-modo-sufijo`). El modo es `tune` si busca, `tuned` si
reutiliza el archivo guardado y `fixed` si usa los valores por defecto. Ese
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

1. **Retrieval**: popularidad, item-item por coseno, ALS, BPR y perfil TF-IDF.
   Las fuentes se combinan con Reciprocal Rank Fusion ponderado. Un corte de
   ese ranking entrega los candidatos al ranker (por defecto 200; entre 100 y
   1.000 si se busca).
2. **Ranking**: LightGBM LambdaRank ordena los candidatos con relevancia lineal
   0–10 y señales colaborativas, de contenido y secuenciales.

Retrieval se controla con Recall@M, recall ponderado por rating y el techo
oracle de NDCG@20. La evaluación final usa NDCG@20 macro, Recall@20, cobertura,
exposición long-tail y bootstrap pareado contra popularidad.

Los artefactos de evaluación se escriben en `artifacts/`; la entrega final se
genera en `submission.csv`. Cada corrida también deja el log en
`artifacts/run.log`. Con `--tune`, los hiperparámetros elegidos por Optuna
quedan en `artifacts/tuned_hyperparameters.json` y las entregas posteriores los
reutilizan. La definición de cada archivo
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
