# Metodologías de retrieval y ranking

El recomendador es un pipeline de dos etapas. **Retrieval** arma un ranking
fusionado y se queda con un prefijo (por defecto 200 libros; la búsqueda mueve
ese corte entre 100 y 1.000). **Ranking** los ordena y deja el top-20. Los
libros ya leídos se excluyen en ambas etapas.

El entrenamiento usa solo el historial anterior al corte temporal. La
evaluación reserva las últimas 20 interacciones de cada lector (ver
[`evaluacion.md`](evaluacion.md)).

## Retrieval

Cada fuente devuelve un ranking propio. Después se fusionan.

### Popularidad

`PopularityRecommender` (`src/quelibroleo/baselines.py`).

Ordena el catálogo elegible por cantidad de interacciones en el train, con
desempate estable por `id_libro`. El mismo ranking se aplica a todos los
lectores, saltando los libros que cada uno ya leyó. Sirve de baseline y de
red de seguridad cuando las fuentes personalizadas no cubren al usuario.

Librerías: pandas.

### Item-item por coseno

`ItemCosineRetriever` (`src/quelibroleo/two_stage/retrieval.py`), fuente `item_cosine`.

Construye una matriz dispersa ítem × usuario y normaliza cada fila a norma L2.
El score de un candidato es el producto interno entre el perfil del lector y el
vector del ítem. No se materializa la matriz cuadrada ítem × ítem: el score se
calcula como `(perfil · vectores_ítem) · vectores_ítemᵀ`. El peso de cada
lectura es `rating / 10` por un factor de recencia.

La recencia es una semivida exponencial: si `half_life_days` es 365, una
lectura de un año antes del último evento del lector pesa la mitad
(`2^(-edad_en_días / semivida)`).

Un modo `adjusted` (rating centrado en la media del lector, sin recencia) se
probó y quedó afuera: en la ablación del último fold, quitarlo dejaba el recall
y el oracle del conjunto candidato en el mismo valor.

Librerías: NumPy, SciPy (`scipy.sparse.csr_matrix`), scikit-learn
(`sklearn.preprocessing.normalize`), pandas.

### Factorización implícita: ALS y BPR

`ImplicitMFRetriever`.

Trata cada lectura como una señal implícita con confianza

`(1 + rating / 10) × recencia`

(la misma semivida que el coseno de ocurrencia). Dos algoritmos sobre esa
matriz usuario × ítem:

- **ALS** (`implicit.als.AlternatingLeastSquares`): mínimos cuadrados
  alternados. Ajusta factores de usuario e ítem para reconstruir la confianza.
- **BPR** (`implicit.bpr.BayesianPersonalizedRanking`): optimiza que un ítem
  observado quede por encima de negativos muestreados. El objetivo es el orden,
  no el valor de la confianza.

Por defecto, 64 factores y 20 iteraciones. En inferencia se usa
`model.recommend` con `filter_already_liked_items=True`.

Librería: [implicit](https://github.com/benfred/implicit) ≥ 0.7.3, SciPy sparse,
NumPy, pandas.

### Perfil TF-IDF

`TfidfRetriever`.

Vectoriza el texto de cada libro (título, autor, género, editorial, resumen y,
como respaldo, el `id_libro`) con unigramas y bigramas, `min_df=2`,
`max_df=0.98`, hasta 60.000 términos y TF sublineal. El perfil del lector es
la suma de los vectores de sus libros, ponderada por `rating / 10` y recencia,
normalizada a L2. El score es el coseno entre ese perfil y cada libro.

Librería: scikit-learn (`TfidfVectorizer`), SciPy sparse, NumPy, pandas.

### Fusión: Reciprocal Rank Fusion y un corte

`CandidateFusion`.

No hay cupos por fuente. El ranker de la etapa siguiente ya puede bajar un
candidato débil; un cupo mínimo lo metería igual, desplazando a otro con mejor
evidencia, y pelearía con el peso por la misma decisión.

1. Cada fuente aporta un ranking más largo que el corte final. No se emite el
   catálogo entero (hay unos 48.000 libros con alguna interacción): un puesto
   tan atrás no puede entrar al top mergeado. La profundidad es
   `(C + k) * n_fuentes - k`, con pesos parejos. Así, los C primeros de una
   sola fuente le ganan a un libro que en todas las fuentes aparece recién
   después de ese puesto. Con 5 fuentes y `k = 60`, un corte de 200 lee 1.240
   por fuente; si la búsqueda lleva el corte a 1.000, lee 5.240. Durante el
   study se lee una sola vez, a la profundidad del corte más grande, y cada
   trial corta el ranking ya fusionado.
2. Esos rankings se combinan con Reciprocal Rank Fusion ponderado (Cormack,
   Clarke y Buettcher, 2009):

   `score(u, i) = Σ_s  w_s / (k + rango_s(u, i))`

   con `k = 60` fijo. `k` no se busca junto con los pesos: los dos cambian
   cuánto pesa el tope de cada lista respecto del fondo, y la búsqueda no
   podría separarlos. Un libro que no aparece en una fuente no suma ese
   término. Los pesos se normalizan para sumar 1. Escalarlos a todos por una
   constante no cambia el orden, y un árbol tampoco parte distinto `rrf_score`
   si solo cambia la escala. Los pesos por defecto, antes de normalizar, son
   0,5 (popularidad), 1,0 (coseno, ALS, BPR) y 0,8 (TF-IDF).
3. El corte `rrf_cutoff` se aplica solo después del merge: se queda con los
   primeros C libros de ese ranking (`rrf_score` descendente, desempate por
   `id_libro`). No recorta las listas de cada fuente. Un corte mayor es un
   prefijo más largo del mismo ranking: solo agrega candidatos. El defecto es
   200. Con `--tune`, Optuna recorre C como entero en escala logarítmica entre
   100 y 1.000 (`RetrievalSearch.rrf_cutoff`) y un peso por fuente
   (`RetrievalSearch.source_weight`, 0,1–3, log). La escala logarítmica sigue
   el rendimiento decreciente de agrandar el conjunto.

El orden de la fusión (`retrieval_rank`, `rrf_score`) no es el ranking final:
solo define quién entra al ranker. Los scores y rangos de cada fuente se
conservan como features.

Librerías: NumPy y pandas. No hay una librería externa de RRF.

## Ranking

### LambdaRank (LightGBM)

`LambdaRanker` (`src/quelibroleo/two_stage/ranker.py`) envuelve `lightgbm.LGBMRanker`.

LambdaRank entrena árboles de boosting para optimizar NDCG, no el error de
rating. Cada lector es un grupo (`query_id`). Dentro del grupo, la etiqueta
`relevance` es el rating futuro (1–10) si el candidato está en el hold-out, y
0 si no. `label_gain` elige cuánto aporta cada rating y no entra a Optuna.
`linear` (el defecto, y el de los tuneos ya guardados) es la identidad `0..10`.
`exponential` es `2^rating - 1`, la misma ganancia que
`sklearn.metrics.ndcg_score`. Se pasa con `--label-gain`. Cambiarlo deja
desalineados los hiperparámetros guardados, así que la corrida correspondiente
es `evaluate --tune --label-gain exponential`. `lambdarank_truncation_level` y
`eval_at` están en 20, alineados con la métrica de entrega.

Parámetros por defecto: 200 árboles, learning rate 0,04, 31 hojas, profundidad
libre, `min_child_samples=30`, subsample y colsample 0,9, `reg_lambda=1`. Si
hay validación temporal, early stopping a 30 rondas sobre NDCG.

El desempate del score predicho es estable por `id_libro`.

Librerías: LightGBM ≥ 4.7, NumPy, pandas. NDCG de evaluación:
`sklearn.metrics.ndcg_score` en `src/quelibroleo/metrics.py`.

La búsqueda opcional (`main.py evaluate --tune`) es un solo study de Optuna,
`PipelineConfig.optuna_trials` trials. Recorre juntos el corte de RRF, un peso
por fuente y `RankerSearch`. El objetivo es el NDCG@20 del top-20
de LambdaRank en el último fold rolling, entrenando con los folds anteriores.
Ese fold también corta el early stopping. El hold-out oficial no entra al
study. `n_estimators`, `early_stopping_rounds` y `rrf_k` quedan fijos. La
profundidad de cada fuente sale del corte, de `k` y de la cantidad de fuentes;
no es otro hiperparámetro. Un trial se descarta si `num_leaves` supera
`2^max_depth`, o si `min_child_samples * num_leaves` supera las filas de
entrenamiento. El resultado queda en `artifacts/tuned_hyperparameters.json`,
con los pesos ya normalizados. Sin `--tune`, el entrenamiento de la evaluación
y de la entrega usa ese archivo. Un archivo de una búsqueda anterior, con
cuotas, no se reutiliza.

### Columnas de two_stage

`ColumnBuilder` (`src/quelibroleo/two_stage/columns.py`) arma, con el historial
anterior al corte:

| Grupo | Qué aporta |
| --- | --- |
| Fuentes de retrieval | Score, rango, recíproco `1/(60+rango)` y un indicador de presencia por cada fuente, más `rrf_score`. |
| Lector | Cantidad de lecturas, media y desvío de rating, días activos, autores y géneros distintos. |
| Ítem | Cantidad de lecturas, usuarios distintos, media, desvío y media bayesiana (`smoothing=10` hacia la media global). También `log1p` del conteo. |
| Afinidad | Cuántas veces el lector ya eligió ese autor o género, y con qué rating medio. |
| Secuencia | Coincidencia con el último autor y género; proporción de ese autor o género en las últimas 10 lecturas; probabilidad empírica de pasar del autor/género anterior al del candidato; edad de publicación respecto del corte del lector. |

No se codifican texto libre ni identificadores categóricos de alta cardinalidad
dentro del booster: autor y género entran solo a través de estos agregados
numéricos. La editorial queda en el texto de TF-IDF. Valores infinitos se
pasan a NaN; LightGBM los trata como faltantes.

## EASE

`EaseModel` (`src/quelibroleo/ease.py`), modelo `ease`. Es un recomendador de
una etapa: no arma candidatos ni llama a LambdaRank. Se elige con
`--model ease` y usa el mismo corte, el mismo top-20 y el mismo NDCG@20.

Aprende una matriz ítem-ítem **B** que reconstruye la fila del lector,
`x̂ = x B`, con la diagonal en cero (Steck, WWW 2019). La solución es cerrada:
se diagonaliza `XᵀX` una vez y cada `λ` solo reescala los autovalores. En
inferencia se enmascaran los libros ya leídos. Si el lector no tiene lecturas
entre los libros del modelo, el ranking es el de popularidad; si EASE no
completa 20, popularidad rellena el resto.

Tres hiperparámetros, los únicos que toca su study:

| Parámetro | Valores | Qué cambia |
| --- | --- | --- |
| `reg_lambda` | 1–100.000, log. Defecto 500 | Penalización L2 de **B** |
| `min_interactions` | 10–100, log. Defecto 20 | Libros que entran a la matriz. Por debajo de 10 el catálogo pasa de unos 12.000 y la Gram densa no entra cómoda |
| `weight_mode` | `binary`, `rating`, `confidence` | Unos, el rating 1–10, o `(1 + rating / 10)` por la misma recencia que ALS |

`confidence` usa `recency_half_life_days` de la config de retrieval (365). Ese
valor no se busca. `--label-gain` sigue siendo solo de LambdaRank.

El study maximiza el NDCG@20 del último fold rolling y escribe
`artifacts/ease_hyperparameters.json`. No lee ni reemplaza
`tuned_hyperparameters.json`. Sin `--tune`, la corrida siguiente reutiliza ese
archivo. En MLflow el tag `model` vale `ease`, el nombre de la corrida lo
incluye, y `ease_selected.*` guarda el trío elegido.
