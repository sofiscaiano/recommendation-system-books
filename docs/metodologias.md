# Metodologías de retrieval y ranking

El recomendador es un pipeline de dos etapas. **Retrieval** propone un conjunto
chico de candidatos por lector (por defecto 200, configurable entre 100 y
1.000). **Ranking** los ordena y deja el top-20. Los libros ya leídos se
excluyen en ambas etapas.

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

### Item-item por coseno (dos variantes)

`ItemCosineRetriever` (`src/quelibroleo/retrieval.py`).

Construye una matriz dispersa ítem × usuario y normaliza cada fila a norma L2.
El score de un candidato es el producto interno entre el perfil del lector y el
vector del ítem. No se materializa la matriz cuadrada ítem × ítem: el score se
calcula como `(perfil · vectores_ítem) · vectores_ítemᵀ`.

- **occurrence** (`item_cosine`): el peso de cada lectura es
  `rating / 10` por un factor de recencia.
- **adjusted** (`item_adjusted_cosine`): el peso es el rating centrado en la
  media del lector (coseno ajustado). No usa recencia.

La recencia es una semivida exponencial: si `half_life_days` es 365, una
lectura de un año antes del último evento del lector pesa la mitad
(`2^(-edad_en_días / semivida)`).

Librerías: NumPy, SciPy (`scipy.sparse.csr_matrix`), scikit-learn
(`sklearn.preprocessing.normalize`), pandas.

### SVD explícito

`SurpriseSVDRetriever`.

Factoriza la matriz de ratings observados (escala 1–10) con el SVD sesgado de
Funk / Koren. El score predicho de un par usuario–ítem es

`μ + b_u + b_i + p_u · q_i`

con media global, sesgos y factores latentes (por defecto 64 factores y 20
épocas). Solo puntúa ítems y usuarios vistos en el trainset.

Librería: [scikit-surprise](https://surpriselib.com/) (`surprise.SVD`,
`Dataset`, `Reader`), más NumPy y pandas.

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

### Fusión: cuotas y Reciprocal Rank Fusion

`CandidateFusion`.

1. Cada fuente aporta una cuota mínima de sus mejores rangos (por defecto, para
   un presupuesto de 200: popularidad 25, cada coseno 60, SVD 75, ALS 75,
   BPR 75, TF-IDF 60). Si la suma supera el presupuesto, las cuotas se escalan
   en proporción.
2. El resto del presupuesto se llena con Reciprocal Rank Fusion ponderado:

   `score(u, i) = Σ_s  peso_s / (k + rango_s(u, i))`

   con `k = 60`. Los pesos por defecto son 0,5 (popularidad), 1,0 (cosenos,
   SVD, ALS, BPR) y 0,8 (TF-IDF). Un libro que no aparece en una fuente no
   suma ese término. Con `--tune`, Optuna recorre un peso y una cuota relativa
   por fuente; los intervalos son `RetrievalConfig.fusion_weight` (0,1–3, log)
   y `fusion_quota_share` (0–1). La grilla de presupuestos sigue siendo
   `candidate_budgets`.

El orden de la fusión (`retrieval_rank`, `rrf_score`) no es el ranking final:
solo define quién entra al ranker. Los scores y rangos de cada fuente se
conservan como features.

Librerías: NumPy y pandas. No hay una librería externa de RRF.

## Ranking

### LambdaRank (LightGBM)

`LambdaRanker` (`src/quelibroleo/ranker.py`) envuelve `lightgbm.LGBMRanker`.

LambdaRank entrena árboles de boosting para optimizar NDCG, no el error de
rating. Cada lector es un grupo (`query_id`). Dentro del grupo, la etiqueta
`relevance` es el rating futuro (1–10) si el candidato está en el hold-out, y
0 si no. `label_gain` es la identidad `0..10`, así que un 10 aporta más al
NDCG que un 1. `lambdarank_truncation_level` y `eval_at` están en 20, alineados
con la métrica de entrega.

Parámetros por defecto: 200 árboles, learning rate 0,04, 31 hojas, profundidad
libre, `min_child_samples=30`, subsample y colsample 0,9, `reg_lambda=1`. Si
hay validación temporal, early stopping a 30 rondas sobre NDCG.

El desempate del score predicho es estable por `id_libro`.

Librerías: LightGBM ≥ 4.7, NumPy, pandas. NDCG de evaluación:
`sklearn.metrics.ndcg_score` en `src/quelibroleo/metrics.py`.

La búsqueda opcional (`main.py evaluate --tune`) usa Optuna. Qué parámetros
entra a la búsqueda y con qué intervalo está en `RankerConfig.search_space` y
en `RetrievalConfig.fusion_weight` / `fusion_quota_share`
(`src/quelibroleo/config.py`). LambdaRank se elige por NDCG@20 en un fold
interno, sin tocar el hold-out final. `n_estimators` y `early_stopping_rounds`
quedan fijos.

### Features del ranker

`FeatureBuilder` (`src/quelibroleo/features.py`) arma, con el historial
anterior al corte:

| Grupo | Qué aporta |
| --- | --- |
| Fuentes de retrieval | Score, rango, recíproco `1/(60+rango)` y un indicador de presencia por cada fuente, más `rrf_score`. |
| Lector | Cantidad de lecturas, media y desvío de rating, días activos, autores y géneros distintos. |
| Ítem | Cantidad de lecturas, usuarios distintos, media, desvío y media bayesiana (`smoothing=10` hacia la media global). También `log1p` del conteo. |
| Afinidad | Cuántas veces el lector ya eligió ese autor o género, y con qué rating medio. |
| Secuencia | Coincidencia con el último autor y género; proporción de ese autor o género en las últimas 10 lecturas; probabilidad empírica de pasar del autor/género anterior al del candidato; edad de publicación respecto del corte del lector. |

No se codifican texto libre ni identificadores categóricos de alta cardinalidad
dentro del booster: autor, género y editorial entran solo a través de estos
agregados numéricos. Valores infinitos se pasan a NaN; LightGBM los trata como
faltantes.
