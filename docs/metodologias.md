# Metodologías de retrieval y ranking

El recomendador es un pipeline de dos etapas. **Retrieval** arma un ranking
fusionado y se queda con un prefijo. El corte y los pesos son los del último
`--tune`; la búsqueda mueve el corte entre 100 y 1.000. **Ranking** los ordena
y deja el top-20. Los libros ya leídos se excluyen en ambas etapas.

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

### EASE

`EaseRetriever` (`src/quelibroleo/two_stage/retrieval.py`), fuente `ease`.

Misma matriz ítem-ítem que el modelo de una etapa (Steck, WWW 2019): en el
ajuste se diagonaliza `XᵀX` y el score es el de esa solución cerrada. λ y
`min_interactions` no entran al study de `two_stage`. Si existe
`artifacts/ease_hyperparameters.json`, se usan esos valores; si no, los de
`EaseConfig`. Cambiarlos pide un `--tune` nuevo. Optuna solo elige el peso
de la lista. La recencia del perfil es `recency_half_life_days`, la misma
del resto del retrieval.

No rellena con popularidad. Esa lista ya es otra fuente, y mezclarla acá
haría que los mismos libros sumen dos veces en la fusión. Un lector sin
lecturas entre los libros del modelo no aporta candidatos.

`RetrievalConfig.ease_source` la prende (por defecto, sí). `--ease-source` y
`--no-ease-source` la pisan en esa corrida. La primera vez que queda prendida
hace falta `--tune`: un tuneo anterior no tiene peso para esta fuente.

El detalle de la solución cerrada y del study propio está más abajo, en EASE
como modelo único.

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
   si solo cambia la escala. No hay pesos fijos: salen de un `--tune` con esta config.
3. El corte `rrf_cutoff` se aplica solo después del merge: se queda con los
   primeros C libros de ese ranking (`rrf_score` descendente, desempate por
   `id_libro`). No recorta las listas de cada fuente. Un corte mayor es un
   prefijo más largo del mismo ranking: solo agrega candidatos. Tampoco hay
   un corte fijo. Con `--tune`, Optuna recorre C como entero en escala
   logarítmica entre 100 y 1.000 (`RetrievalSearch.rrf_cutoff`) y un peso por
   fuente (`RetrievalSearch.source_weight`, 0,1–3, log). La escala logarítmica
   sigue el rendimiento decreciente de agrandar el conjunto. Sin un tuneo de
   esta config, la corrida pide `evaluate --tune` y no arranca.

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
`linear` (el defecto de `RankerConfig.label_gain`) es la identidad `0..10`.
`exponential` es `2^rating - 1`, la misma ganancia que
`sklearn.metrics.ndcg_score`. Se cambia en `config.py`. `--label-gain` solo
pisa ese valor en una corrida. Cambiarlo deja desalineado el tuneo guardado
con esa config, así que hace falta un `evaluate --tune` nuevo.
`lambdarank_truncation_level` y
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
Ese fold también corta el early stopping. Con `tune_on_example_users` o
`--tune-on-example-users`, el objetivo y el early stopping usan solo los
lectores de `ejemplo.csv` que entran a ese fold. El hold-out oficial no entra al
study. `n_estimators`, `early_stopping_rounds` y `rrf_k` quedan fijos. La
profundidad de cada fuente sale del corte, de `k` y de la cantidad de fuentes;
no es otro hiperparámetro. Un trial se descarta si `num_leaves` supera
`2^max_depth`, o si `min_child_samples * num_leaves` supera las filas de
entrenamiento. Cada búsqueda se agrega a `artifacts/tuning_history.json`,
con la config usada y los pesos ya normalizados. Sin `--tune`, la evaluación
y la entrega reutilizan la entrada de esa misma config, aunque no sea la
última. Si no hay ninguna, hay que buscar antes. Una entrada vieja, con
cuotas, no se reutiliza.

### Columnas de two_stage

`ColumnBuilder` (`src/quelibroleo/two_stage/columns.py`) arma, con el historial
anterior al corte:

| Grupo | Qué aporta |
| --- | --- |
| Fuentes de retrieval | Score, rango, recíproco `1/(60+rango)` y un indicador de presencia por cada fuente, más `rrf_score`. |
| Lector | Cantidad de lecturas, media y desvío de rating, días activos, autores y géneros distintos. También el promedio de días entre lecturas consecutivas, el año de la primera y de la última lectura, el género (`user_gender`: 1 mujer, 0 hombre, faltante si es `-` o vacío) y la edad en años en esas dos fechas (año de la lectura menos año de nacimiento). La diversificación es autores distintos, géneros distintos o décadas de edición distintas, dividido por las lecturas del historial. |
| Ítem | Cantidad de lecturas, usuarios distintos, media, desvío y media bayesiana (`smoothing=10` hacia la media global). También `log1p` del conteo, la diferencia entre el año de edición y el año promedio de las lecturas del lector, el rating medio del autor, el ratio entre el rating medio del libro y el del autor, y el share de las lecturas del autor o del género que concentra ese libro. Un autor o género vacío no tiene share. |
| Afinidad | Cuántas veces el lector ya eligió ese autor o género, con qué rating medio, y ese conteo dividido por todas las lecturas del lector (`user_author_share`, `user_genre_share`). |
| Secuencia | Coincidencia con el último autor y género; proporción de ese autor o género en las últimas 10 lecturas; probabilidad de que el libro, el autor o el género del candidato aparezca entre las 5 lecturas siguientes a uno ya leído, promediada sobre las últimas 5 lecturas del lector con pesos 5, 4, 3, 2 y 1 (un par no observado aporta 0 y un puesto que no existe no entra al denominador); diferencia en años entre la última lectura y el año de edición, y entre el año de edición y el año promedio de las lecturas. |

No se codifican texto libre ni identificadores categóricos de alta cardinalidad
dentro del booster: autor y género entran solo a través de estos agregados
numéricos. La editorial queda en el texto de TF-IDF. Valores infinitos se
pasan a NaN; LightGBM los trata como faltantes.

## EASE

`EaseModel` (`src/quelibroleo/ease.py`), modelo `ease`. Es un recomendador de
una etapa: no arma candidatos ni llama a LambdaRank. Se elige con
`--model ease` y usa el mismo corte, el mismo top-20 y el mismo NDCG@20.
La misma matriz también puede ser una fuente de `two_stage`; en ese papel
no completa con popularidad (ver Retrieval).

Aprende una matriz ítem-ítem **B** que reconstruye la fila del lector,
`x̂ = x B`, con la diagonal en cero (Steck, WWW 2019). La solución es cerrada:
se diagonaliza `XᵀX` una vez y cada `λ` solo reescala los autovalores.

`X` guarda el rating 1–10. Una lectura ausente vale 0. Esa matriz no lleva
recencia: dos libros leídos juntos pesan lo mismo aunque haya pasado un año.
Al puntuar, el perfil sí la lleva. Cada rating se multiplica por
`2^(-edad / semivida)`, con la edad medida contra la última lectura de ese
lector y la semivida de `RetrievalConfig.recency_half_life_days` (365). En
inferencia se enmascaran los libros ya leídos. Si el lector no tiene lecturas
entre los libros del modelo, el ranking es el de popularidad; si EASE no
completa 20, popularidad rellena el resto.

En el hold-out interno, con los libros de al menos 20 lecturas en el train,
ese perfil sube el NDCG@20 de 0,141 (el mismo rating en la matriz y en la
consulta, λ = 17.000) a 0,155. Popularidad, en el mismo corte, queda en 0,072.
Bajar el mínimo de 20 a 10 casi no mueve el NDCG@20 (0,156) y agranda mucho
la Gram. Normalizar cada lector para que no dominen los que más leen tampoco
mejoró el número una vez elegido λ. Un `weight_mode` que metía la recencia o
una confianza adentro de `X` queda afuera: cambiaba el valor a reconstruir.

Dos hiperparámetros, los únicos que toca su study:

| Parámetro | Valores | Qué cambia |
| --- | --- | --- |
| `reg_lambda` | 100–100.000, log. Defecto 15.000 | Penalización L2 de **B**. El defecto está en la escala de una Gram de ratings |
| `min_interactions` | 10–100, log. Defecto 20 | Libros que entran a la matriz. Con 5 lecturas el catálogo pasa de unos 12.000 y la Gram densa no entra cómoda. Con el mínimo en 15, cerca del 30% de las lecturas del hold-out ni siquiera son columnas |

La semivida no se busca. `--label-gain` sigue siendo solo de LambdaRank.
Un JSON viejo que todavía traiga `weight_mode` se lee igual: esa clave se
ignora.

El study maximiza el NDCG@20 del último fold rolling y escribe
`artifacts/ease_hyperparameters.json`. No lee ni reemplaza
`tuned_hyperparameters.json`. Sin `--tune`, la corrida siguiente reutiliza ese
archivo. En MLflow el tag `model` vale `ease`, el nombre de la corrida lo
incluye, y `ease_selected.*` guarda λ y el mínimo.
