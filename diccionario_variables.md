# Diccionario de variables del modelo

El `LGBMRegressor` usa las **13 columnas** de `FEATURE_COLUMNS` (`src/quelibroleo/features.py`). El target es `rating`; los identificadores no entran al modelo.

## Categóricas

LightGBM las trata como `category`. Los niveles se fijan en train.

| Variable | Descripción |
| --- | --- |
| `autor` | Autor del libro (catálogo). |
| `genero` | Género literario del libro. |
| `editorial` | Editorial del libro. |
| `sexo` | Género/sexo del lector (columna `genero` de `lectores`). |
| `vive_en_ciudad` | Ciudad extraída de `vive_en` (parte antes del `-`). |
| `vive_en_pais` | País extraído de `vive_en` (parte después del `-`, o el texto completo si no hay guion). |

## Numéricas

| Variable | Descripción |
| --- | --- |
| `anio_edicion` | Año de edición del libro. |
| `nacimiento` | Año de nacimiento del lector. |
| `edad` | Año de referencia − `nacimiento`. En train/eval: año de `fecha`. En inferencia: último año de train (`ref_year`). |
| `item_mean_rating` | Rating medio histórico del libro en train. Sin interacciones en train: ausente (`NaN`). |
| `item_count` | Número de interacciones del libro en train. Sin interacciones en train: ausente (`NaN`). |
| `user_genero_mean` | Rating medio **de ese lector** en ese género. Si el lector no lo leyó en train: ausente (`NaN`). |
| `user_autor_mean` | Rating medio **de ese lector** para ese autor. Si el lector no lo leyó en train: ausente (`NaN`). |

Las estadísticas de interacciones (`item_*`, afinidades) se calculan **solo con train**. No se imputan con medias globales ni de género/autor.

## Fuera del modelo

Se usan al armar features o el ranking, pero no como inputs de LightGBM:

| Variable | Rol |
| --- | --- |
| `id_lector`, `id_libro` | Claves; salida de la submission. |
| `titulo`, `isbn`, `resumen`, `img_src`, `nombre` | Catálogo / perfil; no se usan en el modelo. |
| `vive_en` | Texto crudo; se parte en ciudad y país. |
| `fecha` | Split temporal y cálculo de `edad` en train/eval. |
| `rating` | Target del regresor. |
