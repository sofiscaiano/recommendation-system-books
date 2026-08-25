from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, ndcg_score, root_mean_squared_error


def rating_errors(y_true, y_pred) -> tuple[float, float]:
    """MSE y RMSE sobre predicciones de rating (hold-out punto a punto)."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return mse, rmse


def ndcg_at_k(
    df_test: pd.DataFrame,
    df_pred: pd.DataFrame,
    k: int = 20,
    user_col: str = "id_lector",
    item_col: str = "id_libro",
    score_col: str = "score_pred",
    rating_col: str = "rating",
) -> float:
    """NDCG@k medio con ``sklearn.metrics.ndcg_score``.

    Por usuario se une el top-k recomendado con los ítems de test. Los
    recomendados llevan su ``score_pred``; los relevantes no recomendados
    reciben un score menor al mínimo predicho (sklearn no admite ``-inf``).
    La relevancia de un recomendado ausente en test es 0. El IDCG usa los
    ratings verdaderos del hold-out (ganancia exponencial de sklearn).
    """
    scores: list[float] = []
    pred_groups = df_pred.groupby(user_col, sort=False)
    test_groups = df_test.groupby(user_col, sort=False)
    usuarios = set(pred_groups.groups).intersection(test_groups.groups)

    for usuario in usuarios:
        pred = pred_groups.get_group(usuario)[[item_col, score_col]]
        verdadero = test_groups.get_group(usuario)[[item_col, rating_col]]
        union = pred.merge(verdadero, on=item_col, how="outer")
        y_true = union[rating_col].fillna(0.0).to_numpy(dtype=float).reshape(1, -1)
        min_pred = pred[score_col].min()
        filler = float(min_pred) - 1.0 if pd.notna(min_pred) else -1e12
        y_score = union[score_col].fillna(filler).to_numpy(dtype=float).reshape(1, -1)

        if y_true.shape[1] == 0 or not np.any(y_true > 0):
            continue

        k_eval = min(k, y_true.shape[1])
        scores.append(float(ndcg_score(y_true, y_score, k=k_eval)))

    return float(np.mean(scores)) if scores else 0.0
