"""Métricas por etapa y diagnósticos del ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from .data import ITEM_COL, RATING_COL, USER_COL


def _recommendation_order(recommendations: pd.DataFrame) -> pd.DataFrame:
    if "rank" in recommendations:
        return recommendations.sort_values([USER_COL, "rank", ITEM_COL])
    if "score" in recommendations:
        return recommendations.sort_values(
            [USER_COL, "score", ITEM_COL],
            ascending=[True, False, True],
        )
    ordered = recommendations.copy()
    ordered["_input_order"] = np.arange(len(ordered))
    return ordered.sort_values([USER_COL, "_input_order"])


def ndcg_per_user(
    recommendations: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    k: int = 20,
) -> pd.Series:
    """NDCG correcto aun cuando un positivo no fue recuperado.

    Se evalúa la unión entre recomendaciones y verdad. Los positivos ausentes
    reciben score menor que cualquier recomendación, por lo que penalizan el
    DCG y siguen formando parte del IDCG.
    """

    ordered = _recommendation_order(recommendations)
    truth_ratings = (
        truth.groupby([USER_COL, ITEM_COL], as_index=False)[RATING_COL].max()
    )
    results: dict[str, float] = {}

    for user in sorted(truth_ratings[USER_COL].astype(str).unique()):
        user_truth = truth_ratings[truth_ratings[USER_COL] == user]
        user_recs = ordered[ordered[USER_COL] == user].head(k)
        rec_items = list(user_recs[ITEM_COL].astype(str))
        truth_items = list(user_truth[ITEM_COL].astype(str))
        padding = [f"__padding_{index}" for index in range(k - len(rec_items))]
        universe = list(dict.fromkeys(rec_items + padding + truth_items))
        relevance_by_item = dict(
            zip(
                user_truth[ITEM_COL].astype(str),
                user_truth[RATING_COL].astype(float),
                strict=True,
            )
        )
        relevance = np.array(
            [relevance_by_item.get(item, 0.0) for item in universe],
            dtype=float,
        )
        score_by_item = {
            item: float(k - rank + 1)
            for rank, item in enumerate(rec_items)
        }
        score_by_item.update({item: 0.0 for item in padding})
        predicted = np.array(
            [score_by_item.get(item, -1.0) for item in universe],
            dtype=float,
        )
        if relevance.sum() == 0:
            results[user] = 0.0
        else:
            results[user] = float(
                ndcg_score(
                    relevance.reshape(1, -1),
                    predicted.reshape(1, -1),
                    k=k,
                    ignore_ties=True,
                )
            )
    return pd.Series(results, name=f"ndcg@{k}", dtype=float)


def ndcg_at_k(
    recommendations: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    k: int = 20,
) -> float:
    values = ndcg_per_user(recommendations, truth, k=k)
    return float(values.mean()) if not values.empty else 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    """Cobertura de positivos antes del ranker."""

    recall: float
    weighted_recall: float
    oracle_ndcg: float
    mean_candidates: float


def retrieval_metrics(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    k: int = 20,
) -> RetrievalMetrics:
    users = sorted(truth[USER_COL].astype(str).unique())
    candidates_by_user = candidates.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
    recalls: list[float] = []
    weighted_recalls: list[float] = []
    oracle_rows: list[dict[str, object]] = []

    for user in users:
        user_truth = truth[truth[USER_COL] == user]
        available = candidates_by_user.get(user, set())
        hit_mask = user_truth[ITEM_COL].isin(available)
        recalls.append(float(hit_mask.mean()) if len(user_truth) else 0.0)
        total_gain = float(user_truth[RATING_COL].sum())
        captured_gain = float(user_truth.loc[hit_mask, RATING_COL].sum())
        weighted_recalls.append(
            captured_gain / total_gain if total_gain > 0 else 0.0
        )
        captured = user_truth.loc[hit_mask].sort_values(
            [RATING_COL, ITEM_COL], ascending=[False, True]
        )
        oracle_rows.extend(
            {
                USER_COL: user,
                ITEM_COL: row[ITEM_COL],
                "rank": rank,
            }
            for rank, (_, row) in enumerate(captured.iterrows(), start=1)
        )

    oracle = pd.DataFrame.from_records(
        oracle_rows, columns=[USER_COL, ITEM_COL, "rank"]
    )
    counts = candidates.groupby(USER_COL)[ITEM_COL].nunique()
    return RetrievalMetrics(
        recall=float(np.mean(recalls)) if recalls else 0.0,
        weighted_recall=float(np.mean(weighted_recalls))
        if weighted_recalls
        else 0.0,
        oracle_ndcg=ndcg_at_k(oracle, truth, k=k),
        mean_candidates=float(counts.reindex(users, fill_value=0).mean())
        if users
        else 0.0,
    )


def rows_against_reference(
    rows: list[dict[str, object]],
    *,
    stage: str,
    ndcg_at_20: float,
    recall: float | None = None,
    features: float | None = None,
) -> list[dict[str, object]]:
    """Antepone la referencia y resta sus métricas a cada ablación.

    El delta es ablación menos referencia: negativo significa que sacar esa
    fuente o esa familia dejó un resultado peor.
    """

    reference: dict[str, object] = {
        "stage": stage,
        "ablation": "full",
        "ndcg_at_20": ndcg_at_20,
        "features": features,
        "recall": recall,
        "delta_ndcg_at_20": 0.0,
        "delta_recall": 0.0 if recall is not None else None,
    }
    compared = [reference]
    for row in rows:
        item = dict(row)
        item["delta_ndcg_at_20"] = float(row["ndcg_at_20"]) - ndcg_at_20
        row_recall = row.get("recall")
        if recall is None or row_recall is None:
            item["delta_recall"] = None
        else:
            item["delta_recall"] = float(row_recall) - recall
        compared.append(item)
    return compared


def recall_at_k(
    recommendations: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    k: int = 20,
) -> float:
    ordered = _recommendation_order(recommendations)
    top = ordered.groupby(USER_COL, sort=False).head(k)
    recalls = []
    top_sets = top.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
    for user, frame in truth.groupby(USER_COL):
        relevant = set(frame[ITEM_COL])
        recalls.append(len(relevant & top_sets.get(user, set())) / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


def catalog_coverage(
    recommendations: pd.DataFrame,
    eligible_items: set[str],
    *,
    k: int = 20,
) -> float:
    if not eligible_items:
        return 0.0
    ordered = _recommendation_order(recommendations)
    top = ordered.groupby(USER_COL, sort=False).head(k)
    return len(set(top[ITEM_COL]) & eligible_items) / len(eligible_items)


def long_tail_exposure(
    recommendations: pd.DataFrame,
    train: pd.DataFrame,
    *,
    k: int = 20,
    head_quantile: float = 0.8,
) -> float:
    popularity = train.groupby(ITEM_COL).size()
    threshold = popularity.quantile(head_quantile)
    tail = set(popularity[popularity <= threshold].index)
    ordered = _recommendation_order(recommendations)
    top = ordered.groupby(USER_COL, sort=False).head(k)
    if top.empty:
        return 0.0
    return float(top[ITEM_COL].isin(tail).mean())


def activity_segments(
    train: pd.DataFrame,
    users: pd.Index | list[str] | tuple[str, ...],
) -> dict[str, pd.Index]:
    """Cuartiles de actividad en train y el grupo de una sola fecha."""

    index = pd.Index(users, dtype=str)
    counts = train.groupby(USER_COL).size().reindex(index)
    segments: dict[str, pd.Index] = {}
    valid = counts.dropna()
    if len(valid) >= 4:
        buckets = pd.qcut(
            valid.rank(method="first"),
            q=4,
            labels=["activity_q1", "activity_q2", "activity_q3", "activity_q4"],
        )
        for label in buckets.cat.categories:
            segments[str(label)] = buckets[buckets == label].index
    dates = train.groupby(USER_COL)["fecha"].nunique().reindex(index)
    segments["single_date_history"] = dates[dates == 1].index
    return segments


def _nan_to_zero(value: float) -> float:
    return 0.0 if np.isnan(value) else float(value)


def segment_metrics(
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    train: pd.DataFrame,
    eligible_items: set[str],
    *,
    k: int = 20,
) -> dict[str, dict[str, float]]:
    """NDCG, recall de retrieval, cobertura y cola larga por segmento."""

    per_user_ndcg = ndcg_per_user(recommendations, truth, k=k)
    users = per_user_ndcg.index
    report: dict[str, dict[str, float]] = {}
    for name, member_index in activity_segments(train, users).items():
        members = [user for user in member_index if user in users]
        recs = recommendations[recommendations[USER_COL].isin(members)]
        cand = candidates[candidates[USER_COL].isin(members)]
        holdout = truth[truth[USER_COL].isin(members)]
        retrieval = retrieval_metrics(cand, holdout, k=k)
        report[name] = {
            "users": float(len(members)),
            "ndcg_at_20": _nan_to_zero(per_user_ndcg.reindex(members).mean()),
            "model_recall_at_20": recall_at_k(recs, holdout, k=k),
            "retrieval_recall": retrieval.recall,
            "retrieval_weighted_recall": retrieval.weighted_recall,
            "catalog_coverage_at_20": catalog_coverage(
                recs, eligible_items, k=k
            ),
            "long_tail_exposure_at_20": long_tail_exposure(recs, train, k=k),
        }
    return report


def paired_bootstrap_delta(
    challenger: pd.Series,
    baseline: pd.Series,
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    common = challenger.index.intersection(baseline.index)
    deltas = (challenger.loc[common] - baseline.loc[common]).to_numpy(dtype=float)
    if len(deltas) == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(random_state)
    draws = rng.choice(deltas, size=(samples, len(deltas)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(deltas.mean()),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
    }
