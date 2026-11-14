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
    ordered = _recommendation_order(recommendations)
    truth_ratings = truth.groupby([USER_COL, ITEM_COL], as_index=False)[
        RATING_COL
    ].max()
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
            item: float(k - rank + 1) for rank, item in enumerate(rec_items)
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
    recall: float
    weighted_recall: float
    oracle_ndcg: float
    mean_candidates: float


def retrieval_hits(candidates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth_rows = truth.groupby([USER_COL, ITEM_COL], as_index=False)[RATING_COL].max()
    truth_rows[USER_COL] = truth_rows[USER_COL].astype(str)
    truth_rows[ITEM_COL] = truth_rows[ITEM_COL].astype(str)
    pool = candidates[[USER_COL, ITEM_COL]].drop_duplicates().copy()
    pool[USER_COL] = pool[USER_COL].astype(str)
    pool[ITEM_COL] = pool[ITEM_COL].astype(str)
    pool["in_retrieval"] = 1
    merged = truth_rows.merge(pool, on=[USER_COL, ITEM_COL], how="left")
    merged["in_retrieval"] = merged["in_retrieval"].fillna(0).astype(int)
    return merged[[USER_COL, ITEM_COL, RATING_COL, "in_retrieval"]]


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
        weighted_recalls.append(captured_gain / total_gain if total_gain > 0 else 0.0)
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
        weighted_recall=float(np.mean(weighted_recalls)) if weighted_recalls else 0.0,
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
