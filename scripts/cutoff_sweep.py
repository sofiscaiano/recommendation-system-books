"""NDCG@20 al achicar el pool que entra a LambdaRank.

Usa el ranker de ``artifacts/evaluated_pipeline.joblib``, entrenado con el
corte tuneado. Recupera ese pool una vez por split y, para cada corte menor,
deja entrar solo el prefijo del RRF. Las features de un libro no dependen de
los otros candidatos, así que puntuar el pool completo y descartar la cola es
lo mismo que volver a fusionar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from quelibroleo.config import PipelineConfig
from quelibroleo.data import ITEM_COL, USER_COL
from quelibroleo.metrics import ndcg_at_k, retrieval_metrics
from quelibroleo.pipeline import TrainedPipeline, _collect_material
from quelibroleo.split_data import example_users, load_split, validation_folds
from quelibroleo.two_stage.retrieval import CandidateFusion, per_source_depth

LOGGER = logging.getLogger(__name__)

CUTOFFS = (20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600, 700, 843)


def _score_pool(
    trained: TrainedPipeline,
    train: pd.DataFrame,
    target: pd.DataFrame,
    users: tuple[str, ...],
    books: pd.DataFrame,
    readers: pd.DataFrame,
    config: PipelineConfig,
    fold: int,
) -> pd.DataFrame:
    """Puntúa el pool del corte tuneado y conserva el rango de retrieval."""

    n_sources = len(config.retrieval.enabled_sources())
    depth = per_source_depth(
        trained.rrf_cutoff,
        rrf_k=config.retrieval.rrf_k,
        n_sources=n_sources,
    )
    LOGGER.info(
        "Recuperando fold %s: %s usuarios, corte %s, profundidad %s",
        fold,
        len(users),
        trained.rrf_cutoff,
        depth,
    )
    material = _collect_material(
        train,
        target,
        users,
        books,
        config,
        fold,
        depth,
        readers,
    )
    fusion = CandidateFusion(weights=trained.weights, rrf_k=config.retrieval.rrf_k)
    fused = fusion.fuse(material.source_candidates, trained.rrf_cutoff)
    features = material.feature_builder.transform(
        fused.candidates,
        fused.source_candidates,
    )
    scored = trained.ranker.predict(features)
    ranks = fused.candidates[[USER_COL, ITEM_COL, "retrieval_rank"]]
    return scored.merge(ranks, on=[USER_COL, ITEM_COL], how="left", validate="one_to_one")


def _rows_for_split(
    scored: pd.DataFrame,
    truth: pd.DataFrame,
    example: set[str],
    *,
    split: str,
    cutoffs: tuple[int, ...],
) -> list[dict[str, object]]:
    """NDCG, recall y oráculo para cada corte y cada cohorte."""

    rows: list[dict[str, object]] = []
    cohorts = {
        "todos": truth,
        "ejemplo": truth[truth[USER_COL].astype(str).isin(example)],
    }
    for cutoff in cutoffs:
        pool = scored.loc[
            scored["retrieval_rank"] <= cutoff, [USER_COL, ITEM_COL, "score"]
        ].copy()
        pool = pool.sort_values(
            [USER_COL, "score", ITEM_COL],
            ascending=[True, False, True],
            kind="mergesort",
        )
        pool["rank"] = pool.groupby(USER_COL, sort=False).cumcount() + 1
        top = pool.loc[pool["rank"] <= 20, [USER_COL, ITEM_COL, "rank"]]
        for cohort, cohort_truth in cohorts.items():
            if cohort_truth.empty:
                continue
            coverage = retrieval_metrics(pool, cohort_truth)
            rows.append(
                {
                    "split": split,
                    "cohort": cohort,
                    "cutoff": cutoff,
                    "users": int(cohort_truth[USER_COL].nunique()),
                    "ndcg_at_20": ndcg_at_k(top, cohort_truth),
                    "recall": coverage.recall,
                    "weighted_recall": coverage.weighted_recall,
                    "oracle_ndcg_at_20": coverage.oracle_ndcg,
                    "mean_candidates": coverage.mean_candidates,
                }
            )
            LOGGER.info(
                "%s %s corte %s: ndcg=%.4f recall=%.4f oraculo=%.4f",
                split,
                cohort,
                cutoff,
                rows[-1]["ndcg_at_20"],
                rows[-1]["recall"],
                rows[-1]["oracle_ndcg_at_20"],
            )
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = PipelineConfig()
    artifact = Path(config.artifacts_dir) / "evaluated_pipeline.joblib"
    trained = joblib.load(artifact)
    if not isinstance(trained, TrainedPipeline):
        raise TypeError(f"{artifact} no contiene el pipeline evaluado")
    cutoffs = tuple(sorted({*CUTOFFS, int(trained.rrf_cutoff)}))
    if cutoffs[-1] != int(trained.rrf_cutoff) and trained.rrf_cutoff not in cutoffs:
        cutoffs = tuple(sorted({*cutoffs, int(trained.rrf_cutoff)}))
    cutoffs = tuple(value for value in cutoffs if value <= int(trained.rrf_cutoff))

    split = load_split(config, holdout=True)
    example = example_users(config)
    folds = validation_folds(split.history, config)
    validation = folds[-1]

    rows: list[dict[str, object]] = []
    validation_scores = _score_pool(
        trained,
        validation.train,
        validation.target,
        validation.query_users,
        split.books,
        split.readers,
        config,
        validation.fold,
    )
    rows.extend(
        _rows_for_split(
            validation_scores,
            validation.target,
            example,
            split="validacion",
            cutoffs=cutoffs,
        )
    )
    del validation_scores

    test_scores = _score_pool(
        trained,
        split.history,
        split.test,
        split.evaluation_users,
        split.books,
        split.readers,
        config,
        999,
    )
    rows.extend(
        _rows_for_split(
            test_scores,
            split.test,
            example,
            split="test",
            cutoffs=cutoffs,
        )
    )

    frame = pd.DataFrame(rows)
    reference = frame.loc[frame["cutoff"] == int(trained.rrf_cutoff), ["split", "cohort", "ndcg_at_20", "recall"]]
    reference = reference.rename(
        columns={"ndcg_at_20": "ndcg_full", "recall": "recall_full"}
    )
    frame = frame.merge(reference, on=["split", "cohort"], how="left")
    frame["delta_ndcg_at_20"] = frame["ndcg_at_20"] - frame["ndcg_full"]
    frame["delta_recall"] = frame["recall"] - frame["recall_full"]
    destination = Path(config.artifacts_dir) / "cutoff_sweep.csv"
    frame.to_csv(destination, index=False)
    LOGGER.info("Barrido escrito en %s", destination)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
