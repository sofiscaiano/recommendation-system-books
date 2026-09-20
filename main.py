"""CLI del recomendador en dos etapas."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from quelibroleo.config import PipelineConfig
from quelibroleo.pipeline import evaluate, train_final_and_submit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval multi-fuente y ranking LambdaRank para Qué Libro Leo"
    )
    parser.add_argument(
        "command",
        choices=("evaluate", "submit", "all"),
        nargs="?",
        default="evaluate",
    )
    parser.add_argument("--db", type=Path, default=Path("data.db"))
    parser.add_argument("--example", type=Path, default=Path("ejemplo.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Ejecuta Optuna para fusión y LambdaRank; es considerablemente más lento",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = replace(
        PipelineConfig(),
        db_path=args.db,
        example_path=args.example,
        output_path=args.output,
        artifacts_dir=args.artifacts,
        random_seed=args.seed,
    )
    if args.command in {"evaluate", "all"}:
        report, _ = evaluate(config, tune=args.tune)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    if args.command in {"submit", "all"}:
        submission = train_final_and_submit(config, tune=args.tune)
        print(f"Entrega guardada en {args.output}: {len(submission)} filas")


if __name__ == "__main__":
    main()
