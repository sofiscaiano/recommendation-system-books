"""CLI del recomendador en dos etapas."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated

import typer

from quelibroleo.config import PipelineConfig
from quelibroleo.pipeline import evaluate as evaluate_pipeline
from quelibroleo.pipeline import train_final_and_submit

app = typer.Typer(
    help="Retrieval multi-fuente y ranking LambdaRank para Qué Libro Leo",
    invoke_without_command=True,
)

DbOption = Annotated[Path, typer.Option("--db", help="Base SQLite de solo lectura")]
ExampleOption = Annotated[Path, typer.Option("--example", help="Lectores de la entrega")]
OutputOption = Annotated[Path, typer.Option("--output", help="CSV de entrega")]
ArtifactsOption = Annotated[
    Path, typer.Option("--artifacts", help="Directorio de artefactos y log")
]
TuneOption = Annotated[
    bool, typer.Option("--tune", help="Ajusta fusión y LambdaRank con Optuna")
]
SeedOption = Annotated[int, typer.Option("--seed")]
NoteOption = Annotated[
    str,
    typer.Option(
        "--note",
        help="Descripción obligatoria de esta corrida",
    ),
]


def _configure_logging(artifacts: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(artifacts / "run.log", encoding="utf-8"),
        ],
        force=True,
    )


def _config(
    db: Path,
    example: Path,
    output: Path,
    artifacts: Path,
    seed: int,
    note: str,
) -> PipelineConfig:
    return replace(
        PipelineConfig(),
        db_path=db,
        example_path=example,
        output_path=output,
        artifacts_dir=artifacts,
        random_seed=seed,
        run_note=note,
    )


def _run(
    command: str,
    *,
    db: Path,
    example: Path,
    output: Path,
    artifacts: Path,
    tune: bool,
    seed: int,
    note: str,
) -> None:
    if not note.strip():
        raise typer.BadParameter("La corrida exige --note con una descripción.")
    _configure_logging(artifacts)
    config = _config(db, example, output, artifacts, seed, note)
    logging.getLogger(__name__).info(
        "Comando %s (tune=%s). Log en %s",
        command,
        tune,
        artifacts / "run.log",
    )
    if command in {"evaluate", "all"}:
        report, _ = evaluate_pipeline(config, tune=tune)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    if command in {"submit", "all"}:
        submission = train_final_and_submit(config, tune=tune)
        print(f"Entrega guardada en {output}: {len(submission)} filas")


@app.callback()
def main(
    ctx: typer.Context,
    db: DbOption = Path("data.db"),
    example: ExampleOption = Path("ejemplo.csv"),
    output: OutputOption = Path("submission.csv"),
    artifacts: ArtifactsOption = Path("artifacts"),
    tune: TuneOption = False,
    seed: SeedOption = 42,
    note: NoteOption = "",
) -> None:
    """Sin subcomando, corre la evaluación con los hiperparámetros fijos."""

    if ctx.invoked_subcommand is None:
        _run(
            "evaluate",
            db=db,
            example=example,
            output=output,
            artifacts=artifacts,
            tune=tune,
            seed=seed,
            note=note,
        )


@app.command()
def evaluate(
    db: DbOption = Path("data.db"),
    example: ExampleOption = Path("ejemplo.csv"),
    output: OutputOption = Path("submission.csv"),
    artifacts: ArtifactsOption = Path("artifacts"),
    tune: TuneOption = False,
    seed: SeedOption = 42,
    note: NoteOption = ...,
) -> None:
    """Evalúa el leave-last-20 y escribe artifacts/."""

    _run(
        "evaluate",
        db=db,
        example=example,
        output=output,
        artifacts=artifacts,
        tune=tune,
        seed=seed,
        note=note,
    )


@app.command()
def submit(
    db: DbOption = Path("data.db"),
    example: ExampleOption = Path("ejemplo.csv"),
    output: OutputOption = Path("submission.csv"),
    artifacts: ArtifactsOption = Path("artifacts"),
    tune: TuneOption = False,
    seed: SeedOption = 42,
    note: NoteOption = ...,
) -> None:
    """Reentrena con toda la historia y escribe submission.csv."""

    _run(
        "submit",
        db=db,
        example=example,
        output=output,
        artifacts=artifacts,
        tune=tune,
        seed=seed,
        note=note,
    )


@app.command(name="all")
def run_all(
    db: DbOption = Path("data.db"),
    example: ExampleOption = Path("ejemplo.csv"),
    output: OutputOption = Path("submission.csv"),
    artifacts: ArtifactsOption = Path("artifacts"),
    tune: TuneOption = False,
    seed: SeedOption = 42,
    note: NoteOption = ...,
) -> None:
    """Evalúa y después genera la entrega."""

    _run(
        "all",
        db=db,
        example=example,
        output=output,
        artifacts=artifacts,
        tune=tune,
        seed=seed,
        note=note,
    )


if __name__ == "__main__":
    app()
