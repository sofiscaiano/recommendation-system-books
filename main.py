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
from quelibroleo.serving import export_serving_artifact

app = typer.Typer(
    help="Recomendador de Qué Libro Leo. --model elige two_stage, popularity, ease o two_tower",
    invoke_without_command=True,
)

DbOption = Annotated[Path, typer.Option("--db", help="Base SQLite de solo lectura")]
ExampleOption = Annotated[
    Path, typer.Option("--example", help="Lectores de la entrega")
]
OutputOption = Annotated[Path, typer.Option("--output", help="CSV de entrega")]
ArtifactsOption = Annotated[
    Path, typer.Option("--artifacts", help="Directorio de artefactos y log")
]
TuneOption = Annotated[
    bool,
    typer.Option(
        "--tune",
        help="Busca con Optuna. Sin este flag se reutiliza el tuneo guardado",
    ),
]
SeedOption = Annotated[int, typer.Option("--seed")]
NoteOption = Annotated[
    str,
    typer.Option(
        "--note",
        help="Descripción obligatoria de esta corrida",
    ),
]
LabelGainOption = Annotated[
    str,
    typer.Option(
        "--label-gain",
        help="Ganancia de LambdaRank: linear (0..10) o exponential (2^rel-1). Cambiarla exige --tune",
    ),
]
TwoTowerSourceOption = Annotated[
    bool,
    typer.Option(
        "--two-tower-source",
        help="two_stage suma el two-tower como fuente de retrieval. Exige --tune la primera vez",
    ),
]
TuneOnExampleUsersOption = Annotated[
    bool | None,
    typer.Option(
        "--tune-on-example-users/--no-tune-on-example-users",
        help=(
            "Optuna y el early stopping del último fold puntúan solo los lectores de --example. "
            "Sin el flag se usa PipelineConfig.tune_on_example_users, apagado por defecto"
        ),
    ),
]
ModelOption = Annotated[
    str,
    typer.Option(
        "--model",
        help="Algoritmo: two_stage, popularity, ease o two_tower. El corte train/val/test no cambia",
    ),
]


def _configure_logging(artifacts: Path) -> None:
    """Log a consola y a ``artifacts/run.log``."""

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
    label_gain: str,
    model: str,
    two_tower_source: bool = False,
    tune_on_example_users: bool | None = None,
) -> PipelineConfig:
    """Arma la config de la corrida con las rutas, la nota y el modelo."""

    base = PipelineConfig()
    config = replace(
        base,
        db_path=db,
        example_path=example,
        output_path=output,
        artifacts_dir=artifacts,
        random_seed=seed,
        run_note=note,
        model=model,
        ranker=replace(base.ranker, label_gain=label_gain),
        retrieval=replace(base.retrieval, two_tower_source=two_tower_source),
    )
    if tune_on_example_users is None:
        return config
    return replace(config, tune_on_example_users=tune_on_example_users)


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
    label_gain: str,
    model: str,
    two_tower_source: bool = False,
    tune_on_example_users: bool | None = None,
) -> None:
    """Ejecuta evaluate, submit o ambos con la misma configuración."""

    if not note.strip():
        raise typer.BadParameter("La corrida exige --note con una descripción.")
    _configure_logging(artifacts)
    config = _config(
        db,
        example,
        output,
        artifacts,
        seed,
        note,
        label_gain,
        model,
        two_tower_source,
        tune_on_example_users,
    )
    logging.getLogger(__name__).info(
        "Comando %s (model=%s, tune=%s, tune_on_example_users=%s). Log en %s",
        command,
        model,
        tune,
        config.tune_on_example_users,
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
    label_gain: LabelGainOption = "linear",
    model: ModelOption = "two_stage",
) -> None:
    """Sin subcomando, evalúa con el tuneo guardado o con los valores fijos."""

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
            label_gain=label_gain,
            model=model,
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
    label_gain: LabelGainOption = "linear",
    model: ModelOption = "two_stage",
    two_tower_source: TwoTowerSourceOption = False,
    tune_on_example_users: TuneOnExampleUsersOption = None,
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
        label_gain=label_gain,
        model=model,
        two_tower_source=two_tower_source,
        tune_on_example_users=tune_on_example_users,
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
    label_gain: LabelGainOption = "linear",
    model: ModelOption = "two_stage",
    two_tower_source: TwoTowerSourceOption = False,
    tune_on_example_users: TuneOnExampleUsersOption = None,
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
        label_gain=label_gain,
        model=model,
        two_tower_source=two_tower_source,
        tune_on_example_users=tune_on_example_users,
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
    label_gain: LabelGainOption = "linear",
    model: ModelOption = "two_stage",
    two_tower_source: TwoTowerSourceOption = False,
    tune_on_example_users: TuneOnExampleUsersOption = None,
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
        label_gain=label_gain,
        model=model,
        two_tower_source=two_tower_source,
        tune_on_example_users=tune_on_example_users,
    )


@app.command(name="export-serving")
def export_serving(
    db: DbOption = Path("data.db"),
    artifacts: ArtifactsOption = Path("artifacts"),
    seed: SeedOption = 42,
) -> None:
    """Ajusta las fuentes sobre toda la historia y escribe artifacts/serving.joblib."""

    _configure_logging(artifacts)
    config = replace(
        PipelineConfig(),
        db_path=db,
        artifacts_dir=artifacts,
        random_seed=seed,
    )
    destination = export_serving_artifact(config)
    print(f"Artefacto de serving guardado en {destination}")


if __name__ == "__main__":
    app()
