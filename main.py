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
    help="Recomendador de Qué Libro Leo. --model elige two_stage, popularity o una fuente",
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
        help=(
            "Busca con Optuna y lo agrega al historial. Sin este flag, two_stage "
            "reutiliza un tuneo de la misma config; si no hay, pide correr --tune"
        ),
    ),
]
SeedOption = Annotated[
    int | None,
    typer.Option(
        "--seed",
        help="Semilla. Sin el flag se usa PipelineConfig.random_seed",
    ),
]
NoteOption = Annotated[
    str,
    typer.Option(
        "--note",
        help="Descripción obligatoria de esta corrida",
    ),
]
LabelGainOption = Annotated[
    str | None,
    typer.Option(
        "--label-gain",
        help=(
            "Ganancia de LambdaRank: linear (0..10) o exponential (2^rel-1). "
            "Sin el flag se usa RankerConfig.label_gain. Cambiarla exige --tune"
        ),
    ),
]
TwoTowerSourceOption = Annotated[
    bool | None,
    typer.Option(
        "--two-tower-source/--no-two-tower-source",
        help=(
            "Prende o apaga el two-tower como fuente de retrieval. "
            "Sin el flag se usa RetrievalConfig.two_tower_source. "
            "Prenderlo por primera vez exige --tune"
        ),
    ),
]
EaseSourceOption = Annotated[
    bool | None,
    typer.Option(
        "--ease-source/--no-ease-source",
        help=(
            "Prende o apaga EASE como fuente de retrieval. "
            "Sin el flag se usa RetrievalConfig.ease_source. "
            "Prenderlo por primera vez exige --tune. "
            "λ y el mínimo de lecturas salen del JSON de EASE o de EaseConfig. "
            "La consulta pondera el rating por recencia"
        ),
    ),
]
TuneOnExampleUsersOption = Annotated[
    bool | None,
    typer.Option(
        "--tune-on-example-users/--no-tune-on-example-users",
        help=(
            "Optuna y el early stopping del último fold puntúan solo los lectores de --example. "
            "Sin el flag se usa PipelineConfig.tune_on_example_users, prendido por defecto"
        ),
    ),
]
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        help=(
            "Algoritmo: two_stage, popularity, item_cosine, als, bpr, tfidf, "
            "ease o two_tower. Sin el flag se usa PipelineConfig.model. "
            "El corte train/val/test no cambia"
        ),
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
    seed: int | None,
    note: str,
    label_gain: str | None,
    model: str | None,
    two_tower_source: bool | None = None,
    ease_source: bool | None = None,
    tune_on_example_users: bool | None = None,
) -> PipelineConfig:
    base = PipelineConfig()
    retrieval = base.retrieval
    if two_tower_source is not None:
        retrieval = replace(retrieval, two_tower_source=two_tower_source)
    if ease_source is not None:
        retrieval = replace(retrieval, ease_source=ease_source)
    ranker = base.ranker
    if label_gain is not None:
        ranker = replace(ranker, label_gain=label_gain)
    config = replace(
        base,
        db_path=db,
        example_path=example,
        output_path=output,
        artifacts_dir=artifacts,
        run_note=note,
        ranker=ranker,
        retrieval=retrieval,
    )
    if seed is not None:
        config = replace(config, random_seed=seed)
    if model is not None:
        config = replace(config, model=model)
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
    seed: int | None,
    note: str,
    label_gain: str | None,
    model: str | None,
    two_tower_source: bool | None = None,
    ease_source: bool | None = None,
    tune_on_example_users: bool | None = None,
) -> None:
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
        ease_source,
        tune_on_example_users,
    )
    logger = logging.getLogger(__name__)
    if config.model == "two_stage":
        logger.info(
            "Modelo two_stage. Fuentes: %s",
            ", ".join(config.retrieval.enabled_sources()),
        )
    else:
        logger.info("Modelo %s", config.model)
    logger.info(
        "Comando %s (tune=%s, seed=%s, label_gain=%s, tune_on_example_users=%s). Log en %s",
        command,
        tune,
        config.random_seed,
        config.ranker.label_gain,
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
    seed: SeedOption = None,
    note: NoteOption = "",
    label_gain: LabelGainOption = None,
    model: ModelOption = None,
) -> None:
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
    seed: SeedOption = None,
    note: NoteOption = ...,
    label_gain: LabelGainOption = None,
    model: ModelOption = None,
    two_tower_source: TwoTowerSourceOption = None,
    ease_source: EaseSourceOption = None,
    tune_on_example_users: TuneOnExampleUsersOption = None,
) -> None:
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
        ease_source=ease_source,
        tune_on_example_users=tune_on_example_users,
    )


@app.command()
def submit(
    db: DbOption = Path("data.db"),
    example: ExampleOption = Path("ejemplo.csv"),
    output: OutputOption = Path("submission.csv"),
    artifacts: ArtifactsOption = Path("artifacts"),
    tune: TuneOption = False,
    seed: SeedOption = None,
    note: NoteOption = ...,
    label_gain: LabelGainOption = None,
    model: ModelOption = None,
    two_tower_source: TwoTowerSourceOption = None,
    ease_source: EaseSourceOption = None,
    tune_on_example_users: TuneOnExampleUsersOption = None,
) -> None:
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
        ease_source=ease_source,
        tune_on_example_users=tune_on_example_users,
    )


@app.command(name="all")
def run_all(
    db: DbOption = Path("data.db"),
    example: ExampleOption = Path("ejemplo.csv"),
    output: OutputOption = Path("submission.csv"),
    artifacts: ArtifactsOption = Path("artifacts"),
    tune: TuneOption = False,
    seed: SeedOption = None,
    note: NoteOption = ...,
    label_gain: LabelGainOption = None,
    model: ModelOption = None,
    two_tower_source: TwoTowerSourceOption = None,
    ease_source: EaseSourceOption = None,
    tune_on_example_users: TuneOnExampleUsersOption = None,
) -> None:
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
        ease_source=ease_source,
        tune_on_example_users=tune_on_example_users,
    )


if __name__ == "__main__":
    app()
