"""API de un proceso: el modelo se carga al arrancar y cada request solo predice."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from quelibroleo.serving import ServingIndex, UnknownBooksError, load_serving_index

LOGGER = logging.getLogger("uvicorn.error")
_LOCK = threading.Lock()
_INDEX: ServingIndex | None = None


class RatingIn(BaseModel):
    id_libro: str = Field(min_length=1)
    rating: int = Field(ge=1, le=10)

    @field_validator("id_libro")
    @classmethod
    def strip_book(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("id_libro vacío")
        return text


class RecommendIn(BaseModel):
    id_lector: str = Field(min_length=1)
    valoraciones: list[RatingIn] = Field(default_factory=list)

    @field_validator("id_lector")
    @classmethod
    def strip_user(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("ID de lector requerido")
        return text


class BookOut(BaseModel):
    id_libro: str
    titulo: str
    autor: str
    genero: str
    editorial: str
    anio_edicion: str
    isbn: str
    resumen: str
    img_src: str
    rank: int


class RecommendOut(BaseModel):
    id_lector: str
    conocido: bool
    libros: list[BookOut]


def _origins() -> list[str]:
    raw = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _INDEX
    artifact = Path(os.environ.get("SERVING_ARTIFACT", "artifacts/serving.joblib"))
    db_path = Path(os.environ.get("SERVING_DB", "data.db"))
    LOGGER.info("Cargando artefacto %s", artifact)
    _INDEX = load_serving_index(artifact, db_path)
    yield
    _INDEX = None


app = FastAPI(title="Qué Libro Leo", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ready": _INDEX is not None}


@app.post("/recomendaciones", response_model=RecommendOut)
def recomendaciones(body: RecommendIn) -> dict[str, object]:
    if _INDEX is None:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado")
    ratings: dict[str, int] = {}
    for rating in body.valoraciones:
        ratings[rating.id_libro] = rating.rating
    started = time.perf_counter()
    try:
        with _LOCK:
            result = _INDEX.recommend(body.id_lector, list(ratings.items()))
    except UnknownBooksError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Libros fuera del catálogo: {', '.join(exc.book_ids)}",
        ) from exc
    LOGGER.info(
        "Recomendación de %s en %.3f s",
        body.id_lector,
        time.perf_counter() - started,
    )
    return result
