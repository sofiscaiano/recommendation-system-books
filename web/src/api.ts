export type Book = {
  id_libro: string;
  titulo: string;
  autor: string;
  genero: string;
  editorial: string;
  resumen: string;
  img_src: string;
  rank: number;
};

export type RecommendationResponse = {
  id_lector: string;
  conocido: boolean;
  libros: Book[];
};

export type SessionRating = {
  id_libro: string;
  rating: number;
};

function apiUrl(): string {
  const configured = import.meta.env.VITE_API_URL;
  return (configured && configured.trim()) || "http://localhost:8000";
}

export async function fetchRecommendations(
  idLector: string,
  valoraciones: SessionRating[],
): Promise<RecommendationResponse> {
  const response = await fetch(`${apiUrl()}/recomendaciones`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id_lector: idLector, valoraciones }),
  });
  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    const detail =
      payload &&
      typeof payload === "object" &&
      "detail" in payload &&
      typeof payload.detail === "string"
        ? payload.detail
        : "No se pudieron cargar las recomendaciones";
    throw new Error(detail);
  }
  return response.json() as Promise<RecommendationResponse>;
}
