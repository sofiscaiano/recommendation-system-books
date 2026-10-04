import type { SessionRating } from "./api";

function storageKey(idLector: string): string {
  return `quelibroleo:ratings:${idLector}`;
}

export function loadRatings(idLector: string): SessionRating[] {
  const raw = localStorage.getItem(storageKey(idLector));
  if (!raw) {
    return [];
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.flatMap((item) => {
      if (!item || typeof item !== "object") {
        return [];
      }
      const idLibro = "id_libro" in item ? item.id_libro : undefined;
      const rating = "rating" in item ? item.rating : undefined;
      if (typeof idLibro !== "string" || typeof rating !== "number") {
        return [];
      }
      if (rating < 1 || rating > 10) {
        return [];
      }
      return [{ id_libro: idLibro, rating }];
    });
  } catch {
    return [];
  }
}

export function saveRatings(idLector: string, ratings: SessionRating[]): void {
  localStorage.setItem(storageKey(idLector), JSON.stringify(ratings));
}

export function mergeRatings(
  current: SessionRating[],
  drafts: Record<string, string>,
): SessionRating[] {
  const next = new Map(current.map((item) => [item.id_libro, item.rating]));
  for (const [idLibro, value] of Object.entries(drafts)) {
    const rating = Number(value);
    if (rating >= 1 && rating <= 10) {
      next.set(idLibro, rating);
    }
  }
  return [...next.entries()].map(([id_libro, rating]) => ({ id_libro, rating }));
}
