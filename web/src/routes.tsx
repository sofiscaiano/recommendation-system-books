import { Button } from "@base-ui/react/button";
import { Input } from "@base-ui/react/input";
import { useQuery } from "@tanstack/react-query";
import {
  createRootRoute,
  createRoute,
  createRouter,
  Link,
  Outlet,
  useNavigate,
  useParams,
} from "@tanstack/react-router";
import { type FormEvent, useState } from "react";

import { fetchRecommendations } from "./api";
import { RatingSelect } from "./RatingSelect";
import { loadRatings, mergeRatings, saveRatings } from "./ratings";

function Cover({ src, title }: { src: string; title: string }) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return (
      <div className="flex h-56 w-36 shrink-0 items-center justify-center bg-slate-100 px-2 text-center text-xs text-slate-500">
        Sin tapa
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={`Tapa de ${title}`}
      className="h-56 w-36 shrink-0 bg-slate-100 object-cover"
      onError={() => setFailed(true)}
    />
  );
}

function Attribute({ label, value }: { label: string; value: string }) {
  if (!value) {
    return null;
  }
  return (
    <div className="flex gap-2">
      <dt className="shrink-0 font-medium text-slate-500">{label}</dt>
      <dd className="min-w-0">{value}</dd>
    </div>
  );
}

function LoginPage() {
  const navigate = useNavigate();
  const [idLector, setIdLector] = useState("");

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const id = idLector.trim();
    if (!id) {
      return;
    }
    void navigate({ to: "/recomendar/$idLector", params: { idLector: id } });
  }

  return (
    <form
      onSubmit={onSubmit}
      className="mx-auto mt-16 w-full max-w-md rounded border bg-white p-4 shadow-xl"
    >
      <h1 className="mb-4 text-xl font-bold">Entrar al sistema</h1>
      <label className="mb-4 flex flex-col gap-1 text-sm">
        Usuario
        <Input
          value={idLector}
          onChange={(event) => setIdLector(event.target.value)}
          minLength={1}
          maxLength={80}
          required
          className="rounded border p-2"
        />
      </label>
      <div className="flex justify-end">
        <Button
          type="submit"
          className="rounded bg-teal-500 px-4 py-2 font-bold text-white hover:bg-teal-700"
        >
          Entrar
        </Button>
      </div>
    </form>
  );
}

function RecommendPage() {
  const { idLector } = useParams({ from: "/recomendar/$idLector" });
  const [submitted, setSubmitted] = useState(() => loadRatings(idLector));
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const query = useQuery({
    queryKey: ["recomendaciones", idLector, submitted],
    queryFn: () => fetchRecommendations(idLector, submitted),
  });

  function onMore() {
    const next = mergeRatings(submitted, drafts);
    saveRatings(idLector, next);
    setSubmitted(next);
    setDrafts({});
  }

  const hasDraft = Object.values(drafts).some((value) => Number(value) >= 1);
  const coldStart = query.data && !query.data.conocido && submitted.length === 0;

  return (
    <main className="mx-auto max-w-6xl px-6 py-4">
      <div className="mb-4 flex items-start justify-between gap-4">
        <h1 className="text-lg font-bold">Recomendaciones para {idLector}</h1>
        <Link to="/" className="text-sm text-teal-700 underline">
          Cambiar lector
        </Link>
      </div>
      {coldStart ? (
        <p className="mb-4 rounded border border-teal-200 bg-teal-50 p-3 text-sm">
          Este lector no está en la base. La lista es la de popularidad hasta que
          puntúes alguno.
        </p>
      ) : null}
      {submitted.length > 0 ? (
        <p className="mb-4 text-sm text-slate-600">
          Incluye {submitted.length} puntaje{submitted.length === 1 ? "" : "s"} de
          esta sesión.
        </p>
      ) : null}
      {query.isPending ? <p>Armando recomendaciones…</p> : null}
      {query.isError ? (
        <p className="text-red-700">
          {query.error instanceof Error
            ? query.error.message
            : "No se pudieron cargar las recomendaciones"}
        </p>
      ) : null}
      {query.data ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {query.data.libros.map((book) => (
            <article
              key={book.id_libro}
              className="flex overflow-hidden rounded border bg-white shadow-sm"
            >
              <Cover src={book.img_src} title={book.titulo} />
              <div className="flex min-w-0 flex-1 flex-col p-3">
                <h2 className="font-semibold">{book.titulo}</h2>
                <dl className="mt-2 space-y-1 text-sm text-slate-600">
                  <Attribute label="Autor" value={book.autor} />
                  <Attribute label="Género" value={book.genero} />
                  <Attribute label="Editorial" value={book.editorial} />
                  <Attribute label="Año" value={book.anio_edicion} />
                  <Attribute label="ISBN" value={book.isbn} />
                </dl>
                {book.resumen ? (
                  <p className="mt-2 line-clamp-5 text-xs text-slate-700">{book.resumen}</p>
                ) : null}
                <div className="mt-auto flex items-center justify-end gap-2 pt-3">
                  <RatingSelect
                    value={drafts[book.id_libro] ?? null}
                    onChange={(value) =>
                      setDrafts((current) => {
                        const next = { ...current };
                        if (value) {
                          next[book.id_libro] = value;
                        } else {
                          delete next[book.id_libro];
                        }
                        return next;
                      })
                    }
                  />
                </div>
              </div>
            </article>
          ))}
        </div>
      ) : null}
      <div className="mt-4 flex justify-end">
        <Button
          type="button"
          disabled={!hasDraft || query.isFetching}
          onClick={onMore}
          className="rounded bg-teal-400 px-4 py-2 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          Más recomendaciones
        </Button>
      </div>
    </main>
  );
}

const rootRoute = createRootRoute({
  component: () => (
    <div className="min-h-screen">
      <Outlet />
    </div>
  ),
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: LoginPage,
});

const recommendRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/recomendar/$idLector",
  component: RecommendPage,
});

const routeTree = rootRoute.addChildren([indexRoute, recommendRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
