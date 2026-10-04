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
import { useState } from "react";

import { fetchRecommendations, type Book } from "./api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { LoginForm } from "@/components/login-form";
import { Skeleton } from "@/components/ui/skeleton";
import { RatingSelect } from "./RatingSelect";
import { loadRatings, mergeRatings, saveRatings } from "./ratings";

function Cover({ src, title }: { src: string; title: string }) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return (
      <div className="flex aspect-[2/3] w-28 shrink-0 items-center justify-center self-start bg-muted px-2 text-center text-xs text-muted-foreground sm:w-36">
        Sin tapa
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={`Tapa de ${title}`}
      className="aspect-[2/3] w-28 shrink-0 self-start bg-muted object-cover sm:w-36"
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
      <dt className="shrink-0 text-muted-foreground">{label}</dt>
      <dd className="min-w-0">{value}</dd>
    </div>
  );
}

function BookCard({
  book,
  draft,
  onDraft,
}: {
  book: Book;
  draft: string | null;
  onDraft: (value: string | null) => void;
}) {
  return (
    <Card className="h-full gap-0 py-0">
      <div className="flex min-h-56 flex-1">
        <Cover src={book.img_src} title={book.titulo} />
        <div className="flex min-w-0 flex-1 flex-col gap-3 py-4">
          <CardHeader className="px-4">
            <CardTitle className="line-clamp-3">{book.titulo}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-1 flex-col gap-3 px-4">
            <dl className="space-y-1 text-sm">
              <Attribute label="Autor" value={book.autor} />
              <Attribute label="Género" value={book.genero} />
              <Attribute label="Editorial" value={book.editorial} />
              <Attribute label="Año" value={book.anio_edicion} />
              <Attribute label="ISBN" value={book.isbn} />
            </dl>
            {book.resumen ? (
              <p className="line-clamp-4 text-xs text-muted-foreground">{book.resumen}</p>
            ) : null}
          </CardContent>
        </div>
      </div>
      <CardFooter className="justify-end">
        <RatingSelect value={draft} onChange={onDraft} />
      </CardFooter>
    </Card>
  );
}

function LoginPage() {
  const navigate = useNavigate();

  return (
    <main className="flex min-h-[calc(100svh-3.5rem)] w-full items-center justify-center p-6 md:p-10">
      <div className="w-full max-w-sm">
        <LoginForm
          onSubmit={(idLector) => {
            void navigate({ to: "/recomendar/$idLector", params: { idLector } });
          }}
        />
      </div>
    </main>
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
    <main className="mx-auto max-w-6xl px-6 py-6">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            Recomendaciones
          </h1>
          <p className="truncate text-sm text-muted-foreground">{idLector}</p>
        </div>
        <Button variant="outline" nativeButton={false} render={<Link to="/" />}>
          Cambiar lector
        </Button>
      </div>
      {coldStart ? (
        <Alert className="mb-4">
          <AlertTitle>Lector nuevo</AlertTitle>
          <AlertDescription>
            Este lector no está en la base. La lista es la de popularidad hasta que
            puntúes alguno.
          </AlertDescription>
        </Alert>
      ) : null}
      {submitted.length > 0 ? (
        <Badge variant="secondary" className="mb-4">
          {submitted.length} puntaje{submitted.length === 1 ? "" : "s"} de esta sesión
        </Badge>
      ) : null}
      {query.isPending ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }, (_, index) => (
            <Skeleton key={index} className="h-72 rounded-xl" />
          ))}
        </div>
      ) : null}
      {query.isError ? (
        <Alert variant="destructive">
          <AlertTitle>No se pudieron cargar las recomendaciones</AlertTitle>
          <AlertDescription>
            {query.error instanceof Error
              ? query.error.message
              : "Probá de nuevo en un momento."}
          </AlertDescription>
        </Alert>
      ) : null}
      {query.data ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {query.data.libros.map((book) => (
            <BookCard
              key={book.id_libro}
              book={book}
              draft={drafts[book.id_libro] ?? null}
              onDraft={(value) =>
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
          ))}
        </div>
      ) : null}
      <div className="mt-4 flex justify-end">
        <Button type="button" disabled={!hasDraft || query.isFetching} onClick={onMore}>
          Más recomendaciones
        </Button>
      </div>
    </main>
  );
}

const rootRoute = createRootRoute({
  component: () => (
    <div className="min-h-svh">
      <header className="border-b">
        <div className="mx-auto flex h-14 max-w-6xl items-center px-6">
          <span className="font-heading text-sm font-semibold">Qué Libro Leo</span>
        </div>
      </header>
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
