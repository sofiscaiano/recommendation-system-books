import { cn } from "cn";
import { type FormEvent, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";

type LoginFormProps = Omit<React.ComponentProps<"div">, "onSubmit"> & {
  onSubmit: (idLector: string) => void;
};

export function LoginForm({ className, onSubmit, ...props }: LoginFormProps) {
  const [idLector, setIdLector] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const id = idLector.trim();
    if (!id) {
      return;
    }
    onSubmit(id);
  }

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card>
        <CardHeader>
          <CardTitle>Entrar</CardTitle>
          <CardDescription>
            Ingresá el usuario para ver sus recomendaciones.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="username">Usuario</FieldLabel>
                <Input
                  id="username"
                  name="username"
                  value={idLector}
                  onChange={(event) => setIdLector(event.target.value)}
                  minLength={1}
                  maxLength={80}
                  autoComplete="username"
                  required
                />
              </Field>
              <Field>
                <Button type="submit">Entrar</Button>
              </Field>
            </FieldGroup>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
