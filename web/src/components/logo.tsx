import { cn } from "cn";

type LogoProps = {
  className?: string;
};

export function Logo({ className }: LogoProps) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <img src="/logo.svg" alt="" width={20} height={20} className="size-5 shrink-0" />
      <span className="font-heading text-sm font-semibold">Qué Libro Leo</span>
    </span>
  );
}
