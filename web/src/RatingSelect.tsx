import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

const SCORES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] as const;

type RatingSelectProps = {
  value: string | null;
  onChange: (value: string | null) => void;
};

export function RatingSelect({ value, onChange }: RatingSelectProps) {
  return (
    <Label className="font-normal text-muted-foreground">
      Mi puntuación
      <Select
        value={value}
        onValueChange={(next) => onChange(next)}
        items={SCORES.map((score) => ({ value: score, label: score }))}
      >
        <SelectTrigger className="w-20" aria-label="Mi puntuación">
          <SelectValue placeholder="—" />
        </SelectTrigger>
        <SelectContent align="end">
          {SCORES.map((score) => (
            <SelectItem key={score} value={score}>
              {score}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Label>
  );
}
