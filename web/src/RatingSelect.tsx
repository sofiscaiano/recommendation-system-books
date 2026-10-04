import { Select } from "@base-ui/react/select";

const SCORES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] as const;

type RatingSelectProps = {
  value: string | null;
  onChange: (value: string | null) => void;
};

export function RatingSelect({ value, onChange }: RatingSelectProps) {
  return (
    <Select.Root
      value={value}
      onValueChange={(next) => onChange(next)}
      items={SCORES.map((score) => ({ value: score, label: score }))}
    >
      <Select.Label className="text-sm">Mi puntuación</Select.Label>
      <Select.Trigger className="rounded border bg-white px-3 py-1 text-sm">
        <Select.Value placeholder="—" />
      </Select.Trigger>
      <Select.Portal>
        <Select.Positioner className="z-10" sideOffset={4}>
          <Select.Popup className="max-h-60 overflow-y-auto rounded border bg-white p-1 shadow-lg">
            <Select.List>
              {SCORES.map((score) => (
                <Select.Item
                  key={score}
                  value={score}
                  className="cursor-pointer rounded px-3 py-1 text-sm data-[highlighted]:bg-teal-50"
                >
                  <Select.ItemText>{score}</Select.ItemText>
                </Select.Item>
              ))}
            </Select.List>
          </Select.Popup>
        </Select.Positioner>
      </Select.Portal>
    </Select.Root>
  );
}
