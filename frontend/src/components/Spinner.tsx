import { cn } from "@/lib/utils";

interface SpinnerProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

const SIZES = { sm: "w-4 h-4", md: "w-6 h-6", lg: "w-8 h-8" } as const;

export default function Spinner({ className, size = "md" }: SpinnerProps) {
  return (
    <svg
      className={cn("animate-spin text-brand-600", SIZES[size], className)}
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}
