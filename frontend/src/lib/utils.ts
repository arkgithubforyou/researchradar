import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}

/** Truncate text to maxLen chars, adding ellipsis. */
export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "\u2026";
}

/** Format a large number with K/M suffix. */
export function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

/** Color palette for charts. */
export const CHART_COLORS = [
  "#4263eb", // brand-700
  "#f76707", // orange
  "#37b24d", // green
  "#ae3ec9", // purple
  "#1098ad", // teal
  "#e8590c", // deep orange
  "#74b816", // lime
  "#d6336c", // pink
  "#495057", // gray
  "#2b8a3e", // dark green
] as const;

/** All known venues for filter dropdowns. */
export const VENUES = [
  { value: "acl", label: "ACL" },
  { value: "emnlp", label: "EMNLP" },
  { value: "naacl", label: "NAACL" },
  { value: "findings-acl", label: "Findings of ACL" },
  { value: "findings-emnlp", label: "Findings of EMNLP" },
  { value: "findings-naacl", label: "Findings of NAACL" },
  { value: "coling", label: "COLING" },
  { value: "eacl", label: "EACL" },
  { value: "tacl", label: "TACL" },
  { value: "cl", label: "CL" },
  { value: "semeval", label: "SemEval" },
  { value: "conll", label: "CoNLL" },
  { value: "workshop", label: "Workshop" },
] as const;

const VENUE_LABEL_MAP: Record<string, string> = Object.fromEntries(
  VENUES.map((v) => [v.value, v.label]),
);

/** Venue display name lookup. */
export function venueLabel(venue: string | null): string {
  if (!venue) return "Unknown";
  return VENUE_LABEL_MAP[venue.toLowerCase()] ?? venue.toUpperCase();
}
