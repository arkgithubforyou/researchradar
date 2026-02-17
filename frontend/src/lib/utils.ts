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

/** Venue display name cleanup. */
export function venueLabel(venue: string | null): string {
  if (!venue) return "Unknown";
  return venue.toUpperCase();
}
