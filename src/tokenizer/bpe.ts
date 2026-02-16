export function splitBpe(text: string): string[] {
  return text.trim().split(/\s+/).filter(Boolean);
}
