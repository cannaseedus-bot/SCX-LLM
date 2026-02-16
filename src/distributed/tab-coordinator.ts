export function deterministicMerge(values: number[][]): number[] {
  return values.flat().sort((a, b) => a - b);
}
