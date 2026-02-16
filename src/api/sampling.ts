export function sampleGreedy(logits: number[]): number {
  let bestIndex = 0;
  for (let i = 1; i < logits.length; i += 1) {
    if (logits[i] > logits[bestIndex]) {
      bestIndex = i;
    }
  }
  return bestIndex;
}
