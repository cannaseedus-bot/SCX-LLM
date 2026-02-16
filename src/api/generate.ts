import { sampleGreedy } from './sampling';

export function generate(logits: number[]): number {
  return sampleGreedy(logits);
}
