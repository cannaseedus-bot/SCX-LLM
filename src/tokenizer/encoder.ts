import { splitBpe } from './bpe';

export function encode(text: string): number[] {
  return splitBpe(text).map((token) => token.length);
}
