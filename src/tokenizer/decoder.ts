export function decode(tokens: number[]): string {
  return tokens.map((n) => 'x'.repeat(n)).join(' ');
}
