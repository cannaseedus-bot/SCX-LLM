import { describe, expect, it } from 'vitest';
import { decode } from '../src/tokenizer/decoder';
import { encode } from '../src/tokenizer/encoder';

describe('tokenizer', () => {
  it('roundtrips by token lengths', () => {
    const input = 'alpha beta';
    const encoded = encode(input);
    const decoded = decode(encoded);

    expect(decoded).toBe('xxxxx xxxx');
  });
});
