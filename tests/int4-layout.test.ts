import { describe, expect, it } from 'vitest';
import { decodeSignedInt4, dequantInt4Value, unpackInt4Pair } from '../src/runtime/int4';

describe('int4 canonical layout', () => {
  it('decodes signed int4 nibble range [-8, 7]', () => {
    expect(decodeSignedInt4(0x0)).toBe(0);
    expect(decodeSignedInt4(0x7)).toBe(7);
    expect(decodeSignedInt4(0x8)).toBe(-8);
    expect(decodeSignedInt4(0xf)).toBe(-1);
  });

  it('unpacks two int4 values per byte (high then low)', () => {
    const [a, b] = unpackInt4Pair(0xe3); // high=14(-2), low=3
    expect(a).toBe(-2);
    expect(b).toBe(3);
  });

  it('applies canonical dequant rule (value - zp) * scale', () => {
    const byte = 0x3e; // high=3, low=14(-2)
    expect(dequantInt4Value(byte, true, 0.5, 1)).toBe(1.0); // (3-1)*0.5
    expect(dequantInt4Value(byte, false, 0.5, 1)).toBe(-1.5); // (-2-1)*0.5
  });
});
