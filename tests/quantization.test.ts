import { describe, expect, it } from 'vitest';
import { dequantizeInt4, quantizeInt4, relativeError } from '../scripts/quantize-int4';

describe('quantization', () => {
  it('clamps to int4 range', () => {
    const values = new Float32Array([-20, -8.1, -2.3, 0, 4.7, 9.9]);
    expect(Array.from(quantizeInt4(values))).toEqual([-8, -8, -2, 0, 5, 7]);
  });

  it('keeps dequant relative error under 5% for reference-aligned vectors', () => {
    const reference = new Float32Array([-8, -4, -1, 0, 2, 6, 7]);
    const quantized = quantizeInt4(reference);
    const dequantized = dequantizeInt4(quantized);

    const error = relativeError(reference, dequantized);
    expect(error).toBeLessThan(0.05);
  });
});
