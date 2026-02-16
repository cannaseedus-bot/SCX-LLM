export function quantizeInt4(values: Float32Array): Int8Array {
  const out = new Int8Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    out[i] = Math.max(-8, Math.min(7, Math.round(values[i])));
  }
  return out;
}

export function dequantizeInt4(values: Int8Array): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    out[i] = values[i];
  }
  return out;
}

export function relativeError(reference: Float32Array, predicted: Float32Array): number {
  if (reference.length !== predicted.length) {
    throw new Error('Mismatched vector lengths');
  }

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < reference.length; i += 1) {
    const diff = reference[i] - predicted[i];
    numerator += diff * diff;
    denominator += reference[i] * reference[i];
  }

  if (denominator === 0) {
    return 0;
  }

  return Math.sqrt(numerator) / Math.sqrt(denominator);
}
