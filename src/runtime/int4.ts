export function decodeSignedInt4(nibble: number): number {
  const v = nibble & 0x0f;
  return v >= 8 ? v - 16 : v;
}

export function unpackInt4Pair(byteValue: number): [number, number] {
  const high = decodeSignedInt4((byteValue >> 4) & 0x0f);
  const low = decodeSignedInt4(byteValue & 0x0f);
  return [high, low];
}

export function dequantInt4Value(byteValue: number, isHigh: boolean, scale: number, zeroPoint: number): number {
  const nibble = isHigh ? (byteValue >> 4) & 0x0f : byteValue & 0x0f;
  return (decodeSignedInt4(nibble) - zeroPoint) * scale;
}
