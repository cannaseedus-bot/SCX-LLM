function append(target: Uint8Array, source: Uint8Array, offset: number): number {
  target.set(source, offset);
  return offset + source.byteLength;
}

export async function layerHash(
  input: ArrayBuffer,
  router: ArrayBuffer,
  selected: Uint32Array,
  output: ArrayBuffer,
): Promise<ArrayBuffer> {
  const selectedBytes = new Uint8Array(selected.buffer, selected.byteOffset, selected.byteLength);
  const combined = new Uint8Array(input.byteLength + router.byteLength + selectedBytes.byteLength + output.byteLength);

  let cursor = 0;
  cursor = append(combined, new Uint8Array(input), cursor);
  cursor = append(combined, new Uint8Array(router), cursor);
  cursor = append(combined, selectedBytes, cursor);
  append(combined, new Uint8Array(output), cursor);

  return crypto.subtle.digest('SHA-256', combined);
}
