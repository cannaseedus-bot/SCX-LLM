function append(target: Uint8Array, source: Uint8Array, offset: number): number {
  target.set(source, offset);
  return offset + source.byteLength;
}

function toBytes(buffer: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (ArrayBuffer.isView(buffer)) {
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }

  return new Uint8Array(buffer);
}

export function toHex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export async function routeHash(input: ArrayBuffer, routerWeights: ArrayBuffer, layerId: number): Promise<ArrayBuffer> {
  const layer = new Uint32Array([layerId]);
  const layerBytes = toBytes(layer);
  const combined = new Uint8Array(input.byteLength + routerWeights.byteLength + layerBytes.byteLength);

  let cursor = 0;
  cursor = append(combined, toBytes(input), cursor);
  cursor = append(combined, toBytes(routerWeights), cursor);
  append(combined, layerBytes, cursor);

  return crypto.subtle.digest('SHA-256', combined);
}

export async function layerHash(
  input: ArrayBuffer,
  router: ArrayBuffer,
  selected: Uint32Array,
  output: ArrayBuffer,
): Promise<ArrayBuffer> {
  const selectedBytes = toBytes(selected);
  const combined = new Uint8Array(input.byteLength + router.byteLength + selectedBytes.byteLength + output.byteLength);

  let cursor = 0;
  cursor = append(combined, toBytes(input), cursor);
  cursor = append(combined, toBytes(router), cursor);
  cursor = append(combined, selectedBytes, cursor);
  append(combined, toBytes(output), cursor);

  return crypto.subtle.digest('SHA-256', combined);
}

export async function verifyLayerDeterminism(
  baselineHash: ArrayBuffer,
  input: ArrayBuffer,
  router: ArrayBuffer,
  selected: Uint32Array,
  output: ArrayBuffer,
): Promise<boolean> {
  const current = await layerHash(input, router, selected, output);
  return toHex(current) === toHex(baselineHash);
}

export type ReplayTraceEntry = {
  layer: number;
  hashHex: string;
  kind: 'route' | 'layer';
};

export class ReplayTrace {
  private readonly entries: ReplayTraceEntry[] = [];

  public async recordRoute(layer: number, input: ArrayBuffer, routerWeights: ArrayBuffer): Promise<ReplayTraceEntry> {
    const hash = await routeHash(input, routerWeights, layer);
    const entry = { layer, hashHex: toHex(hash), kind: 'route' as const };
    this.entries.push(entry);
    return entry;
  }

  public async record(
    layer: number,
    input: ArrayBuffer,
    router: ArrayBuffer,
    selected: Uint32Array,
    output: ArrayBuffer,
  ): Promise<ReplayTraceEntry> {
    const hash = await layerHash(input, router, selected, output);
    const entry = { layer, hashHex: toHex(hash), kind: 'layer' as const };
    this.entries.push(entry);
    return entry;
  }

  public all(): ReplayTraceEntry[] {
    return [...this.entries];
  }

  public compare(expected: ReplayTraceEntry[]): { ok: boolean; mismatchLayer?: number } {
    if (expected.length !== this.entries.length) {
      return { ok: false, mismatchLayer: Math.min(expected.length, this.entries.length) };
    }

    for (let i = 0; i < expected.length; i += 1) {
      if (
        expected[i].layer !== this.entries[i].layer
        || expected[i].hashHex !== this.entries[i].hashHex
        || expected[i].kind !== this.entries[i].kind
      ) {
        return { ok: false, mismatchLayer: expected[i].layer };
      }
    }

    return { ok: true };
  }
}
