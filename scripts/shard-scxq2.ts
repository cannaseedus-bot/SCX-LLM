import { createHash } from 'node:crypto';

export type Scxq2ShardMeta = {
  id: number;
  file: string;
  offset: number;
  size: number;
  hashHex: string;
};

export function sha256Hex(buffer: ArrayBuffer): string {
  return createHash('sha256')
    .update(Buffer.from(buffer))
    .digest('hex');
}

export function buildShardHeader(id: number, offset: number, payload: Uint8Array): Uint8Array {
  const header = new Uint8Array(44);
  const view = new DataView(header.buffer);
  const hash = createHash('sha256').update(payload).digest();

  view.setUint32(0, id, true);
  view.setUint32(4, offset, true);
  view.setUint32(8, payload.byteLength, true);
  header.set(hash, 12);

  return header;
}

export function shardScxq2(experts: number): string[] {
  return Array.from({ length: experts }, (_, i) => `expert_${i}.scxq2`);
}

export function buildShardMetadata(buffers: ArrayBuffer[]): Scxq2ShardMeta[] {
  let offset = 0;

  return buffers.map((buffer, id) => {
    const size = buffer.byteLength;
    const meta: Scxq2ShardMeta = {
      id,
      file: `expert_${id}.scxq2`,
      offset,
      size,
      hashHex: sha256Hex(buffer),
    };

    offset += size;
    return meta;
  });
}
