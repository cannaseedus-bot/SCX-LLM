import { sha256 } from './merkle';

export type Scxq2ShardDescriptor = {
  id: number;
  url: string;
  expectedHashHex: string;
};

export type Scxq2ShardHeader = {
  id: number;
  offset: number;
  size: number;
  merkleHash: Uint8Array;
};

export function hexToBytes(hex: string): Uint8Array {
  const normalized = hex.trim().toLowerCase();
  if (normalized.length % 2 !== 0) {
    throw new Error('hex string must be even length');
  }

  const bytes = new Uint8Array(normalized.length / 2);
  for (let i = 0; i < normalized.length; i += 2) {
    bytes[i / 2] = Number.parseInt(normalized.slice(i, i + 2), 16);
  }

  return bytes;
}

export function bytesToHex(bytes: ArrayBuffer | Uint8Array): string {
  const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return Array.from(source)
    .map((v) => v.toString(16).padStart(2, '0'))
    .join('');
}

export async function computeSha256Hex(buffer: ArrayBuffer): Promise<string> {
  return bytesToHex(await sha256(buffer));
}

export function parseShardHeader(buffer: ArrayBuffer): Scxq2ShardHeader {
  if (buffer.byteLength < 44) {
    throw new Error('invalid shard: expected at least 44-byte header');
  }

  const view = new DataView(buffer);
  const hash = new Uint8Array(buffer, 12, 32);

  return {
    id: view.getUint32(0, true),
    offset: view.getUint32(4, true),
    size: view.getUint32(8, true),
    merkleHash: new Uint8Array(hash),
  };
}

export function concatBuffers(buffers: ArrayBuffer[]): ArrayBuffer {
  const total = buffers.reduce((acc, b) => acc + b.byteLength, 0);
  const out = new Uint8Array(total);
  let cursor = 0;

  for (const buffer of buffers) {
    const bytes = new Uint8Array(buffer);
    out.set(bytes, cursor);
    cursor += bytes.byteLength;
  }

  return out.buffer;
}

export async function loadShard(url: string, expectedHashHex: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch shard ${url}: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  const digest = await computeSha256Hex(buffer);
  if (digest !== expectedHashHex.toLowerCase()) {
    throw new Error(`shard integrity failure for ${url}`);
  }

  return buffer;
}

export async function streamModel(shards: Scxq2ShardDescriptor[]): Promise<ArrayBuffer> {
  const ordered = [...shards].sort((a, b) => a.id - b.id);
  const assembled: ArrayBuffer[] = [];

  for (const shard of ordered) {
    assembled.push(await loadShard(shard.url, shard.expectedHashHex));
  }

  return concatBuffers(assembled);
}

export async function streamShard(name: string): Promise<string> {
  const url = `model/shards/${name}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to stream shard: ${name}`);
  }

  return `streamed:${name}`;
}
