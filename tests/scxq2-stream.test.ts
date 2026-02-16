import { describe, expect, it, vi } from 'vitest';
import { concatBuffers, parseShardHeader, streamModel } from '../src/runtime/scxq2-stream';

function buildHeader(id: number, offset: number, size: number): ArrayBuffer {
  const header = new Uint8Array(44);
  const view = new DataView(header.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, offset, true);
  view.setUint32(8, size, true);
  return header.buffer;
}

describe('scxq2 stream', () => {
  it('parses shard headers', () => {
    const header = parseShardHeader(buildHeader(7, 128, 1024));
    expect(header.id).toBe(7);
    expect(header.offset).toBe(128);
    expect(header.size).toBe(1024);
    expect(header.merkleHash.length).toBe(32);
  });

  it('concats buffers in order', () => {
    const a = new Uint8Array([1, 2]).buffer;
    const b = new Uint8Array([3]).buffer;

    expect(Array.from(new Uint8Array(concatBuffers([a, b])))).toEqual([1, 2, 3]);
  });

  it('streams shards in deterministic id order', async () => {
    const shardA = new Uint8Array([1, 1, 1]).buffer;
    const shardB = new Uint8Array([2, 2]).buffer;

    const digestA = await crypto.subtle.digest('SHA-256', shardA);
    const digestB = await crypto.subtle.digest('SHA-256', shardB);

    const hashA = Buffer.from(new Uint8Array(digestA)).toString('hex');
    const hashB = Buffer.from(new Uint8Array(digestB)).toString('hex');

    const fetchMock = vi.fn(async (url: string) => {
      if (url === '/b') {
        return { ok: true, arrayBuffer: async () => shardB };
      }
      return { ok: true, arrayBuffer: async () => shardA };
    });

    vi.stubGlobal('fetch', fetchMock);

    const out = await streamModel([
      { id: 1, url: '/b', expectedHashHex: hashB },
      { id: 0, url: '/a', expectedHashHex: hashA },
    ]);

    expect(Array.from(new Uint8Array(out))).toEqual([1, 1, 1, 2, 2]);
    vi.unstubAllGlobals();
  });
});
