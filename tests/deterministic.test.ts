import { describe, expect, it } from 'vitest';
import { layerHash } from '../src/runtime/replay-verifier';

describe('deterministic replay', () => {
  it('returns same hash for same inputs', async () => {
    const input = new Uint8Array([1, 2, 3]).buffer;
    const router = new Uint8Array([4, 5]).buffer;
    const selected = new Uint32Array([2, 7]);
    const output = new Uint8Array([6, 9]).buffer;

    const first = await layerHash(input, router, selected, output);
    const second = await layerHash(input, router, selected, output);

    expect(new Uint8Array(first)).toEqual(new Uint8Array(second));
  });
});
