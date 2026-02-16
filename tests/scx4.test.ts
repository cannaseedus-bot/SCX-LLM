import { describe, expect, it } from 'vitest';
import { loadSCX4, parseScx4Header } from '../src/runtime/scx4-loader';

describe('scx4 loader', () => {
  it('reads version byte', () => {
    const buffer = new Uint8Array([3]).buffer;
    expect(parseScx4Header(buffer).version).toBe(3);
  });

  it('loads valid SCX4 metadata', async () => {
    const model = await loadSCX4();
    expect(model.hidden_size).toBe(640);
    expect(model.layers).toBe(16);
    expect(model.experts).toBe(8);
  });
});
