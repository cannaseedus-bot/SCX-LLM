import { describe, expect, it } from 'vitest';
import { ReplayTrace, layerHash, routeHash, verifyLayerDeterminism } from '../src/runtime/replay-verifier';

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

  it('produces deterministic route hash', async () => {
    const input = new Uint8Array([3, 3, 3]).buffer;
    const routerWeights = new Uint8Array([9, 1]).buffer;

    const first = await routeHash(input, routerWeights, 4);
    const second = await routeHash(input, routerWeights, 4);
    const changedLayer = await routeHash(input, routerWeights, 5);

    expect(new Uint8Array(first)).toEqual(new Uint8Array(second));
    expect(new Uint8Array(first)).not.toEqual(new Uint8Array(changedLayer));
  });

  it('detects mismatch against baseline hash', async () => {
    const input = new Uint8Array([1, 2]).buffer;
    const router = new Uint8Array([3, 4]).buffer;
    const selected = new Uint32Array([0, 1]);

    const baselineOutput = new Uint8Array([5, 6]).buffer;
    const baseline = await layerHash(input, router, selected, baselineOutput);

    const same = await verifyLayerDeterminism(baseline, input, router, selected, baselineOutput);
    const changed = await verifyLayerDeterminism(
      baseline,
      input,
      router,
      selected,
      new Uint8Array([5, 99]).buffer,
    );

    expect(same).toBe(true);
    expect(changed).toBe(false);
  });

  it('records route+layer trace entries and compares', async () => {
    const trace = new ReplayTrace();
    const input = new Uint8Array([1]).buffer;
    const router = new Uint8Array([2]).buffer;
    const selected = new Uint32Array([1]);
    const output = new Uint8Array([3]).buffer;

    const routeEntry = await trace.recordRoute(0, input, router);
    const layerEntry = await trace.record(0, input, router, selected, output);
    const comparison = trace.compare([
      { layer: 0, hashHex: routeEntry.hashHex, kind: 'route' },
      { layer: 0, hashHex: layerEntry.hashHex, kind: 'layer' },
    ]);

    expect(comparison.ok).toBe(true);
  });
});
