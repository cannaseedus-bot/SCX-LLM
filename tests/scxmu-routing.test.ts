import { describe, expect, it } from 'vitest';
import { SCXMU_OPCODES, deterministicMerge, deterministicTopK } from '../src/distributed/tab-coordinator';

describe('scxmu routing primitives', () => {
  it('exposes stable execution opcodes', () => {
    expect(SCXMU_OPCODES.ROUTE).toBe(0x20);
    expect(SCXMU_OPCODES.SELECT).toBe(0x21);
    expect(SCXMU_OPCODES.DISPATCH).toBe(0x22);
    expect(SCXMU_OPCODES.COMBINE).toBe(0x23);
  });

  it('deterministicTopK tie-breaks by lower expert index', () => {
    const scores = new Float32Array([0.8, 0.9, 0.9, 0.4]);
    const top2 = deterministicTopK(scores, 2);

    expect(top2).toEqual([
      { expertId: 1, score: 0.9 },
      { expertId: 2, score: 0.9 },
    ]);
  });

  it('deterministicMerge sorts experts before weighted combine', () => {
    const out = deterministicMerge([
      { expertId: 5, score: 0.25, output: new Float32Array([4, 4]) },
      { expertId: 1, score: 0.75, output: new Float32Array([2, 2]) },
    ]);

    expect(Array.from(out)).toEqual([2.5, 2.5]);
  });
});
