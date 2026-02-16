import { describe, expect, it } from 'vitest';
import { sha256 } from '../src/runtime/merkle';

describe('merkle primitives', () => {
  it('detects tampering by hash mismatch', async () => {
    const clean = await sha256(new TextEncoder().encode('section-a').buffer);
    const tampered = await sha256(new TextEncoder().encode('section-b').buffer);

    expect(new Uint8Array(clean)).not.toEqual(new Uint8Array(tampered));
  });
});
