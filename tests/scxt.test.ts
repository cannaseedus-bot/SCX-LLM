import { describe, expect, it } from 'vitest';
import { SCXT_OPCODES, flatten4DIndex, opcodeValues, parseScxtHeader } from '../src/runtime/scxt';

describe('scxt runtime primitives', () => {
  it('keeps opcode values unique', () => {
    const values = opcodeValues();
    expect(new Set(values).size).toBe(values.length);
    expect(SCXT_OPCODES.ADD).toBe(0x01);
    expect(SCXT_OPCODES.HALT).toBe(0x0f);
  });

  it('flattens 4d index canonically', () => {
    const dims: [number, number, number, number] = [2, 3, 4, 5];
    const idx = flatten4DIndex(1, 2, 3, 4, dims);
    expect(idx).toBe((((1 * 3) + 2) * 4 + 3) * 5 + 4);
  });

  it('parses deterministic header layout', () => {
    const bytes = new Uint8Array(0x20 + 32);
    const view = new DataView(bytes.buffer);

    bytes.set([0x53, 0x43, 0x58, 0x54], 0); // SCXT
    bytes[4] = 1;
    bytes[5] = 1;
    view.setUint16(0x06, 0x0003, true);
    view.setUint32(0x08, 2, true);
    view.setUint32(0x0c, 3, true);
    view.setUint32(0x10, 4, true);
    view.setUint32(0x14, 5, true);
    view.setUint32(0x18, 120, true);
    view.setUint32(0x1c, 0x7, true);

    const header = parseScxtHeader(bytes.buffer);
    expect(header.magic).toBe('SCXT');
    expect(header.major).toBe(1);
    expect(header.minor).toBe(1);
    expect(header.dims).toEqual([2, 3, 4, 5]);
    expect(header.cellCount).toBe(120);
  });
});
