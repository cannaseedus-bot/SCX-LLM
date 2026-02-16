import { describe, expect, it } from 'vitest';
import { loadSCX4, parseScx4Header, parseScx4Sections } from '../src/runtime/scx4-loader';

describe('scx4 loader', () => {
  it('parses valid SCX4 header', () => {
    const bytes = new Uint8Array(128);
    bytes.set([0x53, 0x43, 0x58, 0x34], 0);
    bytes[4] = 1;
    bytes[5] = 0;
    new DataView(bytes.buffer).setUint32(0x08, bytes.byteLength, true);

    const header = parseScx4Header(bytes.buffer);
    expect(header.magic).toBe('SCX4');
    expect(header.major).toBe(1);
  });

  it('parses section directory bounds safely', () => {
    const bytes = new Uint8Array(144);
    bytes.set([0x53, 0x43, 0x58, 0x34], 0);
    const view = new DataView(bytes.buffer);
    view.setUint32(0x08, bytes.byteLength, true);
    view.setUint32(0x34, 1, true);

    view.setUint32(128, 0x01, true);
    view.setUint32(132, 140, true);
    view.setUint32(136, 4, true);

    const sections = parseScx4Sections(bytes.buffer, 1);
    expect(sections[0].type).toBe(0x01);
  });

  it('loads valid SCX4 metadata', async () => {
    const model = await loadSCX4();
    expect(model.hidden_size).toBe(640);
    expect(model.layers).toBe(16);
    expect(model.experts).toBe(8);
  });
});
