export const SCXT_MAGIC = 'SCXT';
export const SCXT_HEADER_SIZE = 0x20;
export const SCXT_MERKLE_SIZE = 32;

export const SCXT_OPCODES = {
  NOP: 0x00,
  ADD: 0x01,
  MUL: 0x02,
  WMIX: 0x03,
  ENC: 0x04,
  DEC: 0x05,
  SLICE: 0x06,
  ACT_TANH: 0x07,
  ACT_STEP: 0x08,
  ACT_NORM: 0x09,
  SEARCH: 0x0a,
  ANALYZE: 0x0b,
  LOAD: 0x0c,
  SAVE: 0x0d,
  CMP: 0x0e,
  HALT: 0x0f,
} as const;

export type ScxtHeader = {
  magic: 'SCXT';
  major: number;
  minor: number;
  flags: number;
  dims: [number, number, number, number];
  cellCount: number;
  layerMask: number;
};

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) {
    throw new Error(msg);
  }
}

export function flatten4DIndex(
  w: number,
  x: number,
  y: number,
  z: number,
  dims: [number, number, number, number],
): number {
  const [dw, dx, dy, dz] = dims;
  assert(w >= 0 && w < dw, 'w out of range');
  assert(x >= 0 && x < dx, 'x out of range');
  assert(y >= 0 && y < dy, 'y out of range');
  assert(z >= 0 && z < dz, 'z out of range');

  return (((w * dx) + x) * dy + y) * dz + z;
}

export function parseScxtHeader(buffer: ArrayBuffer): ScxtHeader {
  assert(buffer.byteLength >= SCXT_HEADER_SIZE + SCXT_MERKLE_SIZE, 'SCXT file too small');
  const view = new DataView(buffer);

  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  assert(magic === SCXT_MAGIC, 'Invalid SCXT magic');

  const dims: [number, number, number, number] = [
    view.getUint32(0x08, true),
    view.getUint32(0x0c, true),
    view.getUint32(0x10, true),
    view.getUint32(0x14, true),
  ];

  return {
    magic: 'SCXT',
    major: view.getUint8(0x04),
    minor: view.getUint8(0x05),
    flags: view.getUint16(0x06, true),
    dims,
    cellCount: view.getUint32(0x18, true),
    layerMask: view.getUint32(0x1c, true),
  };
}

export function opcodeValues(): number[] {
  return Object.values(SCXT_OPCODES);
}
