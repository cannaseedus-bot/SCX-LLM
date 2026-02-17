import { readFile } from 'node:fs/promises';

export const SCX4_HEADER_SIZE = 128;
export const SCX4_SECTION_ALIGNMENT = 64;

export const SCX4_FLAGS = {
  INT4_WEIGHTS: 0x0001,
  MOE_ENABLED: 0x0002,
  KV_INT8: 0x0004,
  SCXQ2_COMPRESSED: 0x0008,
  SHARED_ATTENTION: 0x0010,
} as const;

export const SCX4_SECTION_TYPES = {
  EMBEDDING: 0x01,
  ATTENTION: 0x02,
  EXPERT: 0x03,
  ROUTER: 0x04,
  LAYERNORM: 0x05,
  QUANT_TABLE: 0x06,
  SCXQ2_LANE: 0x07,
  MERKLE_TREE: 0x08,
} as const;

export type Scx4Header = {
  magic: 'SCX4';
  major: number;
  minor: number;
  flags: number;
  fileSize: number;
  layers: number;
  hiddenSize: number;
  heads: number;
  headDim: number;
  mlpDim: number;
  experts: number;
  topK: number;
  vocabSize: number;
  seqLen: number;
  blockSize: number;
  sectionCount: number;
  merkleRoot: Uint8Array;
};

export type Scx4Section = {
  type: number;
  offset: number;
  size: number;
  merkleOffset: number;
};

export type Scx4ModelMetadata = {
  format: 'scx4';
  version: number;
  layers: number;
  hidden_size: number;
  experts: number;
  top_k: number;
  quantization: string;
  tokenizer: string;
};

function assert(cond: unknown, message: string): asserts cond {
  if (!cond) {
    throw new Error(message);
  }
}

function validateNoOverlap(sections: Scx4Section[]): void {
  const sorted = [...sections].sort((a, b) => a.offset - b.offset);
  for (let i = 1; i < sorted.length; i += 1) {
    const prev = sorted[i - 1];
    const cur = sorted[i];
    const prevEnd = prev.offset + prev.size;
    assert(cur.offset >= prevEnd, `SCX4 overlapping sections: ${prev.type} overlaps ${cur.type}`);
  }
}

export function parseScx4Header(buffer: ArrayBuffer): Scx4Header {
  assert(buffer.byteLength >= SCX4_HEADER_SIZE, 'SCX4 header too small');
  const view = new DataView(buffer);

  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  assert(magic === 'SCX4', 'Invalid SCX4 magic');

  const major = view.getUint8(0x04);
  const minor = view.getUint8(0x05);
  assert(major === 1, `Unsupported SCX4 major version: ${major}`);
  assert(minor <= 0, `Unsupported SCX4 minor version: ${minor}`);

  const fileSize = view.getUint32(0x08, true);
  assert(fileSize === buffer.byteLength, `SCX4 file size mismatch: header=${fileSize} actual=${buffer.byteLength}`);

  const sectionCount = view.getUint32(0x34, true);
  const directoryEnd = SCX4_HEADER_SIZE + sectionCount * 16;
  assert(directoryEnd <= buffer.byteLength, 'SCX4 section directory out of range');

  return {
    magic: 'SCX4',
    major,
    minor,
    flags: view.getUint16(0x06, true),
    fileSize,
    layers: view.getUint32(0x0c, true),
    hiddenSize: view.getUint32(0x10, true),
    heads: view.getUint32(0x14, true),
    headDim: view.getUint32(0x18, true),
    mlpDim: view.getUint32(0x1c, true),
    experts: view.getUint32(0x20, true),
    topK: view.getUint32(0x24, true),
    vocabSize: view.getUint32(0x28, true),
    seqLen: view.getUint32(0x2c, true),
    blockSize: view.getUint32(0x30, true),
    sectionCount,
    merkleRoot: new Uint8Array(buffer.slice(0x38, 0x58)),
  };
}

export function parseScx4Sections(buffer: ArrayBuffer, sectionCount: number): Scx4Section[] {
  const view = new DataView(buffer);
  const fileSize = buffer.byteLength;
  const directoryOffset = SCX4_HEADER_SIZE;

  const sections: Scx4Section[] = [];
  for (let i = 0; i < sectionCount; i += 1) {
    const base = directoryOffset + i * 16;
    assert(base + 16 <= fileSize, 'SCX4 section directory exceeds file size');

    const section = {
      type: view.getUint32(base + 0, true),
      offset: view.getUint32(base + 4, true),
      size: view.getUint32(base + 8, true),
      merkleOffset: view.getUint32(base + 12, true),
    };

    assert(section.offset % SCX4_SECTION_ALIGNMENT === 0, `SCX4 section ${section.type} is not 64-byte aligned`);
    assert(section.offset + section.size <= fileSize, `SCX4 invalid section bounds for type ${section.type}`);
    sections.push(section);
  }

  validateNoOverlap(sections);
  return sections;
}

export function findScx4Section(buffer: ArrayBuffer, sections: Scx4Section[], type: number): Uint8Array {
  const section = sections.find((s) => s.type === type);
  assert(section, `Missing SCX4 section type ${type}`);
  return new Uint8Array(buffer, section.offset, section.size);
}

export async function loadSCX4(configPath = 'model/config.json'): Promise<Scx4ModelMetadata> {
  const raw = await readFile(configPath, 'utf8');
  const parsed = JSON.parse(raw) as Scx4ModelMetadata;

  if (parsed.format !== 'scx4') {
    throw new Error(`Unsupported model format: ${parsed.format}`);
  }

  return parsed;
}
