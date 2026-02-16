import { readFile } from 'node:fs/promises';

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

export function parseScx4Header(buffer: ArrayBuffer): Scx4Header {
  assert(buffer.byteLength >= 128, 'SCX4 header too small');
  const view = new DataView(buffer);

  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  assert(magic === 'SCX4', 'Invalid SCX4 magic');

  const fileSize = view.getUint32(0x08, true);
  assert(fileSize === buffer.byteLength, `SCX4 file size mismatch: header=${fileSize} actual=${buffer.byteLength}`);

  return {
    magic: 'SCX4',
    major: view.getUint8(0x04),
    minor: view.getUint8(0x05),
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
    sectionCount: view.getUint32(0x34, true),
    merkleRoot: new Uint8Array(buffer.slice(0x38, 0x58)),
  };
}

export function parseScx4Sections(buffer: ArrayBuffer, sectionCount: number): Scx4Section[] {
  const view = new DataView(buffer);
  const fileSize = buffer.byteLength;
  const directoryOffset = 128;

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

    assert(section.offset + section.size <= fileSize, `SCX4 invalid section bounds for type ${section.type}`);
    sections.push(section);
  }

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
