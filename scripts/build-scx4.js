import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';

const DEFAULT_CONFIG = {
  layers: 16,
  hidden: 640,
  heads: 10,
  headDim: 64,
  mlp: 2560,
  experts: 8,
  topk: 2,
  vocab: 32000,
  seq: 2048,
  block: 128,
  output: 'model/model.scx4',
};

export function sha256(buffer) {
  return crypto.createHash('sha256').update(buffer).digest();
}

function clampInt4(v) {
  return Math.max(-8, Math.min(7, v));
}

export function quantizeINT4Block(fp32) {
  if (fp32.length === 0 || fp32.length % 2 !== 0) {
    throw new Error('INT4 blocks must have an even number of values.');
  }

  const absMax = fp32.reduce((acc, v) => Math.max(acc, Math.abs(v)), 0);
  const scale = absMax / 7 || 1e-8;
  const packed = Buffer.alloc(fp32.length / 2);

  for (let i = 0; i < fp32.length; i += 2) {
    const q0 = clampInt4(Math.round(fp32[i] / scale));
    const q1 = clampInt4(Math.round(fp32[i + 1] / scale));

    const n0 = (q0 < 0 ? q0 + 16 : q0) & 0xf;
    const n1 = (q1 < 0 ? q1 + 16 : q1) & 0xf;

    packed[i / 2] = (n0 << 4) | n1;
  }

  const scaleBuf = Buffer.alloc(4);
  scaleBuf.writeFloatLE(scale, 0);

  return { packed, scaleBuf };
}

function randomMatrix(size) {
  const out = new Array(size);
  for (let i = 0; i < size; i += 1) {
    out[i] = (Math.random() - 0.5) * 0.02;
  }
  return out;
}

function blocksToBuffer(fpValues, blockSize) {
  if (blockSize % 2 !== 0) {
    throw new Error('block size must be even for nibble packing.');
  }

  const chunks = [];

  for (let i = 0; i < fpValues.length; i += blockSize) {
    const block = fpValues.slice(i, Math.min(i + blockSize, fpValues.length));
    if (block.length < blockSize) {
      while (block.length < blockSize) {
        block.push(0);
      }
    }

    const { packed, scaleBuf } = quantizeINT4Block(block);
    chunks.push(scaleBuf, packed);
  }

  return Buffer.concat(chunks);
}

function buildSections(config) {
  const sections = [];

  const embedSize = config.vocab * config.hidden;
  sections.push({ type: 0x01, data: blocksToBuffer(randomMatrix(embedSize), config.block) });

  const attnSize = config.hidden * config.hidden * 4;
  sections.push({ type: 0x02, data: blocksToBuffer(randomMatrix(attnSize), config.block) });

  const expertBlobParts = [];
  for (let expert = 0; expert < config.experts; expert += 1) {
    const mlpSize = config.hidden * config.mlp + config.mlp * config.hidden;
    expertBlobParts.push(blocksToBuffer(randomMatrix(mlpSize), config.block));
  }
  sections.push({ type: 0x03, data: Buffer.concat(expertBlobParts) });

  const routerSize = config.layers * config.hidden * config.experts;
  const routerBuf = Buffer.alloc(routerSize * 4);
  for (let i = 0; i < routerSize; i += 1) {
    routerBuf.writeFloatLE((Math.random() - 0.5) * 0.02, i * 4);
  }
  sections.push({ type: 0x04, data: routerBuf });

  return sections;
}

export function buildSCX4File(sections, config) {
  const header = Buffer.alloc(128);
  header.write('SCX4', 0);
  header.writeUInt8(1, 4);
  header.writeUInt8(0, 5);
  header.writeUInt16LE(0x0003, 6);

  header.writeUInt32LE(config.layers, 0x0c);
  header.writeUInt32LE(config.hidden, 0x10);
  header.writeUInt32LE(config.heads, 0x14);
  header.writeUInt32LE(config.headDim, 0x18);
  header.writeUInt32LE(config.mlp, 0x1c);
  header.writeUInt32LE(config.experts, 0x20);
  header.writeUInt32LE(config.topk, 0x24);
  header.writeUInt32LE(config.vocab, 0x28);
  header.writeUInt32LE(config.seq, 0x2c);
  header.writeUInt32LE(config.block, 0x30);

  header.writeUInt32LE(sections.length, 0x34);

  const directory = Buffer.alloc(sections.length * 16);
  const blobs = [];
  let offset = 128 + directory.length;

  sections.forEach((section, index) => {
    const base = index * 16;
    directory.writeUInt32LE(section.type, base);
    directory.writeUInt32LE(offset, base + 4);
    directory.writeUInt32LE(section.data.length, base + 8);
    directory.writeUInt32LE(0, base + 12);
    blobs.push(section.data);
    offset += section.data.length;
  });

  const file = Buffer.concat([header, directory, ...blobs]);
  const merkleRoot = sha256(file);
  merkleRoot.copy(file, 0x38);

  return file;
}

export function buildModel(configOverride = {}) {
  const config = { ...DEFAULT_CONFIG, ...configOverride };
  const sections = buildSections(config);
  const file = buildSCX4File(sections, config);

  fs.mkdirSync(path.dirname(config.output), { recursive: true });
  fs.writeFileSync(config.output, file);

  return { output: config.output, bytes: file.length, sections: sections.length };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = buildModel();
  console.log(`SCX4 built: ${result.output} (${result.bytes} bytes, ${result.sections} sections)`);
}
