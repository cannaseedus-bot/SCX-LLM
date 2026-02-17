import fs from 'node:fs';
import { sha256 } from './build-scx4.js';

export function parseHeader(buffer) {
  if (buffer.length < 128) {
    throw new Error('SCX4 file is smaller than header size.');
  }

  const magic = buffer.subarray(0, 4).toString('ascii');
  const major = buffer.readUInt8(4);
  const minor = buffer.readUInt8(5);
  const flags = buffer.readUInt16LE(6);
  const fileSize = buffer.readUInt32LE(0x08);
  const sectionCount = buffer.readUInt32LE(0x34);
  const merkleRoot = buffer.subarray(0x38, 0x58);

  return { magic, major, minor, flags, fileSize, sectionCount, merkleRoot };
}

export function parseDirectory(buffer, sectionCount) {
  const base = 128;
  const required = base + sectionCount * 16;
  if (buffer.length < required) {
    throw new Error('SCX4 file is too small for declared section directory.');
  }

  const sections = [];
  for (let i = 0; i < sectionCount; i += 1) {
    const off = base + i * 16;
    sections.push({
      type: buffer.readUInt32LE(off),
      offset: buffer.readUInt32LE(off + 4),
      size: buffer.readUInt32LE(off + 8),
      reserved: buffer.readUInt32LE(off + 12),
    });
  }

  return sections;
}

function assertNoOverlap(sections) {
  const sorted = [...sections].sort((a, b) => a.offset - b.offset);
  for (let i = 1; i < sorted.length; i += 1) {
    const prev = sorted[i - 1];
    const cur = sorted[i];
    if (cur.offset < prev.offset + prev.size) {
      throw new Error(`Overlapping sections: ${prev.type} and ${cur.type}`);
    }
  }
}

export function verifySCX4(filePath = 'model/model.scx4') {
  const file = fs.readFileSync(filePath);
  const header = parseHeader(file);

  if (header.magic !== 'SCX4') {
    throw new Error(`Invalid magic: ${header.magic}`);
  }

  if (header.major !== 1 || header.minor > 0) {
    throw new Error(`Unsupported SCX4 version ${header.major}.${header.minor}`);
  }

  if (header.fileSize !== file.length) {
    throw new Error(`File size mismatch: header=${header.fileSize} actual=${file.length}`);
  }

  const sections = parseDirectory(file, header.sectionCount);

  for (const section of sections) {
    if (section.offset % 64 !== 0) {
      throw new Error(`Section type ${section.type} is not 64-byte aligned.`);
    }
    if (section.offset + section.size > file.length) {
      throw new Error(`Section type ${section.type} exceeds file bounds.`);
    }
  }

  assertNoOverlap(sections);

  const digestTarget = Buffer.from(file);
  digestTarget.fill(0, 0x38, 0x58);
  const computed = sha256(digestTarget);

  if (!computed.equals(header.merkleRoot)) {
    throw new Error('Merkle root mismatch.');
  }

  return {
    valid: true,
    version: `${header.major}.${header.minor}`,
    flags: header.flags,
    sectionCount: header.sectionCount,
    totalBytes: file.length,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = verifySCX4(process.argv[2] || 'model/model.scx4');
  console.log(`SCX4 verified: ${JSON.stringify(result)}`);
}
