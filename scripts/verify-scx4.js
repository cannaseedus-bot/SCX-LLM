import fs from 'node:fs';
import { sha256 } from './build-scx4.js';

export function parseHeader(buffer) {
  if (buffer.length < 128) {
    throw new Error('SCX4 file is smaller than header size.');
  }

  const magic = buffer.subarray(0, 4).toString('ascii');
  const version = buffer.readUInt8(4);
  const flags = buffer.readUInt16LE(6);
  const sectionCount = buffer.readUInt32LE(0x34);
  const merkleRoot = buffer.subarray(0x38, 0x58);

  return { magic, version, flags, sectionCount, merkleRoot };
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

export function verifySCX4(filePath = 'model/model.scx4') {
  const file = fs.readFileSync(filePath);
  const header = parseHeader(file);

  if (header.magic !== 'SCX4') {
    throw new Error(`Invalid magic: ${header.magic}`);
  }

  const sections = parseDirectory(file, header.sectionCount);

  for (const section of sections) {
    if (section.offset + section.size > file.length) {
      throw new Error(`Section type ${section.type} exceeds file bounds.`);
    }
  }

  const digestTarget = Buffer.from(file);
  digestTarget.fill(0, 0x38, 0x58);
  const computed = sha256(digestTarget);

  if (!computed.equals(header.merkleRoot)) {
    throw new Error('Merkle root mismatch.');
  }

  return {
    valid: true,
    version: header.version,
    sectionCount: header.sectionCount,
    totalBytes: file.length,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = verifySCX4(process.argv[2] || 'model/model.scx4');
  console.log(`SCX4 verified: ${JSON.stringify(result)}`);
}
