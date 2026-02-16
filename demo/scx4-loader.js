const SECTION_EMBEDDING = 0x01;
const SECTION_ATTENTION = 0x02;
const SECTION_EXPERT = 0x03;
const SECTION_ROUTER = 0x04;
const SECTION_LAYERNORM = 0x05;

function bytesToMagic(bytes) {
  return String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
}

function decodeBase64ToArrayBuffer(base64Text) {
  const normalized = base64Text.replace(/\s+/g, '');
  const binary = atob(normalized);
  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }

  return bytes.buffer;
}

function normalizeScx4Buffer(rawBytes) {
  if (rawBytes.byteLength >= 4 && bytesToMagic(rawBytes.subarray(0, 4)) === 'SCX4') {
    return rawBytes.buffer.slice(rawBytes.byteOffset, rawBytes.byteOffset + rawBytes.byteLength);
  }

  const text = new TextDecoder().decode(rawBytes);
  return decodeBase64ToArrayBuffer(text);
}

function parseHeader(buffer) {
  if (buffer.byteLength < 128) {
    throw new Error('SCX4 header too small');
  }

  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'SCX4') {
    throw new Error('Invalid SCX4 file');
  }

  const fileSize = view.getUint32(0x08, true);
  if (fileSize !== buffer.byteLength) {
    throw new Error(`SCX4 file size mismatch: header=${fileSize} actual=${buffer.byteLength}`);
  }

  return {
    hiddenSize: view.getUint32(0x10, true),
    sectionCount: view.getUint32(0x34, true),
    experts: view.getUint32(0x20, true),
    blockSize: view.getUint32(0x30, true),
  };
}

function parseSections(buffer, sectionCount) {
  const view = new DataView(buffer);
  const directoryOffset = 128;
  const sections = [];

  for (let i = 0; i < sectionCount; i += 1) {
    const base = directoryOffset + i * 16;
    if (base + 16 > buffer.byteLength) {
      throw new Error('SCX4 section directory exceeds file size');
    }

    const section = {
      type: view.getUint32(base + 0, true),
      offset: view.getUint32(base + 4, true),
      size: view.getUint32(base + 8, true),
      merkleOffset: view.getUint32(base + 12, true),
    };

    if (section.offset + section.size > buffer.byteLength) {
      throw new Error(`Invalid section bounds for type ${section.type}`);
    }

    sections.push(section);
  }

  return sections;
}

function getSection(buffer, sections, type, optional = false) {
  const section = sections.find((s) => s.type === type);
  if (!section) {
    if (optional) {
      return new ArrayBuffer(0);
    }
    throw new Error(`Missing section ${type}`);
  }

  return buffer.slice(section.offset, section.offset + section.size);
}

function decodeInt4Vector(sectionData, byteOffset, length, blockSize) {
  const blockBytes = 4 + blockSize / 2;
  const sectionView = new DataView(sectionData.buffer, sectionData.byteOffset, sectionData.byteLength);
  const out = new Float32Array(length);

  for (let index = 0; index < length; index += 1) {
    const blockIndex = Math.floor(index / blockSize);
    const inBlock = index % blockSize;
    const blockBase = byteOffset + blockIndex * blockBytes;

    if (blockBase + blockBytes > sectionData.byteLength) {
      throw new Error('Expert section too small for requested hidden size.');
    }

    const scale = sectionView.getFloat32(blockBase, true);
    const packedOffset = blockBase + 4 + Math.floor(inBlock / 2);
    const packedByte = sectionData[packedOffset];

    const nibble = inBlock % 2 === 0 ? (packedByte >> 4) & 0x0f : packedByte & 0x0f;
    const signed = nibble >= 8 ? nibble - 16 : nibble;

    out[index] = signed * scale;
  }

  return out;
}

function asBytes(data) {
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  return new Uint8Array(data);
}

function uploadToGPU(device, data) {
  const bytes = asBytes(data);
  const gpuBuffer = device.createBuffer({
    size: Math.max(4, bytes.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  new Uint8Array(gpuBuffer.getMappedRange()).set(bytes);
  gpuBuffer.unmap();
  return gpuBuffer;
}

export async function loadSCX4(url, device) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch SCX4: ${response.status}`);
  }

  const raw = new Uint8Array(await response.arrayBuffer());
  const buffer = normalizeScx4Buffer(raw);
  const header = parseHeader(buffer);
  const sections = parseSections(buffer, header.sectionCount);

  const embedding = getSection(buffer, sections, SECTION_EMBEDDING);
  const attention = getSection(buffer, sections, SECTION_ATTENTION);
  const experts = getSection(buffer, sections, SECTION_EXPERT);
  const router = getSection(buffer, sections, SECTION_ROUTER);
  const layernorm = getSection(buffer, sections, SECTION_LAYERNORM, true);

  const routerView = new DataView(router);
  const routerScores = new Float32Array([routerView.getFloat32(0, true), routerView.getFloat32(4, true)]);
  const selectedExpert = routerScores[1] > routerScores[0] ? 1 : 0;

  const expertBytes = new Uint8Array(experts);
  const expertStride = Math.floor(expertBytes.byteLength / Math.max(header.experts, 1));
  const expert0 = decodeInt4Vector(expertBytes, 0, header.hiddenSize, header.blockSize);
  const expert1 = decodeInt4Vector(expertBytes, expertStride, header.hiddenSize, header.blockSize);

  const inputBuffer = device.createBuffer({
    size: header.hiddenSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    size: header.hiddenSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: header.hiddenSize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const embeddingGPU = uploadToGPU(device, embedding);
  const attentionGPU = uploadToGPU(device, attention);
  const expert0Buffer = uploadToGPU(device, expert0);
  const expert1Buffer = uploadToGPU(device, expert1);
  const routerBuffer = uploadToGPU(device, routerScores);
  const layernormGPU = uploadToGPU(device, layernorm);

  const params = new Uint32Array([header.hiddenSize, 0, 0, 0]);
  const paramBuffer = device.createBuffer({
    size: params.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(paramBuffer.getMappedRange()).set(new Uint8Array(params.buffer));
  paramBuffer.unmap();

  const shader = await fetch(new URL('./kernels.wgsl', import.meta.url)).then((r) => r.text());
  const module = device.createShaderModule({ code: shader });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'moe_forward' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: expert0Buffer } },
      { binding: 2, resource: { buffer: expert1Buffer } },
      { binding: 3, resource: { buffer: routerBuffer } },
      { binding: 4, resource: { buffer: outputBuffer } },
      { binding: 5, resource: { buffer: paramBuffer } },
    ],
  });

  async function forward(input) {
    if (input.length !== header.hiddenSize) {
      throw new Error(`Input length ${input.length} does not match hidden size ${header.hiddenSize}.`);
    }

    device.queue.writeBuffer(inputBuffer, 0, input);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(header.hiddenSize / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, header.hiddenSize * 4);

    device.queue.submit([encoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);

    const copy = new Float32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();
    return copy;
  }

  return {
    hiddenSize: header.hiddenSize,
    selectedExpert,
    embeddingGPU,
    attentionGPU,
    expertGPU: expert0Buffer,
    routerGPU: routerBuffer,
    layernormGPU,
    forward,
  };
}
