const SECTION_EXPERT = 0x03;
const SECTION_ROUTER = 0x04;

function decodeBase64ToArrayBuffer(base64Text) {
  const normalized = base64Text.replace(/\s+/g, '');
  const binary = atob(normalized);
  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }

  return bytes.buffer;
}

function readMagic(view) {
  return String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
}

function parseSections(buffer, sectionCount) {
  const view = new DataView(buffer);
  const directoryOffset = 128;
  const sections = [];

  for (let i = 0; i < sectionCount; i += 1) {
    const base = directoryOffset + i * 16;
    sections.push({
      type: view.getUint32(base, true),
      offset: view.getUint32(base + 4, true),
      size: view.getUint32(base + 8, true),
    });
  }

  return sections;
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

function uploadStorage(device, data) {
  const bytes = asBytes(data);
  const buffer = device.createBuffer({
    size: bytes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();

  return buffer;
}

export async function loadSCX4(url, device) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch SCX4: ${response.status}`);
  }

  const base64Model = await response.text();
  const buffer = decodeBase64ToArrayBuffer(base64Model);
  const view = new DataView(buffer);

  if (readMagic(view) !== 'SCX4') {
    throw new Error('Invalid SCX4 file.');
  }

  const hiddenSize = view.getUint32(0x10, true);
  const experts = view.getUint32(0x20, true);
  const blockSize = view.getUint32(0x30, true);
  const sectionCount = view.getUint32(0x34, true);

  const sections = parseSections(buffer, sectionCount);

  const getSection = (type) => {
    const section = sections.find((entry) => entry.type === type);
    if (!section) {
      throw new Error(`Missing section type ${type}.`);
    }

    return new Uint8Array(buffer, section.offset, section.size);
  };

  const expertSection = getSection(SECTION_EXPERT);
  const routerSection = getSection(SECTION_ROUTER);

  const routerView = new DataView(routerSection.buffer, routerSection.byteOffset, routerSection.byteLength);
  const routerScores = new Float32Array([routerView.getFloat32(0, true), routerView.getFloat32(4, true)]);
  const selectedExpert = routerScores[1] > routerScores[0] ? 1 : 0;

  const expertStride = Math.floor(expertSection.byteLength / Math.max(experts, 1));
  const expert0 = decodeInt4Vector(expertSection, 0, hiddenSize, blockSize);
  const expert1 = decodeInt4Vector(expertSection, expertStride, hiddenSize, blockSize);

  const inputBuffer = device.createBuffer({
    size: hiddenSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    size: hiddenSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: hiddenSize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const expert0Buffer = uploadStorage(device, expert0);
  const expert1Buffer = uploadStorage(device, expert1);
  const routerBuffer = uploadStorage(device, routerScores);

  const params = new Uint32Array([hiddenSize, 0, 0, 0]);
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
    compute: {
      module,
      entryPoint: 'moe_forward',
    },
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
    if (input.length !== hiddenSize) {
      throw new Error(`Input length ${input.length} does not match hidden size ${hiddenSize}.`);
    }

    device.queue.writeBuffer(inputBuffer, 0, input);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(hiddenSize / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, hiddenSize * 4);

    device.queue.submit([encoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);

    const copy = new Float32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();
    return copy;
  }

  return {
    hiddenSize,
    selectedExpert,
    forward,
  };
}
