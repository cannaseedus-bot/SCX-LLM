import { readFile } from 'node:fs/promises';

export type Scx4Header = {
  version: number;
  merkleRoot: string;
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

export function parseScx4Header(buffer: ArrayBuffer): Scx4Header {
  return {
    version: new DataView(buffer).getUint8(0),
    merkleRoot: 'placeholder',
  };
}

export async function loadSCX4(configPath = 'model/config.json'): Promise<Scx4ModelMetadata> {
  const raw = await readFile(configPath, 'utf8');
  const parsed = JSON.parse(raw) as Scx4ModelMetadata;

  if (parsed.format !== 'scx4') {
    throw new Error(`Unsupported model format: ${parsed.format}`);
  }

  return parsed;
}
