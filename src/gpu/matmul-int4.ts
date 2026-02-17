import matmulInt4WGSL from '../wgsl/matmul_int4.wgsl?raw';

export type TransformerBlockConfig = {
  hiddenSize: number;
  blockSize: number;
};

export function transformerBlockParams(config: TransformerBlockConfig): Uint32Array {
  return new Uint32Array([config.hiddenSize, config.blockSize, 0, 0]);
}

export function transformerBlockWorkgroups(config: TransformerBlockConfig): number {
  return Math.ceil(config.hiddenSize / 64);
}

export const matmulInt4Pipeline = {
  label: 'scxmu.transformer_block.int4',
  entryPoint: 'transformer_block',
  code: matmulInt4WGSL,
};
