import attentionWGSL from '../wgsl/attention.wgsl?raw';

export type AttentionConfig = {
  seqLen: number;
  headDim: number;
  nHeads: number;
  currentPos: number;
};

export function createAttentionParams(config: AttentionConfig): Float32Array {
  return new Float32Array([
    config.seqLen,
    config.headDim,
    config.nHeads,
    config.currentPos,
    1 / Math.sqrt(config.headDim),
    0,
    0,
    0,
  ]);
}

export const attentionPipeline = {
  label: 'scxmu.attention',
  entryPoint: 'attention',
  code: attentionWGSL,
};
