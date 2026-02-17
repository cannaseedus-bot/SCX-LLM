import moeWGSL from '../wgsl/moe_dispatch.wgsl?raw';

export type MoeDispatchConfig = {
  hiddenSize: number;
  nExperts: number;
  topK: number;
  tokenOffset?: number;
};

export function moeDispatchParams(config: MoeDispatchConfig): Uint32Array {
  return new Uint32Array([config.hiddenSize, config.nExperts, config.topK, config.tokenOffset ?? 0]);
}

export function moeWorkgroups(): number {
  return 1;
}

export const moeDispatchPipeline = {
  label: 'scxmu.moe.dispatch',
  entryPoint: 'moe_dispatch',
  code: moeWGSL,
};
