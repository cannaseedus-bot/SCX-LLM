import moeWGSL from '../wgsl/moe_dispatch.wgsl?raw';

export type MoeDispatchConfig = {
  hiddenSize: number;
};

export function moeWorkgroups(config: MoeDispatchConfig): number {
  return Math.ceil(config.hiddenSize / 64);
}

export const moeDispatchPipeline = {
  label: 'scxmu.moe.forward',
  entryPoint: 'moe_forward',
  code: moeWGSL,
};
