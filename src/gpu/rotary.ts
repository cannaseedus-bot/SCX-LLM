import rotaryWGSL from '../wgsl/rotary.wgsl?raw';

export type RotaryConfig = {
  vectorLength: number;
};

export function rotaryWorkgroups(config: RotaryConfig): number {
  return Math.ceil(config.vectorLength / 64);
}

export const rotaryPipeline = {
  label: 'scxmu.rotary',
  entryPoint: 'rotary',
  code: rotaryWGSL,
};
