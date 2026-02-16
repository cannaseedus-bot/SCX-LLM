export function shardScxq2(experts: number): string[] {
  return Array.from({ length: experts }, (_, i) => `expert_${i}.scxq2`);
}
