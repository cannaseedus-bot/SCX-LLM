export type KvCacheState = {
  nHeads: number;
  seqLen: number;
  headDim: number;
  currentPos: number;
  keys: Float32Array;
  values: Float32Array;
};

export function createKvCache(nHeads: number, seqLen: number, headDim: number): KvCacheState {
  const size = nHeads * seqLen * headDim;
  return {
    nHeads,
    seqLen,
    headDim,
    currentPos: 0,
    keys: new Float32Array(size),
    values: new Float32Array(size),
  };
}

export function appendKv(cache: KvCacheState, key: Float32Array, value: Float32Array): KvCacheState {
  const stride = cache.nHeads * cache.headDim;
  if (key.length !== stride || value.length !== stride) {
    throw new Error('KV append shape mismatch');
  }
  if (cache.currentPos >= cache.seqLen) {
    throw new Error('KV cache is full');
  }

  const offset = cache.currentPos * stride;
  cache.keys.set(key, offset);
  cache.values.set(value, offset);
  cache.currentPos += 1;

  return cache;
}
