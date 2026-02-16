export async function getGpuDevice(): Promise<GPUDevice | null> {
  if (!navigator.gpu) {
    return null;
  }
  const adapter = await navigator.gpu.requestAdapter();
  return adapter?.requestDevice() ?? null;
}
