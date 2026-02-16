import { loadSCX4 } from './scx4-loader.js';

const statusEl = document.getElementById('status');
const outputEl = document.getElementById('output');
const runButton = document.getElementById('run');

function setStatus(message, level = 'ok') {
  statusEl.textContent = message;
  statusEl.dataset.level = level;
}

async function initGPU() {
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU is not available in this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No compatible GPU adapter found.');
  }

  return adapter.requestDevice();
}

async function runInference() {
  runButton.disabled = true;
  setStatus('Initializing WebGPU…');

  try {
    const device = await initGPU();
    setStatus('Loading SCX4 model…');

    const model = await loadSCX4('./model.scx4.b64', device);
    const input = new Float32Array(model.hiddenSize);
    input.fill(0.5);

    setStatus('Running MoE forward pass…');
    const result = await model.forward(input);

    const preview = Array.from(result.slice(0, 16));
    outputEl.textContent = JSON.stringify(
      {
        hiddenSize: model.hiddenSize,
        selectedExpert: model.selectedExpert,
        outputPreview: preview,
      },
      null,
      2,
    );

    setStatus('Inference complete.');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Failed: ${message}`, 'error');
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener('click', () => {
  void runInference();
});
