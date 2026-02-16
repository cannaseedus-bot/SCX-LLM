self.onmessage = (event: MessageEvent<Float32Array>) => {
  self.postMessage(event.data);
};
