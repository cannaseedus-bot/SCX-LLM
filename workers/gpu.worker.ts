self.onmessage = (event: MessageEvent<{ op: string }>) => {
  self.postMessage({ ok: true, op: event.data.op });
};
