export function createChannel(name: string): BroadcastChannel {
  return new BroadcastChannel(name);
}
