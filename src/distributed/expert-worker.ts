import { createChannel, type ExpertMessage } from './broadcast';

export type LocalExpertRunner = (expertId: number, tensor: Float32Array) => Float32Array;

export function attachExpertWorker(localExperts: number[], runLocalExpert: LocalExpertRunner, channelName = 'scxmu-experts'): () => void {
  const channel = createChannel(channelName);
  const owned = new Set(localExperts);

  const handler = (event: MessageEvent<ExpertMessage>) => {
    if (event.data.kind !== 'expert-request') {
      return;
    }

    const { id, expertId, tensor } = event.data.payload;
    if (!owned.has(expertId)) {
      return;
    }

    const result = runLocalExpert(expertId, new Float32Array(tensor));

    channel.postMessage({
      kind: 'expert-response',
      payload: {
        id,
        expertId,
        result: Array.from(result),
      },
    } satisfies ExpertMessage);
  };

  channel.addEventListener('message', handler);

  return () => {
    channel.removeEventListener('message', handler);
    channel.close();
  };
}
