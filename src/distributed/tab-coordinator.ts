import { createChannel, makeRequestId, type ExpertMessage } from './broadcast';

export type RoutedExpert = {
  expertId: number;
  score: number;
};

export type DispatchTensor = Float32Array;

export type DispatchResult = {
  expertId: number;
  score: number;
  output: Float32Array;
};

export class ExpertTabCoordinator {
  private readonly channel: BroadcastChannel;

  public constructor(channelName = 'scxmu-experts') {
    this.channel = createChannel(channelName);
  }

  public destroy(): void {
    this.channel.close();
  }

  public async dispatchExpert(expert: RoutedExpert, tensor: DispatchTensor, timeoutMs = 3_000): Promise<DispatchResult> {
    const id = makeRequestId();

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.channel.removeEventListener('message', onMessage);
        reject(new Error(`Expert ${expert.expertId} timed out`));
      }, timeoutMs);

      const onMessage = (event: MessageEvent<ExpertMessage>) => {
        if (event.data.kind !== 'expert-response' || event.data.payload.id !== id) {
          return;
        }

        clearTimeout(timeout);
        this.channel.removeEventListener('message', onMessage);

        resolve({
          expertId: event.data.payload.expertId,
          score: expert.score,
          output: new Float32Array(event.data.payload.result),
        });
      };

      this.channel.addEventListener('message', onMessage);
      this.channel.postMessage({
        kind: 'expert-request',
        payload: {
          id,
          expertId: expert.expertId,
          score: expert.score,
          tensor: Array.from(tensor),
        },
      } satisfies ExpertMessage);
    });
  }
}

export function deterministicMerge(parts: DispatchResult[]): Float32Array {
  if (parts.length === 0) {
    return new Float32Array(0);
  }

  const sorted = [...parts].sort((a, b) => a.expertId - b.expertId);
  const length = sorted[0].output.length;
  const merged = new Float32Array(length);

  for (const part of sorted) {
    if (part.output.length !== length) {
      throw new Error('Cannot merge expert outputs of different lengths');
    }

    for (let i = 0; i < length; i += 1) {
      merged[i] += part.score * part.output[i];
    }
  }

  return merged;
}
