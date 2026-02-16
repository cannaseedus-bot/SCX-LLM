export type ExpertRequest = {
  id: string;
  expertId: number;
  score: number;
  tensor: number[];
};

export type ExpertResponse = {
  id: string;
  expertId: number;
  result: number[];
};

export type ExpertMessage =
  | { kind: 'expert-request'; payload: ExpertRequest }
  | { kind: 'expert-response'; payload: ExpertResponse };

export function createChannel(name = 'scxmu-experts'): BroadcastChannel {
  return new BroadcastChannel(name);
}

export function makeRequestId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
