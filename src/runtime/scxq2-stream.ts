export async function streamShard(name: string): Promise<string> {
  return `streamed:${name}`;
}
