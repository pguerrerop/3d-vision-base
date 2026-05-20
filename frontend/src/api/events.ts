import { API_BASE_URL } from "./client";

export interface LiveEvent {
  type: string;
  payload: Record<string, unknown>;
}

export function connectEvents(onEvent: (event: LiveEvent) => void, onDisconnect: () => void): EventSource | null {
  if (!("EventSource" in window)) {
    onDisconnect();
    return null;
  }
  const source = new EventSource(`${API_BASE_URL}/api/events/stream`);
  const handle = (type: string) => (message: MessageEvent) => {
    try {
      onEvent({ type, payload: JSON.parse(message.data) });
    } catch {
      onEvent({ type, payload: {} });
    }
  };
  source.addEventListener("connected", handle("connected"));
  source.addEventListener("latest", handle("latest"));
  source.addEventListener("runtime", handle("runtime"));
  source.addEventListener("processed", handle("processed"));
  source.onerror = () => onDisconnect();
  return source;
}
