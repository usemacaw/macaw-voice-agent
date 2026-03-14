/** Conversation message shown in UI. */
export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  isFinal: boolean;
}

/** Connection status. */
export type SessionStatus = "disconnected" | "connecting" | "connected";
