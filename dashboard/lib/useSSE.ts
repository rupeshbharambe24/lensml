"use client";

import { useEffect, useState } from "react";

export type LensmlEvent = {
  step?: number;
  loss?: number;
  run_id?: string;
  kind?: string;
  ts?: number;
  [key: string]: unknown;
};

export type SSEStatus = "idle" | "connecting" | "open" | "error" | "closed";

const MAX_EVENTS = 2000;

/**
 * Base URL of the lensml FastAPI server. Browser code reads it via the
 * `NEXT_PUBLIC_*` env var so it's bundled at build time. Default targets
 * the local dev server. SSE bypasses the Next.js dev proxy because that
 * proxy buffers streaming responses — see notes on CORSMiddleware in
 * `lensml.server.app`.
 */
export const LENSML_API_URL: string =
  process.env.NEXT_PUBLIC_LENSML_API_URL ?? "http://127.0.0.1:8000";

/**
 * Subscribe to a server-sent-events endpoint. Native EventSource
 * auto-reconnects on disconnect. Pass `null` for `url` to detach.
 *
 * Returns the rolling event buffer (last `MAX_EVENTS`) and connection status.
 */
export function useSSE(url: string | null): {
  events: LensmlEvent[];
  status: SSEStatus;
} {
  const [events, setEvents] = useState<LensmlEvent[]>([]);
  const [status, setStatus] = useState<SSEStatus>("idle");

  useEffect(() => {
    if (!url) {
      setStatus("idle");
      return;
    }
    setStatus("connecting");
    const es = new EventSource(url);

    es.onopen = () => setStatus("open");
    es.onerror = () => setStatus("error");
    es.onmessage = (e) => {
      // Heartbeat pings come through as `event: ping\ndata: {}` — they
      // arrive on the named "ping" channel, not the default message channel,
      // so onmessage will not see them. Guards data parsing as well.
      if (!e.data) return;
      try {
        const parsed = JSON.parse(e.data) as LensmlEvent;
        setEvents((prev) => {
          const next = [...prev, parsed];
          return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next;
        });
      } catch {
        // Ignore non-JSON frames silently — protocol is JSON-only by design.
      }
    };

    return () => {
      es.close();
      setStatus("closed");
    };
  }, [url]);

  return { events, status };
}
