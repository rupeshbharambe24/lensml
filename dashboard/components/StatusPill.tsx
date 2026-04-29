"use client";

import type { SSEStatus } from "@/lib/useSSE";

const COPY: Record<SSEStatus, { label: string; cls: string }> = {
  idle: { label: "Idle", cls: "bg-neutral-700 text-neutral-300" },
  connecting: { label: "Connecting…", cls: "bg-amber-600/30 text-amber-300" },
  open: { label: "Watching", cls: "bg-emerald-600/30 text-emerald-300" },
  error: { label: "Reconnecting…", cls: "bg-rose-600/30 text-rose-300" },
  closed: { label: "Closed", cls: "bg-neutral-700 text-neutral-400" },
};

export function StatusPill({ status }: { status: SSEStatus }) {
  const { label, cls } = COPY[status];
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${cls}`}
    >
      <span
        className={`h-2 w-2 rounded-full ${
          status === "open"
            ? "animate-pulse bg-emerald-400"
            : status === "connecting"
              ? "animate-pulse bg-amber-400"
              : status === "error"
                ? "bg-rose-400"
                : "bg-neutral-500"
        }`}
      />
      {label}
    </span>
  );
}
