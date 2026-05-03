"use client";

import { useLang } from "@/lib/i18n";
import type { SSEStatus } from "@/lib/useSSE";

const CLASSES: Record<SSEStatus, string> = {
  idle: "bg-neutral-700 text-neutral-300",
  connecting: "bg-amber-600/30 text-amber-300",
  open: "bg-emerald-600/30 text-emerald-300",
  error: "bg-rose-600/30 text-rose-300",
  closed: "bg-neutral-700 text-neutral-400",
};

export function StatusPill({
  status,
  step,
}: {
  status: SSEStatus;
  step?: number;
}) {
  const { t } = useLang();
  const labels: Record<SSEStatus, string> = {
    idle: t.idle,
    connecting: t.connecting,
    open: t.watching,
    error: t.reconnecting,
    closed: t.closed,
  };
  const cls = CLASSES[status];
  const label = labels[status];
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
      <span>{label}</span>
      {typeof step === "number" && step > 0 && (
        <span className="font-mono opacity-75">· step {step}</span>
      )}
    </span>
  );
}
