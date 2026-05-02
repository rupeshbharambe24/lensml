"use client";

import type { LensmlEvent } from "@/lib/useSSE";

/**
 * Right-top pane: detector anomalies + (eventually) agent reasoning trace.
 *
 * Step 1.7: shows anomaly events as they arrive from the broker. The full
 * agent reasoning view (tool-call cards, hypothesis stream, confidence
 * pill) lands in step 2.3 once the LLM agent loop is implemented.
 */

type AnomalyEvent = LensmlEvent & {
  kind: "anomaly";
  detector?: string;
  signal?: string;
  severity?: string;
  evidence?: Record<string, unknown>;
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-rose-600/30 text-rose-300 border-rose-700/40",
  warning: "bg-amber-600/30 text-amber-300 border-amber-700/40",
  info: "bg-sky-600/30 text-sky-300 border-sky-700/40",
};

function isAnomaly(e: LensmlEvent): e is AnomalyEvent {
  return e.kind === "anomaly";
}

function severityClass(s: string | undefined): string {
  return SEVERITY_COLORS[s ?? "warning"] ?? SEVERITY_COLORS.warning;
}

export function AgentReasoning({ events }: { events: LensmlEvent[] }) {
  const anomalies = events.filter(isAnomaly);
  const latest = anomalies[anomalies.length - 1];

  return (
    <div className="flex h-full flex-col rounded-lg border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-xs uppercase tracking-wider text-neutral-400">
          Agent reasoning
        </h2>
        {latest ? (
          <span
            className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${severityClass(
              latest.severity,
            )}`}
          >
            {anomalies.length} anomal{anomalies.length === 1 ? "y" : "ies"}
          </span>
        ) : (
          <span className="rounded-full bg-neutral-800 px-2 py-0.5 text-[10px] text-neutral-500">
            waiting
          </span>
        )}
      </div>

      <div className="mt-3 flex-1 overflow-y-auto text-sm">
        {!latest && <Empty />}
        {latest && <LatestAnomaly anomaly={latest} />}
        {anomalies.length > 1 && <PastAnomalies anomalies={anomalies.slice(0, -1)} />}
      </div>
    </div>
  );
}

function Empty() {
  return (
    <div className="text-neutral-500">
      <p className="font-mono">
        <span className="text-neutral-600">◯</span> No anomaly detected yet.
      </p>
      <p className="mt-2 text-xs leading-relaxed text-neutral-600">
        The agent will appear here when telemetry crosses a detector threshold.
        Hypothesis, tool calls, and confidence stream live.
      </p>
    </div>
  );
}

function LatestAnomaly({ anomaly }: { anomaly: AnomalyEvent }) {
  const { detector, signal, step, evidence, severity } = anomaly;
  return (
    <div className={`rounded-md border p-3 ${severityClass(severity)}`}>
      <div className="flex items-baseline justify-between font-mono text-xs">
        <span className="font-semibold">⚠ {detector ?? "anomaly"}</span>
        <span className="opacity-70">step {step ?? "—"}</span>
      </div>
      {signal && (
        <div className="mt-1 font-mono text-[11px] uppercase tracking-wider opacity-70">
          {signal}
        </div>
      )}
      {evidence && Object.keys(evidence).length > 0 && (
        <dl className="mt-3 space-y-1 font-mono text-[11px]">
          {Object.entries(evidence).map(([k, v]) => (
            <div key={k} className="flex justify-between gap-4">
              <dt className="opacity-60">{k}</dt>
              <dd className="font-medium">{formatValue(v)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

function PastAnomalies({ anomalies }: { anomalies: AnomalyEvent[] }) {
  return (
    <div className="mt-4">
      <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
        Earlier ({anomalies.length})
      </div>
      <ul className="mt-2 space-y-1 font-mono text-[11px] text-neutral-500">
        {anomalies
          .slice()
          .reverse()
          .slice(0, 10)
          .map((a, i) => (
            <li key={i} className="flex justify-between">
              <span>
                {a.detector} · {a.signal}
              </span>
              <span className="opacity-60">step {a.step}</span>
            </li>
          ))}
      </ul>
    </div>
  );
}

function formatValue(v: unknown): string {
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    return Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0)
      ? v.toExponential(3)
      : v.toFixed(4);
  }
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}
