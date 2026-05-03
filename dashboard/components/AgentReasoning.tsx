"use client";

import { useEffect, useState } from "react";

import { useLang } from "@/lib/i18n";
import { LENSML_API_URL, type LensmlEvent } from "@/lib/useSSE";

/**
 * Right-top pane: detector anomalies + agent reasoning trace.
 *
 * Step 3.5 polish: pop-in animation when a new anomaly arrives, i18n
 * (EN/JA), severity-colored badge.
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

function anomalyKey(a: AnomalyEvent): string {
  return `${a.detector ?? "?"}|${a.signal ?? "?"}|${a.step ?? -1}|${a.ts ?? 0}`;
}

export function AgentReasoning({ events }: { events: LensmlEvent[] }) {
  const { t } = useLang();
  const anomalies = events.filter(isAnomaly);
  const latest = anomalies[anomalies.length - 1];

  // Track the latest key so we can play a pop-in animation each time
  // a new anomaly arrives without re-animating on every parent re-render.
  const [latestKey, setLatestKey] = useState<string | null>(null);
  const [pop, setPop] = useState(false);
  useEffect(() => {
    if (!latest) return;
    const k = anomalyKey(latest);
    if (k === latestKey) return;
    setLatestKey(k);
    setPop(true);
    const tid = setTimeout(() => setPop(false), 600);
    return () => clearTimeout(tid);
  }, [latest, latestKey]);

  return (
    <div className="flex h-full flex-col rounded-lg border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-xs uppercase tracking-wider text-neutral-400">
          {t.agentReasoning}
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
        {latest && <LatestAnomaly anomaly={latest} pop={pop} />}
        {anomalies.length > 1 && <PastAnomalies anomalies={anomalies.slice(0, -1)} />}
      </div>
    </div>
  );
}

function Empty() {
  const { t } = useLang();
  return (
    <div className="text-neutral-500">
      <p className="font-mono">
        <span className="text-neutral-600">◯</span> {t.noAnomaly}
      </p>
      <p className="mt-2 text-xs leading-relaxed text-neutral-600">
        {t.noAnomalyDetail}
      </p>
    </div>
  );
}

type CannedDiagnosis = {
  hypothesis?: string;
  fix_template?: string | null;
  rationale?: string;
  confidence?: string;
  model?: string;
};

function LatestAnomaly({
  anomaly,
  pop,
}: {
  anomaly: AnomalyEvent;
  pop: boolean;
}) {
  const { t } = useLang();
  const { detector, signal, step, evidence, severity } = anomaly;
  const [canned, setCanned] = useState<CannedDiagnosis | null>(null);
  const [loading, setLoading] = useState(false);

  const runCanned = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${LENSML_API_URL}/api/privacy/canned`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ anomaly }),
      });
      if (r.ok) setCanned(await r.json());
    } catch {
      // best-effort — leave canned null on error
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className={`rounded-md border p-3 transition-all duration-300 ${severityClass(severity)} ${pop ? "lensml-popin" : ""}`}
    >
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
      <div className="mt-3 flex justify-end">
        <button
          type="button"
          onClick={() => void runCanned()}
          disabled={loading}
          className="rounded border border-neutral-700 bg-neutral-900/60 px-2 py-1 font-mono text-[10px] text-neutral-300 transition-colors hover:bg-neutral-800 disabled:opacity-50"
        >
          {loading ? "…" : t.privacyRunCanned}
        </button>
      </div>
      {canned && (
        <div className="mt-3 rounded border border-neutral-700 bg-neutral-950/60 p-2 font-mono text-[11px] text-neutral-300">
          <div className="flex justify-between text-[10px] uppercase tracking-wider text-neutral-500">
            <span>local · {canned.model ?? "—"}</span>
            <span>conf: {canned.confidence ?? "—"}</span>
          </div>
          {canned.hypothesis && (
            <p className="mt-1 leading-snug">{canned.hypothesis}</p>
          )}
          {canned.fix_template && (
            <p className="mt-1 text-emerald-300">→ {canned.fix_template}</p>
          )}
          {canned.rationale && (
            <p className="mt-1 leading-snug text-neutral-400">{canned.rationale}</p>
          )}
        </div>
      )}
    </div>
  );
}

function PastAnomalies({ anomalies }: { anomalies: AnomalyEvent[] }) {
  const { t } = useLang();
  return (
    <div className="mt-4">
      <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
        {t.earlier} ({anomalies.length})
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
