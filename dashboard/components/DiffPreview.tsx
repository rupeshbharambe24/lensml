"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { LENSML_API_URL, type LensmlEvent } from "@/lib/useSSE";

/**
 * Right-bottom pane: shows the agent's proposed fix and lets the user
 * approve / reject / undo via REST + keyboard shortcuts (plan §7.3).
 *
 * Polling, not SSE, for the decision payload — the diff is computed
 * server-side against the live script source, so the most recent state
 * is what we want anyway. Polling every 2 s is fine for the v1 demo.
 */

type DecisionRow = {
  id: number;
  run_id: string;
  ts: number;
  fix_template: string | null;
  diff_text: string | null;
  approved: number | null;
  applied_at: number | null;
};

type DecisionDetail = DecisionRow & {
  preview?: {
    applicable: boolean;
    reason?: string | null;
    template?: string;
    rationale?: string;
    params?: Record<string, unknown>;
    old_source?: string;
    new_source?: string | null;
    diff?: string | null;
  };
};

type Phase =
  | { kind: "idle" }
  | { kind: "pending"; detail: DecisionDetail }
  | { kind: "applying"; detail: DecisionDetail }
  | { kind: "applied"; detail: DecisionDetail; commit: string }
  | { kind: "rejected"; detail: DecisionDetail }
  | { kind: "error"; message: string; detail?: DecisionDetail };

const POLL_MS = 2000;

async function fetchDecisions(runId: string): Promise<DecisionRow[]> {
  const r = await fetch(`${LENSML_API_URL}/api/runs/${runId}/decisions`);
  if (!r.ok) throw new Error(`GET decisions: ${r.status}`);
  return (await r.json()) as DecisionRow[];
}

async function fetchDecisionDetail(runId: string, id: number): Promise<DecisionDetail> {
  const r = await fetch(`${LENSML_API_URL}/api/runs/${runId}/decisions/${id}`);
  if (!r.ok) throw new Error(`GET decision: ${r.status}`);
  return (await r.json()) as DecisionDetail;
}

async function postAction(
  runId: string,
  id: number,
  action: "approve" | "reject" | "undo",
): Promise<{ status: string; commit?: string; diff?: string }> {
  const r = await fetch(`${LENSML_API_URL}/api/runs/${runId}/decisions/${id}/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  if (!r.ok) {
    let detail = `${action}: ${r.status}`;
    try {
      const body = await r.json();
      if (body.detail) detail = String(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return await r.json();
}

export function DiffPreview({ runId, events }: { runId: string; events: LensmlEvent[] }) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });

  // Cheap signal to re-poll: number of anomaly events in the live stream.
  // When a new anomaly fires the agent typically queues a decision shortly
  // after; this avoids a long lag between event arrival and UI update.
  const anomalySignal = useMemo(
    () => events.filter((e) => e.kind === "anomaly" || e.kind === "decision_applied").length,
    [events],
  );

  const refresh = useCallback(async () => {
    try {
      const decisions = await fetchDecisions(runId);
      const pending = decisions.filter((d) => d.approved === null);
      if (pending.length === 0) {
        // If nothing pending, but we just saw an applied/rejected one in
        // the current render, keep that state visible briefly.
        setPhase((p) =>
          p.kind === "applied" || p.kind === "rejected" || p.kind === "error" ? p : { kind: "idle" },
        );
        return;
      }
      const latest = pending[pending.length - 1];
      const detail = await fetchDecisionDetail(runId, latest.id);
      setPhase({ kind: "pending", detail });
    } catch (e) {
      setPhase({ kind: "error", message: String((e as Error).message ?? e) });
    }
  }, [runId]);

  // Initial fetch + polling
  useEffect(() => {
    void refresh();
    const id = setInterval(() => {
      void refresh();
    }, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  // Re-fetch immediately when a new anomaly / applied event arrives
  useEffect(() => {
    void refresh();
  }, [anomalySignal, refresh]);

  // Action dispatch
  const doAction = useCallback(
    async (action: "approve" | "reject" | "undo") => {
      const detail =
        phase.kind === "pending" || phase.kind === "applied" || phase.kind === "rejected"
          ? phase.detail
          : phase.kind === "applying"
            ? phase.detail
            : null;
      if (!detail) return;
      // For undo, only act on an applied state.
      if (action === "undo" && phase.kind !== "applied") return;
      // For approve/reject, only act on a pending state.
      if ((action === "approve" || action === "reject") && phase.kind !== "pending") return;

      try {
        if (action === "approve") {
          setPhase({ kind: "applying", detail });
          const out = await postAction(runId, detail.id, "approve");
          setPhase({ kind: "applied", detail, commit: out.commit ?? "" });
        } else if (action === "reject") {
          await postAction(runId, detail.id, "reject");
          setPhase({ kind: "rejected", detail });
        } else if (action === "undo") {
          await postAction(runId, detail.id, "undo");
          setPhase({ kind: "idle" });
          await refresh();
        }
      } catch (e) {
        setPhase({ kind: "error", message: String((e as Error).message ?? e), detail });
      }
    },
    [phase, runId, refresh],
  );

  // Keyboard shortcuts (plan §7.3)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ignore when typing in an input
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (e.key === "y" || e.key === "Y") {
        void doAction("approve");
      } else if (e.key === "n" || e.key === "N") {
        void doAction("reject");
      } else if (e.key === "u" || e.key === "U") {
        void doAction("undo");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [doAction]);

  return (
    <div className="flex h-full flex-col rounded-lg border border-neutral-800 bg-neutral-900/50 p-4">
      <Header phase={phase} />
      <div className="mt-3 flex-1 overflow-y-auto text-sm">
        {phase.kind === "idle" && <Empty />}
        {phase.kind === "error" && <ErrorBox message={phase.message} />}
        {phase.kind !== "idle" && phase.kind !== "error" && <Body phase={phase} />}
      </div>
      <Footer phase={phase} onAction={doAction} />
    </div>
  );
}

function Header({ phase }: { phase: Phase }) {
  const status =
    phase.kind === "idle"
      ? { label: "empty", cls: "bg-neutral-800 text-neutral-500" }
      : phase.kind === "pending"
        ? { label: "needs review", cls: "bg-amber-600/30 text-amber-300 border border-amber-700/40" }
        : phase.kind === "applying"
          ? { label: "applying…", cls: "bg-sky-600/30 text-sky-300 border border-sky-700/40" }
          : phase.kind === "applied"
            ? { label: "applied", cls: "bg-emerald-600/30 text-emerald-300 border border-emerald-700/40" }
            : phase.kind === "rejected"
              ? { label: "rejected", cls: "bg-neutral-800 text-neutral-400" }
              : { label: "error", cls: "bg-rose-600/30 text-rose-300 border border-rose-700/40" };
  return (
    <div className="flex items-center justify-between">
      <h2 className="font-mono text-xs uppercase tracking-wider text-neutral-400">Diff preview</h2>
      <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${status.cls}`}>{status.label}</span>
    </div>
  );
}

function Empty() {
  return (
    <div className="text-neutral-500">
      <p className="font-mono text-xs">No proposed fix.</p>
      <p className="mt-2 text-xs leading-relaxed text-neutral-600">
        A proposed patch will appear here when the agent calls{" "}
        <code className="rounded bg-neutral-800 px-1 py-0.5 text-neutral-400">propose_fix</code>. Press{" "}
        <Kbd>y</Kbd> to accept, <Kbd>n</Kbd> to reject, <Kbd>u</Kbd> to undo.
      </p>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-rose-700/40 bg-rose-600/10 p-3 font-mono text-[11px] text-rose-300">
      {message}
    </div>
  );
}

function Body({
  phase,
}: {
  phase: Exclude<Phase, { kind: "idle" } | { kind: "error" }>;
}) {
  const detail = phase.detail;
  const preview = detail.preview;
  const template = preview?.template ?? detail.fix_template ?? "?";
  const rationale = preview?.rationale ?? "";
  const diff = preview?.diff ?? "";
  const applicable = preview?.applicable ?? false;
  const reason = preview?.reason;

  return (
    <div>
      <div className="font-mono text-xs">
        <span className="font-semibold text-neutral-200">{template}</span>
        <span className="mx-2 text-neutral-700">·</span>
        <span className="text-neutral-500">decision #{detail.id}</span>
        {phase.kind === "applied" && phase.commit && (
          <>
            <span className="mx-2 text-neutral-700">·</span>
            <span className="text-emerald-400">commit {phase.commit.slice(0, 8)}</span>
          </>
        )}
      </div>

      {rationale && (
        <p className="mt-2 text-xs leading-relaxed text-neutral-400">{rationale}</p>
      )}

      {!applicable && (
        <div className="mt-3 rounded-md border border-amber-700/40 bg-amber-600/10 p-3 font-mono text-[11px] text-amber-300">
          Cannot apply: {reason ?? "(no reason given)"}
        </div>
      )}

      {applicable && diff && (
        <pre className="mt-3 overflow-x-auto rounded-md border border-neutral-800 bg-neutral-950 p-3 font-mono text-[11px] leading-relaxed">
          {diff.split("\n").map((line, i) => {
            const cls = line.startsWith("+++")
              ? "text-neutral-400"
              : line.startsWith("---")
                ? "text-neutral-400"
                : line.startsWith("+")
                  ? "text-emerald-400"
                  : line.startsWith("-")
                    ? "text-rose-400"
                    : line.startsWith("@@")
                      ? "text-sky-400"
                      : "text-neutral-300";
            return (
              <div key={i} className={cls}>
                {line || " "}
              </div>
            );
          })}
        </pre>
      )}
    </div>
  );
}

function Footer({
  phase,
  onAction,
}: {
  phase: Phase;
  onAction: (action: "approve" | "reject" | "undo") => void | Promise<void>;
}) {
  if (phase.kind === "idle") return null;

  if (phase.kind === "pending") {
    const applicable = phase.detail.preview?.applicable ?? false;
    return (
      <div className="mt-3 flex items-center justify-between border-t border-neutral-800 pt-3">
        <div className="font-mono text-[10px] text-neutral-500">
          <Kbd>y</Kbd> accept · <Kbd>n</Kbd> reject
        </div>
        <div className="flex gap-2">
          <Button onClick={() => onAction("reject")} variant="ghost">
            Reject
          </Button>
          <Button onClick={() => onAction("approve")} variant="primary" disabled={!applicable}>
            Accept
          </Button>
        </div>
      </div>
    );
  }
  if (phase.kind === "applied") {
    return (
      <div className="mt-3 flex items-center justify-between border-t border-neutral-800 pt-3">
        <div className="font-mono text-[10px] text-neutral-500">
          <Kbd>u</Kbd> undo
        </div>
        <Button onClick={() => onAction("undo")} variant="ghost">
          Undo
        </Button>
      </div>
    );
  }
  return null;
}

function Kbd({ children }: { children: React.ReactNode }) {
  return <kbd className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px]">{children}</kbd>;
}

function Button({
  children,
  onClick,
  variant,
  disabled,
}: {
  children: React.ReactNode;
  onClick: () => void;
  variant: "primary" | "ghost";
  disabled?: boolean;
}) {
  const base =
    "rounded px-3 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40";
  const styles =
    variant === "primary"
      ? "bg-emerald-600 text-white hover:bg-emerald-500"
      : "border border-neutral-700 text-neutral-300 hover:bg-neutral-800";
  return (
    <button type="button" className={`${base} ${styles}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}
