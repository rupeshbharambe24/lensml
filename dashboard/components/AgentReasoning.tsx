"use client";

/**
 * Placeholder pane for the agent reasoning trace.
 * Real impl lands in step 2.3 (lensml-implementation-sequence.md):
 *   - Tool-call cards "pop" in via Framer Motion
 *   - Reasoning trace streams token-by-token
 *   - Confidence pill at the top
 */
export function AgentReasoning() {
  return (
    <div className="flex h-full flex-col rounded-lg border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-xs uppercase tracking-wider text-neutral-400">
          Agent reasoning
        </h2>
        <span className="rounded-full bg-neutral-800 px-2 py-0.5 text-[10px] text-neutral-500">
          waiting
        </span>
      </div>
      <div className="mt-3 flex-1 overflow-y-auto text-sm text-neutral-500">
        <p className="font-mono">
          <span className="text-neutral-600">◯</span> No anomaly detected yet.
        </p>
        <p className="mt-2 text-xs leading-relaxed text-neutral-600">
          The agent will appear here when telemetry crosses a detector threshold.
          Hypothesis, tool calls, and confidence stream live.
        </p>
      </div>
    </div>
  );
}
