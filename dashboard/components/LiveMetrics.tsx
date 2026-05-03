"use client";

import { useLang } from "@/lib/i18n";
import type { LensmlEvent } from "@/lib/useSSE";

import { LossChart } from "./LossChart";

/**
 * Left pane: live training telemetry plots.
 * v1: just the loss curve. v0.2 adds grad-L2 norm + activation stats sub-charts.
 */
export function LiveMetrics({ events }: { events: LensmlEvent[] }) {
  const { t } = useLang();
  const lossCount = events.filter((e) => typeof e.loss === "number").length;
  const lastLoss = [...events]
    .reverse()
    .find((e) => typeof e.loss === "number")?.loss;

  return (
    <div className="flex h-full flex-col rounded-lg border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-baseline justify-between">
        <h2 className="font-mono text-xs uppercase tracking-wider text-neutral-400">
          {t.liveMetrics}
        </h2>
        <div className="font-mono text-xs text-neutral-500">
          {lossCount} {t.pts}
          {lastLoss !== undefined && (
            <>
              <span className="mx-2 text-neutral-700">·</span>
              <span className="text-neutral-300">loss = {lastLoss.toFixed(4)}</span>
            </>
          )}
        </div>
      </div>
      <div className="mt-4 flex-1">
        <LossChart events={events} />
      </div>
    </div>
  );
}
