"use client";

import { useState } from "react";

import { AgentReasoning } from "@/components/AgentReasoning";
import { DiffPreview } from "@/components/DiffPreview";
import { LiveMetrics } from "@/components/LiveMetrics";
import { StatusPill } from "@/components/StatusPill";
import { LENSML_API_URL, useSSE } from "@/lib/useSSE";

const DEFAULT_RUN_ID = "demo-run";

export default function HomePage() {
  // For step 1.5 the run id is hardcoded; the URL bar / API will drive it
  // in step 2.x once /api/runs returns active runs.
  const [runId] = useState(DEFAULT_RUN_ID);
  // Absolute URL bypasses the Next.js dev proxy (which buffers SSE).
  // FastAPI is configured with CORS for localhost:3000.
  const sseUrl = `${LENSML_API_URL}/api/runs/${runId}/metrics`;
  const { events, status } = useSSE(sseUrl);

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <header className="flex items-center justify-between border-b border-neutral-800 px-6 py-3">
        <div className="flex items-center gap-3 font-mono text-sm">
          <span className="font-semibold tracking-tight">lensml</span>
          <span className="text-neutral-700">▸</span>
          <span className="text-neutral-300">{runId}</span>
        </div>
        <StatusPill status={status} />
      </header>

      <div className="grid h-[calc(100vh-57px)] grid-cols-1 gap-4 p-4 md:grid-cols-2">
        <LiveMetrics events={events} />
        <div className="grid grid-rows-2 gap-4">
          <AgentReasoning events={events} />
          <DiffPreview />
        </div>
      </div>
    </main>
  );
}
