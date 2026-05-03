"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { AgentReasoning } from "@/components/AgentReasoning";
import { CommandPalette, type PaletteAction } from "@/components/CommandPalette";
import { DiffPreview } from "@/components/DiffPreview";
import { KeyboardHelp } from "@/components/KeyboardHelp";
import { LiveMetrics } from "@/components/LiveMetrics";
import { PrivacyToggle, dispatchPrivacyToggle } from "@/components/PrivacyToggle";
import { StatusPill } from "@/components/StatusPill";
import { useLang } from "@/lib/i18n";
import { LENSML_API_URL, useSSE } from "@/lib/useSSE";

const DEFAULT_RUN_ID = "demo-run";

export default function HomePage() {
  const [runId] = useState(DEFAULT_RUN_ID);
  const sseUrl = `${LENSML_API_URL}/api/runs/${runId}/metrics`;
  const { events, status } = useSSE(sseUrl);
  const { lang, toggle } = useLang();

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  // Most-recent step (any kind of metric event with `step`).
  const latestStep = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const s = events[i].step;
      if (typeof s === "number") return s;
    }
    return undefined;
  }, [events]);

  // Global keyboard shortcuts: ⌘K / Ctrl+K opens palette, ? opens help,
  // g toggles language, Esc closes overlays.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTyping =
        !!target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable);

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
        return;
      }
      if (isTyping) return;
      if (e.key === "?") {
        e.preventDefault();
        setHelpOpen(true);
        return;
      }
      if (e.key === "Escape") {
        setPaletteOpen(false);
        setHelpOpen(false);
        return;
      }
      if (e.key === "g") {
        toggle();
        return;
      }
      if (e.key === "p") {
        dispatchPrivacyToggle();
        return;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  const handlePalette = useCallback(
    (a: PaletteAction) => {
      switch (a) {
        case "toggle-lang":
          toggle();
          break;
        case "show-help":
          setHelpOpen(true);
          break;
        case "reload":
          if (typeof window !== "undefined") window.location.reload();
          break;
        case "toggle-privacy":
          dispatchPrivacyToggle();
          break;
        case "close":
          setPaletteOpen(false);
          break;
      }
    },
    [toggle],
  );

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <header className="flex items-center justify-between border-b border-neutral-800 px-6 py-3">
        <div className="flex items-center gap-3 font-mono text-sm">
          <span className="font-semibold tracking-tight">lensml</span>
          <span className="text-neutral-700">▸</span>
          <span className="text-neutral-300">{runId}</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={toggle}
            title="Toggle language (g)"
            className="rounded border border-neutral-800 bg-neutral-900 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-neutral-400 transition-colors hover:bg-neutral-800 hover:text-neutral-200"
          >
            {lang.toUpperCase()}
          </button>
          <button
            onClick={() => setPaletteOpen(true)}
            title="Command palette (⌘K)"
            className="rounded border border-neutral-800 bg-neutral-900 px-2 py-1 font-mono text-[10px] text-neutral-400 transition-colors hover:bg-neutral-800 hover:text-neutral-200"
          >
            ⌘K
          </button>
          <button
            onClick={() => setHelpOpen(true)}
            title="Keyboard shortcuts (?)"
            className="rounded border border-neutral-800 bg-neutral-900 px-2 py-1 font-mono text-[10px] text-neutral-400 transition-colors hover:bg-neutral-800 hover:text-neutral-200"
          >
            ?
          </button>
          <PrivacyToggle />
          <StatusPill status={status} step={latestStep} />
        </div>
      </header>

      <div className="grid h-[calc(100vh-57px)] grid-cols-1 gap-4 p-4 md:grid-cols-2">
        <LiveMetrics events={events} />
        <div className="grid grid-rows-2 gap-4">
          <AgentReasoning events={events} />
          <DiffPreview runId={runId} events={events} />
        </div>
      </div>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onAction={handlePalette}
      />
      <KeyboardHelp open={helpOpen} onClose={() => setHelpOpen(false)} />
    </main>
  );
}
