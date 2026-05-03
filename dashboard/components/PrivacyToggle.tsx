"use client";

import { useCallback, useEffect, useState } from "react";

import { useLang } from "@/lib/i18n";
import { LENSML_API_URL } from "@/lib/useSSE";

/**
 * Privacy-mode header pill (plan §3.7 — Plan C).
 *
 * Hits ``GET /api/privacy`` once on mount, then toggles via ``POST``.
 * State is server-side authoritative — the pill always shows whatever
 * the server reported back from the most recent toggle / fetch. If the
 * server can't be reached, the pill greys out.
 */

type Status = "loading" | "ok" | "error";

export function PrivacyToggle() {
  const { t } = useLang();
  const [enabled, setEnabled] = useState<boolean>(false);
  const [status, setStatus] = useState<Status>("loading");

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${LENSML_API_URL}/api/privacy`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      setEnabled(Boolean(body.enabled));
      setStatus("ok");
    } catch {
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const toggle = useCallback(async () => {
    const next = !enabled;
    try {
      const r = await fetch(`${LENSML_API_URL}/api/privacy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: next }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      setEnabled(Boolean(body.enabled));
      setStatus("ok");
    } catch {
      setStatus("error");
    }
  }, [enabled]);

  // Listen for global "privacy:toggle" events so the command palette
  // and keyboard shortcut can fire the same path without prop drilling.
  useEffect(() => {
    const handler = () => void toggle();
    window.addEventListener("lensml-privacy-toggle", handler);
    return () => window.removeEventListener("lensml-privacy-toggle", handler);
  }, [toggle]);

  const label = enabled ? t.privacyOn : t.privacyOff;
  const statusCls =
    status === "error"
      ? "border-rose-700/50 bg-rose-900/30 text-rose-300"
      : enabled
        ? "border-emerald-700/50 bg-emerald-900/30 text-emerald-300 hover:bg-emerald-900/50"
        : "border-neutral-800 bg-neutral-900 text-neutral-400 hover:bg-neutral-800";

  return (
    <button
      type="button"
      onClick={() => void toggle()}
      title="Toggle privacy mode (p)"
      disabled={status === "loading"}
      className={`rounded border px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-colors disabled:opacity-50 ${statusCls}`}
    >
      {status === "loading" ? "…" : label}
    </button>
  );
}

/** Imperative trigger used by the command palette + keyboard shortcut. */
export function dispatchPrivacyToggle(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent("lensml-privacy-toggle"));
}
