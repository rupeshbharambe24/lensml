"use client";

import { useEffect } from "react";

import { useLang } from "@/lib/i18n";

const SHORTCUTS: { keys: string[]; description: string }[] = [
  { keys: ["?"], description: "Open this help overlay" },
  { keys: ["Esc"], description: "Close overlay / palette" },
  { keys: ["⌘ K", "Ctrl K"], description: "Open command palette" },
  { keys: ["y"], description: "Approve the proposed fix" },
  { keys: ["n"], description: "Reject the proposed fix" },
  { keys: ["u"], description: "Undo the last applied fix" },
  { keys: ["g"], description: "Toggle language (EN ↔ JA)" },
  { keys: ["p"], description: "Toggle privacy mode (local LLM)" },
];

export function KeyboardHelp({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { t } = useLang();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="lensml-popin w-[420px] rounded-lg border border-neutral-800 bg-neutral-950 p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="font-mono text-xs uppercase tracking-wider text-neutral-400">
          {t.shortcutsTitle}
        </h3>
        <ul className="mt-4 space-y-2">
          {SHORTCUTS.map((s) => (
            <li
              key={s.description}
              className="flex items-center justify-between gap-4 text-sm text-neutral-200"
            >
              <span>{s.description}</span>
              <span className="flex gap-1 font-mono text-[11px]">
                {s.keys.map((k) => (
                  <kbd
                    key={k}
                    className="rounded border border-neutral-700 bg-neutral-900 px-1.5 py-0.5 text-neutral-300"
                  >
                    {k}
                  </kbd>
                ))}
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-[11px] text-neutral-500">
          Press{" "}
          <kbd className="rounded border border-neutral-700 bg-neutral-900 px-1 py-0.5">
            Esc
          </kbd>{" "}
          to close.
        </p>
      </div>
    </div>
  );
}
