"use client";

import { Command } from "cmdk";
import { useEffect } from "react";

import { useLang } from "@/lib/i18n";

/**
 * Command palette (plan §3.5). Opens on ⌘K / Ctrl+K. Hosts a small set
 * of v1 actions: toggle language, show keyboard shortcuts, reload.
 *
 * The palette state is owned by the parent (page.tsx) so the same
 * keyboard handler can also coordinate the help overlay and other
 * single-key shortcuts.
 */

export type PaletteAction =
  | "toggle-lang"
  | "show-help"
  | "reload"
  | "close";

export function CommandPalette({
  open,
  onClose,
  onAction,
}: {
  open: boolean;
  onClose: () => void;
  onAction: (a: PaletteAction) => void;
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
      className="fixed inset-0 z-40 flex items-start justify-center bg-black/60 pt-[12vh] backdrop-blur-sm"
      onClick={onClose}
    >
      <Command
        className="lensml-popin w-[480px] overflow-hidden rounded-lg border border-neutral-800 bg-neutral-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        label={t.paletteTitle}
      >
        <Command.Input
          autoFocus
          placeholder={t.palettePlaceholder}
          className="w-full border-b border-neutral-800 bg-transparent px-4 py-3 text-sm text-neutral-100 outline-none placeholder:text-neutral-600"
        />
        <Command.List className="max-h-[320px] overflow-y-auto p-2">
          <Command.Empty className="px-3 py-6 text-center text-xs text-neutral-500">
            {t.paletteEmpty}
          </Command.Empty>

          <Command.Group
            heading={
              <span className="px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-neutral-500">
                Actions
              </span>
            }
          >
            <PaletteItem
              hint="g"
              label={t.paletteCmdToggleLang}
              onSelect={() => {
                onAction("toggle-lang");
                onClose();
              }}
            />
            <PaletteItem
              hint="?"
              label={t.paletteCmdShowHelp}
              onSelect={() => {
                onAction("show-help");
                onClose();
              }}
            />
            <PaletteItem
              hint="R"
              label={t.paletteCmdReload}
              onSelect={() => {
                onAction("reload");
                onClose();
              }}
            />
          </Command.Group>
        </Command.List>
      </Command>
    </div>
  );
}

function PaletteItem({
  label,
  hint,
  onSelect,
}: {
  label: string;
  hint: string;
  onSelect: () => void;
}) {
  return (
    <Command.Item
      onSelect={onSelect}
      className="flex cursor-pointer items-center justify-between rounded px-3 py-2 text-sm text-neutral-200 aria-selected:bg-neutral-800 aria-selected:text-white"
    >
      <span>{label}</span>
      <kbd className="rounded border border-neutral-700 bg-neutral-900 px-1.5 py-0.5 font-mono text-[10px] text-neutral-400">
        {hint}
      </kbd>
    </Command.Item>
  );
}
