"use client";

import { useEffect, useState } from "react";

/**
 * Minimal EN/JA i18n scaffold (plan §3.5 — JA toggle scaffold).
 *
 * This is intentionally tiny: the v1 demo audience reads English; JA is a
 * "we thought about it" gesture for the FAR AWAY hackathon. We ship a
 * handful of high-traffic strings translated and let everything else
 * fall back to the EN string.
 */

export type Lang = "en" | "ja";

const STORAGE_KEY = "lensml.lang";

type Strings = {
  watching: string;
  connecting: string;
  reconnecting: string;
  closed: string;
  idle: string;
  liveMetrics: string;
  agentReasoning: string;
  diff: string;
  noAnomaly: string;
  noAnomalyDetail: string;
  earlier: string;
  approve: string;
  reject: string;
  undo: string;
  pts: string;
  shortcutsTitle: string;
  paletteTitle: string;
  palettePlaceholder: string;
  paletteCmdToggleLang: string;
  paletteCmdShowHelp: string;
  paletteCmdReload: string;
  paletteEmpty: string;
};

const EN: Strings = {
  watching: "Watching",
  connecting: "Connecting…",
  reconnecting: "Reconnecting…",
  closed: "Closed",
  idle: "Idle",
  liveMetrics: "Live metrics",
  agentReasoning: "Agent reasoning",
  diff: "Proposed fix",
  noAnomaly: "No anomaly detected yet.",
  noAnomalyDetail:
    "The agent will appear here when telemetry crosses a detector threshold. Hypothesis, tool calls, and confidence stream live.",
  earlier: "Earlier",
  approve: "Approve",
  reject: "Reject",
  undo: "Undo",
  pts: "pts",
  shortcutsTitle: "Keyboard shortcuts",
  paletteTitle: "Command palette",
  palettePlaceholder: "Type a command…",
  paletteCmdToggleLang: "Toggle language (EN ↔ JA)",
  paletteCmdShowHelp: "Show keyboard shortcuts",
  paletteCmdReload: "Reload dashboard",
  paletteEmpty: "No commands match",
};

const JA: Strings = {
  watching: "監視中",
  connecting: "接続中…",
  reconnecting: "再接続中…",
  closed: "切断",
  idle: "待機中",
  liveMetrics: "ライブメトリクス",
  agentReasoning: "エージェント推論",
  diff: "提案された修正",
  noAnomaly: "異常はまだ検出されていません。",
  noAnomalyDetail:
    "テレメトリが検出器のしきい値を超えると、ここにエージェントが表示されます。仮説、ツール呼び出し、信頼度がライブで流れます。",
  earlier: "以前",
  approve: "承認",
  reject: "却下",
  undo: "取り消し",
  pts: "点",
  shortcutsTitle: "キーボードショートカット",
  paletteTitle: "コマンドパレット",
  palettePlaceholder: "コマンドを入力…",
  paletteCmdToggleLang: "言語を切り替え (EN ↔ JA)",
  paletteCmdShowHelp: "ショートカットを表示",
  paletteCmdReload: "ダッシュボードを再読み込み",
  paletteEmpty: "一致するコマンドはありません",
};

const TABLE: Record<Lang, Strings> = { en: EN, ja: JA };

export function useLang(): {
  lang: Lang;
  t: Strings;
  setLang: (l: Lang) => void;
  toggle: () => void;
} {
  const [lang, setLangState] = useState<Lang>("en");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "ja") setLangState(saved);
  }, []);

  const setLang = (l: Lang) => {
    setLangState(l);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, l);
    }
  };

  return {
    lang,
    t: TABLE[lang],
    setLang,
    toggle: () => setLang(lang === "en" ? "ja" : "en"),
  };
}
