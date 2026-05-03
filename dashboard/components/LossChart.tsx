"use client";

import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";

import type { LensmlEvent } from "@/lib/useSSE";

const CHART_HEIGHT = 220;
const FALLBACK_WIDTH = 600;

type Marker = { step: number; severity: string; detector?: string };

const SEVERITY_STROKE: Record<string, string> = {
  critical: "rgba(244, 63, 94, 0.65)", // rose-500
  warning: "rgba(251, 191, 36, 0.55)", // amber-400
  info: "rgba(56, 189, 248, 0.55)", // sky-400
};

/**
 * uPlot loss-vs-step chart with vertical intervention markers (plan §3.5).
 *
 * Markers are drawn from every anomaly event in the buffer — one
 * dashed vertical line per anomaly, color-coded by severity. They sit
 * on top of the loss curve so the user can correlate "what happened
 * here?" at a glance.
 */
export function LossChart({ events }: { events: LensmlEvent[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);
  const markersRef = useRef<Marker[]>([]);
  const [ready, setReady] = useState(false);

  // Lifecycle: observe container size, init chart once non-zero width
  // arrives, resize on subsequent changes, destroy on unmount.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const drawMarkers = (u: uPlot) => {
      const markers = markersRef.current;
      if (markers.length === 0) return;
      const ctx = u.ctx;
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      const top = u.bbox.top;
      const bottom = u.bbox.top + u.bbox.height;
      for (const mk of markers) {
        const x = u.valToPos(mk.step, "x", true);
        if (x < u.bbox.left || x > u.bbox.left + u.bbox.width) continue;
        ctx.strokeStyle =
          SEVERITY_STROKE[mk.severity] ?? SEVERITY_STROKE.warning;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.stroke();
      }
      ctx.restore();
    };

    const opts: uPlot.Options = {
      width: FALLBACK_WIDTH,
      height: CHART_HEIGHT,
      legend: { show: true },
      series: [
        { label: "step" },
        {
          label: "loss",
          stroke: "#06b6d4",
          width: 1.5,
          points: { show: false },
        },
      ],
      axes: [
        {
          stroke: "#71717a",
          grid: { stroke: "#27272a", width: 0.5 },
          ticks: { stroke: "#404040" },
        },
        {
          stroke: "#71717a",
          grid: { stroke: "#27272a", width: 0.5 },
          ticks: { stroke: "#404040" },
        },
      ],
      scales: { x: { time: false } },
      hooks: {
        draw: [drawMarkers],
      },
    };

    const init = (width: number) => {
      if (plotRef.current) return;
      const u = new uPlot({ ...opts, width }, [[0], [0]], el);
      plotRef.current = u;
      setReady(true);
    };

    const ro = new ResizeObserver((entries) => {
      const width = Math.max(0, entries[0]?.contentRect.width ?? el.clientWidth);
      if (width === 0) return;
      if (!plotRef.current) {
        init(width);
      } else {
        plotRef.current.setSize({ width, height: CHART_HEIGHT });
      }
    });
    ro.observe(el);

    if (el.clientWidth > 0) {
      init(el.clientWidth);
    }

    return () => {
      ro.disconnect();
      plotRef.current?.destroy();
      plotRef.current = null;
    };
  }, []);

  // Push events into the chart whenever they update or chart becomes ready.
  useEffect(() => {
    const u = plotRef.current;
    if (!u) return;
    const xs: number[] = [];
    const ys: number[] = [];
    const markers: Marker[] = [];
    for (const e of events) {
      if (typeof e.step === "number" && typeof e.loss === "number") {
        xs.push(e.step);
        ys.push(e.loss);
      }
      if (
        e.kind === "anomaly" &&
        typeof e.step === "number"
      ) {
        markers.push({
          step: e.step,
          severity: typeof e.severity === "string" ? e.severity : "warning",
          detector: typeof e.detector === "string" ? e.detector : undefined,
        });
      }
    }
    markersRef.current = markers;
    if (xs.length > 0) {
      u.setData([xs, ys]);
    } else {
      u.redraw();
    }
  }, [events, ready]);

  return <div ref={containerRef} className="h-full w-full" />;
}
