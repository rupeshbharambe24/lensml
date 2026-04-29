"use client";

import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";

import type { LensmlEvent } from "@/lib/useSSE";

const CHART_HEIGHT = 220;
const FALLBACK_WIDTH = 600;

/**
 * uPlot loss-vs-step chart. Receives the rolling event buffer from useSSE
 * and re-renders on update.
 *
 * Layout-stability concerns:
 *   - At mount, the flex/grid parent may not yet have a non-zero
 *     clientWidth — uPlot would then render an invisible 0-pixel canvas.
 *     We use a ResizeObserver to defer initialization until the container
 *     reports a real width, and to track size changes after that.
 */
export function LossChart({ events }: { events: LensmlEvent[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);
  const [ready, setReady] = useState(false);

  // Lifecycle: observe container size, init chart once non-zero width
  // arrives, resize on subsequent changes, destroy on unmount.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

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

    // Try once immediately in case the layout is already settled.
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
    for (const e of events) {
      if (typeof e.step === "number" && typeof e.loss === "number") {
        xs.push(e.step);
        ys.push(e.loss);
      }
    }
    if (xs.length > 0) {
      u.setData([xs, ys]);
    }
  }, [events, ready]);

  return <div ref={containerRef} className="h-full w-full" />;
}
