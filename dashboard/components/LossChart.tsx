"use client";

import { useEffect, useRef } from "react";
import uPlot from "uplot";

import type { LensmlEvent } from "@/lib/useSSE";

const CHART_HEIGHT = 220;

/**
 * uPlot loss-vs-step chart. Receives the rolling event buffer from useSSE
 * and re-renders on update. Only events with both `step` and `loss` fields
 * contribute to the series.
 */
export function LossChart({ events }: { events: LensmlEvent[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);

  // Init the chart once on mount, destroy on unmount.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const opts: uPlot.Options = {
      width: el.clientWidth,
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

    const u = new uPlot(opts, [[0], [0]], el);
    plotRef.current = u;

    const onResize = () => {
      if (plotRef.current && el) {
        plotRef.current.setSize({ width: el.clientWidth, height: CHART_HEIGHT });
      }
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      u.destroy();
      plotRef.current = null;
    };
  }, []);

  // Push new events into the chart.
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
  }, [events]);

  return <div ref={containerRef} className="w-full" />;
}
