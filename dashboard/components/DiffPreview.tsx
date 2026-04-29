"use client";

/**
 * Placeholder for the proposed-fix diff preview.
 * Real impl lands in step 2.5 (lensml-implementation-sequence.md):
 *   - react-diff-viewer-continued rendering
 *   - [y]/[n]/[u] keyboard shortcuts
 *   - POSTs to /api/runs/{id}/decisions/{decision_id}/{approve|reject|undo}
 */
export function DiffPreview() {
  return (
    <div className="flex h-full flex-col rounded-lg border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-xs uppercase tracking-wider text-neutral-400">
          Diff preview
        </h2>
        <span className="rounded-full bg-neutral-800 px-2 py-0.5 text-[10px] text-neutral-500">
          empty
        </span>
      </div>
      <div className="mt-3 flex-1 overflow-y-auto text-sm text-neutral-500">
        <p className="font-mono text-xs">No proposed fix.</p>
        <p className="mt-2 text-xs leading-relaxed text-neutral-600">
          A proposed patch will appear here when the agent calls{" "}
          <code className="rounded bg-neutral-800 px-1 py-0.5 text-neutral-400">
            propose_fix
          </code>
          . Press <kbd className="rounded bg-neutral-800 px-1.5 py-0.5">y</kbd> to accept,{" "}
          <kbd className="rounded bg-neutral-800 px-1.5 py-0.5">n</kbd> to reject,{" "}
          <kbd className="rounded bg-neutral-800 px-1.5 py-0.5">u</kbd> to undo.
        </p>
      </div>
    </div>
  );
}
