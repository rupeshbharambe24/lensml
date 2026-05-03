"""lensml command-line interface (entry point: `lensml`).

Subcommands (per plan §3.2):
    lensml dashboard              start the FastAPI server
    lensml watch <script.py>      run a training script under supervision (TBD)
    lensml bench run              execute the lensml-bench fixture suite (TBD)
    lensml replay <session.jsonl> deterministic demo playback (plan §3.6)
    lensml replay --scenario <name>
    lensml record <out.jsonl>     write a JSONL recording from a built-in scenario
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="lensml",
    help="Supervise your ML training runs with an LLM agent.",
    no_args_is_help=True,
)


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", help="Bind interface (default 127.0.0.1)."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn auto-reload (dev only)."),
    db_path: str = typer.Option(
        ":memory:", "--db", help="SQLite path; ':memory:' for ephemeral runs."
    ),
    zmq_endpoint: str | None = typer.Option(
        None,
        "--zmq",
        help="If set, bridge events from this ZMQ PUB endpoint (e.g. tcp://127.0.0.1:5555).",
    ),
) -> None:
    """Start the FastAPI server.

    The Next.js dashboard is run separately (``cd dashboard && pnpm dev``).
    The dashboard talks directly to this server via CORS — see plan §3.1.
    """
    import os

    import uvicorn

    if zmq_endpoint is not None:
        os.environ["LENSML_ZMQ_ENDPOINT"] = zmq_endpoint
    if db_path != ":memory:":
        os.environ["LENSML_DB_PATH"] = db_path

    uvicorn.run(
        "lensml.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def watch(script: str) -> None:
    """Run a training script under lensml supervision (v0.2)."""
    typer.echo(
        "lensml watch is not yet implemented. For now, add\n\n"
        "    import lensml\n"
        "    lensml.watch(model, optimizer, dataloader)\n\n"
        "to your training script and run it directly.",
        err=True,
    )
    raise typer.Exit(code=2)


bench_app = typer.Typer(
    name="bench",
    help="lensml-bench: run fixtures, list fixtures, render reports.",
    no_args_is_help=True,
)
app.add_typer(bench_app, name="bench")


@bench_app.command("list")
def bench_list(
    env: str | None = typer.Option(None, "--env", help="Filter by host env."),
) -> None:
    """List packaged bench fixtures (plan §6.1)."""
    from lensml.bench.fixtures import discover_fixtures

    fixtures = discover_fixtures()
    if env is not None:
        fixtures = [f for f in fixtures if f.meta.host_env == env]
    if not fixtures:
        typer.echo("(no fixtures match)")
        return
    for f in fixtures:
        typer.echo(f"{f.slug:40s}  {f.meta.difficulty:7s}  loss<{f.meta.loss_threshold}")


@bench_app.command("run")
def bench_run(
    env: str | None = typer.Option(None, "--env", help="Filter by host env."),
    pathology: str | None = typer.Option(None, "--pathology", help="Filter by pathology_class."),
    force: bool = typer.Option(
        False, "--force", help="Re-run all fixtures (ignore cached passes)."
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write report.md/report.json under this directory."
    ),
    cache: str | None = typer.Option(None, "--cache", help="Path to a JSON cache file for resume."),
    timeout: float = typer.Option(
        120.0, "--timeout", help="Per-script subprocess timeout (seconds)."
    ),
) -> None:
    """Run packaged fixtures end-to-end and print summary (plan §6, §8 D24)."""
    from lensml.bench.report import build_report
    from lensml.bench.runner import BenchRunner

    runner = BenchRunner(
        timeout=timeout,
        cache_path=Path(cache) if cache else None,
    )
    summary = runner.run_all(env=env, pathology=pathology, force=force)
    typer.echo(
        f"bench: {summary.overall_pass}/{summary.total} overall pass "
        f"({summary.overall_pass_rate:.0%}); fix passes "
        f"{summary.fix_pass}/{summary.total}; elapsed {summary.elapsed:.1f}s"
    )
    for run in summary.runs:
        flag = "✓" if run.overall_pass else "✗"
        typer.echo(f"  {flag} {run.fixture_slug:40s}  ({run.elapsed:.1f}s)")
    if out is not None:
        report = build_report(summary)
        md, js = report.write(Path(out))
        typer.echo(f"  → wrote {md} and {js}")


@bench_app.command("baseline")
def bench_baseline(
    provider: str = typer.Option(
        "groq", "--provider", help="LLM provider label (e.g. groq, gemini, cerebras)."
    ),
    env: str | None = typer.Option(None, "--env", help="Filter by host env."),
    pathology: str | None = typer.Option(None, "--pathology", help="Filter by pathology_class."),
    out: str | None = typer.Option(
        None, "--out", help="Write baseline-<provider>.json under this directory."
    ),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-script subprocess timeout."),
) -> None:
    """Run the LLM-only baseline (plan §6.3) — the credibility lynchpin.

    For each fixture, single-shots the bug.py at the configured LLM
    provider with a "diagnose + propose unified diff" prompt. Parses
    the response, applies the diff, grades the patched script. Output
    is shaped exactly like ``lensml bench run`` so reports can compare
    arms side by side.
    """
    import asyncio
    import json as _json

    from lensml.bench.baseline import BaselineRunner
    from lensml.llm.client import LensmlLLM
    from lensml.llm.routing import Task

    llm = LensmlLLM()

    def _complete(prompt: str) -> str:
        async def _go() -> str:
            resp = await llm.complete(
                Task.DIAGNOSIS,
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
            )
            return resp.content or ""

        return asyncio.run(_go())

    runner = BaselineRunner(provider=provider, complete=_complete, timeout=timeout)
    summary = runner.run_all(env=env, pathology=pathology)
    typer.echo(
        f"baseline[{provider}]: {summary.fix_passes}/{summary.total} fixes pass; "
        f"{summary.diagnosis_correct}/{summary.total} diagnoses correct; "
        f"{summary.diff_applied}/{summary.total} diffs applied; "
        f"elapsed {summary.elapsed:.1f}s"
    )
    for r in summary.results:
        flag = "✓" if r.fix_passes else ("⊘" if r.diff_applied else "✗")
        diag = r.diagnosed_class or "(unparsed)"
        typer.echo(f"  {flag} {r.fixture_slug:40s}  {diag}")
    if out is not None:
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)
        target = out_path / f"baseline-{provider}.json"
        target.write_text(
            _json.dumps(summary.to_dict(), default=str, indent=2),
            encoding="utf-8",
        )
        typer.echo(f"  → wrote {target}")


def _post_event(server: str, event: dict[str, Any]) -> None:
    """POST an event to the running dashboard server's `_test_publish`
    endpoint. Uses the stdlib so we don't add an httpx dep to the CLI
    happy path."""
    import urllib.error
    import urllib.request

    run_id = str(event.get("run_id", "demo"))
    url = f"{server.rstrip('/')}/api/runs/{run_id}/_test_publish"
    body = json.dumps(event, default=str).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as r:
            r.read()
    except urllib.error.URLError as e:
        typer.echo(f"  (server POST failed: {e})", err=True)


@app.command()
def replay(
    session: str = typer.Argument(
        None,
        help="Path to a JSONL recording to replay. Use --scenario for a built-in instead.",
    ),
    scenario: str | None = typer.Option(
        None,
        "--scenario",
        help="Built-in scenario: exploding_grad | dead_relu | missing_normalize.",
    ),
    server_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--server",
        help="Dashboard server base URL (POSTs to /api/runs/<id>/_test_publish).",
    ),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed multiplier; 2 = 2× faster."),
    fixed_delay: float = typer.Option(
        0.05,
        "--fixed-delay",
        help="Fallback delay (s) between events when ts deltas are unavailable.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Override every event's run_id (so the replay shows up under this id).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Don't POST — just print event count + kinds."
    ),
) -> None:
    """Deterministic playback of a recorded session for demos (plan §3.6).

    Either pass a JSONL file or use ``--scenario <name>`` for a built-in.
    """
    from lensml.replay import Player, read_jsonl
    from lensml.replay.scenarios import generate, list_scenarios

    if session is None and scenario is None:
        typer.echo(
            f"Pass a JSONL path or --scenario <name>. Built-in scenarios: {', '.join(list_scenarios())}",
            err=True,
        )
        raise typer.Exit(code=2)

    if scenario is not None:
        try:
            events = generate(scenario)
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=2) from e
    else:
        path = Path(session) if session else None
        if path is None or not path.exists():
            typer.echo(f"file not found: {session}", err=True)
            raise typer.Exit(code=2)
        events = read_jsonl(path)

    typer.echo(
        f"replay: {len(events)} events; speed={speed}× fixed_delay={fixed_delay}s"
        + (f" run_id={run_id}" if run_id else "")
    )

    if dry_run:
        kinds: dict[str, int] = {}
        for e in events:
            k = str(e.get("kind", "unknown"))
            kinds[k] = kinds.get(k, 0) + 1
        for k, n in sorted(kinds.items()):
            typer.echo(f"  {k}: {n}")
        return

    # Re-stamp ts so the dashboard sees fresh wall-clock times — otherwise
    # the chart x-axis (which is step) is fine, but downstream consumers
    # that filter by recent ts would skip a year-old event.
    now = time.time()
    base = events[0].get("ts") if events else 0
    base_f = float(base) if isinstance(base, (int, float)) else 0.0
    rebased: list[dict[str, Any]] = []
    for e in events:
        ev_ts = e.get("ts")
        if isinstance(ev_ts, (int, float)) and base_f:
            new_ts = now + (float(ev_ts) - base_f)
        else:
            new_ts = now
        rebased.append({**e, "ts": new_ts})

    player = Player(fixed_delay=fixed_delay, speed=speed, run_id=run_id)
    stats = player.play(rebased, dispatch=lambda ev: _post_event(server_url, ev))
    typer.echo(
        f"replay: dispatched {stats.dispatched} events in {stats.elapsed:.1f}s"
        f" (skipped {stats.skipped})"
    )
    for k, n in sorted(stats.kinds.items()):
        typer.echo(f"  {k}: {n}")


@app.command()
def record(
    out: str = typer.Argument(..., help="Path to write the JSONL recording."),
    scenario: str = typer.Option(
        ...,
        "--scenario",
        help="Built-in scenario to materialize: exploding_grad | dead_relu | missing_normalize.",
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Override the run_id field on every event."
    ),
) -> None:
    """Materialize a built-in scenario as a JSONL file (plan §3.6).

    Use the resulting file as input to ``lensml replay <file>``, or
    ship it as part of a demo bundle.
    """
    from lensml.replay.scenarios import generate, list_scenarios

    try:
        events = generate(scenario)
    except ValueError as e:
        typer.echo(str(e), err=True)
        typer.echo(f"choices: {', '.join(list_scenarios())}", err=True)
        raise typer.Exit(code=2) from e

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ev in events:
            if run_id is not None:
                ev = {**ev, "run_id": run_id}
            f.write(json.dumps(ev, default=str))
            f.write("\n")
    typer.echo(f"wrote {len(events)} events → {out}")


if __name__ == "__main__":
    app()
