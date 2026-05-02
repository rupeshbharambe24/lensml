"""lensml command-line interface (entry point: `lensml`).

Subcommands (per plan §3.2):
    lensml dashboard              start the FastAPI server (real)
    lensml watch <script.py>      run a training script under supervision (TBD)
    lensml bench run              execute the lensml-bench fixture suite (TBD)
    lensml replay <session.jsonl> deterministic demo playback (TBD)
"""

from __future__ import annotations

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

    # Use string import + factory so --reload works (uvicorn re-imports
    # the factory on file changes).
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


@app.command(name="bench")
def bench_cmd(action: str = "run") -> None:
    """Run the lensml-bench fixture suite (v0.1 build week 4)."""
    typer.echo("lensml bench lands in week 4 of the build calendar.", err=True)
    raise typer.Exit(code=2)


@app.command()
def replay(session_jsonl: str) -> None:
    """Deterministic playback of a recorded session for demos (v0.1 day 20)."""
    typer.echo("lensml replay lands later in the build calendar.", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
