"""lensml command-line interface (entry point: `lensml`).

Subcommands (per plan §3.2):
    lensml watch <script.py>      run a training script under supervision
    lensml dashboard              start the FastAPI server + open browser
    lensml bench run              execute the lensml-bench fixture suite
    lensml replay <session.jsonl> deterministic demo playback
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="lensml",
    help="Supervise your ML training runs with an LLM agent.",
    no_args_is_help=True,
)


@app.command()
def watch(script: str) -> None:
    """Run a training script under lensml supervision."""
    raise NotImplementedError("Implemented Day 5 — see lensml-plan.md §8")


@app.command()
def dashboard(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server and open the dashboard in a browser."""
    raise NotImplementedError("Implemented Day 3 — see lensml-plan.md §8")


@app.command(name="bench")
def bench_cmd(action: str = "run") -> None:
    """Run the lensml-bench fixture suite."""
    raise NotImplementedError("Implemented Day 24 — see lensml-plan.md §8")


@app.command()
def replay(session_jsonl: str) -> None:
    """Deterministic playback of a recorded session for demos."""
    raise NotImplementedError("Implemented Day 20 — see lensml-plan.md §8")


if __name__ == "__main__":
    app()
