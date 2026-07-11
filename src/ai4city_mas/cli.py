from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ai4city_mas.config import load_config
from ai4city_mas.domain import PipelineStage, STAGE_TITLES
from ai4city_mas.pipeline import Pipeline


app = typer.Typer(no_args_is_help=True, help="AI4City multi-agent pipeline backbone.")
console = Console()


@app.command()
def plan(config: Path = typer.Option(Path("configs/default.yaml"))) -> None:
    """Show the configured 11-stage execution plan."""
    loaded = load_config(config)
    context_count = (
        len(loaded.context.times)
        * len(loaded.context.crowd_levels)
        * len(loaded.context.noise_db)
    )
    table = Table(title=f"{loaded.project.name} ({context_count} contexts per scene)")
    table.add_column("Stage", style="cyan")
    table.add_column("Owner / action")
    for stage in PipelineStage:
        table.add_row(stage.value, STAGE_TITLES[stage])
    console.print(table)


@app.command()
def run(
    config: Path = typer.Option(Path("configs/default.yaml")),
    run_id: str | None = typer.Option(None),
) -> None:
    """Execute the pipeline and write a traceable run directory."""
    state = Pipeline(load_config(config)).run(run_id=run_id)
    console.print(f"status={state['status']}")
    console.print(f"stage={state['stage']}")
    console.print(f"artifacts={len(state['artifacts'])}")
    console.print(f"run_dir={state['run_dir']}")


@app.command()
def status(run_dir: Path) -> None:
    """Print a prior run manifest."""
    manifest = run_dir.resolve() / "manifest.json"
    if not manifest.exists():
        raise typer.BadParameter(f"Manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8") as stream:
        console.print_json(json.dumps(json.load(stream), ensure_ascii=False))


@app.command("test")
def run_tests() -> None:
    """Run the standard-library unit test suite in ai4city-mas."""
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    source = str(root / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=root,
        env=env,
        check=False,
    )
    raise typer.Exit(result.returncode)
