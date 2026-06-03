"""
langtrain CLI — lt / langtrain command.

Usage:
    lt                               # interactive REPL (Claude Code-style)
    lt login                         # browser-based auth (no API key needed)
    lt logout                        # clear credentials
    lt whoami                        # account info + GPU availability
    lt gpu                           # list GPU options
    lt fine-tune <model> <dataset>   # one-shot fine-tuning job
    lt jobs                          # list training jobs
    lt analyze <file>                # dataset intelligence
    lt models                        # list your models
"""

from __future__ import annotations

import os
import sys

import click


def _client():
    from langtrain.client import LangtrainClient
    return LangtrainClient()


# ─────────────────────────────────────────────────────────────────────────────
# Default command: launch REPL
# ─────────────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.version_option(package_name="langtrain-ai")
@click.pass_context
def main(ctx):
    """⚡ Langtrain — fine-tune LLMs from your terminal."""
    if ctx.invoked_subcommand is None:
        # No sub-command → launch the interactive REPL
        from langtrain.repl import start
        try:
            import importlib.metadata
            version = importlib.metadata.version("langtrain-ai")
        except Exception:
            version = ""
        start(version=version)


# ─────────────────────────────────────────────────────────────────────────────
# Auth commands
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
def login():
    """Authenticate via browser — no API key needed."""
    from langtrain.auth import browser_login
    browser_login()


@main.command()
def logout():
    """Clear stored credentials."""
    from pathlib import Path
    creds = Path.home() / ".langtrain" / "credentials.json"
    if creds.exists():
        creds.unlink()
        click.echo(click.style("✓ Logged out.", fg="green"))
    else:
        click.echo("Not logged in.")


@main.command()
def whoami():
    """Show account info and GPU availability."""
    try:
        me   = _client().me()
        gpus = _client().gpu.available()
        click.echo(click.style("\n⚡ Langtrain Account", bold=True))
        click.echo(f"   Email  : {me.get('email', '—')}")
        click.echo(f"   Plan   : {me.get('plan', '—').title()}")
        click.echo(f"   Credits: {me.get('credits', '—')}")
        click.echo()
        if gpus:
            click.echo(click.style("  Available GPUs:", bold=True))
            for g in gpus:
                click.echo(f"   • {g.get('name','?')}  ×{g.get('count',1)}  ({g.get('vram_gb','?')}GB)")
        click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# One-shot commands (non-interactive use)
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
def gpu():
    """List available GPU instances."""
    try:
        gpus = _client().gpu.available()
        if not gpus:
            click.echo("No GPUs available.")
            return
        click.echo(click.style("\n  GPU Options:\n", bold=True))
        for g in gpus:
            click.echo(
                f"  {g.get('name','?'):30s} "
                f"{str(g.get('vram_gb','?'))+'GB':8s} "
                f"×{g.get('count',1)}  "
                f"${g.get('price_per_hour','?')}/hr"
            )
        click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))


@main.command("fine-tune")
@click.argument("model")
@click.argument("dataset")
@click.option("--method", default="adaptive_rank")
@click.option("--rank",   default=16, type=int)
@click.option("--epochs", default=3,  type=int)
@click.option("--stream/--no-stream", default=True)
def fine_tune(model, dataset, method, rank, epochs, stream):
    """Launch a fine-tuning job (one-shot, non-interactive)."""
    client     = _client()
    dataset_id = dataset
    if os.path.isfile(dataset):
        click.echo(f"Uploading {dataset}…")
        ds         = client.datasets.upload(dataset)
        dataset_id = ds["id"]
        click.echo(f"Uploaded: {dataset_id}")

    click.echo(f"\nLaunching {method} job  model={model}  rank={rank}  epochs={epochs}\n")
    job = client.fine_tune(model=model, dataset_id=dataset_id, method=method,
                           config={"lora_r": rank, "num_epochs": epochs})
    click.echo(click.style(f"✓ Job started: {job.job_id}", fg="green", bold=True))
    if stream:
        try:
            for step in job.stream():
                click.echo(f"  {step}")
        except KeyboardInterrupt:
            click.echo("\nDetached. Job continues in cloud. Check: lt jobs")


@main.command()
def jobs():
    """List recent training jobs."""
    try:
        job_list = _client().jobs()
        if not job_list:
            click.echo("No jobs found.")
            return
        click.echo(click.style(f"\n  {'JOB ID':24s} {'MODEL':30s} {'STATUS':12s} METHOD", bold=True))
        click.echo("  " + "─" * 80)
        for j in job_list[:20]:
            click.echo(
                f"  {j.get('id','?'):24s} "
                f"{j.get('base_model','?')[:28]:30s} "
                f"{j.get('status','?'):12s} "
                f"{j.get('method','?')}"
            )
        click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))


@main.command()
@click.argument("file")
def analyze(file):
    """Analyze a dataset and get model + config recommendations."""
    from langtrain.intelligence import DatasetIntelligence
    click.echo(f"\nAnalyzing: {file}\n")
    try:
        report = DatasetIntelligence.analyze(file)
        report.print_summary()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))


@main.command()
def models():
    """List your fine-tuned models."""
    try:
        ml = _client().models.list()
        if not ml:
            click.echo("No models found. Fine-tune one first.")
            return
        click.echo(click.style(f"\n  {'ID':28s} {'STATUS':12s} BASE", bold=True))
        click.echo("  " + "─" * 70)
        for m in ml:
            click.echo(f"  {m.get('id','?')[:26]:28s} {m.get('status','?'):12s} {m.get('base_model','?')}")
        click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))


if __name__ == "__main__":
    main()
