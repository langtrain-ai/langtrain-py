"""
langtrain CLI — lt / langtrain command.

Usage:
    lt login                         # authenticate
    lt whoami                        # account info + GPU availability
    lt gpu                           # list available GPUs
    lt fine-tune <model> <dataset>   # launch a training job
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


@click.group()
@click.version_option(package_name="langtrain")
def main():
    """⚡ Langtrain — train, align, and deploy LLMs."""
    pass


@main.command()
@click.argument("api_key", required=False)
def login(api_key):
    """Save your API key. Get one at https://app.langtrain.xyz/home/settings"""
    if not api_key:
        api_key = click.prompt("Paste your Langtrain API key", hide_input=True)
    # Write to ~/.langtrain/credentials
    import json
    from pathlib import Path
    creds_path = Path.home() / ".langtrain" / "credentials.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text(json.dumps({"api_key": api_key}, indent=2))
    os.environ["LANGTRAIN_API_KEY"] = api_key
    try:
        me = _client().me()
        click.echo(click.style(f"✓ Logged in as {me.get('email', 'unknown')}", fg="green", bold=True))
        click.echo(f"  Plan: {me.get('plan', 'free').title()}")
    except Exception as e:
        click.echo(click.style(f"✗ Auth failed: {e}", fg="red"))
        sys.exit(1)


@main.command()
def whoami():
    """Show account info and GPU availability."""
    try:
        me = _client().me()
        gpus = _client().gpu.available()
        click.echo(click.style("\n⚡ Langtrain Account", bold=True))
        click.echo(f"   Email : {me.get('email', '—')}")
        click.echo(f"   Plan  : {me.get('plan', '—').title()}")
        click.echo(f"   Credits: {me.get('credits', '—')}")
        click.echo()
        if gpus:
            click.echo(click.style("  Available GPUs:", bold=True))
            for g in gpus:
                vram = g.get('vram_gb', '?')
                name = g.get('name', g.get('type', '?'))
                count = g.get('count', 1)
                click.echo(f"   • {name}  ×{count}  ({vram}GB VRAM)")
        else:
            click.echo("  No GPU instances available right now.")
        click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)


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
                f"  {g.get('name', '?'):30s} "
                f"{str(g.get('vram_gb', '?')) + 'GB':8s} "
                f"×{g.get('count', 1)}  "
                f"${g.get('price_per_hour', '?')}/hr"
            )
        click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))


@main.command("fine-tune")
@click.argument("model")
@click.argument("dataset")
@click.option("--method", default="adaptive_rank", help="Training method (adaptive_rank, qlora, dpo, grpo)")
@click.option("--rank", default=16, help="LoRA rank")
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--stream/--no-stream", default=True, help="Stream training steps")
def fine_tune(model, dataset, method, rank, epochs, stream):
    """Launch a fine-tuning job. MODEL: model name or ID. DATASET: dataset ID or file path."""
    import json
    client = _client()
    dataset_id = dataset

    # If it looks like a file path, upload it first
    if os.path.isfile(dataset):
        click.echo(f"Uploading dataset: {dataset}")
        ds = client.datasets.upload(dataset)
        dataset_id = ds["id"]
        click.echo(f"Dataset uploaded: {dataset_id}")

    click.echo(f"\nLaunching {method} training job…")
    click.echo(f"  Model : {model}")
    click.echo(f"  Method: {method}  rank={rank}  epochs={epochs}\n")

    job = client.fine_tune(
        model=model,
        dataset_id=dataset_id,
        method=method,
        config={"lora_r": rank, "num_epochs": epochs},
    )
    click.echo(click.style(f"Job started: {job.job_id}", fg="green", bold=True))

    if stream:
        try:
            for step in job.stream():
                click.echo(f"  {step}")
        except KeyboardInterrupt:
            click.echo("\nInterrupted. Job continues running. Check status with: lt jobs")
    else:
        click.echo(f"  Track at: https://app.langtrain.xyz/home/training-jobs")


@main.command()
def jobs():
    """List recent training jobs."""
    try:
        job_list = _client().jobs()
        if not job_list:
            click.echo("No jobs found.")
            return
        click.echo(click.style(f"\n  {'JOB ID':24s} {'MODEL':30s} {'STATUS':12s} {'METHOD'}", bold=True))
        click.echo("  " + "─" * 80)
        for j in job_list[:20]:
            click.echo(
                f"  {j.get('id', '?'):24s} "
                f"{j.get('base_model', '?')[:28]:30s} "
                f"{j.get('status', '?'):12s} "
                f"{j.get('method', '?')}"
            )
        click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))


@main.command()
@click.argument("file")
def analyze(file):
    """Analyze a dataset file and get model + training recommendations."""
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
        model_list = _client().models.list()
        if not model_list:
            click.echo("No models found. Fine-tune one first.")
            return
        click.echo(click.style(f"\n  {'MODEL ID':28s} {'NAME':28s} {'STATUS':12s} {'BASE'}", bold=True))
        click.echo("  " + "─" * 80)
        for m in model_list:
            click.echo(
                f"  {m.get('id', '?')[:26]:28s} "
                f"{m.get('name', '?')[:26]:28s} "
                f"{m.get('status', '?'):12s} "
                f"{m.get('base_model', '?')}"
            )
        click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))


if __name__ == "__main__":
    main()
