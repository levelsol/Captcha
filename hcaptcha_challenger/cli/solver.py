from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Note: For Ollama version, cost calculation would need to be adapted
# since there are no API costs for local models
app = typer.Typer()

DEFAULT_CHALLENGE_DIR = Path("tmp")


@app.callback()
def dataset_callback(ctx: typer.Context):
    """
    Dataset subcommand callback. Shows help if no command is provided.
    """
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


@app.command(name="cost")
def check_cost(
    challenge_dir: Annotated[
        Path,
        typer.Option(
            help="Challenge directory to analyze", envvar="CHALLENGE_DIR", show_default=True
        ),
    ] = DEFAULT_CHALLENGE_DIR,
    output_file: Annotated[
        Optional[Path], typer.Option(help="Save stats to JSON file (optional)")
    ] = None,
    show_all_models: Annotated[
        bool, typer.Option("--all", "-a", help="Show details for all models, even with low usage")
    ] = False,
    threshold: Annotated[
        int,
        typer.Option(help="Minimum usage count to show detailed model stats", show_default=True),
    ] = 5,
):
    """
    Display model usage statistics for challenges (Ollama version - no costs)
    """
    console = Console()

    try:
        # Check if directory exists
        challenge_path = Path(challenge_dir).resolve()
        if not challenge_path.exists():
            console.print(
                Panel(
                    f"[bold red]Directory not found: {challenge_path}",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(1)

        # Search for model answer files
        search_pattern = "**/*_model_answer.json"
        answer_files = list(challenge_path.glob(search_pattern))
        
        if not answer_files:
            console.print(
                Panel(
                    f"[bold yellow]No model answer files found in {challenge_path}[/bold yellow]\n"
                    f"Make sure the directory contains challenge data with *_model_answer.json files.",
                    title="No Data Found",
                    border_style="yellow",
                )
            )
            raise typer.Exit(1)

        with console.status(f"[bold blue]Analyzing {len(answer_files)} model answer files..."):
            # For Ollama version, we'll just count usage instead of calculating costs
            stats = analyze_ollama_usage(challenge_path)

        # Create summary table
        summary_table = Table(
            title="[bold blue]Ollama Model Usage Analysis[/bold blue]",
            box=box.ROUNDED,
            border_style="blue",
            padding=(0, 1),
            width=None,
        )

        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        # Add summary information
        summary_table.add_row("Total Challenges", f"{stats['total_challenges']:,}")
        summary_table.add_row("Total API Calls", f"{stats['total_calls']:,}")
        summary_table.add_row("Most Used Model", stats['most_used_model'])

        # Model usage table
        model_table = Table(
            title="Model Usage Breakdown",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 1),
            width=None,
        )

        model_table.add_column("Model", style="magenta", no_wrap=True)
        model_table.add_column("Calls", style="yellow", justify="right")
        model_table.add_column("% of Total", style="red", justify="right")

        # Display model usage
        for model, count in stats['model_usage'].items():
            percentage = (count / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0
            model_table.add_row(
                model,
                f"{count:,}",
                f"{percentage:.1f}%",
            )

        console.print(summary_table)
        console.print(model_table)

        console.print(
            Panel(
                "[bold green]Note:[/bold green] Ollama runs locally, so there are no API costs! "
                "This analysis shows usage patterns only.",
                title="Ollama Usage",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(Panel(f"[bold red]Error: {str(e)}", title="Error", border_style="red"))
        raise typer.Exit(1)


def analyze_ollama_usage(challenge_path: Path) -> dict:
    """Analyze Ollama model usage patterns"""
    import json
    from collections import defaultdict
    
    model_usage = defaultdict(int)
    total_calls = 0
    challenge_dirs = set()
    
    for answer_file in challenge_path.rglob("*_model_answer.json"):
        try:
            # Track unique challenges by parent directory
            challenge_dirs.add(str(answer_file.parent))
            
            # Try to read the response file to get model info
            with open(answer_file, 'r') as f:
                data = json.load(f)
                
                # Try to infer model from filename or content
                model_name = "llava:latest"  # default
                if 'model' in data:
                    model_name = data['model']
                elif 'response' in data and isinstance(data['response'], dict):
                    if 'model' in data['response']:
                        model_name = data['response']['model']
                
                model_usage[model_name] += 1
                total_calls += 1
                
        except Exception:
            # If we can't read the file, still count it
            model_usage["unknown"] += 1
            total_calls += 1
    
    most_used_model = max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else "none"
    
    return {
        'total_challenges': len(challenge_dirs),
        'total_calls': total_calls,
        'model_usage': dict(model_usage),
        'most_used_model': most_used_model
    }
