import typer # type: ignore
from pathlib import Path
from typing import Optional

from src.core.dispatchers import get_model_info_dispatch, get_thetas_dispatch
from src.utils.common import init_logger

app = typer.Typer(help="Commands to query trained models")


@app.command("model-info")
def model_info(
    model_path: str = typer.Option(..., help="Path to the trained model directory"),
    config: Optional[str] = typer.Option("static/config/config.yaml", help="Path to YAML config file"),
):
    """Show full topic model info as records"""
    logger = init_logger(config_file=config)
    model_info = get_model_info_dispatch(
        model_path=model_path,
        config_path=Path(config),
        logger=logger
    )
    typer.echo("Full model info:\n")
    for topic in model_info:
        typer.echo(topic)

@app.command("topic-distribution")
def topic_distribution(
    model_path: str = typer.Option(..., help="Path to the trained model directory"),
    config: Optional[str] = typer.Option("static/config/config.yaml", help="Path to YAML config file"),
):
    """Show topic distribution for the model"""
    logger = init_logger(config_file=config)
    
    thetas = get_thetas_dispatch(
        model_path=model_path,
        config_path=Path(config),
        logger=logger
    )
    if thetas is not None:
        typer.echo("Topic distribution:\n")
        for topic in thetas:
            typer.echo(topic)
    else:
        typer.echo("No topic distribution available.")
        