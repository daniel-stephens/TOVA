import typer # type: ignore
from pathlib import Path
from typing import Optional

from src.core.dispatchers import get_model_info_dispatch, get_topic_info_dispatch
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
    typer.echo(model_info)

@app.command("topic-info")
def topic_info(
    topic_id: int = typer.Option(..., help="ID of the topic to query"),
    model_path: str = typer.Option(..., help="Path to the trained model directory"),
    config: Optional[str] = typer.Option("static/config/config.yaml", help="Path to YAML config file"),
):
    """Show full topic model info as records"""
    logger = init_logger(config_file=config)
    topic_info = get_topic_info_dispatch(
        topic_id=topic_id,
        model_path=model_path,
        config_path=Path(config),
        logger=logger
    ) 
    if topic_info is not None:
        typer.echo(f"Topic {topic_id} info:\n")
        typer.echo(topic_info)
    else:
        typer.echo(f"No information available for topic {topic_id}.")
