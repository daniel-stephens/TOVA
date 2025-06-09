import typer # type: ignore
from pathlib import Path
from typing import Optional

from src.core.dispatchers import get_model_info_dispatch, get_thetas_documents_by_id_dispatch, get_topic_info_dispatch
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

@app.command("get-thetas-docs-by-id")
def get_thetas_by_docs_ids(
    model_path: str = typer.Option(..., help="Path to the trained model directory"),
    docs_ids: str = typer.Option(..., help="Comma-separated list of document IDs to query"),
    config: Optional[str] = typer.Option("static/config/config.yaml", help="Path to YAML config file"),
):
    """Get topic weights for specific documents by their IDs"""
    logger = init_logger(config_file=config)
    doc_ids_list = [doc_id.strip() for doc_id in docs_ids.split(",")]
    
    docs_thetas_info = get_thetas_documents_by_id_dispatch(
        model_path=model_path,
        docs_ids=doc_ids_list,
        config_path=Path(config),
        logger=logger
    )
    
    if docs_thetas_info is not None:
        typer.echo("Documents topic weights:\n")
        for doc_id, thetas in docs_thetas_info.items():
            typer.echo(f"Document ID: {doc_id}")
            for topic_id, weight in thetas.items():
                typer.echo(f"  Topic {topic_id}: {weight}")
    else:
        typer.echo("No topic weights found for the specified document IDs.")