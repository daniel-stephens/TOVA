import typer
from src.topic_models.traditional.tomotopy_lda_tm_model import TomotopyLDATMmodel
from pathlib import Path
import time

# Temporary registry for future extension
MODEL_REGISTRY = {
    "tomotopy": TomotopyLDATMmodel
}

def run(
    model: str = typer.Option(..., help="Model name (e.g., 'tomotopy')"),
    data: str = typer.Option(..., help="Path to the training data"),
    text_col: str = typer.Option("tokenized_text", help="Text column name"),
    output: str = typer.Option(..., help="Path to save the model"),
):
    typer.echo(f"Running training with model: {model}")
    model_cls = MODEL_REGISTRY.get(model)
    if model_cls is None:
        raise typer.BadParameter(f"Unknown model: {model}")

    tm_model = model_cls(model_path=output)
    start = time.perf_counter()
    duration = tm_model.train_model(data, text_col=text_col)
    typer.echo(f"Training completed in {duration:.2f} seconds")
