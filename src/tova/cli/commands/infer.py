import typer # type: ignore
from tova.core.dispatchers import infer_model_dispatch
from tova.utils.common import init_logger
from tova.utils.tm_utils import prepare_training_data, normalize_json_data
from pathlib import Path
from typing import Optional

app = typer.Typer()

@app.command()
def run(
    model_path: str = typer.Option(...),
    data: str = typer.Option(...),
    text_col: str = typer.Option("tokenized_text"),
    id_col: Optional[str] = typer.Option("id"),
    config: Optional[str] = Path("./static/config/config.yaml"),
):  
    logger = init_logger(config_file=config)

    if Path(data).is_file():
        normalized_data = prepare_training_data(
            path=data,
            logger=logger,
            text_col=text_col,
            id_col=id_col,
            get_embeddings=True
        )
    else:
        normalized_data = normalize_json_data(
            raw_data=data,
            text_col=text_col,
            id_col=id_col
        )

    thetas, duration = infer_model_dispatch(model_path, normalized_data)
    
    # stringify thetas
    
    typer.echo(f"Thetas:\n{thetas}")
    
    typer.echo(f"Training completed in {duration:.2f} seconds")
