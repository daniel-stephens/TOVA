import typer # type: ignore
from src.core.dispatchers import train_model_dispatch
from src.utils.common import init_logger
from src.utils.tm_utils import prepare_training_data, normalize_json_data
from pathlib import Path
from typing import Optional

app = typer.Typer()

@app.command()
def run(
    model: str = typer.Option(...),
    data: str = typer.Option(...),
    text_col: str = typer.Option("tokenized_text"),
    id_col: Optional[str] = typer.Option("id"),
    output: str = typer.Option(...),
    config: Optional[str] = Path("./static/config/config.yaml"),
    tr_params: Optional[str] = typer.Option(None)
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

    duration = train_model_dispatch(
        model=model,
        data=normalized_data,
        output=output,
        config_path=config,
        tr_params=tr_params,
        logger=logger
    )
    
    typer.echo(f"Training completed in {duration:.2f} seconds")
