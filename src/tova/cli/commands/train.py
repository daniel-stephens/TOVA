import json
import pathlib
from pathlib import Path
from typing import Optional

import typer # type: ignore

from tova.core.dispatchers import train_model_dispatch
from tova.utils.common import init_logger
from tova.utils.tm_utils import prepare_training_data, normalize_json_data

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
            id_col=id_col,
            logger=logger
        )

    parsed_tr_params = {}
    if tr_params:
        raw_params = tr_params
        candidate_path = Path(tr_params)
        if candidate_path.is_file():
            raw_params = candidate_path.read_text(encoding="utf8")
        try:
            parsed_tr_params = json.loads(raw_params)
            if not isinstance(parsed_tr_params, dict):
                raise typer.BadParameter("--tr-params must decode to a JSON object")
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(
                f"Unable to parse --tr-params as JSON: {exc.msg}") from exc

    duration = train_model_dispatch(
        model=model,
        data=normalized_data,
        output=output,
        model_name=pathlib.Path(output).name,
        config_path=config,
        tr_params=parsed_tr_params,
        logger=logger
    )
    
    typer.echo(f"Training completed in {duration:.2f} seconds")
