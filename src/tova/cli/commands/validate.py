import typer  # type: ignore
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List
from tova.utils.common import init_logger

app = typer.Typer()

ACCEPTED_FILETYPES = {"csv", "xls", "xlsx", "json", "jsonl"}

def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

def validate_filetype(filepath: Path) -> bool:
    return get_extension(filepath.name) in ACCEPTED_FILETYPES

def read_file(filepath: Path) -> pd.DataFrame:
    ext = get_extension(filepath.name)
    try:
        if ext == "csv":
            return pd.read_csv(filepath)
        elif ext in {"xls", "xlsx"}:
            return pd.read_excel(filepath)
        elif ext == "json":
            return pd.json_normalize(json.load(open(filepath)))
        elif ext == "jsonl":
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            return pd.DataFrame(lines)
        else:
            raise ValueError(f"Unsupported file type: {filepath.name}")
    except Exception as e:
        raise ValueError(f"Could not read {filepath.name}: {e}")


@app.command()
def run(
    file: Path = typer.Option(..., exists=True, help="Path to the input file"),
    text_columns: str = typer.Option(..., help='JSON list of text columns, e.g. \'["text"]\''),
    id_column: str = typer.Option(..., help="Name of the ID column"),
    label_column: str = typer.Option(..., help="Name of the label column"),
    config: Optional[Path] = Path("./static/config/config.yaml")
):
    """
    Validate the input file for expected columns and print a preview.
    """
    logger = init_logger(config_file=config)

    if not validate_filetype(file):
        typer.echo(f"❌ Unsupported file type: {file.name}")
        raise typer.Exit(code=1)

    try:
        df = read_file(file)
    except Exception as e:
        typer.echo(f"❌ Error reading file: {e}")
        raise typer.Exit(code=1)

    # Parse text_columns
    try:
        text_cols: List[str] = json.loads(text_columns)
        if not isinstance(text_cols, list) or not all(isinstance(x, str) for x in text_cols):
            raise ValueError
    except Exception:
        typer.echo("❌ Invalid text_columns. Must be a JSON list of strings.")
        raise typer.Exit(code=1)

    # Check required columns
    required_cols = text_cols + [id_column, label_column]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        typer.echo(f"❌ Missing required columns in {file.name}: {missing_cols}")
        raise typer.Exit(code=1)

    # Print preview
    typer.echo(f"✅ {file.name} validated successfully.")
    typer.echo("\nPreview (first 5 rows):")
    typer.echo(df.head(5).to_string(index=False))


