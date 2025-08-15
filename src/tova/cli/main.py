import typer # type: ignore
from tova.cli.commands import train, infer, model_query

app = typer.Typer()
app.add_typer(train.app, name="train")
app.add_typer(infer.app, name="infer")
app.add_typer(model_query.app, name="model_query")


if __name__ == "__main__":
    app()
