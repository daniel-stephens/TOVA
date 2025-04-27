import typer # type: ignore
from src.commands import train, infer, visualize

app = typer.Typer()

#-------------------------------------------------------
# CLI for Topic Modeling Training
#-------------------------------------------------------
train_app = typer.Typer()
train_app.command("run")(train.run)  # maps to: cli train run
app.add_typer(train_app, name="train")

#-------------------------------------------------------
# CLI for Topic Modeling Inference
#-------------------------------------------------------
infer_app = typer.Typer()
infer_app.command("run")(infer.run)
app.add_typer(infer_app, name="infer")

#-------------------------------------------------------
# CLI for Topic Modeling Visualization
#-------------------------------------------------------
# @TODO

if __name__ == "__main__":
    app()
