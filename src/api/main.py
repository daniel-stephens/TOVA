# python -m uvicorn src.api.main:app --reload --port 8989
from fastapi import FastAPI # type: ignore
from src.api.routers import training, inference, queries

app = FastAPI(
    title="TOVA API",
    version="1.0.0",
    swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}, "deepLink": True},
)

app.include_router(training.router, prefix="/train", tags=["Training"])
app.include_router(inference.router, prefix="/infer", tags=["Inference"])
app.include_router(queries.router, prefix="/queries", tags=["Queries"])