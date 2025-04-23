from fastapi import FastAPI # type: ignore
from src.api.routers import training, inference

app = FastAPI()
app.include_router(training.router, prefix="/train")
app.include_router(inference.router, prefix="/infer")
