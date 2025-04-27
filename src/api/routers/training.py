from fastapi import APIRouter # type: ignore
from pydantic import BaseModel # type: ignore
from src.commands import train

router = APIRouter()

class TrainRequest(BaseModel):
    model: str
    data: str
    text_col: str = "tokenized_text"
    output: str

@router.post("/")
def train_model(req: TrainRequest):
    train.run(
        model=req.model,
        data=req.data,
        text_col=req.text_col,
        output=req.output
    )
    return {"status": "Training started"}