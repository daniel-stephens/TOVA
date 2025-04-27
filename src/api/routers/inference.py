from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def infer_model():
    return {"message": "Inference initiated."}
