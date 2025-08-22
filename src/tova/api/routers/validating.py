from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel
import pandas as pd
import io, json

# Initialize the API router
router = APIRouter()

# Allowed file extensions
ACCEPTED_FILETYPES = {"csv", "xls", "xlsx", "json", "jsonl"}

# ---------- Response models (for OpenAPI / Swagger docs) ----------
class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: str
    file: Optional[str] = None

class SingleSuccess(BaseModel):
    status: Literal["success"] = "success"
    preview: List[Dict[str, Any]]

class MultiSuccess(BaseModel):
    status: Literal["success"] = "success"
    previews: Dict[str, List[Dict[str, Any]]]

# ---------- Utility functions ----------
def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

def validate_filetype(file: UploadFile) -> bool:
    return get_extension(file.filename) in ACCEPTED_FILETYPES

def read_file(file: UploadFile) -> pd.DataFrame:
    ext = get_extension(file.filename)
    content = file.file.read()
    try:
        if ext == "csv":
            return pd.read_csv(io.BytesIO(content))
        elif ext in {"xls", "xlsx"}:
            return pd.read_excel(io.BytesIO(content))
        elif ext == "json":
            return pd.json_normalize(json.loads(content))
        elif ext == "jsonl":
            lines = content.decode("utf-8").splitlines()
            return pd.DataFrame([json.loads(line) for line in lines if line.strip()])
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read {file.filename}: {e}")

# ---------- Routes ----------
@router.post(
    "/single",
    summary="Validate a single file and return 5-row preview",
    response_model=SingleSuccess,
    responses={400: {"model": ErrorResponse}},
)
async def validate_single(file: UploadFile = File(..., description="One file (csv, xls, xlsx, json, jsonl)")):
    if not validate_filetype(file):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
    df = read_file(file)
    preview = df.head(5).to_dict(orient="records")
    return {"status": "success", "preview": preview}

@router.post(
    "/multiple",
    summary="Validate multiple files and return 5-row preview per file",
    response_model=MultiSuccess,
    responses={400: {"model": ErrorResponse}},
)
async def validate_multiple(
    files: List[UploadFile] = File(..., description="Multiple files (csv/xls/xlsx/json/jsonl)"),
    text_columns: str = Form(..., description='JSON list of text columns, e.g. ["text"]'),
    id_column: str = Form(..., description="Name of the ID column"),
    label_column: str = Form(..., description="Name of the label column"),
):
    # Parse and validate text_columns
    try:
        text_cols = json.loads(text_columns)
        if not isinstance(text_cols, list) or not all(isinstance(x, str) for x in text_cols):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid text_columns. Must be a JSON list of strings.")

    previews: Dict[str, List[Dict[str, Any]]] = {}
    for f in files:
        if not validate_filetype(f):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")
        df = read_file(f)
        missing_cols = [c for c in (text_cols + [id_column, label_column]) if c not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"{f.filename} missing required columns: {missing_cols}")
        previews[f.filename] = df.head(5).to_dict(orient="records")

    return {"status": "success", "previews": previews}
