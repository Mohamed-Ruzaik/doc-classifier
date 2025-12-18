# app/main.py

from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ocr import ocr_file
from .ml_model import classifier

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# App
app = FastAPI(title="Document Classifier API")

# ✅ Static files (must be AFTER app + PROJECT_ROOT exist)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

# CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from typing import List, Optional
from pydantic import BaseModel

class TopKItem(BaseModel):
    label: str
    confidence: float

class ClassificationResult(BaseModel):
    id: str
    filename: str
    predicted_class: str
    confidence: Optional[float] = None
    top_k: Optional[List[TopKItem]] = None

@app.post("/classify", response_model=ClassificationResult)
async def classify_document(file: UploadFile = File(...)):
    filename = file.filename
    suffix = Path(filename).suffix.lower()

    if suffix not in [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    doc_id = str(uuid.uuid4())
    save_path = UPLOADS_DIR / f"{doc_id}{suffix}"

    with save_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    text = ocr_file(save_path)
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract any text from document")

    label, confidence, top_k = classifier.predict_text(text, top_k=3, unknown_threshold=0.40)

    return ClassificationResult(
        id=doc_id,
        filename=filename,
        predicted_class=label,
        confidence=confidence,
        top_k=top_k,
    )



@app.get("/", response_class=HTMLResponse)
def root():
    return "<h3>Document Classifier API running</h3><p>Frontend at /static/index.html</p>"
