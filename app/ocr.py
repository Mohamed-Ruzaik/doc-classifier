import os
from pathlib import Path
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

PDF_DPI = int(os.getenv("PDF_DPI", "220"))
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "2"))

def _ocr_image(image: Image.Image) -> str:
    text = pytesseract.image_to_string(image, lang="eng", config="--oem 1 --psm 6")
    return " ".join(text.split()).strip()

def ocr_file(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            pages = convert_from_path(str(path), dpi=PDF_DPI, first_page=1, last_page=PDF_MAX_PAGES)
            parts = [_ocr_image(p) for p in pages]
            return " ".join([t for t in parts if t]).strip()

        return _ocr_image(Image.open(path).convert("RGB"))
    except Exception:
        return ""
