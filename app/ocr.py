# app/ocr.py

from pathlib import Path
from PIL import Image
import pytesseract
from pdf2image import convert_from_path


# ===== OCR CONFIG =====
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_BIN = r"D:\Tools\poppler-25.12.0\Library\bin"

PDF_DPI = 220          # balance between speed & OCR quality
PDF_MAX_PAGES = 2      # STEP 4: OCR first 2 pages only

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


def _ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image and normalize text."""
    text = pytesseract.image_to_string(
        image,
        lang="eng",
        config="--oem 1 --psm 6"
    )
    return " ".join(text.split()).strip()


def ocr_file(path: Path) -> str:
    """
    OCR helper used at inference time.

    - PDFs: OCR first 2 pages (joined)
    - Images: OCR once
    """
    suffix = path.suffix.lower()

    try:
        # ===== PDF =====
        if suffix == ".pdf":
            pages = convert_from_path(
                str(path),
                dpi=PDF_DPI,
                first_page=1,
                last_page=PDF_MAX_PAGES,
                poppler_path=POPPLER_BIN
            )

            if not pages:
                return ""

            parts = []
            for img in pages:
                t = _ocr_image(img)
                if t:
                    parts.append(t)

            return " ".join(parts).strip()

        # ===== IMAGE =====
        image = Image.open(path).convert("RGB")
        return _ocr_image(image)

    except Exception:
        # Never crash the API due to OCR failure
        return ""
