import hashlib
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pdf2image import convert_from_path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion

import joblib


# ====== CONFIG ======
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_BIN = r"D:\Tools\poppler-25.12.0\Library\bin"

PDF_DPI = 220                # 200 faster, 250 better OCR
PDF_MAX_PAGES = 2            # keep 1 for speed; try 2 later
MIN_CHARS = 40               # skip too-short OCR
RANDOM_STATE = 42

# Use more features than 5000; still light-weight
TFIDF_MAX_FEATURES = 20000

# OCR parallelism (Ryzen sweet spot usually 4–8)
OCR_WORKERS = 8

VALID_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ====== PATHS ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = DATA_DIR / "ocr_cache"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.as_posix()}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = img.convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    img = img.point(lambda p: 255 if p > 160 else 0)  # binarize
    return img


def _tesseract_ocr(img: Image.Image) -> str:
    # LSTM + block of text
    config = "--oem 1 --psm 6"
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    return " ".join(text.split()).strip()


def _ocr_one_file(full_path_str: str, tesseract_exe: str, poppler_bin: str,
                  pdf_dpi: int, pdf_max_pages: int) -> str:
    """
    Runs in worker process. Returns OCR text (may be empty).
    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
    path = Path(full_path_str)
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            pages = convert_from_path(
                str(path),
                dpi=pdf_dpi,
                first_page=1,
                last_page=pdf_max_pages,
                poppler_path=poppler_bin,
                fmt="png"
            )
            if not pages:
                return ""
            # OCR first rendered page(s)
            parts = []
            for p in pages:
                img = _preprocess_for_ocr(p)
                t = _tesseract_ocr(img)
                if t:
                    parts.append(t)
            return " ".join(parts).strip()

        else:
            img = Image.open(path)
            img = _preprocess_for_ocr(img)
            return _tesseract_ocr(img)

    except Exception:
        return ""


def ocr_with_cache(full_path: Path) -> str:
    """
    Cache wrapper around worker OCR.
    """
    key = _cache_key(full_path)
    cache_path = CACHE_DIR / f"{key}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore").strip()
    return ""  # means "not cached yet"


def write_cache(full_path: Path, text: str) -> None:
    key = _cache_key(full_path)
    cache_path = CACHE_DIR / f"{key}.txt"
    cache_path.write_text(text, encoding="utf-8")


def load_dataset_parallel(labels_csv: Path):
    df = pd.read_csv(labels_csv)

    tasks = []
    kept_texts, kept_labels = [], []

    empty_log = DATA_DIR / "empty_or_failed.txt"
    skipped = 0
    processed = 0

    # Build task list: only valid files; use cache when available
    rows = df.to_dict("records")
    total = len(rows)

    with empty_log.open("w", encoding="utf-8") as log:
        # First pass: serve from cache immediately; queue uncached for workers
        for row in rows:
            file_path = row["file_path"]
            label = row["label"]
            full_path = (PROJECT_ROOT / file_path)

            if not full_path.exists():
                log.write(f"[MISSING] {full_path}\n")
                skipped += 1
                continue

            if full_path.suffix.lower() not in VALID_EXTS:
                log.write(f"[SKIP_EXT] {full_path}\n")
                skipped += 1
                continue

            cached = ocr_with_cache(full_path)
            if cached and len(cached) >= MIN_CHARS:
                kept_texts.append(cached)
                kept_labels.append(label)
            elif cached and len(cached) < MIN_CHARS:
                log.write(f"[EMPTY_CACHED] {full_path}\n")
                skipped += 1
            else:
                tasks.append((str(full_path), label))

        print(f"[INFO] Total rows: {total}")
        print(f"[INFO] From cache kept: {len(kept_texts)}")
        print(f"[INFO] To OCR (uncached): {len(tasks)}")
        print(f"[INFO] Skipped so far: {skipped}")

        # Worker OCR for uncached
        if tasks:
            with ProcessPoolExecutor(max_workers=OCR_WORKERS) as ex:
                future_to_item = {
                    ex.submit(
                        _ocr_one_file,
                        full_path_str,
                        TESSERACT_EXE,
                        POPPLER_BIN,
                        PDF_DPI,
                        PDF_MAX_PAGES
                    ): (full_path_str, label)
                    for (full_path_str, label) in tasks
                }

                for fut in as_completed(future_to_item):
                    full_path_str, label = future_to_item[fut]
                    processed += 1

                    text = fut.result() or ""
                    full_path = Path(full_path_str)

                    # cache whatever we got (even empty) so reruns are fast
                    write_cache(full_path, text)

                    if not text or len(text) < MIN_CHARS:
                        log.write(f"[EMPTY] {full_path}\n")
                        skipped += 1
                    else:
                        kept_texts.append(text)
                        kept_labels.append(label)

                    if processed % 100 == 0 or processed == len(tasks):
                        print(
                            f"OCR processed {processed}/{len(tasks)} | "
                            f"total kept={len(kept_texts)} | skipped={skipped}"
                        )

    print(f"Finished dataset load: kept={len(kept_texts)}, skipped={skipped}")
    print(f"Empty/missing log: {empty_log}")
    return kept_texts, kept_labels


def main():
    # sanity checks
    if not Path(TESSERACT_EXE).exists():
        raise FileNotFoundError(f"Tesseract exe not found: {TESSERACT_EXE}")
    if not Path(POPPLER_BIN).exists():
        raise FileNotFoundError(f"Poppler bin folder not found: {POPPLER_BIN}")

    labels_csv = DATA_DIR / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at: {labels_csv}")

    print("[DEBUG] PROJECT_ROOT:", PROJECT_ROOT)
    print("[DEBUG] labels_csv:", labels_csv)
    print("[DEBUG] OCR cache dir:", CACHE_DIR)
    print("[DEBUG] OCR workers:", OCR_WORKERS)

    texts, labels = load_dataset_parallel(labels_csv)

    if len(texts) < 200:
        raise RuntimeError("Too few usable OCR texts. Likely OCR failing or MIN_CHARS too high.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    vectorizer = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            lowercase=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
            lowercase=True
        )),
    ])


    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred))

    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    clf_path = MODELS_DIR / "classifier.pkl"
    joblib.dump(vectorizer, tfidf_path)
    joblib.dump(clf, clf_path)

    print(f"Saved vectorizer to {tfidf_path}")
    print(f"Saved classifier to {clf_path}")


if __name__ == "__main__":
    main()
