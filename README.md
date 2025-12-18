# Document Classifier (FastAPI + OCR + TF-IDF)

A production-oriented **document classification** system that accepts **PDFs and images**, performs **OCR**, and predicts the **document type** using a **TF-IDF + Logistic Regression** model.

---

## Features

* FastAPI backend with `/classify` endpoint (JSON output)
* Simple HTML/JavaScript frontend
* OCR pipeline:

  * PDFs processed using `pdf2image` + **Poppler**
  * Images processed using **Pillow (PIL)**
  * Text extraction via **Tesseract OCR** (`--oem 1 --psm 6`)
* Machine learning model:

  * TF-IDF (word + character n-grams)
  * Logistic Regression classifier
* Prediction confidence score
* Top-3 class predictions
* Optional **"unknown"** label using a confidence threshold (recommended for production)

---

## Supported Document Classes (RVL-CDIP style)

* advertisement
* budget
* email
* file folder
* form
* handwritten
* invoice
* letter
* memo
* news article
* presentation
* questionnaire
* resume
* scientific publication
* scientific report
* specification

---

## Project Structure

```
app/        # FastAPI app, OCR pipeline, model loader
scripts/    # Training scripts (OCR caching + model training)
models/     # Saved model artifacts (vectorizer + classifier)
static/     # Frontend (index.html, app.js, styles.css)
uploads/    # Uploaded documents (gitignored)
data/       # labels.csv, OCR cache, dataset (dataset gitignored)
```

---

## Setup (Windows)

### 1. Create Virtual Environment and Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Install External Tools

#### Tesseract OCR

* Install using the official Windows installer

#### Poppler

* Required for PDF to image conversion (`pdf2image`)

After installation, update paths in the following files:

* `app/ocr.py`
* `scripts/train_text_classifier.py`

Set:

* `TESSERACT_EXE`
* `POPPLER_BIN`

---

## Run the API and Frontend

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Access URLs

* Swagger UI:
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

* Frontend:
  [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)

---

## Training the Model (Optional)

> **Note:** This repository does **not** include the RVL-CDIP dataset.

### Expected Dataset Layout

```
data/docs/<class_name>/**
```

---

### Step 1: Generate `labels.csv`

```powershell
python rebuild_labels_csv.py
```

---

### Step 2: Train the Model

```powershell
python scripts/train_text_classifier.py
```

---

## Generated Model Artifacts

After training, the following files are saved:

```
models/tfidf_vectorizer.pkl
models/classifier.pkl
```

---

## Notes

* Do **NOT** commit the dataset (`data/docs/`) to GitHub
* Do **NOT** commit OCR cache (`data/ocr_cache/`)
* For multi-page PDFs, OCR only the **first 2 pages** for better performance
* Using an **"unknown"** class for low-confidence predictions improves real-world reliability

---

## License

MIT License © 2025 Mohamed Ruzaik

