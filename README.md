# Document Classifier (FastAPI + OCR + TF-IDF)

A production-oriented **document classification system** that accepts **PDFs and images**, performs **OCR**, and predicts the **document type** using a **TF-IDF + Logistic Regression** model.

---

## Features

* FastAPI backend with `/classify` endpoint (JSON response)
* Simple HTML + JavaScript frontend
* OCR pipeline:

  * PDFs processed using `pdf2image` + **Poppler**
  * Images processed using **Pillow (PIL)**
  * Text extraction via **Tesseract OCR** (`--oem 1 --psm 6`)
* Machine learning model:

  * TF-IDF (word + character n-grams)
  * Logistic Regression classifier
* Prediction confidence score
* Top-3 class predictions
* Optional **"unknown"** label for low-confidence predictions (recommended)

---

## Supported Document Classes

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
app/        # FastAPI app, OCR logic, model loader
scripts/    # Training scripts (OCR caching + training)
models/     # Trained model artifacts
static/     # Frontend (HTML, JS, CSS)
uploads/    # Uploaded documents (gitignored)
data/       # labels.csv, OCR cache, dataset (dataset gitignored)
```

---

## Setup (Windows – Local Run)

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

* Required for PDF-to-image conversion (`pdf2image`)

Ensure both tools are available in your **system PATH**, or configure their paths directly in the training and OCR scripts.

---

## Run the API and Frontend

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Access URLs

* **Swagger UI**:
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

* **Frontend UI**:
  [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)

---

## Training the Model (Optional)

> **Note:** This repository does **not** include the RVL-CDIP dataset.

### Expected Dataset Layout

```bash
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

## Model Artifacts

After training, the following files are created:

```bash
models/tfidf_vectorizer.pkl
models/classifier.pkl
```

---

## Run with Docker (Recommended)

### Build the Image

```bash
docker build -t doc-classifier .
```

### Run the Container

```bash
docker run -p 8000:8000 doc-classifier
```

### Open

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)

---

## Optional OCR Configuration

You can control PDF OCR behavior using environment variables:

* `PDF_MAX_PAGES` (default: `2`)
* `PDF_DPI` (default: `220`)

### Example

```bash
docker run -p 8000:8000 -e PDF_MAX_PAGES=1 doc-classifier
```

---

## Notes

* Do **NOT** commit the dataset (`data/docs/`) to GitHub
* Do **NOT** commit OCR cache (`data/ocr_cache/`)
* OCR only the first **2 pages** of PDFs for performance
* Using an **"unknown"** label for low-confidence predictions improves real-world reliability

---

## Usage Notice

This project is open-source under the **MIT License**.

If you plan to use this project (or a modified version of it) in a **production**, **commercial**, or **large-scale academic** system, please consider contacting the author first.

---

## Author

**Mohamed Ruzaik**
GitHub: [https://github.com/Mohamed-Ruzaik](https://github.com/Mohamed-Ruzaik)

---

## License

MIT License © 2025 Mohamed Ruzaik
