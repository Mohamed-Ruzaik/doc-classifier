# Base image
FROM python:3.11-slim

# System dependencies for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY static ./static
COPY models ./models

# Runtime folders
RUN mkdir -p /app/uploads

# Expose API port
EXPOSE 8000

# Environment defaults (can be overridden)
ENV PDF_DPI=220
ENV PDF_MAX_PAGES=2

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
