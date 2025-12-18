# app/ml_model.py

from pathlib import Path
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"

class DocumentClassifier:
    def __init__(self):
        if not VECTORIZER_PATH.exists() or not CLASSIFIER_PATH.exists():
            raise RuntimeError("Model files not found. Train first.")
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.clf = joblib.load(CLASSIFIER_PATH)

    def predict_text(self, text: str, top_k: int = 3, unknown_threshold: float = 0.40):
        """
        Return (label, confidence, top_k_list).
        top_k_list = [{"label": ..., "confidence": ...}, ...]
        """
        X = self.vectorizer.transform([text])

        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(X)[0]
            classes = self.clf.classes_

            # top-k
            k = min(top_k, len(classes))
            top_idx = np.argsort(probs)[::-1][:k]
            top_list = [
                {"label": str(classes[i]), "confidence": float(probs[i])}
                for i in top_idx
            ]

            best = top_list[0]
            label = best["label"]
            confidence = best["confidence"]

            if confidence < unknown_threshold:
                return "unknown", confidence, top_list

            return label, confidence, top_list

        # fallback (no probs)
        label = self.clf.predict(X)[0]
        return str(label), None, None

# Single global instance (loaded once at app startup)
classifier = DocumentClassifier()
