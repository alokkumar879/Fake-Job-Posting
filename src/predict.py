"""
predict.py
----------
Loads the trained TF-IDF vectorizer + best model and exposes a simple
function/CLI to classify a new, unseen job posting as fraudulent or
legitimate.

Usage (CLI):
    python src/predict.py --title "Work From Home Data Entry" \\
        --description "Earn $500/day no experience needed, just send your bank details" \\
        --requirements "None" --company_profile "" --benefits "Daily cash payouts"

Usage (as a library):
    from src.predict import JobPostingClassifier
    clf = JobPostingClassifier()
    result = clf.predict({
        "title": "...", "company_profile": "...", "description": "...",
        "requirements": "...", "benefits": "...",
        "telecommuting": 0, "has_company_logo": 1, "has_questions": 1,
    })
    print(result)  # {'label': 'Legitimate', 'fraud_probability': 0.03, ...}
"""

import argparse
import json
import joblib
from scipy.sparse import hstack, csr_matrix
import numpy as np

from preprocessing import clean_text, TEXT_COLUMNS, STRUCTURED_COLUMNS

MODELS_DIR = "models"


class JobPostingClassifier:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.vectorizer = joblib.load(f"{models_dir}/tfidf_vectorizer.joblib")
        self.model = joblib.load(f"{models_dir}/best_model.joblib")
        with open(f"{models_dir}/model_metadata.json") as f:
            self.metadata = json.load(f)

    def _prepare_features(self, posting: dict):
        raw_text = " ".join(str(posting.get(col, "") or "") for col in TEXT_COLUMNS)
        text = clean_text(raw_text)
        X_text = self.vectorizer.transform([text])
        struct_values = np.array([[int(posting.get(col, 0) or 0) for col in STRUCTURED_COLUMNS]])
        X_struct = csr_matrix(struct_values.astype(float))
        return hstack([X_text, X_struct]).tocsr()

    def predict(self, posting: dict) -> dict:
        """
        posting: dict with keys among
            title, company_profile, description, requirements, benefits,
            telecommuting, has_company_logo, has_questions
        """
        X = self._prepare_features(posting)
        pred = int(self.model.predict(X)[0])
        proba = float(self.model.predict_proba(X)[0][1]) if hasattr(self.model, "predict_proba") else None

        return {
            "label": "Fraudulent" if pred == 1 else "Legitimate",
            "fraud_probability": round(proba, 4) if proba is not None else None,
            "model_used": self.metadata.get("best_model"),
        }


def _cli():
    parser = argparse.ArgumentParser(description="Classify a job posting as fraudulent or legitimate.")
    parser.add_argument("--title", default="")
    parser.add_argument("--company_profile", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--requirements", default="")
    parser.add_argument("--benefits", default="")
    parser.add_argument("--telecommuting", type=int, default=0)
    parser.add_argument("--has_company_logo", type=int, default=0)
    parser.add_argument("--has_questions", type=int, default=0)
    args = parser.parse_args()

    clf = JobPostingClassifier()
    result = clf.predict(vars(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
