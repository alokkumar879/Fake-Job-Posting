"""
train.py
--------
Trains and compares multiple classification models for Fake Job Posting
Detection, using TF-IDF text features combined with a handful of structured
meta-features (telecommuting, has_company_logo, has_questions).

Models compared:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest
    - Decision Tree

Because the dataset is heavily imbalanced (~4.8% fraudulent), models are
evaluated on precision, recall, F1-score and ROC-AUC in addition to
accuracy, and `class_weight='balanced'` is used where supported.

Outputs (written to models/ and reports/):
    - models/tfidf_vectorizer.joblib
    - models/best_model.joblib
    - models/model_metadata.json
    - reports/model_comparison.csv
    - reports/confusion_matrices.png
    - reports/roc_curves.png
    - reports/eda_class_balance.png
    - reports/eda_text_length.png
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from preprocessing import load_data, preprocess, split_data, STRUCTURED_COLUMNS

DATA_PATH = "data/fake_job_postings.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def run_eda(df, y):
    """Generate a couple of exploratory plots used in the README."""
    sns.set_theme(style="whitegrid")

    # Class balance
    plt.figure(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    ax = sns.barplot(x=["Legitimate (0)", "Fraudulent (1)"], y=counts.values,
                      hue=["Legitimate (0)", "Fraudulent (1)"],
                      palette=["#3b82f6", "#ef4444"], legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v} ({v/len(y)*100:.1f}%)", ha="center")
    plt.title("Class Balance: Legitimate vs Fraudulent Job Postings")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/eda_class_balance.png", dpi=150)
    plt.close()

    # Text length distribution by class
    lengths = df["clean_text"].str.split().apply(len)
    plt.figure(figsize=(6, 4))
    sns.histplot(lengths[y == 0], color="#3b82f6", label="Legitimate", kde=True, stat="density", alpha=0.5, bins=40)
    sns.histplot(lengths[y == 1], color="#ef4444", label="Fraudulent", kde=True, stat="density", alpha=0.5, bins=40)
    plt.xlim(0, 600)
    plt.title("Cleaned Text Length Distribution by Class")
    plt.xlabel("Number of tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/eda_text_length.png", dpi=150)
    plt.close()


def build_feature_matrix(vectorizer, text_series, structured_df, fit=False):
    """TF-IDF-transform text and concatenate with structured meta-features."""
    if fit:
        X_text = vectorizer.fit_transform(text_series)
    else:
        X_text = vectorizer.transform(text_series)
    X_struct = csr_matrix(structured_df.values.astype(float))
    return hstack([X_text, X_struct]).tocsr()


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"])
    return metrics, cm, y_proba, report


def main():
    print("Loading data...")
    df_raw = load_data(DATA_PATH)

    print("Preprocessing text and features...")
    X, y = preprocess(df_raw)
    run_eda(X, y)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
    )
    X_train_vec = build_feature_matrix(vectorizer, X_train["clean_text"], X_train[STRUCTURED_COLUMNS], fit=True)
    X_test_vec = build_feature_matrix(vectorizer, X_test["clean_text"], X_test[STRUCTURED_COLUMNS], fit=False)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=25, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced",
                                                 n_jobs=-1, random_state=42),
    }

    results = []
    roc_data = {}
    cms = {}
    reports_text = {}

    for name, model in models.items():
        print(f"Training {name}...")
        t0 = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - t0

        metrics, cm, y_proba, report = evaluate_model(name, model, X_test_vec, y_test)
        metrics["train_time_sec"] = round(train_time, 2)
        results.append(metrics)
        cms[name] = cm
        reports_text[name] = report
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics["roc_auc"])

        joblib.dump(model, f"{MODELS_DIR}/{name.lower().replace(' ', '_')}.joblib")

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_df.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)
    print("\n=== Model Comparison (sorted by F1) ===")
    print(results_df.to_string(index=False))

    # Confusion matrices grid
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred Legit", "Pred Fraud"],
                    yticklabels=["True Legit", "True Fraud"])
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/confusion_matrices.png", dpi=150)
    plt.close()

    # ROC curves
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/roc_curves.png", dpi=150)
    plt.close()

    # Persist best model + vectorizer + metadata
    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.joblib")
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.joblib")

    metadata = {
        "best_model": best_name,
        "structured_columns": STRUCTURED_COLUMNS,
        "metrics": results_df.to_dict(orient="records"),
    }
    with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(f"{REPORTS_DIR}/classification_reports.txt", "w") as f:
        for name, report in reports_text.items():
            f.write(f"=== {name} ===\n{report}\n\n")

    print(f"\nBest model by F1-score: {best_name}")
    print("Artifacts saved to models/ and reports/")


if __name__ == "__main__":
    main()
