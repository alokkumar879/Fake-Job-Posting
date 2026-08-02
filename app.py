"""
Streamlit demo for the Fake Job Posting Detection project.

Run locally with:
    streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import JobPostingClassifier

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="centered")

st.title("🕵️ Fake Job Posting Detection")
st.caption("ML pipeline: TF-IDF text features + structured meta-features → classifier ensemble comparison")

@st.cache_resource
def load_classifier():
    return JobPostingClassifier()

try:
    clf = load_classifier()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model artifacts not found. Run `python src/train.py` first to generate models/.")

tab1, tab2 = st.tabs(["🔎 Try a Prediction", "📊 Model Comparison"])

with tab1:
    st.subheader("Paste a job posting")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Job Title", "Work From Home Data Entry Clerk")
        company_profile = st.text_area("Company Profile", "", height=80)
        telecommuting = st.checkbox("Telecommuting allowed", value=True)
    with col2:
        has_logo = st.checkbox("Company logo present", value=False)
        has_questions = st.checkbox("Screening questions present", value=False)

    description = st.text_area(
        "Job Description",
        "Earn $500 a day. No experience needed. Just send your bank account details "
        "and a copy of your ID to get started immediately.",
        height=120,
    )
    requirements = st.text_area("Requirements", "None", height=60)
    benefits = st.text_area("Benefits", "Daily cash payouts", height=60)

    if st.button("Classify Posting", type="primary", disabled=not model_loaded):
        posting = {
            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "telecommuting": int(telecommuting),
            "has_company_logo": int(has_logo),
            "has_questions": int(has_questions),
        }
        result = clf.predict(posting)
        if result["label"] == "Fraudulent":
            st.error(f"⚠️ Likely **FRAUDULENT** — probability {result['fraud_probability']:.1%}")
        else:
            st.success(f"✅ Likely **LEGITIMATE** — fraud probability only {result['fraud_probability']:.1%}")
        st.caption(f"Model used: {result['model_used']}")

with tab2:
    st.subheader("Model comparison (held-out test set)")
    try:
        comparison = pd.read_csv("reports/model_comparison.csv")
        st.dataframe(comparison.set_index("model").style.format({
            "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
            "f1_score": "{:.3f}", "roc_auc": "{:.3f}", "train_time_sec": "{:.2f}",
        }), use_container_width=True)
        st.image("reports/roc_curves.png", caption="ROC Curves")
        st.image("reports/confusion_matrices.png", caption="Confusion Matrices")
    except FileNotFoundError:
        st.warning("Run `python src/train.py` first to generate the comparison report.")

st.divider()
st.caption(
    "Dataset: EMSCAD (Employment Scam Aegean Dataset), 17,880 postings, ~4.8% fraudulent. "
    "This tool is a portfolio/internship project and should not be used as the sole basis "
    "for real-world hiring-fraud decisions."
)
