"""
preprocessing.py
-----------------
Data cleaning and text preprocessing utilities for the Fake Job Posting
Detection project.

Responsibilities:
    1. Load the raw EMSCAD job postings dataset.
    2. Handle missing values across text and structured columns.
    3. Combine the most informative textual fields (title, company_profile,
       description, requirements, benefits) into a single text blob.
    4. Clean the text (lowercasing, punctuation/number removal, stopword
       removal, whitespace normalization).
    5. Provide a train/test split ready for vectorization.
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

# A compact stopword list (avoids an nltk download dependency so the
# pipeline runs anywhere without extra corpora downloads).
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
STRUCTURED_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]
TARGET_COLUMN = "fraudulent"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    return df


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/punctuation/digits, remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # strip URLs
    text = re.sub(r"[^a-z\s]", " ", text)        # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Fill missing text fields and merge them into one document per row."""
    df = df.copy()
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("")
    combined = df[TEXT_COLUMNS].agg(" ".join, axis=1)
    return combined


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
      - build combined raw text
      - clean it
      - fill structured/meta features
    Returns a DataFrame with `clean_text` and structured feature columns,
    plus the target vector.
    """
    df = df.copy()

    # Target sanity check
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Combine + clean text
    df["raw_text"] = build_combined_text(df)
    df["clean_text"] = df["raw_text"].apply(clean_text)

    # Structured / binary meta-features already present in the dataset
    for col in STRUCTURED_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)

    # Drop rows that ended up with empty text after cleaning (very rare)
    df = df[df["clean_text"].str.len() > 0]

    features = df[["clean_text"] + STRUCTURED_COLUMNS]
    target = df[TARGET_COLUMN]
    return features, target


def split_data(features, target, test_size=0.2, random_state=42):
    """Stratified train/test split (stratify keeps the ~4.8% fraud ratio
    consistent between train and test sets, which matters heavily for
    imbalanced classification)."""
    return train_test_split(
        features, target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


if __name__ == "__main__":
    df = load_data("data/fake_job_postings.csv")
    X, y = preprocess(df)
    print(f"Rows after preprocessing: {len(X)}")
    print(f"Fraud ratio: {y.mean():.4f}")
    print(X.head())
