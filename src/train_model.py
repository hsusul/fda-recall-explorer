import os
import requests
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

import joblib


#API
BASE_URL = "https://api.fda.gov/drug/enforcement.json"
MODEL_PATH = "model.joblib"

#CSV
DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "recalls_train.csv"

def fetch_pages(api_key: str | None, per_class_pages: int = 4, limit: int = 100) -> pd.DataFrame:
    """
    Fetch recalls by class separately so we don't accidentally get only one class.
    per_class_pages=4 & limit=100 => up to ~1200 rows total (3 classes).
    """
    classes = ["Class I", "Class II", "Class III"]
    all_rows = []

    for cls in classes:
        for i in range(per_class_pages):
            skip = i * limit
            search = f'classification:"{cls}"'
            params = {"search": search, "limit": limit, "skip": skip}
            if api_key:
                params["api_key"] = api_key

            r = requests.get(BASE_URL, params=params, timeout=25)
            if r.status_code != 200:
                raise RuntimeError(f"openFDA request failed: HTTP {r.status_code}\n{r.text[:300]}")

            data = r.json()
            results = data.get("results", [])
            if not results:
                break

            all_rows.extend(results)
            print(f"Fetched {cls} page {i+1}/{per_class_pages} (rows so far: {len(all_rows)})")

    df = pd.DataFrame([{
        "classification": row.get("classification"),
        "reason_for_recall": row.get("reason_for_recall"),
        "product_description": row.get("product_description"),
        "recall_number": row.get("recall_number"),
    } for row in all_rows])

    return df



def make_xy(df: pd.DataFrame):
    # Build input text X from recall text fields
    text = (
        df["reason_for_recall"].fillna("") + " " +
        df["product_description"].fillna("")
    ).str.strip()

    # Label y = Class I/II/III
    y = df["classification"].astype(str)

    # Drop empty text rows (rare, but possible)
    mask = text.str.len() > 0
    return text[mask], y[mask]


def main():
    # Optional: use environment variable if you have a key
    api_key = os.getenv("OPENFDA_API_KEY")

    if DATA_PATH.exists():
        print(f"Loading cached training data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        df = fetch_pages(api_key=api_key, per_class_pages=4, limit=100)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved training data -> {DATA_PATH}")
    print("Raw df shape:", df.shape)


    # Save CSV during training
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved training data -> {DATA_PATH}")



    # Drop rows missing the label
    df = df.dropna(subset=["classification"])
    X, y = make_xy(df)

    print("Usable rows:", len(X))
    print("Label counts:\n", y.value_counts())
    if y.nunique() < 2:
        raise RuntimeError(f"Need at least 2 classes to train, but got: {y.unique()}")


    # Split (stratify keeps class proportions similar in train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ML pipeline = text -> numbers -> classifier
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nReport:\n", classification_report(y_test, preds))

    # Save the whole pipeline (vectorizer + model)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
