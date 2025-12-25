import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://api.fda.gov/drug/enforcement.json"  # drug recall enforcement endpoint

import joblib

MODEL_PATH = "model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def make_text_for_model(df: pd.DataFrame) -> pd.Series:
    return (
        df["reason_for_recall"].fillna("") + " " +
        df["product_description"].fillna("")
    ).str.strip()

def build_search(term: str) -> str:
    # openFDA search syntax is field:"term" (Lucene-style) :contentReference[oaicite:2]{index=2}
    term = term.strip()
    if not term:
        return ""  # no search
    return (
        f'(product_description:"{term}" OR '
        f'openfda.generic_name:"{term}" OR '
        f'reason_for_recall:"{term}")'
    )


@st.cache_data(show_spinner=False)
def fetch_recalls(search: str, limit: int, skip: int, api_key: str | None):
    params = {"limit": limit, "skip": skip}
    if search:
        params["search"] = search
    if api_key:
        params["api_key"] = api_key  # optional key improves daily quota :contentReference[oaicite:3]{index=3}

    r = requests.get(BASE_URL, params=params, timeout=20)

    # Helpful debugging if openFDA returns HTML or an error
    content_type = r.headers.get("content-type", "")
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "content_type": content_type, "text": r.text[:400]}

    try:
        return r.json()
    except Exception:
        return {"error": "JSON decode failed", "content_type": content_type, "text": r.text[:400]}


def to_table(payload: dict) -> pd.DataFrame:
    results = payload.get("results", [])
    rows = []
    for item in results:
        rows.append({
            "classification": item.get("classification"),
            "status": item.get("status"),
            "recall_initiation_date": item.get("recall_initiation_date"),
            "recall_number": item.get("recall_number"),
            "recalling_firm": item.get("recalling_firm"),
            "product_description": item.get("product_description"),
            "reason_for_recall": item.get("reason_for_recall"),
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="FDA Recall Explorer", layout="wide")
st.title("FDA Drug Recall Explorer (openFDA)")

with st.sidebar:
    st.header("Query")
    term = st.text_input("Drug keyword (try: ibuprofen, metformin, ozempic)", value="ibuprofen")
    limit = st.slider("Records per page (limit)", min_value=1, max_value=100, value=25, step=1)
    skip = st.number_input("Skip (pagination)", min_value=0, value=0, step=25)

    api_key = st.text_input("openFDA API key (optional)", value="", type="password")
    api_key = api_key.strip() or None

    run = st.button("Fetch recalls")

st.caption(
    "Data source: openFDA drug enforcement reports API (recall data). "
    "Uses query params like search/limit/skip."  # :contentReference[oaicite:4]{index=4}
)

# --- Fetch on button click and SAVE results ---
if run:
    search = build_search(term)
    payload = fetch_recalls(search=search, limit=int(limit), skip=int(skip), api_key=api_key)

    if "error" in payload:
        st.error(payload["error"])
        st.write("content-type:", payload.get("content_type"))
        st.code(payload.get("text", ""), language="html")
        st.stop()

    meta = payload.get("meta", {}).get("results", {})
    df = to_table(payload)

    if df.empty:
        st.warning("No results returned for that term (try a broader keyword).")
        st.stop()

    # ✅ Persist across reruns (checkbox clicks)
    st.session_state["df"] = df
    st.session_state["meta"] = meta
    st.session_state["last_term"] = term


# --- DISPLAY if we have saved results ---
if "df" in st.session_state:
    df = st.session_state["df"]
    meta = st.session_state.get("meta", {})

    st.subheader("Results")
    st.write(f"Last search term: **{st.session_state.get('last_term', '')}**")
    st.write(f"Total matches (approx): **{meta.get('total', 'unknown')}**")
    st.write(f"Showing **{meta.get('skip', skip)} → {meta.get('skip', skip) + meta.get('limit', limit) - 1}**")

    # ML predictions (if model exists)
    try:
        model = load_model()
        X_text = make_text_for_model(df)

        df["predicted_class"] = model.predict(X_text)
        proba = model.predict_proba(X_text)
        df["confidence"] = proba.max(axis=1).round(3)
        df["match"] = (df["predicted_class"] == df["classification"])
    except FileNotFoundError:
        st.warning("model.joblib not found yet. Train it first with: python -m src.train_model")
    except Exception as e:
        st.warning(f"Model prediction failed: {e}")

    # Filters (these rerun safely now)
    only_mismatch = st.checkbox("Show only mismatches (predicted != actual)", value=False)
    if "match" in df.columns and only_mismatch:
        df = df[df["match"] == False]

    if "confidence" in df.columns:
        df = df.sort_values("confidence", ascending=False)

    # Confidence threshold filter
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    if "confidence" in df.columns:
        df = df[df["confidence"] >= min_conf]

    # Truncate long text for readability (display-only)
    truncate = st.checkbox("Truncate long text in table", value=True)
    if truncate:
        if "product_description" in df.columns:
            df["product_description"] = df["product_description"].astype(str).str.slice(0, 120) + "..."
        if "reason_for_recall" in df.columns:
            df["reason_for_recall"] = df["reason_for_recall"].astype(str).str.slice(0, 120) + "..."

    # Simple quick insights
    left, right = st.columns(2)
    with left:
        st.write("**Counts by classification**")
        st.dataframe(df["classification"].value_counts(dropna=False).rename_axis("classification").to_frame("count"))
    with right:
        st.write("**Counts by status**")
        st.dataframe(df["status"].value_counts(dropna=False).rename_axis("status").to_frame("count"))

    st.write("**Records**")
    show_cols = [
        "classification", "predicted_class", "confidence", "match",
        "status", "recall_initiation_date", "recall_number",
        "recalling_firm", "product_description", "reason_for_recall"
    ]
    # Only show columns that exist (in case model didn't load)
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True)

else:
    st.info("Use the sidebar, then click **Fetch recalls**.")
