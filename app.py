import requests
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt



BASE_URL = "https://api.fda.gov/drug/enforcement.json"  # drug recall enforcement endpoint


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

def get_api_key(sidebar_value: str) -> str | None:
    # 1) sidebar has priority
    v = (sidebar_value or "").strip()
    if v and v.upper() not in {"YOUR_KEY_HERE", "NONE", "NULL"}:
        return v

    # 2) fall back to secrets (also stripped)
    s = (st.secrets.get("OPENFDA_API_KEY", "") or "").strip()
    if s and s.upper() not in {"YOUR_KEY_HERE", "NONE", "NULL"}:
        return s

    return None

with st.sidebar:
    st.header("Query")

    with st.form("query_form"):
        term = st.text_input("Drug keyword (try: ibuprofen, metformin, ozempic)", value="ibuprofen")
        limit = st.slider("Records per page (limit)", 1, 100, 25, 1)
        skip = st.number_input("Skip (pagination)", min_value=0, value=0, step=25)

        api_key_input = st.text_input("openFDA API key (optional)", value="", type="password")

        submitted = st.form_submit_button("Fetch recalls")

api_key = get_api_key(api_key_input)
secret_present = bool((st.secrets.get("OPENFDA_API_KEY", "") or "").strip())
typed_present = bool((api_key_input or "").strip())

if typed_present:
    st.sidebar.caption("API key source: sidebar")
elif secret_present:
    st.sidebar.caption("API key source: Streamlit secrets")
else:
    st.sidebar.caption("API key source: none (unauthenticated)")


st.caption(
    "Data source: openFDA drug enforcement reports API (recall data). "
    "Uses query params like search/limit/skip."  # :contentReference[oaicite:4]{index=4}
)


if submitted:
    search = build_search(term)
    payload = fetch_recalls(search=search, limit=int(limit), skip=int(skip), api_key=api_key)

    if "error" in payload:
        st.error(payload["error"])
        st.write("content-type:", payload.get("content_type"))
        st.code(payload.get("text", ""), language="json")
        st.stop()

    meta = payload.get("meta", {}).get("results", {})
    df_new = to_table(payload)

    st.session_state["meta"] = meta
    st.session_state["df_raw"] = df_new
    st.session_state["last_term"] = term

# If we have previous results saved, use them even on reruns
if "df_raw" not in st.session_state:
    st.info("Use the sidebar, then click **Fetch recalls**.")
    st.stop()

df = st.session_state["df_raw"].copy()
meta = st.session_state.get("meta", {})

# ML predictions (if model exists)
try:
    model = load_model()
    X_text = make_text_for_model(df)

    df["predicted_class"] = model.predict(X_text)
    proba = model.predict_proba(X_text)
    df["confidence"] = proba.max(axis=1).round(3)
    df["match"] = (df["predicted_class"] == df["classification"])
except FileNotFoundError:
    st.warning("model.joblib not found. Train it first with: python -m src.train_model")
except Exception as e:
    st.warning(f"Model prediction failed: {e}")


tab1, tab2, tab3 = st.tabs(["Explore", "Model", "About"])

with tab1:
    # Work on a copy so tab2 can use the original df
    df_view = df.copy()

    # Parse date for filtering + charting (on df_view)
    df_view["recall_initiation_date_dt"] = pd.to_datetime(
        df_view["recall_initiation_date"], format="%Y%m%d", errors="coerce"
    )

    # (Optional but recommended) drop future dates so weird data doesn't mess with filtering
    today_ts = pd.Timestamp.today().normalize()
    df_view = df_view[
        df_view["recall_initiation_date_dt"].isna()
        | (df_view["recall_initiation_date_dt"] <= today_ts)
    ]

    # Default date range = last 5 years
    default_start = (today_ts - pd.DateOffset(years=5)).date()
    default_end = today_ts.date()

    date_val = st.date_input(
        "Recall initiation date range",
        value=(default_start, default_end),
    )

    # Safe unpack (handles 1-date return)
    if isinstance(date_val, (list, tuple)) and len(date_val) == 2:
        start_date, end_date = date_val
    else:
        start_date, end_date = None, None
        st.info("Pick an end date to apply the date range filter.")

    st.subheader("Results")
    st.write(f"Total matches (approx): **{meta.get('total', 'unknown')}**")

    st.markdown("### Filters")
    col1, col2, col3, col4 = st.columns(4)
    has_conf = "confidence" in df_view.columns
    has_match = "match" in df_view.columns

    with col1:
        if has_match:
            only_mismatch = st.checkbox("Show only mismatches", value=False)
        else:
            only_mismatch = False
            st.caption("Mismatches require model predictions.")

    with col2:
        if has_conf:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.01)
        else:
            min_conf = 0.0
            st.caption("Confidence requires model predictions.")

    with col3:
        status_choice = st.selectbox(
            "Status",
            ["(all)"] + sorted([x for x in df_view["status"].dropna().unique().tolist()])
        )

    with col4:
        class_choice = st.selectbox(
            "Classification",
            ["(all)"] + sorted([x for x in df_view["classification"].dropna().unique().tolist()])
        )

    # Apply filters (to df_view)
    if status_choice != "(all)":
        df_view = df_view[df_view["status"] == status_choice]

    if class_choice != "(all)":
        df_view = df_view[df_view["classification"] == class_choice]

    if start_date and end_date:
        # If the user is filtering by date range, rows without a valid date shouldn't remain
        df_view = df_view.dropna(subset=["recall_initiation_date_dt"])
        df_view = df_view[
            (df_view["recall_initiation_date_dt"].dt.date >= start_date) &
            (df_view["recall_initiation_date_dt"].dt.date <= end_date)
        ]

    if "confidence" in df_view.columns:
        df_view = df_view[df_view["confidence"] >= float(min_conf)]

    if only_mismatch and "match" in df_view.columns:
        df_view = df_view[df_view["match"] == False]

    st.write(f"Filtered rows: **{len(df_view)}**")

    # Simple Chart
    st.markdown("### Trend (filtered)")
    trend = df_view.dropna(subset=["recall_initiation_date_dt"]).copy()

    if trend.empty:
        st.info("No valid dates to plot for the current filters.")
    else:
        trend["month"] = trend["recall_initiation_date_dt"].dt.to_period("M").dt.to_timestamp()
        monthly = trend.groupby("month").size().reset_index(name="count")

        fig, ax = plt.subplots()
        ax.plot(monthly["month"], monthly["count"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Recalls")
        ax.set_title("Recalls over time (monthly)")
        fig.autofmt_xdate()  
        st.pyplot(fig)

    # Table
    show_cols = [
        "classification", "predicted_class", "confidence", "match",
        "status", "recall_initiation_date", "recall_number",
        "recalling_firm", "product_description", "reason_for_recall"
    ]
    show_cols = [c for c in show_cols if c in df_view.columns]
    st.dataframe(df_view[show_cols], use_container_width=True)


with tab2:
    # Model Health Code
    st.markdown("### Model health (on current page)")

    if "predicted_class" in df.columns and "classification" in df.columns:
        # Only evaluate rows that actually have labels
        eval_df = df.dropna(subset=["classification", "predicted_class"]).copy()

        if len(eval_df) == 0:
            st.info("No labeled rows to evaluate on this page.")
        else:
            acc = accuracy_score(eval_df["classification"], eval_df["predicted_class"])
            mismatches = int((eval_df["classification"] != eval_df["predicted_class"]).sum())

            avg_conf = None
            if "confidence" in eval_df.columns:
                avg_conf = float(np.mean(eval_df["confidence"]))

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Mismatches", str(mismatches))
            if avg_conf is not None:
                c3.metric("Avg confidence", f"{avg_conf:.3f}")
            else:
                c3.metric("Avg confidence", "n/a")

            # Confusion matrix (table)
            labels = sorted(eval_df["classification"].dropna().unique().tolist())
            cm = confusion_matrix(eval_df["classification"], eval_df["predicted_class"], labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"true: {l}" for l in labels], columns=[f"pred: {l}" for l in labels])

            with st.expander("Show confusion matrix"):
                st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Train/load the model to see model health.")


with tab3:
    st.markdown("""
    **FDA Recall Explorer**
    - Pulls recall records from openFDA
    - Predicts recall class (I/II/III) from text using TF-IDF + Logistic Regression
    - Lets you filter, inspect mismatches, and view trends
    """)




