# FDA Recall Explorer (openFDA) + ML Classifier

A Streamlit web app that pulls FDA drug recall records from the openFDA API and uses an ML model to predict recall classification (Class I / II / III) from recall text.

## Features
- Search recalls by keyword (product description / generic name / recall reason)
- Filter by status, actual classification, model confidence, mismatches
- Trend chart over time (monthly)
- ML predictions + “explain prediction” (top TF-IDF terms contributing to predicted class)

## Tech Stack
- Python, Streamlit
- openFDA API (`/drug/enforcement.json`)
- scikit-learn: TF-IDF + Logistic Regression (tuned with GridSearchCV)
- pandas, matplotlib

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
