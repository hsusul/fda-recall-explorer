# FDA Recall Explorer (openFDA) + ML

Streamlit web app that pulls FDA drug recall records from the openFDA API and uses an ML model to predict recall classification (Class I / II / III) from recall text.

## Demo
- Live app: https://fdarecallexplorer.streamlit.app/

## Features
- Search recalls by keyword (product description / generic name / recall reason)
- Filters: status, classification, confidence, mismatches
- Trend chart over time (monthly)
- ML predictions: predicted class + confidence + match
- Explain prediction: top TF-IDF terms driving the predicted class

## ML (How it works)
- Input text = `reason_for_recall + product_description`
- Vectorizer = TF-IDF (unigrams + bigrams)
- Model = Logistic Regression (multi-class)
- Tuning = GridSearchCV over `C`
- Output = predicted class + confidence (max probability)

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
