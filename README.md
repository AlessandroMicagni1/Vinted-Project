# Vinted Trust Radar - Streamlit Lite

Unofficial educational analytics dashboard. This project is not affiliated with Vinted.

This version avoids `gensim`, `scipy`, and `nltk`, because those packages can fail to build on some Mac/Python combinations.
It is designed as a lightweight dashboard for exploring trust signals in Vinted Google Play reviews.

## Run

```bash
cd /Users/alessandrocardinal/Desktop/vinted_trust_radar_streamlit_lite
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

## What it includes

- Data Input: loads the Google Play CSV
- Sentiment & Aspects: VADER compound scores, polarity pie chart, aspect bars
- Topic Clusters: displays the notebook/deck LDA topic outputs and assigns sample reviews using the LDA topic keywords
- Temporal Trend: weekly sentiment and % negative reviews
- Business Actions: recommended actions based on trust barriers

