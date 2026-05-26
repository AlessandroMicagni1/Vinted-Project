from __future__ import annotations

from pathlib import Path
import re
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    SentimentIntensityAnalyzer = None
    VADER_AVAILABLE = False

DEFAULT_CSV = Path(__file__).with_name("vinted_googleplay_clean.csv")

NAVY = "#0A1628"
BLUE = "#1A7FA0"
SKY = "#378ADD"
TEAL = "#168B9F"
GREEN = "#1D9E75"
RED = "#E24B4A"
PURPLE = "#7F77DD"
GREY = "#888888"
BG = "#F4F6F9"

ASPECTS = {
    "authenticity": ["falso", "truffa", "fake", "contraffatto", "autentico", "originale"],
    "refunds": ["rimborso", "reso", "restituzione", "restituire", "rimborsato"],
    "support": ["assistenza", "supporto", "operatore", "bot", "servizio clienti"],
    "shipping": ["spedizione", "pacco", "consegna", "corriere", "tracking", "spedire"],
    "fees": ["commissione", "costo", "prezzo", "tariffa", "protezione acquisti"],
    "app_usability": ["app", "bug", "interfaccia", "lenta", "notifiche", "funziona", "ricerca"],
    "seller": ["venditore", "venditrice", "acquirente", "affidabile", "seria"],
}

TOPIC_REFERENCE = pd.DataFrame(
    [
        {
            "topic": "T1",
            "label": "Customer Support & Problems",
            "top_words": "assistenza, account, pessima, mai, vendita, pacco",
            "reviews": 873,
            "avg_stars": 2.62,
            "color": RED,
            "keywords": ["assistenza", "account", "pessima", "mai", "problema", "bloccato", "supporto", "bot", "operatore", "pacco"],
        },
        {
            "topic": "T2",
            "label": "App Usability & Search",
            "top_words": "facile, semplice, intuitiva, usare, veloce, comprare",
            "reviews": 654,
            "avg_stars": 4.67,
            "color": BLUE,
            "keywords": ["facile", "semplice", "intuitiva", "usare", "veloce", "comprare", "ricerca", "trovare", "app"],
        },
        {
            "topic": "T3",
            "label": "Positive General Experience",
            "top_words": "ottima, esperienza, consiglio, positiva, super, top",
            "reviews": 1274,
            "avg_stars": 4.87,
            "color": GREEN,
            "keywords": ["ottima", "ottimo", "esperienza", "consiglio", "positiva", "positivo", "super", "top", "perfetto", "fantastica"],
        },
        {
            "topic": "T4",
            "label": "Shipping & Returns",
            "top_words": "venditore, reso, servizio, oggetto, pacco, acquisto",
            "reviews": 569,
            "avg_stars": 4.49,
            "color": TEAL,
            "keywords": ["venditore", "reso", "servizio", "oggetto", "pacco", "acquisto", "spedizione", "rimborso", "consegna", "corriere"],
        },
        {
            "topic": "T5",
            "label": "Item Quality & Delivery",
            "top_words": "perfetto, spedizione, fantastica, articoli, nuova",
            "reviews": 541,
            "avg_stars": 4.41,
            "color": PURPLE,
            "keywords": ["perfetto", "perfetta", "spedizione", "fantastica", "articoli", "nuova", "qualità", "qualita", "prodotto"],
        },
    ]
)

POS_WORDS = {
    "ottimo", "ottima", "ottimi", "ottime", "buono", "buona", "perfetto", "perfetta", "facile", "semplice",
    "veloce", "fantastico", "fantastica", "consiglio", "soddisfatto", "soddisfatta", "super", "top", "adoro",
}
NEG_WORDS = {
    "pessimo", "pessima", "male", "truffa", "falso", "fake", "problema", "problemi", "bug", "lenta", "lento",
    "mai", "bloccato", "bloccata", "rimborso", "reso", "assistenza", "scam", "contraffatto", "deluso", "delusa",
    "impossibile", "vergogna", "soldi", "perso", "perdere",
}

st.set_page_config(page_title="Vinted Trust Radar", page_icon="🛍️", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background: #0E1117; }}
    .main {{ background: #0E1117; }}
    .block-container {{ padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1420px; }}
    h2 {{ font-size: 1.85rem !important; line-height: 1.2 !important; margin: .2rem 0 1rem 0 !important; }}
    h3 {{ font-size: 1.55rem !important; line-height: 1.22 !important; margin: .35rem 0 1rem 0 !important; }}
    h4 {{ line-height: 1.25 !important; }}
    div[data-testid="stHorizontalBlock"] {{ gap: 1.35rem; }}
    div[data-testid="stVerticalBlock"] {{ gap: .85rem; }}
    .section-gap {{ height: 1.1rem; }}
    .hero {{ background:{NAVY}; border-radius:8px; padding:24px 28px; color:white; margin-bottom:1.35rem; border-left:6px solid {RED}; }}
    .hero h1 {{ margin:0; font-size:2.25rem; line-height:1.08; }}
    .hero p {{ margin:.55rem 0 0 0; color:#C8D2DC; font-size:1rem; line-height:1.45; max-width:920px; }}
    .kpi-card {{ background:white; border:1px solid #DDE3EA; border-radius:8px; padding:15px 17px; box-shadow:0 1px 7px rgba(10,22,40,.06); min-height:104px; }}
    .kpi-label {{ color:#607080; font-size:.86rem; margin-bottom:7px; line-height:1.25; }}
    .kpi-value {{ color:{NAVY}; font-size:1.62rem; font-weight:850; line-height:1.08; }}
    .kpi-note {{ color:#7B8794; font-size:.82rem; margin-top:8px; line-height:1.35; }}
    .kpi-card.compact {{ min-height:98px; }}
    .kpi-card.compact .kpi-value {{ font-size:1.45rem; white-space:nowrap; }}
    .topic-card {{ background:white; border:1px solid #DDE3EA; border-radius:8px; padding:17px; min-height:280px; box-shadow:0 1px 7px rgba(10,22,40,.05); }}
    .topic-card h4 {{ color:{NAVY}; margin:0 0 12px 0; font-size:1.15rem; }}
    .topic-card h3 {{ color:{NAVY}; margin:16px 0 0 0 !important; font-size:1.55rem !important; }}
    .topic-pill {{ display:inline-block; color:white; font-weight:800; border-radius:8px; padding:7px 10px; margin-bottom:12px; }}
    .small-note {{ color:#607080; font-size:.88rem; line-height:1.45; }}
    .stDownloadButton button {{ border-radius:8px; padding:.45rem .85rem; margin-top:.25rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(label: str, value: str, note: str = "", compact: bool = False):
    class_name = "kpi-card compact" if compact else "kpi-card"
    st.markdown(
        f"""
        <div class="{class_name}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "score", "at"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = raw.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].ne("") & df["text"].str.lower().ne("nan")]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df = df.dropna(subset=["score", "at"])
    df["score"] = df["score"].clip(1, 5).round().astype(int)
    df = df.drop_duplicates(subset=["text", "score", "at"])
    return df.reset_index(drop=True)


class FallbackSentiment:
    def polarity_scores(self, text: str):
        tokens = re.findall(r"[a-zA-ZÀ-ÿ]{3,}", str(text).lower())
        if not tokens:
            return {"compound": 0.0}
        pos = sum(t in POS_WORDS for t in tokens)
        neg = sum(t in NEG_WORDS for t in tokens)
        compound = (pos - neg) / max(4, pos + neg + 1)
        return {"compound": max(-1.0, min(1.0, float(compound)))}


def get_analyzer():
    if VADER_AVAILABLE and SentimentIntensityAnalyzer is not None:
        return SentimentIntensityAnalyzer(), "VADER"
    return FallbackSentiment(), "fallback Italian lexicon"


def classify(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


@st.cache_data(show_spinner=False)
def read_default_data():
    return pd.read_csv(DEFAULT_CSV)


@st.cache_data(show_spinner="Cleaning data and scoring sentiment...")
def prepare(raw: pd.DataFrame):
    df = clean_data(raw)
    analyzer, _ = get_analyzer()
    df["compound"] = df["text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    df["sentiment"] = df["compound"].apply(classify)
    return df


def aspect_summary(df: pd.DataFrame):
    rows = []
    lower = df["text"].astype(str).str.lower()
    for aspect, keywords in ASPECTS.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        sub = df[lower.str.contains(pattern, regex=True, na=False)]
        if sub.empty:
            continue
        rows.append({
            "aspect": aspect,
            "reviews": len(sub),
            "mean_sentiment": sub["compound"].mean(),
            "avg_stars": sub["score"].mean(),
            "pct_negative": (sub["compound"] <= -0.05).mean() * 100,
            "keywords": ", ".join(keywords),
        })
    return pd.DataFrame(rows).sort_values("avg_stars") if rows else pd.DataFrame()


def assign_reference_topic(text: str, score: int):
    text_l = str(text).lower()
    best_topic = "T3"
    best_hits = -1
    for _, row in TOPIC_REFERENCE.iterrows():
        hits = sum(1 for k in row["keywords"] if k in text_l)
        if hits > best_hits:
            best_hits = hits
            best_topic = row["topic"]
    if best_hits == 0:
        if score <= 2:
            return "T1"
        if score >= 5:
            return "T3"
        if "app" in text_l or "ricerca" in text_l:
            return "T2"
    return best_topic


def weekly_summary(df: pd.DataFrame):
    temp = df.copy()
    temp["week"] = temp["at"].dt.to_period("W")
    weekly = temp.groupby("week").agg(
        reviews=("text", "count"),
        mean_sentiment=("compound", "mean"),
        mean_score=("score", "mean"),
        pct_negative=("compound", lambda x: (x <= -0.05).mean() * 100),
    ).reset_index()
    weekly["week_start"] = weekly["week"].dt.start_time
    weekly["week_str"] = weekly["week"].astype(str)
    return weekly


def setup_axis(ax):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=NAVY)


def download_fig(fig, filename, label="Download chart"):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    st.download_button(label, buffer.getvalue(), file_name=filename, mime="image/png")


def show_chart(fig):
    fig.tight_layout(pad=1.25)
    st.pyplot(fig, width="stretch")


st.markdown(
    """
    <div class="hero">
      <h1>Vinted Trust Radar</h1>
      <p>A trust analytics dashboard for Vinted reviews, turning Google Play feedback into sentiment trends, topic signals, and clear action priorities.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data Input")
    uploaded = st.file_uploader("Upload Google Play reviews CSV", type=["csv"])
    st.caption("Leave empty to use the bundled cleaned Vinted Italy dataset.")
    _, analyzer_name = get_analyzer()
    st.info(f"Sentiment engine: {analyzer_name}")
    st.caption("This lite build avoids gensim/scipy/nltk so it installs reliably on Mac.")

try:
    raw_df = pd.read_csv(uploaded) if uploaded is not None else read_default_data()
    df = prepare(raw_df)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

# Topic assignment for samples and interactive exploration. The reference LDA results remain the notebook/deck results.
df["topic_ref"] = df.apply(lambda r: assign_reference_topic(r["text"], int(r["score"])), axis=1)
topic_lookup = TOPIC_REFERENCE.set_index("topic")
df["topic_label"] = df["topic_ref"].map(topic_lookup["label"].to_dict())

st.subheader("Dataset overview")
a, b, c, d = st.columns(4, gap="large")
with a:
    kpi("Reviews after cleaning", f"{len(df):,}", "Google Play Italy reviews")
with b:
    kpi("Date range", f"{df['at'].min().date()} → {df['at'].max().date()}", "timestamp field: at")
with c:
    kpi("Average star rating", f"{df['score'].mean():.2f}★", "Google Play score")
with d:
    kpi("Avg compound", f"{df['compound'].mean():.3f}", "VADER baseline score")

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
tab_data, tab_sent, tab_topics, tab_time, tab_actions = st.tabs([
    "Data Input", "Sentiment & Aspects", "Topic Clusters", "Temporal Trend", "Business Actions"
])

with tab_data:
    st.markdown("### 1. Data Input — load the CSV")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        display_cols = [c for c in ["text", "score", "at", "thumbsUpCount", "language", "compound", "sentiment"] if c in df.columns]
        st.dataframe(df[display_cols].head(100), width="stretch", height=430, hide_index=True)
    with right:
        st.markdown("#### Star rating distribution")
        counts = df["score"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4.35), facecolor=BG)
        setup_axis(ax)
        ax.bar(counts.index.astype(str), counts.values, color=BLUE)
        ax.set_xlabel("Stars", color=NAVY)
        ax.set_ylabel("Reviews", color=NAVY)
        ax.set_title("Google Play ratings", color=NAVY, fontweight="bold")
        for x, y in zip(counts.index.astype(str), counts.values):
            ax.text(x, y + max(counts.values) * .01, f"{int(y):,}", ha="center", color=NAVY, fontsize=9)
        show_chart(fig)
        download_fig(fig, "score_distribution.png")
        plt.close(fig)
    st.markdown("#### Column check")
    st.dataframe(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns], "non_null": [int(df[c].notna().sum()) for c in df.columns]}), width="stretch", hide_index=True)

with tab_sent:
    st.markdown("### 2. Sentiment & Aspects — pie chart + aspect sentiment bars")
    dist = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi("Positive", f"{int(dist['positive']):,}", f"{dist['positive']/len(df)*100:.1f}% of reviews")
    with c2:
        kpi("Neutral", f"{int(dist['neutral']):,}", f"{dist['neutral']/len(df)*100:.1f}% of reviews")
    with c3:
        kpi("Negative", f"{int(dist['negative']):,}", f"{dist['negative']/len(df)*100:.1f}% of reviews")
    with c4:
        kpi("Avg compound", f"{df['compound'].mean():.3f}", "mild positive / neutral baseline")

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        fig, ax = plt.subplots(figsize=(5.9, 4.35), facecolor=BG)
        labels = [f"{s.title()}\n{dist[s]/len(df)*100:.0f}%" for s in dist.index]
        ax.pie(dist.values, labels=labels, startangle=90, colors=[GREEN, GREY, RED], wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Overall Sentiment Distribution", color=NAVY, fontweight="bold")
        ax.axis("equal")
        show_chart(fig)
        download_fig(fig, "sentiment_pie.png")
        plt.close(fig)
    with right:
        aspects = aspect_summary(df)
        aspects_plot = aspects.sort_values("mean_sentiment", ascending=True)
        fig, ax = plt.subplots(figsize=(7.1, 4.35), facecolor=BG)
        setup_axis(ax)
        colors = [RED if v < 0.05 else BLUE for v in aspects_plot["mean_sentiment"]]
        ax.barh(aspects_plot["aspect"], aspects_plot["mean_sentiment"], color=colors)
        ax.set_xlabel("Mean VADER compound score", color=NAVY)
        ax.set_title("Aspect-Based Sentiment", color=NAVY, fontweight="bold")
        for i, v in enumerate(aspects_plot["mean_sentiment"]):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", color=NAVY, fontsize=9)
        show_chart(fig)
        download_fig(fig, "aspect_sentiment.png")
        plt.close(fig)
    st.markdown("#### Aspect evidence table")
    st.dataframe(aspects[["aspect", "reviews", "avg_stars", "pct_negative", "mean_sentiment", "keywords"]], width="stretch", hide_index=True)

with tab_topics:
    st.markdown("### 3. Topic Clusters — LDA topic distribution + top words per topic")
    st.info("Demo-safe mode: the app reuses the LDA topic labels, top words, review counts, and average stars from the notebook/deck. It then uses those topic keywords to show sample reviews interactively, without requiring gensim to compile on your Mac.")

    cols = st.columns(5, gap="large")
    for col, (_, row) in zip(cols, TOPIC_REFERENCE.iterrows()):
        with col:
            st.markdown(
                f"""
                <div class="topic-card" style="border-top:8px solid {row['color']};">
                    <div class="topic-pill" style="background:{row['color']};">{row['topic']}</div>
                    <h4>{row['label']}</h4>
                    <p class="small-note">{row['top_words']}</p>
                    <h3>{int(row['reviews']):,}</h3>
                    <p class="small-note">reviews · {row['avg_stars']:.2f}★ avg</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        fig, ax = plt.subplots(figsize=(7, 4.35), facecolor=BG)
        setup_axis(ax)
        ref_sorted = TOPIC_REFERENCE.sort_values("reviews")
        ax.barh(ref_sorted["topic"] + ": " + ref_sorted["label"], ref_sorted["reviews"], color=ref_sorted["color"])
        ax.set_xlabel("Number of reviews", color=NAVY)
        ax.set_title("Reviews per LDA Topic", color=NAVY, fontweight="bold")
        for i, v in enumerate(ref_sorted["reviews"]):
            ax.text(v + 15, i, f"{int(v):,}", va="center", color=NAVY, fontsize=9)
        show_chart(fig)
        download_fig(fig, "topic_distribution.png")
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4.35), facecolor=BG)
        setup_axis(ax)
        ref_sorted = TOPIC_REFERENCE.sort_values("avg_stars")
        ax.barh(ref_sorted["topic"] + ": " + ref_sorted["label"], ref_sorted["avg_stars"], color=ref_sorted["color"])
        ax.axvline(4.22, color=GREY, linestyle="--", linewidth=1)
        ax.set_xlim(0, 5)
        ax.set_xlabel("Average Star Rating", color=NAVY)
        ax.set_title("Avg Star Rating per Topic", color=NAVY, fontweight="bold")
        for i, v in enumerate(ref_sorted["avg_stars"]):
            ax.text(v + .04, i, f"{v:.2f}★", va="center", color=NAVY, fontsize=9)
        show_chart(fig)
        download_fig(fig, "topic_avg_stars.png")
        plt.close(fig)

    st.markdown("#### Sample reviews by topic")
    selected_topic = st.selectbox("Choose a topic", TOPIC_REFERENCE["topic"] + " — " + TOPIC_REFERENCE["label"])
    selected_code = selected_topic.split(" — ")[0]
    samples = df[df["topic_ref"] == selected_code].sort_values(["score", "compound"]).head(8)
    for _, row in samples.iterrows():
        st.markdown(f"**{int(row['score'])}★ · {row['compound']:.3f} · {row['at'].date()}**")
        st.write(row["text"])
        st.divider()

with tab_time:
    st.markdown("### 4. Temporal Trend — weekly sentiment chart + % negative reviews")
    weekly = weekly_summary(df)
    for start in range(0, len(weekly), 4):
        metric_cols = st.columns(min(4, len(weekly) - start), gap="large")
        for col, (_, row) in zip(metric_cols, weekly.iloc[start:start + 4].iterrows()):
            with col:
                kpi(row["week_str"].split("/")[0], f"{row['mean_sentiment']:.3f}", f"{int(row['reviews'])} rev · {row['pct_negative']:.1f}% neg", compact=True)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    fig, ax1 = plt.subplots(figsize=(11, 4.55), facecolor=BG)
    setup_axis(ax1)
    ax1.plot(weekly["week_start"], weekly["mean_sentiment"], marker="o", linewidth=2.5, color=BLUE, label="Mean sentiment")
    ax1.fill_between(weekly["week_start"], weekly["mean_sentiment"], color=SKY, alpha=.15)
    ax1.set_ylabel("Mean VADER Compound", color=BLUE)
    ax1.set_title("Weekly Sentiment Trend — Vinted Italy", color=NAVY, fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(weekly["week_start"], weekly["pct_negative"], marker="s", linewidth=2, linestyle="--", color=RED, label="% negative")
    ax2.set_ylabel("% Negative Reviews", color=RED)
    ax2.tick_params(colors=RED)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    show_chart(fig)
    download_fig(fig, "weekly_temporal_trend.png")
    plt.close(fig)

    st.dataframe(weekly[["week_str", "reviews", "mean_sentiment", "mean_score", "pct_negative"]], width="stretch", hide_index=True)

with tab_actions:
    st.markdown("### Business Actions — what Vinted should prioritize")
    aspects = aspect_summary(df)
    action_map = {
        "authenticity": "Strengthen counterfeit detection and make buyer-protection signals visible before purchase.",
        "refunds": "Simplify refund status communication and reduce uncertainty during disputes.",
        "support": "Escalate bot failures to human support faster and publish response-time expectations.",
        "shipping": "Improve carrier-status transparency and flag likely delivery delays earlier.",
        "fees": "Explain protection fees and total costs earlier in the checkout journey.",
        "app_usability": "Prioritize bug triage, search usability, notifications and crash reports.",
        "seller": "Surface seller reliability signals more prominently before checkout.",
    }
    critical = aspects.sort_values("avg_stars").head(5).copy()
    critical["recommended_action"] = critical["aspect"].map(action_map)
    critical["evidence"] = critical.apply(lambda r: f"{int(r['reviews'])} reviews · {r['avg_stars']:.2f}★ avg · {r['pct_negative']:.1f}% negative", axis=1)
    st.dataframe(critical[["aspect", "evidence", "recommended_action"]], width="stretch", hide_index=True)
    st.markdown(
        """
        **Presentation line:** The prototype does not add a new model; it operationalizes the Python analysis into a dashboard. A manager can load the CSV, see which trust dimensions are most problematic, explore LDA topic clusters, and monitor weekly sentiment changes.
        """
    )
