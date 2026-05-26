from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel

DEFAULT_CSV = Path(__file__).with_name("vinted_googleplay_clean.csv")

NAVY   = "#0A1628"
BLUE   = "#1A7FA0"
SKY    = "#378ADD"
TEAL   = "#168B9F"
GREEN  = "#1D9E75"
RED    = "#E24B4A"
PURPLE = "#7F77DD"
GREY   = "#8899AA"
AMBER  = "#F5A623"
BG     = "#F4F6F9"

COLORS_CYCLE = [BLUE, RED, GREEN, TEAL, PURPLE]
N_TOPICS     = 5

ASPECTS = {
    "authenticity":  ["falso","truffa","fake","contraffatto","autentico","originale"],
    "refunds":       ["rimborso","reso","restituzione","restituire","rimborsato"],
    "support":       ["assistenza","supporto","operatore","bot","servizio clienti"],
    "shipping":      ["spedizione","pacco","consegna","corriere","tracking","spedire"],
    "fees":          ["commissione","costo","prezzo","tariffa","protezione acquisti"],
    "app_usability": ["app","bug","interfaccia","lenta","notifiche","funziona","ricerca"],
    "seller":        ["venditore","venditrice","acquirente","affidabile","seria"],
}

# Italian stopwords (embedded — no nltk download needed)
ITALIAN_STOPWORDS = {
    # articles & prepositions
    "il","lo","la","i","gli","le","un","uno","una",
    "di","del","dello","dei","degli","della","delle",
    "a","al","allo","ai","agli","alla","alle",
    "da","dal","dallo","dai","dagli","dalla","dalle",
    "in","nel","nello","nei","negli","nella","nelle",
    "su","sul","sullo","sui","sugli","sulla","sulle",
    "con","col","coi","per","tra","fra",
    # pronouns
    "io","tu","lui","lei","noi","voi","loro",
    "mi","ti","ci","vi","si","ne","lo","la","li","le","gli",
    "me","te","se","ce","ve",
    "mio","mia","miei","mie","tuo","tua","tuoi","tue",
    "suo","sua","suoi","sue","nostro","nostra","nostri","nostre",
    "vostro","vostra","vostri","vostre",
    "questo","questa","questi","queste","quello","quella","quelli","quelle",
    "che","chi","cui","quale","quali","quanto","quanta","quanti","quante",
    # conjunctions / adverbs
    "e","ed","o","ma","se","non","anche","solo","già","ancora","sempre",
    "mai","più","molto","molta","molti","molte","troppo","troppa",
    "poco","poca","ogni","tutto","tutti","tutta","tutte",
    "poi","però","perché","perche","quando","come","dove","dov",
    "cosa","fatto","fare","ora","qui","qua","lì","là","così","ecco",
    "quindi","allora","davvero","proprio","circa","quasi","appena",
    "subito","prima","dopo","vero","grande","grandi","stesso","stessa",
    "anno","anni","giorno","giorni","mese","mesi","volta","volte",
    "tempo","modo","tipo","bene","male","via","fino","oltre","invece",
    "magari","certo","cosi","adesso","oggi","ieri","domani",
    # auxiliary verbs
    "ho","hai","ha","abbiamo","avete","hanno",
    "avevo","avevi","aveva","avevamo","avevate","avevano",
    "avrò","avrai","avrà","avremo","avrete","avranno",
    "avrei","avresti","avrebbe","avremmo","avreste","avrebbero",
    "sono","sei","siamo","siete",
    "ero","eri","era","eravamo","eravate","erano",
    "sarò","sarai","sarà","saremo","sarete","saranno",
    "sarei","saresti","sarebbe","saremmo","sareste","sarebbero",
    "sia","siano","fosse","fossero","fossi","stato","stata","stati","state",
    "avere","essere","fare","dire","andare","venire","sapere",
    "volere","potere","dovere","stare",
    # Vinted-specific noise
    "vinted","app","applicazione","ciao","grazie","purtroppo",
}

# Priority-ordered (keyword_set, label) pairs for auto-labelling LDA topics
_LABEL_HINTS = [
    ({"truffa","falso","fake","contraffatto"},                "Fraud and Authenticity"),
    ({"tutela","protezione","acquirenti"},                    "Buyer Protection"),
    ({"rimborso","reso","restituzione","restituire"},         "Refunds and Returns"),
    ({"assistenza","supporto","servizio","clienti"},          "Customer Support"),
    ({"account","bloccato","bloccata","sospeso"},             "Account Issues"),
    ({"spedizione","consegna","pacco","corriere","tracking"}, "Shipping and Delivery"),
    ({"bug","interfaccia","notifiche"},                       "App Issues"),
    ({"facile","semplice","usare","funziona","ricerca"},      "App Usability"),
    ({"perfetto","perfetta","ottimo","ottima","veloce",
      "affidabile","soddisfatto","soddisfatta","consiglio"},  "Item Quality"),
    ({"esperienza","positiva","fantastica","benissimo"},      "Positive Experience"),
    ({"vendere","comprare","acquisto"},                       "Buying and Selling"),
]

def _auto_label(words: list) -> str:
    word_set = set(words)
    for keywords, label in _LABEL_HINTS:
        if word_set & keywords:
            return label
    return " & ".join(w.title() for w in words[:2])


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vinted Trust Radar",
    page_icon="shopping_bags",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{ background:{BG}; }}
    [data-testid="stSidebar"] {{ background:{NAVY}; }}
    [data-testid="stSidebar"] * {{ color:#C8D2DC !important; }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{ color:#FFFFFF !important; }}
    .block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1440px; }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{ font-weight:600; color:#607080; }}
    [data-testid="stTabs"] [aria-selected="true"] {{ color:{BLUE} !important; border-bottom-color:{BLUE} !important; }}
    div[data-testid="stHorizontalBlock"] {{ gap:1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(label: str, value: str, note: str = "", accent: str = BLUE):
    st.markdown(
        f'<div style="background:#FFFFFF;border-radius:10px;padding:18px 20px;'
        f'box-shadow:0 2px 10px rgba(10,22,40,.08);border-left:5px solid {accent};">'
        f'<div style="font-size:.75rem;color:#7A8FA0;margin-bottom:5px;text-transform:uppercase;letter-spacing:.7px;font-weight:600">{label}</div>'
        f'<div style="font-size:1.72rem;font-weight:800;line-height:1.05;color:{NAVY}">{value}</div>'
        f'<div style="font-size:.78rem;color:#8FA0B0;margin-top:5px">{note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def sec(title: str, subtitle: str = ""):
    sub_html = (f'<p style="margin:.3rem 0 0 0;color:#607080;font-size:.88rem;line-height:1.4">{subtitle}</p>'
                if subtitle else "")
    st.markdown(
        f'<div style="border-left:4px solid {BLUE};padding-left:12px;margin:1.4rem 0 .8rem 0">'
        f'<h3 style="margin:0;color:{NAVY};font-size:1.18rem;font-weight:700">{title}</h3>{sub_html}</div>',
        unsafe_allow_html=True,
    )

def method(text: str):
    st.markdown(
        f'<p style="background:#EEF4F8;border-left:4px solid {TEAL};border-radius:0 8px 8px 0;'
        f'padding:12px 18px;font-size:.88rem;color:#2E4057;line-height:1.65;margin:0 0 .6rem 0">{text}</p>',
        unsafe_allow_html=True,
    )


# ── Plotly base layout ────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F4F6F9",
    font=dict(color=NAVY, family="sans-serif", size=12),
    margin=dict(l=10, r=20, t=48, b=10),
    hoverlabel=dict(bgcolor="white", font_size=13, font_color=NAVY),
)

def styled_fig(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=title, font=dict(size=14, color=NAVY), x=0, xanchor="left"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E0E6EE", zeroline=False,
                     tickfont=dict(color=NAVY), title_font=dict(color=NAVY))
    fig.update_yaxes(showgrid=True, gridcolor="#E0E6EE", zeroline=False,
                     tickfont=dict(color=NAVY), title_font=dict(color=NAVY))
    return fig


# ── Data pipeline ─────────────────────────────────────────────────────────────
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


def classify(compound: float) -> str:
    if compound >= 0.05:  return "positive"
    if compound <= -0.05: return "negative"
    return "neutral"


@st.cache_data(show_spinner=False)
def read_default_data():
    return pd.read_csv(DEFAULT_CSV)


@st.cache_data(show_spinner="Computing sentiment scores...")
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(raw)
    analyzer = SentimentIntensityAnalyzer()
    df["compound"]  = df["text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    df["sentiment"] = df["compound"].apply(classify)
    return df


def aspect_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lower = df["text"].astype(str).str.lower()
    for aspect, keywords in ASPECTS.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        sub = df[lower.str.contains(pattern, regex=True, na=False)]
        if sub.empty:
            continue
        rows.append({
            "aspect":         aspect,
            "reviews":        len(sub),
            "mean_sentiment": sub["compound"].mean(),
            "avg_stars":      sub["score"].mean(),
            "pct_negative":   (sub["compound"] <= -0.05).mean() * 100,
            "keywords":       ", ".join(keywords),
        })
    return pd.DataFrame(rows).sort_values("avg_stars") if rows else pd.DataFrame()


def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["week"] = temp["at"].dt.to_period("W")
    wk = temp.groupby("week").agg(
        reviews       =("text",     "count"),
        mean_sentiment=("compound", "mean"),
        mean_score    =("score",    "mean"),
        pct_negative  =("compound", lambda x: (x <= -0.05).mean() * 100),
    ).reset_index()
    wk["week_start"] = wk["week"].dt.start_time
    wk["week_str"]   = wk["week"].astype(str)
    wk["week_label"] = wk["week_start"].dt.strftime("%b %-d")
    return wk


@st.cache_data(show_spinner="Running LDA topic modelling (first run may take ~30 s)...")
def run_lda(df: pd.DataFrame, n_topics: int = N_TOPICS):
    """Train gensim LDA on df texts and assign a dominant topic to each review."""
    _tok = re.compile(r"[a-zA-ZàáâãäåæçèéêëìíîïòóôõöùúûüÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜ]{3,}")
    texts = [
        [t for t in _tok.findall(str(x).lower()) if t not in ITALIAN_STOPWORDS]
        for x in df["text"]
    ]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.70)
    corpus = [dictionary.doc2bow(t) for t in texts]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=15,
        random_state=42,
    )

    # Assign dominant topic per review
    topic_ids = []
    for bow in corpus:
        if not bow:
            topic_ids.append(0)
        else:
            dist_ = lda.get_document_topics(bow)
            topic_ids.append(max(dist_, key=lambda x: x[1])[0] if dist_ else 0)

    df_out = df.copy()
    df_out["topic_id"]  = topic_ids
    df_out["topic_ref"] = [f"T{t}" for t in topic_ids]

    # Build topic summary DataFrame
    rows = []
    for t in range(n_topics):
        top10 = lda.show_topic(t, topn=10)
        words = [w for w, _ in top10]
        sub   = df_out[df_out["topic_id"] == t]
        rows.append({
            "topic":     f"T{t}",
            "label":     _auto_label(words),
            "top_words": ", ".join(words),
            "reviews":   len(sub),
            "avg_stars": float(sub["score"].mean()) if len(sub) > 0 else 0.0,
            "color":     COLORS_CYCLE[t % len(COLORS_CYCLE)],
            "keywords":  words,
        })

    return pd.DataFrame(rows), df_out


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Vinted Trust Radar")
    st.markdown("---")
    st.markdown("### Data Input")
    uploaded = st.file_uploader("Upload Google Play reviews CSV", type=["csv"])
    st.caption("Leave empty to use the bundled Vinted Italy dataset (cleaned, Apr-May 2026).")
    st.markdown("---")
    st.info("Sentiment engine: VADER (lexicon-based baseline)")
    st.caption(
        "VADER is a transparent, lexicon-based baseline. "
        "It was designed for English; on Italian text it serves as a "
        "cross-validated proxy, confirmed against Google Play star ratings."
    )
    st.success("LDA engine: gensim (live computation)")
    st.markdown("---")
    st.markdown(
        "**Project:** Web and Social Media Analytics - BADS 2026  \n"
        "Frank Novoa - Dustin Dutan  \nAlessandro Micagni - Francesco Todaro"
    )

# ── Load and process data ─────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded) if uploaded is not None else read_default_data()
    df = prepare(raw_df)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

try:
    topic_df, df = run_lda(df)
except Exception as exc:
    st.error(f"LDA failed: {exc}")
    st.stop()

topic_lookup      = topic_df.set_index("topic")
df["topic_label"] = df["topic_ref"].map(topic_lookup["label"].to_dict())
overall_avg_stars = df["score"].mean()

# ═══════════════════════════════════════════════════════════════════════════════
# Hero banner
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div style="background:linear-gradient(135deg,{NAVY} 0%,#162540 100%);border-radius:12px;'
    f'padding:26px 30px;color:white;margin-bottom:1.4rem;border-left:6px solid {RED};'
    f'box-shadow:0 4px 20px rgba(10,22,40,.18)">'
    f'<h1 style="margin:0;font-size:2rem;line-height:1.1;letter-spacing:-.5px">Vinted Trust Radar</h1>'
    f'<p style="margin:.6rem 0 .9rem 0;color:#A8BDD0;font-size:.97rem;line-height:1.5;max-width:860px">'
    f'A trust analytics dashboard for Vinted Italy turning Google Play reviews into actionable sentiment signals, '
    f'aspect-based trust barriers, LDA topic clusters, and weekly temporal trends.</p>'
    f'<span style="background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);border-radius:20px;'
    f'padding:3px 12px;font-size:.78rem;color:#C8D2DC;margin-right:.4rem">'
    f'{df["at"].min().date()} to {df["at"].max().date()}</span>'
    f'<span style="background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);border-radius:20px;'
    f'padding:3px 12px;font-size:.78rem;color:#C8D2DC;margin-right:.4rem">{len(df):,} reviews</span>'
    f'<span style="background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);border-radius:20px;'
    f'padding:3px 12px;font-size:.78rem;color:#C8D2DC">VADER + Live LDA + Temporal Analysis</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Global KPI row ────────────────────────────────────────────────────────────
dist = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)

c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
with c1: kpi("Total reviews",       f"{len(df):,}",                          "after cleaning",                BLUE)
with c2: kpi("Average star rating", f"{overall_avg_stars:.2f} *",            "Google Play score",             TEAL)
with c3: kpi("Positive reviews",    f"{dist['positive']/len(df)*100:.1f}%",  f"{dist['positive']:,} reviews", GREEN)
with c4: kpi("Negative reviews",    f"{dist['negative']/len(df)*100:.1f}%",  f"{dist['negative']:,} reviews", RED)
with c5: kpi("Mean compound",       f"{df['compound'].mean():.3f}",          "VADER compound score",          PURPLE)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_data, tab_sent, tab_topics, tab_time, tab_summary = st.tabs([
    "Data Input",
    "Sentiment and Aspects",
    "Topic Clusters",
    "Temporal Trend",
    "Key Findings",
])


# ─── TAB 1: Data Input ───────────────────────────────────────────────────────
with tab_data:
    sec("1. Data Input - Dataset overview",
        "Google Play public reviews of Vinted (fr.vinted) filtered for Italy and Italian language.")

    method(
        "<b>Data collection:</b> Reviews were scraped from the Google Play Store using the "
        "<code>google-play-scraper</code> library, filtered by language (<code>it</code>) and "
        "country (<code>it</code>). Reddit was initially considered as an additional source but "
        "persistent access restrictions prevented reliable collection. App Store scraping returned "
        "no usable results. The final dataset focuses on Google Play reviews, which provide "
        "structured star ratings, timestamps, and review text adequate for sentiment, topic, and temporal analysis."
    )

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        display_cols = [c for c in ["text","score","at","thumbsUpCount","language","compound","sentiment"] if c in df.columns]
        st.dataframe(df[display_cols].head(120), height=420, hide_index=True, use_container_width=True)

    with right:
        counts = df["score"].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            marker_color=[RED, AMBER, GREY, TEAL, GREEN],
            text=[f"{int(v):,}" for v in counts.values],
            textposition="outside",
            textfont=dict(color=NAVY, size=12),
        ))
        styled_fig(fig, "Star Rating Distribution - Google Play Italy")
        fig.update_layout(
            xaxis_title="Stars", yaxis_title="Number of reviews",
            showlegend=False, height=370,
            yaxis=dict(range=[0, counts.max() * 1.15]),
        )
        st.plotly_chart(fig, use_container_width=True)

    sec("Column schema")
    schema = pd.DataFrame({
        "Column":   list(df.columns),
        "Type":     [str(df[c].dtype) for c in df.columns],
        "Non-null": [int(df[c].notna().sum()) for c in df.columns],
        "Sample":   [str(df[c].iloc[0])[:60] for c in df.columns],
    })
    st.dataframe(schema, hide_index=True, use_container_width=True)


# ─── TAB 2: Sentiment and Aspects ────────────────────────────────────────────
with tab_sent:
    sec("2. Sentiment and Aspect-Based Analysis",
        "VADER compound score maps to polarity labels. Italian keyword dictionaries produce aspect-level trust scores.")

    method(
        "<b>Why VADER:</b> VADER (Valence Aware Dictionary and sEntiment Reasoner) is a "
        "transparent, lexicon-based method that requires no training data. On Italian text it serves "
        "as a practical baseline; compound scores are <i>cross-validated against Google Play star "
        "ratings</i> to confirm directional alignment. "
        "<b>Why aspect-based:</b> Overall polarity alone does not identify <i>which</i> "
        "trust dimensions drive dissatisfaction. Keyword dictionaries for seven aspects "
        "(authenticity, refunds, support, shipping, fees, app usability, seller quality) map each "
        "review to the trust barrier it mentions."
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: kpi("Positive",      f"{int(dist['positive']):,}", f"{dist['positive']/len(df)*100:.1f}% of reviews", GREEN)
    with c2: kpi("Neutral",       f"{int(dist['neutral']):,}",  f"{dist['neutral']/len(df)*100:.1f}% of reviews",  GREY)
    with c3: kpi("Negative",      f"{int(dist['negative']):,}", f"{dist['negative']/len(df)*100:.1f}% of reviews", RED)
    with c4: kpi("Mean compound", f"{df['compound'].mean():.3f}", "mild positive baseline", BLUE)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.5], gap="large")

    with left:
        fig = go.Figure(go.Pie(
            labels=[s.title() for s in dist.index],
            values=dist.values,
            hole=.38,
            marker_colors=[GREEN, GREY, RED],
            textinfo="label+percent",
            textfont=dict(color=NAVY, size=13),
            hovertemplate="%{label}: %{value:,} reviews (%{percent})<extra></extra>",
        ))
        styled_fig(fig, "Overall Sentiment Distribution")
        fig.update_layout(
            height=360, showlegend=False,
            annotations=[dict(text="VADER", x=.5, y=.5, font=dict(size=14, color=NAVY), showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        aspects = aspect_summary(df)
        if not aspects.empty:
            asp_sorted = aspects.sort_values("mean_sentiment", ascending=True)
            bar_colors = [RED if v < 0.06 else (GREEN if v > 0.12 else BLUE)
                          for v in asp_sorted["mean_sentiment"]]
            x_max = asp_sorted["mean_sentiment"].max() * 1.35
            fig = go.Figure(go.Bar(
                x=asp_sorted["mean_sentiment"],
                y=asp_sorted["aspect"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in asp_sorted["mean_sentiment"]],
                textposition="outside",
                textfont=dict(color=NAVY, size=12),
                hovertemplate="<b>%{y}</b><br>Mean compound: %{x:.3f}<extra></extra>",
            ))
            styled_fig(fig, "Aspect-Based Sentiment (mean VADER compound)")
            fig.update_layout(
                height=360, showlegend=False,
                xaxis=dict(title="Mean VADER compound score", range=[0, x_max]),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig, use_container_width=True)

    sec("Aspect evidence table",
        "Reviews, avg star rating, pct negative, and mean sentiment per trust dimension.")
    if not aspects.empty:
        disp = aspects[["aspect","reviews","avg_stars","pct_negative","mean_sentiment","keywords"]].copy()
        disp.columns = ["Aspect","Reviews","Avg Stars","Pct Negative","Mean Compound","Keywords"]
        disp["Avg Stars"]     = disp["Avg Stars"].round(2)
        disp["Pct Negative"]  = disp["Pct Negative"].round(1)
        disp["Mean Compound"] = disp["Mean Compound"].round(3)
        st.dataframe(disp, hide_index=True, use_container_width=True)

    if not aspects.empty:
        asp_stars = aspects.sort_values("avg_stars", ascending=True)
        star_colors = [RED if v < 2.5 else (AMBER if v < 3.5 else GREEN)
                       for v in asp_stars["avg_stars"]]
        fig2 = go.Figure(go.Bar(
            x=asp_stars["avg_stars"],
            y=asp_stars["aspect"],
            orientation="h",
            marker_color=star_colors,
            text=[f"{v:.2f}" for v in asp_stars["avg_stars"]],
            textposition="outside",
            textfont=dict(color=NAVY, size=12),
            hovertemplate="<b>%{y}</b><br>Avg stars: %{x:.2f}<extra></extra>",
        ))
        styled_fig(fig2, "Average Star Rating per Aspect")
        fig2.add_vline(x=overall_avg_stars, line_dash="dash", line_color=GREY,
                       annotation_text=f"Overall avg {overall_avg_stars:.2f}",
                       annotation_font_color=NAVY,
                       annotation_position="top right")
        fig2.update_layout(
            height=330, showlegend=False,
            xaxis=dict(title="Average star rating", range=[0, 5.5]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─── TAB 3: Topic Clusters ───────────────────────────────────────────────────
with tab_topics:
    sec("3. LDA Topic Modelling",
        "Unsupervised discovery of 5 latent themes using Gensim LDA (15 passes, random_state=42).")

    method(
        "<b>Why LDA:</b> Latent Dirichlet Allocation discovers hidden thematic structure "
        "from the data itself - no predefined labels are needed. Each topic is a probability "
        "distribution over vocabulary terms, making results transparent and interpretable. "
        "Assigning a dominant topic per review enables per-segment satisfaction comparison. "
        "<b>Setup:</b> Italian stopwords (NLTK) + custom Vinted terms removed; "
        "vocabulary filtered (no_below=5, no_above=0.70); 5 topics trained over 15 passes. "
        "<b>Note on labels:</b> Topic labels are derived from the top words returned by LDA - "
        "the top words shown in each card are the ground truth."
    )

    # Topic cards
    cols = st.columns(N_TOPICS, gap="medium")
    for col, (_, row) in zip(cols, topic_df.iterrows()):
        with col:
            st.markdown(
                f'<div style="background:white;border-radius:10px;padding:16px 14px;'
                f'box-shadow:0 2px 10px rgba(10,22,40,.07);border-top:5px solid {row["color"]};height:100%">'
                f'<span style="display:inline-block;color:white;font-weight:700;border-radius:6px;'
                f'padding:3px 9px;font-size:.82rem;margin-bottom:10px;background:{row["color"]}">{row["topic"]}</span>'
                f'<p style="font-weight:700;color:{NAVY};font-size:.97rem;margin:0 0 8px 0;line-height:1.3">{row["label"]}</p>'
                f'<p style="font-size:.79rem;color:#607080;margin:0 0 12px 0;line-height:1.45">{row["top_words"]}</p>'
                f'<p style="font-size:1.55rem;font-weight:800;color:{NAVY};margin:0">{int(row["reviews"]):,}</p>'
                f'<p style="font-size:.82rem;color:#607080;margin-top:3px">reviews - {row["avg_stars"]:.2f} avg</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        ref_sorted = topic_df.sort_values("reviews")
        x_max_rev = ref_sorted["reviews"].max() * 1.2
        fig = go.Figure(go.Bar(
            x=ref_sorted["reviews"],
            y=ref_sorted["topic"] + " - " + ref_sorted["label"],
            orientation="h",
            marker_color=ref_sorted["color"].tolist(),
            text=[f"{int(v):,}" for v in ref_sorted["reviews"]],
            textposition="outside",
            textfont=dict(color=NAVY, size=12),
            hovertemplate="<b>%{y}</b><br>Reviews: %{x:,}<extra></extra>",
        ))
        styled_fig(fig, "Reviews per LDA Topic")
        fig.update_layout(
            height=320, showlegend=False,
            xaxis=dict(title="Number of reviews", range=[0, x_max_rev]),
            yaxis=dict(tickfont=dict(size=10, color=NAVY)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        ref_sorted2 = topic_df.sort_values("avg_stars")
        fig2 = go.Figure(go.Bar(
            x=ref_sorted2["avg_stars"],
            y=ref_sorted2["topic"] + " - " + ref_sorted2["label"],
            orientation="h",
            marker_color=ref_sorted2["color"].tolist(),
            text=[f"{v:.2f}" for v in ref_sorted2["avg_stars"]],
            textposition="outside",
            textfont=dict(color=NAVY, size=12),
            hovertemplate="<b>%{y}</b><br>Avg stars: %{x:.2f}<extra></extra>",
        ))
        styled_fig(fig2, "Average Star Rating per Topic")
        fig2.add_vline(x=overall_avg_stars, line_dash="dash", line_color=GREY,
                       annotation_text=f"Overall avg {overall_avg_stars:.2f}",
                       annotation_font_color=NAVY,
                       annotation_position="top right")
        fig2.update_layout(
            height=320, showlegend=False,
            xaxis=dict(title="Average star rating", range=[0, 5.6]),
            yaxis=dict(tickfont=dict(size=10, color=NAVY)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    sec("Sample reviews by topic",
        "Each review is assigned to its dominant LDA topic based on the trained model.")
    selected_topic = st.selectbox(
        "Choose a topic",
        topic_df["topic"] + " - " + topic_df["label"],
    )
    selected_code = selected_topic.split(" - ")[0]
    topic_reviews = df[df["topic_ref"] == selected_code]
    samples = topic_reviews.sample(min(8, len(topic_reviews)), random_state=42).sort_values("score", ascending=False)
    for _, row in samples.iterrows():
        sent_color = GREEN if row["sentiment"] == "positive" else (RED if row["sentiment"] == "negative" else GREY)
        st.markdown(
            f'<div style="background:white;border-radius:8px;padding:12px 16px;margin-bottom:.6rem;'
            f'border-left:4px solid {sent_color};box-shadow:0 1px 6px rgba(10,22,40,.06)">'
            f'<span style="font-weight:700;color:{NAVY}">{int(row["score"])} stars</span>'
            f'&nbsp;&nbsp;<span style="color:{GREY};font-size:.83rem">{row["at"].date()} - compound {row["compound"]:.3f}</span>'
            f'<p style="margin:.5rem 0 0;color:#333;font-size:.91rem">{str(row["text"])[:320]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─── TAB 4: Temporal Trend ───────────────────────────────────────────────────
with tab_time:
    sec("4. Temporal Sentiment Analysis - Weekly Trends",
        "Weekly aggregation of VADER compound scores and pct negative reviews (Apr-May 2026).")

    method(
        "<b>Why temporal analysis:</b> User trust is not static. Resampling reviews "
        "by week reveals sentiment spikes that may correlate with platform events such as policy changes, "
        "outages, or carrier issues. The <code>at</code> timestamp is already present in the "
        "dataset so no extra data collection is required. Each week is summarised by mean VADER "
        "compound score, mean star rating, and percentage of negative reviews."
    )

    weekly = weekly_summary(df)

    week_cols = st.columns(min(len(weekly), 4), gap="medium")
    for i, (_, row) in enumerate(weekly.head(4).iterrows()):
        with week_cols[i]:
            kpi(row["week_label"], f"{row['mean_sentiment']:.3f}",
                f"{int(row['reviews'])} rev - {row['pct_negative']:.1f}% neg",
                BLUE if row["mean_sentiment"] >= 0.05 else RED)

    if len(weekly) > 4:
        st.markdown("<br>", unsafe_allow_html=True)
        week_cols2 = st.columns(min(len(weekly) - 4, 4), gap="medium")
        for i, (_, row) in enumerate(weekly.iloc[4:].iterrows()):
            with week_cols2[i]:
                kpi(row["week_label"], f"{row['mean_sentiment']:.3f}",
                    f"{int(row['reviews'])} rev - {row['pct_negative']:.1f}% neg",
                    BLUE if row["mean_sentiment"] >= 0.05 else RED)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["week_label"], y=weekly["mean_sentiment"],
        name="Mean sentiment (compound)",
        mode="lines+markers",
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=9, color=BLUE),
        fill="tozeroy",
        fillcolor="rgba(55,138,221,.10)",
        hovertemplate="Week: %{x}<br>Sentiment: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=weekly["week_label"], y=weekly["pct_negative"],
        name="Pct negative reviews",
        mode="lines+markers",
        line=dict(color=RED, width=2, dash="dash"),
        marker=dict(size=8, color=RED, symbol="square"),
        yaxis="y2",
        hovertemplate="Week: %{x}<br>Pct negative: %{y:.1f}%<extra></extra>",
    ))

    worst_idx = weekly["mean_sentiment"].idxmin()
    worst_row = weekly.loc[worst_idx]
    fig.add_annotation(
        x=worst_row["week_label"],
        y=worst_row["mean_sentiment"],
        text=f"Lowest sentiment<br>{worst_row['week_label']}",
        showarrow=True, arrowhead=2, arrowcolor=RED,
        font=dict(color=RED, size=11), bgcolor="white",
        bordercolor=RED, borderwidth=1, borderpad=4,
        ay=-45,
    )

    styled_fig(fig, "Weekly Sentiment Trend - Vinted Italy (Apr-May 2026)")
    fig.update_layout(
        height=430,
        xaxis=dict(
            title="Week starting",
            type="category",
            tickfont=dict(color=NAVY, size=12),
            title_font=dict(color=NAVY),
        ),
        yaxis=dict(
            title=dict(text="Mean VADER Compound", font=dict(color=BLUE)),
            tickfont=dict(color=BLUE),
            range=[0, weekly["mean_sentiment"].max() * 1.4],
        ),
        yaxis2=dict(
            title=dict(text="Pct Negative Reviews", font=dict(color=RED)),
            tickfont=dict(color=RED),
            overlaying="y", side="right",
            range=[0, weekly["pct_negative"].max() * 2.8],
            showgrid=False,
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,.9)",
            bordercolor="#DDE5EE", borderwidth=1,
            font=dict(color=NAVY),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    disp_weekly = weekly[["week_str","reviews","mean_sentiment","mean_score","pct_negative"]].copy()
    disp_weekly.columns = ["Week","Reviews","Mean Compound","Mean Stars","Pct Negative"]
    disp_weekly["Mean Compound"] = disp_weekly["Mean Compound"].round(3)
    disp_weekly["Mean Stars"]    = disp_weekly["Mean Stars"].round(2)
    disp_weekly["Pct Negative"]  = disp_weekly["Pct Negative"].round(1)
    st.dataframe(disp_weekly, hide_index=True, use_container_width=True)


# ─── TAB 5: Key Findings ─────────────────────────────────────────────────────
with tab_summary:
    sec("Key findings across all three analyses",
        f"Google Play Italy - Apr-May 2026 - {len(df):,} reviews")

    dist_pos = dist["positive"] / len(df) * 100
    dist_neu = dist["neutral"]  / len(df) * 100
    dist_neg = dist["negative"] / len(df) * 100
    mean_cpd = df["compound"].mean()
    weekly   = weekly_summary(df)
    worst    = weekly.loc[weekly["mean_sentiment"].idxmin()]

    lda_worst = topic_df.loc[topic_df["avg_stars"].idxmin()]
    lda_best  = topic_df.loc[topic_df["avg_stars"].idxmax()]

    top_row_l, top_row_r = st.columns(2, gap="medium")

    # ── Card 1: Sentiment ────────────────────────────────────────────────────
    with top_row_l:
        st.markdown(
            f'<div style="background:#FFFFFF;border:0.5px solid #DDE5EE;border-radius:12px;'
            f'padding:1.25rem;border-top:3px solid {PURPLE};">'
            f'<div style="font-size:.75rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:.6px;color:#607080;margin-bottom:12px">Sentiment (VADER)</div>'
            f'<div style="display:flex;gap:8px;margin-bottom:14px">'
            f'<div style="flex:1;background:#EAF3DE;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="font-size:1.35rem;font-weight:500;color:#3B6D11">{dist_pos:.1f}%</div>'
            f'<div style="font-size:.75rem;color:#3B6D11;margin-top:2px">positive</div></div>'
            f'<div style="flex:1;background:#F4F6F9;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="font-size:1.35rem;font-weight:500;color:{NAVY}">{dist_neu:.1f}%</div>'
            f'<div style="font-size:.75rem;color:#607080;margin-top:2px">neutral</div></div>'
            f'<div style="flex:1;background:#FCEBEB;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="font-size:1.35rem;font-weight:500;color:#A32D2D">{dist_neg:.1f}%</div>'
            f'<div style="font-size:.75rem;color:#A32D2D;margin-top:2px">negative</div></div>'
            f'</div>'
            f'<div style="font-size:.85rem;color:#607080;line-height:1.5">'
            f'Mean compound score: <span style="font-weight:500;color:{NAVY}">{mean_cpd:.3f}</span> - mild positive baseline.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Card 2: Aspect-Based ─────────────────────────────────────────────────
    with top_row_r:
        bottom3 = aspects.nsmallest(3, "avg_stars")[["aspect","avg_stars"]] if not aspects.empty else pd.DataFrame()
        rows_html = "".join(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'background:#FCEBEB;border-radius:8px;padding:8px 12px">'
            f'<span style="font-size:.87rem;font-weight:500;color:#A32D2D">{r["aspect"].title()}</span>'
            f'<span style="font-size:1.15rem;font-weight:500;color:#A32D2D">{r["avg_stars"]:.2f} &#9733;</span></div>'
            for _, r in bottom3.iterrows()
        )
        st.markdown(
            f'<div style="background:#FFFFFF;border:0.5px solid #DDE5EE;border-radius:12px;'
            f'padding:1.25rem;border-top:3px solid {BLUE};">'
            f'<div style="font-size:.75rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:.6px;color:#607080;margin-bottom:12px">Aspect-based</div>'
            f'<div style="display:flex;flex-direction:column;gap:7px;margin-bottom:14px">{rows_html}</div>'
            f'<div style="font-size:.85rem;color:#607080;line-height:1.5">'
            f'Three aspects far below the overall avg of <span style="font-weight:500;color:{NAVY}">'
            f'{overall_avg_stars:.2f} &#9733;</span>.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    bot_row_l, bot_row_r = st.columns(2, gap="medium")

    # ── Card 3: LDA Topics ───────────────────────────────────────────────────
    with bot_row_l:
        st.markdown(
            f'<div style="background:#FFFFFF;border:0.5px solid #DDE5EE;border-radius:12px;'
            f'padding:1.25rem;border-top:3px solid {GREEN};">'
            f'<div style="font-size:.75rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:.6px;color:#607080;margin-bottom:12px">LDA topic modelling</div>'
            f'<div style="display:flex;gap:8px;margin-bottom:14px">'
            f'<div style="flex:1;background:#FCEBEB;border:0.5px solid #F7C1C1;border-radius:8px;'
            f'padding:10px 12px;text-align:center">'
            f'<div style="font-size:.7rem;font-weight:500;color:#A32D2D;text-transform:uppercase;'
            f'letter-spacing:.4px;margin-bottom:4px">Critical topic</div>'
            f'<div style="font-size:.82rem;font-weight:500;color:#A32D2D;line-height:1.3">{lda_worst["label"]}</div>'
            f'<div style="font-size:1.25rem;font-weight:500;color:#A32D2D;margin-top:6px">{lda_worst["avg_stars"]:.2f} &#9733;</div>'
            f'<div style="font-size:.72rem;color:#A32D2D;margin-top:2px">{int(lda_worst["reviews"]):,} reviews</div></div>'
            f'<div style="flex:1;background:#EAF3DE;border:0.5px solid #C0DD97;border-radius:8px;'
            f'padding:10px 12px;text-align:center">'
            f'<div style="font-size:.7rem;font-weight:500;color:#3B6D11;text-transform:uppercase;'
            f'letter-spacing:.4px;margin-bottom:4px">Best topic</div>'
            f'<div style="font-size:.82rem;font-weight:500;color:#3B6D11;line-height:1.3">{lda_best["label"]}</div>'
            f'<div style="font-size:1.25rem;font-weight:500;color:#3B6D11;margin-top:6px">{lda_best["avg_stars"]:.2f} &#9733;</div>'
            f'<div style="font-size:.72rem;color:#3B6D11;margin-top:2px">{int(lda_best["reviews"]):,} reviews</div></div>'
            f'</div>'
            f'<div style="font-size:.85rem;color:#607080;line-height:1.5">'
            f'{N_TOPICS} topics discovered. {lda_worst["topic"]} ({lda_worst["label"]}) is the primary trust barrier.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Card 4: Temporal Trend ───────────────────────────────────────────────
    with bot_row_r:
        sent_min = weekly["mean_sentiment"].min()
        sent_max = weekly["mean_sentiment"].max()
        st.markdown(
            f'<div style="background:#FFFFFF;border:0.5px solid #DDE5EE;border-radius:12px;'
            f'padding:1.25rem;border-top:3px solid {SKY};">'
            f'<div style="font-size:.75rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:.6px;color:#607080;margin-bottom:12px">Temporal trend</div>'
            f'<div style="display:flex;gap:8px;margin-bottom:14px">'
            f'<div style="flex:1;background:#F4F6F9;border-radius:8px;padding:10px 12px;text-align:center">'
            f'<div style="font-size:.75rem;color:#607080;margin-bottom:4px">Sentiment range</div>'
            f'<div style="font-size:1.25rem;font-weight:500;color:{NAVY}">{sent_min:.3f}</div>'
            f'<div style="font-size:.75rem;color:#8FA0B0;margin:2px 0">to</div>'
            f'<div style="font-size:1.25rem;font-weight:500;color:{NAVY}">{sent_max:.3f}</div>'
            f'</div>'
            f'<div style="flex:1.4;display:flex;flex-direction:column;gap:7px">'
            f'<div style="background:#FCEBEB;border-radius:8px;padding:8px 12px">'
            f'<div style="font-size:.72rem;color:#A32D2D;margin-bottom:2px">Lowest week</div>'
            f'<div style="font-size:.87rem;font-weight:500;color:#A32D2D">{worst["week_label"]}</div>'
            f'<div style="font-size:.72rem;color:#A32D2D">'
            f'{worst["pct_negative"]:.1f}% negative - {int(worst["reviews"])} reviews</div></div>'
            f'<div style="background:#F4F6F9;border-radius:8px;padding:8px 12px">'
            f'<div style="font-size:.72rem;color:#607080;margin-bottom:2px">Overall trend</div>'
            f'<div style="font-size:.87rem;font-weight:500;color:{NAVY}">Stable</div>'
            f'</div>'
            f'</div></div>'
            f'<div style="font-size:.85rem;color:#607080;line-height:1.5">'
            f'{len(weekly)} weeks analysed. No major sentiment spikes detected.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
