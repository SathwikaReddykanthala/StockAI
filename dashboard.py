import os
import time
import pandas as pd
import streamlit as st


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="StockAI | NSE AI Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #777;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .live-badge {
        background-color: #16a34a;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .small-note {
        color: #777;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">📈 StockAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered NSE stock screening using SMMA, LTQ, ETQ, '
    'market depth, volume and machine-learning probability'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<span class="live-badge">● LIVE SCREENING</span>',
    unsafe_allow_html=True
)


# ==========================================================
# FILES
# ==========================================================

FINAL_FILE = "dataset/final_ai_decisions.csv"
LTQ_FILE = "dataset/live_ltq_features.csv"
SMMA_FILE = "dataset/live_smma_signals.csv"
MARKET_FILE = "dataset/live_market.csv"


# ==========================================================
# LOAD FUNCTION
# ==========================================================

def load_csv(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception as e:

        st.error(f"Could not read {path}: {e}")

        return pd.DataFrame()


final_df = load_csv(FINAL_FILE)
ltq_df = load_csv(LTQ_FILE)
smma_df = load_csv(SMMA_FILE)
market_df = load_csv(MARKET_FILE)


# ==========================================================
# CLEAN SYMBOLS
# ==========================================================

for df in [final_df, ltq_df, smma_df, market_df]:

    if not df.empty and "Symbol" in df.columns:

        df["Symbol"] = (
            df["Symbol"]
            .astype(str)
            .str.replace("\\", "", regex=False)
        )


# ==========================================================
# STOCK NAME MAP
# ==========================================================

STOCK_NAMES = {

    "NSE:BPCL-EQ":
        "Bharat Petroleum Corporation Ltd",

    "NSE:RELIANCE-EQ":
        "Reliance Industries Ltd",

    "NSE:TCS-EQ":
        "Tata Consultancy Services Ltd",

    "NSE:INFY-EQ":
        "Infosys Ltd",

    "NSE:HDFCBANK-EQ":
        "HDFC Bank Ltd",

    "NSE:ICICIBANK-EQ":
        "ICICI Bank Ltd",

    "NSE:SBIN-EQ":
        "State Bank of India",

    "NSE:ITC-EQ":
        "ITC Ltd",

    "NSE:LT-EQ":
        "Larsen & Toubro Ltd",

    "NSE:WIPRO-EQ":
        "Wipro Ltd",

    "NSE:IVZINNIFTY-EQ":
        "Invesco Nifty ETF",

    "NSE:GENESYS-EQ":
        "Genesys International Corporation Ltd",

    "NSE:ATALREAL-EQ":
        "Atal Realtech Ltd",

    "NSE:SPARC-EQ":
        "Sun Pharma Advanced Research Company Ltd",

    "NSE:ZUARI-EQ":
        "Zuari Agro Chemicals Ltd",

    "NSE:SONATSOFTW-EQ":
        "Sonata Software Ltd",

    "NSE:GLOBAL-EQ":
        "Global Education Ltd",

    "NSE:MHRIL-EQ":
        "Mahindra Holidays & Resorts India Ltd",

    "NSE:SHAHALLOYS-EQ":
        "Shah Alloys Ltd",

    "NSE:21STCENMGM-EQ":
        "21st Century Management Services Ltd",

    "NSE:SPLIL-EQ":
        "SPL Industries Ltd",

    "NSE:ICDSLTD-EQ":
        "ICDS Ltd",

    "NSE:GUJRAFFIA-EQ":
        "Gujarat Raffia Industries Ltd",

    "NSE:AEROPLANE-EQ":
        "Aeroplane Company Ltd",

    "NSE:GROWWLIQID-EQ":
        "Groww Liquid Fund",

}


def add_stock_name(df):

    if df.empty or "Symbol" not in df.columns:

        return df

    df = df.copy()

    df["Stock Name"] = (
        df["Symbol"]
        .map(STOCK_NAMES)
        .fillna(
            df["Symbol"]
            .str.replace("NSE:", "", regex=False)
            .str.replace("-EQ", "", regex=False)
        )
    )

    return df


# ==========================================================
# PREPARE FINAL DATA
# ==========================================================

if final_df.empty:

    st.error(
        "dataset/final_ai_decisions.csv not found. "
        "Run the StockAI pipeline first."
    )

    st.stop()


final_df = add_stock_name(final_df)


# ==========================================================
# NUMERIC CONVERSION
# ==========================================================

numeric_columns = [

    "Close",
    "LTQ",
    "LTQ_2min_avg",
    "LTQ_5min_avg",
    "LTQ_Spike_Ratio",

    "ETQ_5min",
    "ETQ_20min",
    "ETQ_60min",

    "BidQty",
    "AskQty",
    "BidAsk_Imbalance",

    "Volume",
    "Return_1",
    "Return_5",

    "ML_Probability",
    "Participation_Score",

    "SMMA20",
    "SMMA120",
    "SMMA_Gap",

    "Volume_Ratio"

]


for col in numeric_columns:

    if col in final_df.columns:

        final_df[col] = pd.to_numeric(
            final_df[col],
            errors="coerce"
        )


# ==========================================================
# PRICE FILTER
# ==========================================================

final_df = final_df[
    final_df["Symbol"]
    .astype(str)
    .str.endswith("-EQ")
].copy()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🔎 StockAI Filters")

st.sidebar.caption(
    "Use filters to identify higher-quality setups."
)


# ==========================================================
# PRICE FILTER
# ==========================================================

st.sidebar.markdown("### 💰 Price")

min_price = float(
    final_df["Close"].min()
)

max_price = float(
    final_df["Close"].max()
)

price_range = st.sidebar.slider(

    "Price range",

    min_value=float(
        max(0, min_price)
    ),

    max_value=float(
        max_price
    ),

    value=(

        float(
            max(0, min_price)
        ),

        float(
            max_price
        )
    )
)


# ==========================================================
# SIGNAL FILTER
# ==========================================================

st.sidebar.markdown("### 📈 SMMA Signal")

signal_filter = st.sidebar.multiselect(

    "Signal",

    [
        "BUY",
        "SELL",
        "NONE"
    ],

    default=[
        "BUY",
        "SELL",
        "NONE"
    ]
)


trend_filter = st.sidebar.multiselect(

    "Trend",

    [
        "BULLISH",
        "BEARISH"
    ],

    default=[
        "BULLISH",
        "BEARISH"
    ]
)


# ==========================================================
# ML FILTER
# ==========================================================

st.sidebar.markdown("### 🤖 Machine Learning")

ml_threshold = st.sidebar.slider(
    "Minimum ML probability",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01,
    format="%.2f"
)

# ==========================================================
# LTQ FILTER
# ==========================================================

st.sidebar.markdown("### ⚡ LTQ")

ltq_threshold = st.sidebar.slider(

    "Minimum LTQ Spike Ratio",

    min_value=0.0,

    max_value=5.0,

    value=0.0,

    step=0.1
)


# ==========================================================
# ETQ FILTER
# ==========================================================

st.sidebar.markdown("### 🔄 Exchange Activity")

etq_threshold = st.sidebar.number_input(

    "Minimum ETQ 5min",

    min_value=0,

    max_value=1_000_000,

    value=0,

    step=100
)


# ==========================================================
# VOLUME FILTER
# ==========================================================

st.sidebar.markdown("### 📊 Volume")

volume_ratio_threshold = st.sidebar.slider(

    "Minimum Volume Ratio",

    min_value=0.0,

    max_value=10.0,

    value=0.0,

    step=0.1
)


# ==========================================================
# RETURN FILTER
# ==========================================================

st.sidebar.markdown("### 📉 Short-Term Return")

return5_min = st.sidebar.number_input(

    "Minimum Return 5",

    value=-100.0,

    step=0.1
)

return5_max = st.sidebar.number_input(

    "Maximum Return 5",

    value=100.0,

    step=0.1
)


# ==========================================================
# LIQUIDITY FILTER
# ==========================================================

st.sidebar.markdown("### 💧 Liquidity")

liquidity_only = st.sidebar.checkbox(

    "Only liquidity-qualified",

    value=False
)


# ==========================================================
# AI FILTER
# ==========================================================

st.sidebar.markdown("### 🧠 AI Decision")

ai_filter = st.sidebar.multiselect(

    "AI Decision",

    [
        "ACCEPT_CANDIDATE",
        "WATCH",
        "AVOID"
    ],

    default=[
        "ACCEPT_CANDIDATE",
        "WATCH",
        "AVOID"
    ]
)


# ==========================================================
# APPLY FILTERS
# ==========================================================

df = final_df.copy()


df = df[
    (df["Close"] >= price_range[0]) &
    (df["Close"] <= price_range[1])
]


if signal_filter:

    df = df[
        df["Signal"].isin(
            signal_filter
        )
    ]


if trend_filter:

    df = df[
        df["Trend"].isin(
            trend_filter
        )
    ]


df = df[
    df["ML_Probability"].fillna(0)
    >= ml_threshold
]


df = df[
    df["LTQ_Spike_Ratio"].fillna(0)
    >= ltq_threshold
]


df = df[
    df["ETQ_5min"].fillna(0)
    >= etq_threshold
]


if "Volume_Ratio" in df.columns:

    df = df[
        df["Volume_Ratio"].fillna(0)
        >= volume_ratio_threshold
    ]


if "Return_5" in df.columns:

    df = df[
        (df["Return_5"].fillna(0) >= return5_min) &
        (df["Return_5"].fillna(0) <= return5_max)
    ]


if liquidity_only:

    df = df[
        (df["BidQty"].fillna(0) > 1_000_000) &
        (df["AskQty"].fillna(0) > 1_000_000)
    ]


if ai_filter:

    df = df[
        df["AI_Decision"].isin(
            ai_filter
        )
    ]


# ==========================================================
# LIQUIDITY COLUMN
# ==========================================================

df["Liquidity_Qualified"] = (

    (df["BidQty"].fillna(0) > 1_000_000) &

    (df["AskQty"].fillna(0) > 1_000_000)

)


# ==========================================================
# AI OPPORTUNITY SCORE
# ==========================================================

def calculate_score(row):

    score = 0

    ml = row.get(
        "ML_Probability",
        0
    )

    ltq = row.get(
        "LTQ_Spike_Ratio",
        0
    )

    etq = row.get(
        "ETQ_5min",
        0
    )

    volume_ratio = row.get(
        "Volume_Ratio",
        0
    )

    imbalance = abs(
        row.get(
            "BidAsk_Imbalance",
            0
        )
    )

    signal = row.get(
        "Signal",
        "NONE"
    )

    liquidity = row.get(
        "Liquidity_Qualified",
        False
    )


    # ML: 0-30
    score += min(
        max(ml, 0) * 30,
        30
    )


    # LTQ: 0-20
    if ltq >= 2:

        score += 20

    elif ltq >= 1.5:

        score += 15

    elif ltq >= 1:

        score += 10


    # ETQ: 0-15
    if etq >= 10000:

        score += 15

    elif etq >= 5000:

        score += 12

    elif etq >= 1000:

        score += 8

    elif etq > 0:

        score += 4


    # Volume: 0-15
    if volume_ratio >= 2:

        score += 15

    elif volume_ratio >= 1.5:

        score += 12

    elif volume_ratio >= 1:

        score += 8


    # Market depth: 0-10
    if imbalance >= 0.5:

        score += 10

    elif imbalance >= 0.25:

        score += 6

    elif imbalance >= 0.1:

        score += 3


    # SMMA: 0-10
    if signal in ["BUY", "SELL"]:

        score += 10


    # Liquidity bonus
    if liquidity:

        score += 10


    return round(
        min(score, 100),
        1
    )


df["AI_Opportunity_Score"] = df.apply(
    calculate_score,
    axis=1
)


# ==========================================================
# OPPORTUNITY LEVEL
# ==========================================================

def opportunity_level(score):

    if score >= 75:

        return "🔥 VERY STRONG"

    if score >= 60:

        return "🟢 STRONG"

    if score >= 45:

        return "🟡 WATCH"

    return "🔴 WEAK"


df["Opportunity_Level"] = (
    df["AI_Opportunity_Score"]
    .apply(opportunity_level)
)


# ==========================================================
# KPI VALUES
# ==========================================================

total_stocks = len(
    final_df
)

filtered_stocks = len(
    df
)

liquidity_count = int(
    df["Liquidity_Qualified"].sum()
)

buy_count = int(
    (df["Signal"] == "BUY").sum()
)

sell_count = int(
    (df["Signal"] == "SELL").sum()
)

accept_count = int(
    (
        df["AI_Decision"]
        == "ACCEPT_CANDIDATE"
    ).sum()
)

watch_count = int(
    (
        df["AI_Decision"]
        == "WATCH"
    ).sum()
)


# ==========================================================
# TOP KPI ROW
# ==========================================================

st.markdown(
    '<div class="section-title">📊 Market Overview</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4, c5, c6, c7 = st.columns(7)


c1.metric(
    "NSE Stocks",
    total_stocks
)


c2.metric(
    "Filtered",
    filtered_stocks
)


c3.metric(
    "💧 Liquidity",
    liquidity_count
)


c4.metric(
    "🟢 BUY",
    buy_count
)


c5.metric(
    "🔴 SELL",
    sell_count
)


c6.metric(
    "🔥 ACCEPT",
    accept_count
)


c7.metric(
    "🟡 WATCH",
    watch_count
)


# ==========================================================
# TOP AI OPPORTUNITIES
# ==========================================================

st.markdown(
    '<div class="section-title">'
    '🔥 Top AI Opportunities'
    '</div>',
    unsafe_allow_html=True
)


opportunities = df[
    df["AI_Opportunity_Score"] >= 45
].copy()


if opportunities.empty:

    st.info(
        "No stocks currently meet the opportunity "
        "score threshold. Relax the filters from "
        "the sidebar if required."
    )

else:

    opportunities = opportunities.sort_values(

        [
            "AI_Opportunity_Score",
            "ML_Probability"
        ],

        ascending=False

    )

    columns = [

        "Symbol",
        "Stock Name",
        "Close",

        "Signal",
        "Trend",

        "ML_Probability",

        "LTQ_Spike_Ratio",

        "ETQ_5min",

        "Volume_Ratio",

        "BidQty",
        "AskQty",

        "Liquidity_Qualified",

        "AI_Opportunity_Score",

        "Opportunity_Level",

        "AI_Decision"

    ]

    columns = [
        c for c in columns
        if c in opportunities.columns
    ]

    st.dataframe(

        opportunities[
            columns
        ].head(20),

        use_container_width=True,

        hide_index=True

    )


# ==========================================================
# CURRENT SMMA CROSSOVERS
# ==========================================================

st.markdown(
    '<div class="section-title">'
    '🎯 Current SMMA Crossover Signals'
    '</div>',
    unsafe_allow_html=True
)


crossovers = df[
    df["Signal"].isin(
        ["BUY", "SELL"]
    )
].copy()


if crossovers.empty:

    st.info(
        "No current SMMA 20/120 crossover "
        "is present in the filtered universe."
    )

else:

    columns = [

        "Symbol",
        "Stock Name",
        "Close",

        "SMMA20",
        "SMMA120",
        "SMMA_Gap",

        "Signal",
        "Trend",

        "LTQ",
        "LTQ_Spike_Ratio",

        "ETQ_5min",

        "BidQty",
        "AskQty",

        "ML_Probability",

        "AI_Opportunity_Score",

        "AI_Decision"

    ]

    columns = [
        c for c in columns
        if c in crossovers.columns
    ]

    st.dataframe(

        crossovers[
            columns
        ].sort_values(
            "AI_Opportunity_Score",
            ascending=False
        ),

        use_container_width=True,

        hide_index=True

    )


# ==========================================================
# LIQUIDITY
# ==========================================================

st.markdown(
    '<div class="section-title">'
    '💧 Liquidity-qualified Stocks'
    '</div>',
    unsafe_allow_html=True
)


liquidity_df = df[
    df["Liquidity_Qualified"]
].copy()


if liquidity_df.empty:

    st.warning(
        "No stock currently satisfies "
        "BidQty > 1,000,000 AND "
        "AskQty > 1,000,000."
    )

else:

    columns = [

        "Symbol",
        "Stock Name",
        "Close",

        "Signal",

        "BidQty",
        "AskQty",

        "BidAsk_Imbalance",

        "ML_Probability",

        "LTQ_Spike_Ratio",

        "ETQ_5min",

        "AI_Opportunity_Score",

        "AI_Decision"

    ]

    columns = [
        c for c in columns
        if c in liquidity_df.columns
    ]

    st.dataframe(

        liquidity_df[
            columns
        ].sort_values(
            "AI_Opportunity_Score",
            ascending=False
        ),

        use_container_width=True,

        hide_index=True

    )


# ==========================================================
# LTQ / ETQ
# ==========================================================

st.markdown(
    '<div class="section-title">'
    '⚡ LTQ / ETQ Analysis'
    '</div>',
    unsafe_allow_html=True
)


if not ltq_df.empty:

    ltq_display = add_stock_name(
        ltq_df.copy()
    )

    for col in [

        "LTQ",
        "LTQ_2min_avg",
        "LTQ_5min_avg",
        "LTQ_Spike_Ratio",
        "ETQ_5min",
        "ETQ_20min",
        "ETQ_60min",
        "BidQty",
        "AskQty",
        "BidAsk_Imbalance"

    ]:

        if col in ltq_display.columns:

            ltq_display[col] = pd.to_numeric(
                ltq_display[col],
                errors="coerce"
            )


    columns = [

        "Symbol",
        "Stock Name",

        "LTQ",
        "LTQ_2min_avg",
        "LTQ_5min_avg",

        "LTQ_Spike_Ratio",

        "ETQ_5min",
        "ETQ_20min",
        "ETQ_60min",

        "BidQty",
        "AskQty",
        "BidAsk_Imbalance"

    ]

    columns = [
        c for c in columns
        if c in ltq_display.columns
    ]


    if "LTQ_Spike_Ratio" in ltq_display.columns:

        ltq_display = ltq_display.sort_values(

            "LTQ_Spike_Ratio",

            ascending=False

        )


    st.dataframe(

        ltq_display[
            columns
        ].head(25),

        use_container_width=True,

        hide_index=True

    )

else:

    st.warning(
        "LTQ data is not available."
    )


# ==========================================================
# FULL STOCK SCREENER
# ==========================================================

st.markdown(
    '<div class="section-title">'
    '📋 Full Stock Screener'
    '</div>',
    unsafe_allow_html=True
)


st.caption(
    f"Showing {len(df)} stocks after applying sidebar filters."
)


columns = [

    "Symbol",
    "Stock Name",

    "Close",

    "Signal",
    "Trend",

    "SMMA20",
    "SMMA120",
    "SMMA_Gap",

    "LTQ",
    "LTQ_Spike_Ratio",

    "ETQ_5min",
    "ETQ_20min",
    "ETQ_60min",

    "BidQty",
    "AskQty",
    "BidAsk_Imbalance",

    "Volume",
    "Volume_Ratio",

    "Return_1",
    "Return_5",

    "ML_Probability",

    "Participation_Score",

    "AI_Opportunity_Score",
    "Opportunity_Level",

    "Liquidity_Qualified",

    "AI_Decision"

]

columns = [
    c for c in columns
    if c in df.columns
]


screen_df = df.sort_values(

    [
        "AI_Opportunity_Score",
        "ML_Probability"
    ],

    ascending=False

)


st.dataframe(

    screen_df[
        columns
    ],

    use_container_width=True,

    hide_index=True

)


# ==========================================================
# AI LOGIC
# ==========================================================

st.markdown(
    '<div class="section-title">'
    '🧠 AI Decision Logic'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
**BUY:** SMMA20 crosses above SMMA120.

**SELL:** SMMA20 crosses below SMMA120.

**AI analysis combines:**

- ML profitability probability
- LTQ spike ratio
- ETQ 5/20/60-minute activity
- Bid/ask imbalance
- Trading volume
- Volume ratio
- Short-term returns
- SMMA trend/crossover
- Liquidity qualification

### Opportunity Score

| Score | Classification |
|---:|---|
| 75–100 | 🔥 VERY STRONG |
| 60–74 | 🟢 STRONG |
| 45–59 | 🟡 WATCH |
| 0–44 | 🔴 WEAK |

**ACCEPT:** Strong supporting evidence.

**WATCH:** Potential setup but insufficient confirmation.

**AVOID:** Weak or conflicting evidence.
"""
)


# ==========================================================
# DISCLAIMER
# ==========================================================

st.markdown("---")

st.caption(
    "StockAI is a market-screening and analysis tool. "
    "AI probabilities and opportunity scores are analytical signals, "
    "not guarantees of future returns."
)


# ==========================================================
# AUTO REFRESH
# ==========================================================

time.sleep(10)

st.rerun()