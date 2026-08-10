## 📈 StockAI — AI Powered RealTime Stock Screening & Decision Intelligence Platform

StockAI is an end-to-end stock market analytics and AI-assisted screening platform designed to process market data, identify technical signals, analyze trading activity and liquidity, generate machine-learning probabilities, and produce a final AI-assisted stock decision.
The platform combines technical analysis, market microstructure, machine learning, liquidity analysis, and an interactive Streamlit dashboard into a single workflow.

### 🚀 Project Overview

StockAI processes market information through the following end-to-end workflow:

```text
Live Market Data
        ↓
Tick Processing
        ↓
1-Minute OHLC Candles
        ↓
Technical Indicators
        ↓
SMMA20 / SMMA120
        ↓
SMMA Crossover Detection
        ↓
LTQ / ETQ Analysis
        ↓
Market Depth Analysis
        ↓
ML Feature Engineering
        ↓
Machine Learning Probability
        ↓
Liquidity Qualification
        ↓
AI Opportunity Scoring
        ↓
Final AI Decision
        ↓
Streamlit Dashboard
```

The final system produces three decision categories:
```
🟢 ACCEPT_CANDIDATE
🟡 WATCH
🔴 AVOID
```

### 🎯 Objectives
The main objectives of StockAI are:

Process large volumes of market data.
Generate short-term technical indicators.
Identify SMMA crossover signals.
Detect unusual trading activity.
Analyze bid/ask market depth.
Estimate market participation.
Use machine learning to calculate probability scores.
Filter stocks based on liquidity.
Rank potentially interesting stocks.
Present the results through an interactive dashboard.

### 🏗️ Complete Architecture

                       ┌─────────────────────┐
                       │   Market Data/API   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │    Live Ticks       │
                       │  live_ticks.csv     │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Candle Builder     │
                       │    1-Min OHLC       │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  live_candles.csv   │
                       └──────────┬──────────┘
                                  │
              ┌───────────────────┼──────────────────┐
              │                   │                  │
              ▼                   ▼                  ▼
       ┌────────────┐      ┌────────────┐     ┌──────────────┐
       │    SMMA    │      │ LTQ / ETQ  │     │ Market Depth │
       │   Engine   │      │   Engine   │     │    Engine    │
       └──────┬─────┘      └──────┬─────┘     └──────┬───────┘
              │                   │                  │
              ▼                   ▼                  ▼
       live_smma_signals   live_ltq_features   Bid/Ask Data
              │                   │                  │
              └───────────────────┼──────────────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ ML Feature Builder  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       live_ml_features.csv
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ ML Prediction Model │
                       └──────────┬──────────┘
                                  │
                                  ▼
                           ML Probability
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Decision Engine     │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       final_ai_decisions.csv
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Streamlit Dashboard │
                       └─────────────────────┘
            
### 📂 Project Structure

```text
StockAI/
│
├── README.md
├── requirements.txt
├── .gitignore
├── dashboard.py
│
├── dataset/
│   │
│   ├── ai_dataset.csv
│   ├── ai_live_decisions.csv
│   ├── ai_predictions.csv
│   │
│   ├── crossover_dataset.csv
│   ├── crossover_dataset_20d.csv
│   ├── crossover_signals.csv
│   │
│   ├── depth_results.csv
│   ├── etq_results.csv
│   ├── etq_snapshot.csv
│   │
│   ├── final_ai_decisions.csv
│   │
│   ├── historical_20days.csv
│   ├── historical_candles.csv
│   ├── historical_liquid_20days.csv
│   ├── historical_trade_outcomes.csv
│   ├── history_with_indicators.csv
│   │
│   ├── indicator_dataset.csv
│   ├── indicator_dataset_20d.csv
│   │
│   ├── liquid_eq_stocks.csv
│   ├── liquid_stocks.csv
│   │
│   ├── live_candles.csv
│   ├── live_liquid_stocks.csv
│   ├── live_ltq_features.csv
│   ├── live_market.csv
│   ├── live_ml_features.csv
│   ├── live_smma_signals.csv
│   ├── live_ticks.csv
│   │
│   ├── ltq_training_dataset.csv
│   ├── ml_training_features.csv
│   │
│   ├── nse_universe.csv
│   ├── price_filtered_stocks.csv
│   │
│   ├── trade_dataset.csv
│   ├── trade_history.csv
│   ├── trade_outcomes.csv
│   │
│   ├── training_data.csv
│   ├── training_dataset.csv
│   └── training_dataset_20d.csv
│
├── models/
│   ├── model.pkl
│   ├── model_3class.pkl
│   ├── model_3class_20d.pkl
│   ├── model_3class_final.pkl
│   └── model_profitability.pkl
│
└── src/
    │
    ├── live/
    │   └── live_candle_builder.py
    │
    ├── smma/
    │   └── live_smma.py
    │
    ├── ml/
    │   └── final/
    │       ├── build_live_features.py
    │       └── decision_engine.py
    │
    └── run/
        └── run_pipeline.py
```

### 📡 Live Market Data
The system works with market tick data containing fields such as:
Timestamp
Symbol
LTP
LTQ
Volume
BidQty
AskQty

### 🕯️ Live Candle Generation
The live candle builder converts individual ticks into 1-minute candles.
OHLC
For every stock and minute:
Open  = first LTP
High  = maximum LTP
Low   = minimum LTP
Close = last LTP

### 📈 SMMA Analysis
StockAI calculates two smoothed moving averages:
SMMA20
SMMA120
The difference is:
SMMA_Gap = SMMA20 - SMMA120
The system also calculates:
SMMA_Spread_Pct
Price_vs_SMMA20
Price_vs_SMMA120

### 🔀 SMMA Crossover Signals
BUY
A bullish crossover occurs when:
SMMA20 crosses above SMMA120
Result:
Signal = BUY
Trend = BULLISH
SELL
A bearish crossover occurs when:
SMMA20 crosses below SMMA120
Result:
Signal = SELL
Trend = BEARISH
No crossover
Signal = NONE
This prevents every stock from being classified as a trade opportunity.

### ⚡ LTQ Analysis
LTQ represents Last Traded Quantity.
StockAI compares current LTQ against historical/recent LTQ activity.
Important fields include:
LTQ
LTQ_2min_avg
LTQ_5min_avg
LTQ_Spike_Ratio
A higher LTQ spike can indicate unusual trading activity.

### 📊 ETQ Analysis
StockAI calculates exchange-traded quantity over several periods:
ETQ_5min
ETQ_20min
ETQ_60min
These measurements help determine whether the stock has meaningful recent market participation.

### 💧 Market Depth
StockAI analyzes:
BidQty
AskQty
BidAsk_Imbalance

A simplified imbalance measure is:
BidAsk_Imbalance =
(BidQty - AskQty) / (BidQty + AskQty)

Interpretation:
Positive → stronger bid-side quantity
Negative → stronger ask-side quantity
Near zero → relatively balanced

### 💦 Liquidity Qualification
Liquidity is an important filter because extremely illiquid stocks can generate unreliable signals.
The system checks market-depth conditions before allowing a stock to become a stronger candidate.
 
The final engine can therefore distinguish between:
Technically interesting
and:
Technically interesting + sufficiently liquid

### 🤖 Machine Learning
The ML engine generates a probability score:
ML_Probability
The model uses features such as:
SMMA20
SMMA120
SMMA_Gap
SMMA_Spread_Pct
Price_vs_SMMA20
Price_vs_SMMA120
Return_1
Return_5
Return_10
Volume
Volume_MA5
Volume_MA20
Volume_Ratio

### 📊 Price & Volume Features
StockAI generates:
Return_1
Return_5
Return_10
and:
Volume
Volume_MA5
Volume_MA20
Volume_Ratio

Volume ratio:
Volume_Ratio = Volume / Volume_MA20

A value greater than 1 indicates volume is above its recent average.

### 🧠 AI Decision Engine
The decision engine combines multiple independent signals.

Conceptually:

ML Probability
       +
SMMA Signal
       +
Trend
       +
LTQ Activity
       +
ETQ Activity
       +
Market Depth
       +
Liquidity
       +
Participation
       ↓
AI Decision

Possible decisions:

🟢 ACCEPT_CANDIDATE

The stock satisfies the configured stronger screening conditions.

🟡 WATCH

The stock shows promising characteristics but requires additional confirmation.

🔴 AVOID

The stock does not satisfy the current screening criteria.

### 🏆 Participation Score
The system also uses a participation score to combine market activity indicators.
Potential contributors include:

ML probability
LTQ activity
ETQ activity
Bid/Ask imbalance
SMMA signal
Liquidity

The resulting score helps rank candidates.
📌 Stock Details
The dashboard can allow the user to select a stock and view its complete information.

🔥 AI Opportunity Ranking
StockAI ranks stocks using an AI opportunity score based on multiple market factors.

🔎 Stock Filtering

The dashboard supports filtering stocks based on different conditions.

📋 Final AI Dataset

The final output is stored in:

dataset/final_ai_decisions.csv

This file contains the consolidated results from the technical, market-activity, liquidity, and machine-learning engines.
 
### 🔄 Complete StockAI Workflow

StockAI follows an end-to-end pipeline from live market data collection to interactive stock analysis:

```text
Live Market Data
        ↓
Live Tick Collection
        ↓
Tick Processing
        ↓
1-Minute Candle Generation
        ↓
OHLC Calculation
        ↓
SMMA20 Calculation
        ↓
SMMA120 Calculation
        ↓
SMMA Gap Calculation
        ↓
SMMA Crossover Detection
        ↓
BUY / SELL / NONE Signal
        ↓
BULLISH / BEARISH Trend
        ↓
LTQ Analysis
        ↓
LTQ Spike Detection
        ↓
ETQ 5-Minute Analysis
        ↓
ETQ 20-Minute Analysis
        ↓
ETQ 60-Minute Analysis
        ↓
Bid Quantity Analysis
        ↓
Ask Quantity Analysis
        ↓
Bid / Ask Imbalance
        ↓
Price Return Features
        ↓
Volume Features
        ↓
ML Feature Engineering
        ↓
Machine Learning Prediction
        ↓
ML Probability
        ↓
Liquidity Qualification
        ↓
Participation Score
        ↓
AI Opportunity Score
        ↓
AI Decision Engine
        ↓
ACCEPT_CANDIDATE / WATCH / AVOID
        ↓
Decision Reason
        ↓
Final AI Dataset
        ↓
Stock Ranking
        ↓
Stock Filtering
        ↓
Stock Selection
        ↓
Individual Stock Analysis
        ↓
Interactive Streamlit Dashboard
```

### 📊 Final Output

The pipeline ultimately converts raw market data into an interactive stock intelligence dashboard:

```text
Live Market Data
       ↓
Technical Analysis
       +
Market Activity
       +
Market Depth
       +
Machine Learning
       +
Liquidity Analysis
       ↓
AI Decision
       ↓
Stock Ranking & Filtering
       ↓
Individual Stock Analysis
       ↓
Streamlit Dashboard
```

### 🎯 AI Decision Categories

| Decision | Description |
|---|---|
| 🟢 **ACCEPT_CANDIDATE** | Stock meets the configured stronger screening conditions |
| 🟡 **WATCH** | Stock shows potentially interesting characteristics but needs confirmation |
| 🔴 **AVOID** | Stock does not currently satisfy the screening conditions |
### 🧪 Historical Data
StockAI also contains historical datasets used for analysis, feature engineering, model development, and strategy evaluation.
Examples:
historical_20days.csv
historical_candles.csv
historical_liquid_20days.csv
historical_trade_outcomes.csv
history_with_indicators.csv

These datasets provide historical market information for research and model development.

### 🤖 ML Training Data
Training-related datasets include:
training_data.csv
training_dataset.csv
training_dataset_20d.csv
ml_training_features.csv
ltq_training_dataset.csv

These datasets are used to develop and evaluate machine-learning components.

### 📈 Trade Analysis
The project contains datasets for analysing historical trading outcomes.

### 🖥️ Streamlit Dashboard
StockAI provides an interactive Streamlit dashboard.
The dashboard can display:
Market Overview
Top AI Opportunities
Current SMMA Crossovers
WATCH Candidates
AI Candidates
Liquidity-qualified Stocks
Stock Details
Technical Indicators
Market Depth
Volume Analysis
ML Probability
AI Decision
Decision Reason
The dashboard allows users to filter and inspect stocks interactively.

## 🔍 Interactive Stock Analysis
A user can select a stock from the dashboard.
For example:
NSE:ATALREAL-EQ
The dashboard can display:
Stock Name
Current Price
Signal
Trend
ML Probability
AI Decision
Opportunity Score
SMMA20
SMMA120
SMMA Gap
LTQ
LTQ Spike Ratio
ETQ
Bid Quantity
Ask Quantity
Volume
Volume Ratio
Returns
This provides a complete view of the selected stock
### 🛠️ Technology Stack
  
  ### Programming Language
  
  Python
  
  ### Data Processing
  
  Pandas
  NumPy
  
  ### Machine Learning
  
  Scikit-learn
  Joblib
  
  ### Visualization
  
  Streamlit
  Plotly
  
  ### Data Storage
  
  CSV
  
  ### Development
  
  VS Code
  PowerShell
  
  ### Version Control
  
  Git
  GitHub
  
  ### Deployment
  
  Streamlit Community Cloud
