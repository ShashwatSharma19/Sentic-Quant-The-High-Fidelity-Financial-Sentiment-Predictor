# 📈 AI Trading Terminal: Transformer vs PPO Reinforcement Agent

A professional-grade, autonomous algorithmic trading pipeline that combines **Deep Learning**, **Reinforcement Learning**, and **Sentiment Analysis** with an institutional-grade validation/backtesting suite.

---

## ⚡ Key Features

*   **Intelligence:** Dual-model architecture featuring a Multi-Head Attention **Transformer** and a **PPO Reinforcement Learning** agent.
*   **NLP Driven:** Real-time news scraping with **FinBERT** sentiment scoring.
*   **Optimization:** **Optuna** hyperparameter tuning for maximum neural network efficiency.
*   **Backtesting:** Professional-grade vectorized backtesting via **VectorBT** (calculating Sharpe, Max Drawdown, etc.).
*   **Architecture:** Built with **Polars** (high-speed data manipulation) and **Gymnasium** (trading environments).
*   **Dashboard:** Comprehensive 5-tab **Streamlit** dashboard for performance visualization and risk audit.

---

## 🛠️ Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone <your-repo-url>
   cd "Algorithimic trading with NLP and Reinforcement learning"
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Guide

### 1. Run the Full Pipeline
Fetches data, runs NLP sentiment, engineers features, trains models, and backtests:
```bash
.venv\Scripts\python.exe run.py --step all
```

### 2. Launch the Dashboard
View the 5-tab quant terminal:
```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

### 3. Switch Stocks/Assets
Open `src/config.py` and change the `ticker`:
```python
ticker: str = "TSLA"  # Works for Stocks, Forex, Crypto, and Indexes
```

---

## 📊 Dashboard Tabs

1.  **Executive Summary:** Top-line ROI, Sharpe, and Monte Carlo robustness simulations.
2.  **Strategy Deep Dive:** $10K equity growth + Underwater drawdown charts.
3.  **Feature Intelligence:** Indicator overlays (RSI, MACD, VIX) + High-conviction News Sentiment.
4.  **Model Internals:** Transformer confidence distributions + RL action breakdowns.
5.  **Data Explorer:** Interactive raw data auditing.

---

## 🏗️ Project Structure

*   `src/data_ingestion.py`: Yahoo Finance + News RSS Scraper.
*   `src/sentiment_analysis.py`: FinBERT score generation.
*   `src/feature_engineering.py`: Indicator math (using Polars).
*   `src/model.py`: Transformer architecture + Optuna tuning.
*   `src/rl_agent.py`: PPO Training logic.
*   `src/backtest.py`: VectorBT simulation layer.
*   `app.py`: Streamlit frontend.

---

## ⚠️ Disclaimer
*This project is for educational purposes only. Past performance does not guarantee future results. Never trade with capital you cannot afford to lose.*
