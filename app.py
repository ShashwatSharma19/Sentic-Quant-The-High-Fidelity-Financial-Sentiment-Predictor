"""
app.py — Professional Quant Trading Terminal
=============================================
5-tab Streamlit dashboard surfacing every insight from the AI trading pipeline.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AI Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design Tokens
TEMPLATE = "plotly_dark"
C_TF     = "#FF9F1C"    # Orange  — Transformer
C_RL     = "#2EC4B6"    # Cyan    — RL Agent
C_BENCH  = "#6C757D"    # Gray    — Benchmark
C_SENT   = "#9B5DE5"    # Purple  — Sentiment
C_POS    = "#06D6A0"    # Green   — Positive
C_NEG    = "#EF476F"    # Red     — Negative

MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_returns():
    path = "data/processed/backtest_results_vbt.csv"
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index('Date').drop(df.columns[0], axis=1)
    df_cum = (1 + df.fillna(0)).cumprod()
    return df, df_cum

def load_features():
    path = "data/processed/raw_features.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)

def load_sentiment():
    path = "data/processed/daily_sentiment.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

def load_predictions():
    path = "data/processed/prediction_details.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_optuna_params():
    path = "models/best_params.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quant Metric Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def sharpe(r):
    return np.sqrt(252) * r.mean() / (r.std() + 1e-9)

def sortino(r):
    down = r[r < 0].std()
    return np.sqrt(252) * r.mean() / (down + 1e-9)

def calmar(cum, r):
    ann = (cum.iloc[-1] ** (252 / max(len(r), 1))) - 1
    mdd = max_dd(cum)
    return ann / (abs(mdd) + 1e-9)

def max_dd(cum):
    return (cum / cum.cummax() - 1).min()

def dd_series(cum):
    return cum / cum.cummax() - 1

def win_rate(r):
    trades = r[r != 0]
    return (trades > 0).sum() / max(len(trades), 1)

def ann_return(cum, n):
    return (cum.iloc[-1] ** (252 / max(n, 1))) - 1

def metrics_dict(daily, cum):
    n = len(daily)
    return {
        "Total ROI":          f"{(cum.iloc[-1] - 1) * 100:.2f}%",
        "Annualized Return":  f"{ann_return(cum, n) * 100:.2f}%",
        "Sharpe Ratio":       f"{sharpe(daily):.2f}",
        "Sortino Ratio":      f"{sortino(daily):.2f}",
        "Calmar Ratio":       f"{calmar(cum, daily):.2f}",
        "Max Drawdown":       f"{max_dd(cum) * 100:.2f}%",
        "Win Rate":           f"{win_rate(daily) * 100:.1f}%",
        "Trading Days":       str(n),
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_sidebar(df_feat):
    with st.sidebar:
        st.markdown("## 📈 AI Trading Terminal")
        st.divider()

        # ── Project Info ──
        st.subheader("📋 Project Info")
        st.caption("**Ticker:** AAPL (Apple Inc.)")
        if df_feat is not None:
            d0 = df_feat['Date'].min().strftime('%Y-%m-%d')
            d1 = df_feat['Date'].max().strftime('%Y-%m-%d')
            st.caption(f"**Range:** {d0} → {d1}")
            st.caption(f"**Days:** {len(df_feat)}")
        st.divider()

        # ── Chart Controls ──
        st.subheader("🎛️ Chart Controls")
        rolling_w = st.slider("Sentiment Smoothing (days)", 5, 90, 30, 5)
        sharpe_w  = st.slider("Rolling Sharpe Window",     20, 120, 60, 10)
        mc_paths  = st.slider("Monte Carlo Paths",         50, 500, 100, 50)
        capital   = st.number_input("Starting Capital ($)", 1000, 1_000_000, 10_000, 1000)
        st.divider()

        # ── Strategy Toggles ──
        st.subheader("📊 Visible Strategies")
        show_tf    = st.checkbox("Transformer",           value=True)
        show_rl    = st.checkbox("RL Agent (PPO)",        value=True)
        show_bench = st.checkbox("Benchmark (Buy & Hold)",value=True)
        st.divider()

        st.caption("Built with PyTorch · Stable‑Baselines3 · VectorBT · FinBERT · Optuna · Polars")

    return dict(rolling_w=rolling_w, sharpe_w=sharpe_w, mc_paths=mc_paths,
                capital=capital, show_tf=show_tf, show_rl=show_rl, show_bench=show_bench)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Executive Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_executive(df_daily, df_cum, ctrl):
    st.header("📊 Executive Tear Sheet")

    # ── Metric Scorecards ──
    m_tf    = metrics_dict(df_daily['Transformer_Return'], df_cum['Transformer_Return'])
    m_rl    = metrics_dict(df_daily['RL_Return'],          df_cum['RL_Return'])
    m_bench = metrics_dict(df_daily['Benchmark_Return'],   df_cum['Benchmark_Return'])

    c1, c2, c3 = st.columns(3)
    def _card(col, title, color, m, delta_key='Total ROI', bench_val=None):
        with col:
            st.markdown(f"### <span style='color:{color}'>{title}</span>", unsafe_allow_html=True)
            roi_val = float(m['Total ROI'].replace('%',''))
            delta_str = None
            if bench_val is not None:
                delta_str = f"{roi_val - bench_val:.2f}% vs Bench"
            st.metric("Total ROI", m['Total ROI'], delta_str)
            for k in ['Annualized Return','Sharpe Ratio','Sortino Ratio',
                       'Calmar Ratio','Max Drawdown','Win Rate','Trading Days']:
                st.caption(f"**{k}:** {m[k]}")

    bench_roi = float(m_bench['Total ROI'].replace('%',''))
    _card(c1, "Deep Learning Transformer", C_TF, m_tf,    bench_val=bench_roi)
    _card(c2, "PPO RL Agent",              C_RL, m_rl,    bench_val=bench_roi)
    _card(c3, "Buy & Hold Benchmark",      C_BENCH, m_bench)

    st.divider()

    # ── Verdict ──
    tf_roi = float(m_tf['Total ROI'].replace('%',''))
    rl_roi = float(m_rl['Total ROI'].replace('%',''))
    best_name, best_roi = ("Transformer", tf_roi) if tf_roi >= rl_roi else ("RL Agent", rl_roi)
    if best_roi > bench_roi:
        st.success(f"✅  **{best_name}** outperformed Buy & Hold by **{best_roi - bench_roi:.2f}%**")
    elif best_roi == bench_roi:
        st.info(f"⚖️  Both AI strategies matched the Benchmark at **{bench_roi:.2f}%** — models converged on a long‑only baseline (expected with limited training data)")
    else:
        st.warning(f"⚠️  Buy & Hold outperformed by **{bench_roi - best_roi:.2f}%** — consider expanding the training dataset")

    st.divider()

    # ── Monte Carlo Simulation ──
    st.subheader("🎲 Monte Carlo Simulation — Strategy Robustness")
    st.caption("Bootstrapped from actual daily returns. If the fan is tight, the result is statistically robust.")

    # Pick the best-performing strategy for simulation
    strat_col = 'Transformer_Return' if tf_roi >= rl_roi else 'RL_Return'
    actual_rets = df_daily[strat_col].dropna().values
    n_days = len(actual_rets)
    cap = ctrl['capital']

    fig_mc = go.Figure()

    # Simulated paths
    terminal_values = []
    np.random.seed(42)
    for i in range(ctrl['mc_paths']):
        sampled = np.random.choice(actual_rets, size=n_days, replace=True)
        path = cap * (1 + sampled).cumprod()
        terminal_values.append(path[-1])
        fig_mc.add_trace(go.Scatter(
            y=path, mode='lines', line=dict(color=C_TF if strat_col=='Transformer_Return' else C_RL, width=0.4),
            opacity=0.15, showlegend=False, hoverinfo='skip'
        ))

    # Actual path
    actual_path = cap * (1 + actual_rets).cumprod()
    fig_mc.add_trace(go.Scatter(
        y=actual_path, mode='lines', name='Actual Path',
        line=dict(color='white', width=3)
    ))

    fig_mc.update_layout(
        template=TEMPLATE, height=420,
        xaxis_title="Trading Days", yaxis_title=f"Portfolio Value ($)",
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Monte Carlo stats
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Median Terminal Value", f"${np.median(terminal_values):,.0f}")
    mc2.metric("5th Percentile (Worst Case)", f"${np.percentile(terminal_values, 5):,.0f}")
    mc3.metric("95th Percentile (Best Case)", f"${np.percentile(terminal_values, 95):,.0f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Strategy Deep Dive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_deep_dive(df_daily, df_cum, ctrl):
    st.header("🔬 Strategy Deep Dive")

    cap = ctrl['capital']

    # ── Equity Curves ($10K growth) ──
    st.subheader(f"Portfolio Growth (${cap:,} initial)")
    fig = go.Figure()
    if ctrl['show_tf']:
        fig.add_trace(go.Scatter(
            x=df_cum.index, y=df_cum['Transformer_Return'] * cap,
            name="Transformer", line=dict(color=C_TF, width=2)))
    if ctrl['show_rl']:
        fig.add_trace(go.Scatter(
            x=df_cum.index, y=df_cum['RL_Return'] * cap,
            name="RL Agent (PPO)", line=dict(color=C_RL, width=2)))
    if ctrl['show_bench']:
        fig.add_trace(go.Scatter(
            x=df_cum.index, y=df_cum['Benchmark_Return'] * cap,
            name="Benchmark", line=dict(color=C_BENCH, width=2, dash='dash')))
    fig.update_layout(template=TEMPLATE, hovermode='x unified', height=400,
                      yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Underwater / Drawdown Chart ──
    st.subheader("📉 Underwater Chart (Drawdown Over Time)")
    st.caption("Shows how deep strategy losses went and how long recovery took.")
    fig_dd = go.Figure()
    if ctrl['show_tf']:
        dd_tf = dd_series(df_cum['Transformer_Return'])
        fig_dd.add_trace(go.Scatter(
            x=dd_tf.index, y=dd_tf * 100, fill='tozeroy',
            name="Transformer", line=dict(color=C_TF, width=1)))
    if ctrl['show_rl']:
        dd_rl = dd_series(df_cum['RL_Return'])
        fig_dd.add_trace(go.Scatter(
            x=dd_rl.index, y=dd_rl * 100, fill='tozeroy',
            name="RL Agent", line=dict(color=C_RL, width=1)))
    if ctrl['show_bench']:
        dd_b = dd_series(df_cum['Benchmark_Return'])
        fig_dd.add_trace(go.Scatter(
            x=dd_b.index, y=dd_b * 100, fill='tozeroy',
            name="Benchmark", line=dict(color=C_BENCH, width=1)))
    fig_dd.update_layout(template=TEMPLATE, hovermode='x unified', height=350,
                         yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Monthly Returns Heatmap ──
    st.subheader("📅 Monthly Returns Heatmap")
    strat_choice = st.selectbox("Strategy", ["Transformer", "RL Agent", "Benchmark"],
                                 key="heatmap_strat")
    col_map = {"Transformer": "Transformer_Return",
               "RL Agent": "RL_Return",
               "Benchmark": "Benchmark_Return"}
    monthly = df_daily[col_map[strat_choice]].copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly_agg = monthly.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    pivot = pd.DataFrame({
        'Year':   monthly_agg.index.year,
        'Month':  monthly_agg.index.month,
        'Return': monthly_agg.values * 100
    }).pivot(index='Year', columns='Month', values='Return')

    fig_hm = px.imshow(
        pivot.values,
        x=[MONTH_LABELS[i-1] for i in pivot.columns],
        y=[str(y) for y in pivot.index],
        color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
        aspect='auto', text_auto='.1f',
        labels=dict(color="Return %")
    )
    fig_hm.update_layout(template=TEMPLATE, height=300)
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Rolling Sharpe Ratio ──
    st.subheader(f"📈 Rolling Sharpe Ratio ({ctrl['sharpe_w']}-day window)")
    fig_rs = go.Figure()
    w = ctrl['sharpe_w']
    if ctrl['show_tf']:
        roll = df_daily['Transformer_Return'].rolling(w).apply(lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-9))
        fig_rs.add_trace(go.Scatter(x=roll.index, y=roll, name="Transformer", line=dict(color=C_TF, width=2)))
    if ctrl['show_rl']:
        roll = df_daily['RL_Return'].rolling(w).apply(lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-9))
        fig_rs.add_trace(go.Scatter(x=roll.index, y=roll, name="RL Agent", line=dict(color=C_RL, width=2)))
    if ctrl['show_bench']:
        roll = df_daily['Benchmark_Return'].rolling(w).apply(lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-9))
        fig_rs.add_trace(go.Scatter(x=roll.index, y=roll, name="Benchmark", line=dict(color=C_BENCH, width=2)))
    fig_rs.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
    fig_rs.update_layout(template=TEMPLATE, hovermode='x unified', height=350,
                         yaxis_title="Sharpe Ratio")
    st.plotly_chart(fig_rs, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Feature & Sentiment Intelligence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_features(df_feat, df_sent, ctrl):
    st.header("🧠 Feature & Sentiment Intelligence")

    if df_feat is None:
        st.error("No feature data found. Run the pipeline first.")
        return

    # ── Price + Indicator Overlay ──
    st.subheader("Price & Technical Indicators")
    overlays = st.multiselect(
        "Overlay Indicators", ["SMA 20", "SMA 50", "RSI", "MACD"],
        default=["SMA 20", "SMA 50"]
    )

    has_rsi  = "RSI"  in overlays
    has_macd = "MACD" in overlays
    rows = 1 + int(has_rsi) + int(has_macd)
    heights = [0.5] + ([0.25] * (rows - 1)) if rows > 1 else [1]
    subtitles = ["Close Price"]
    if has_rsi:  subtitles.append("RSI")
    if has_macd: subtitles.append("MACD")

    fig_ind = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                             row_heights=heights, subplot_titles=subtitles,
                             vertical_spacing=0.06)

    fig_ind.add_trace(go.Scatter(x=df_feat['Date'], y=df_feat['Close'],
                                  name='Close', line=dict(color='white', width=1.5)), row=1, col=1)
    if "SMA 20" in overlays and 'SMA_20' in df_feat.columns:
        fig_ind.add_trace(go.Scatter(x=df_feat['Date'], y=df_feat['SMA_20'],
                                      name='SMA 20', line=dict(color=C_TF, width=1)), row=1, col=1)
    if "SMA 50" in overlays and 'SMA_50' in df_feat.columns:
        fig_ind.add_trace(go.Scatter(x=df_feat['Date'], y=df_feat['SMA_50'],
                                      name='SMA 50', line=dict(color=C_RL, width=1)), row=1, col=1)

    r = 2
    if has_rsi and 'RSI' in df_feat.columns:
        fig_ind.add_trace(go.Scatter(x=df_feat['Date'], y=df_feat['RSI'],
                                      name='RSI', line=dict(color=C_SENT, width=1.5)), row=r, col=1)
        fig_ind.add_hline(y=70, line_dash='dot', line_color=C_NEG, opacity=0.5, row=r, col=1)
        fig_ind.add_hline(y=30, line_dash='dot', line_color=C_POS, opacity=0.5, row=r, col=1)
        r += 1
    if has_macd and 'MACD' in df_feat.columns:
        fig_ind.add_trace(go.Scatter(x=df_feat['Date'], y=df_feat['MACD'],
                                      name='MACD', line=dict(color=C_TF, width=1.5)), row=r, col=1)
        if 'MACD_Signal' in df_feat.columns:
            fig_ind.add_trace(go.Scatter(x=df_feat['Date'], y=df_feat['MACD_Signal'],
                                          name='Signal', line=dict(color=C_NEG, width=1)), row=r, col=1)

    fig_ind.update_layout(template=TEMPLATE, height=250 + 150 * rows, hovermode='x unified')
    st.plotly_chart(fig_ind, use_container_width=True)

    # ── Sentiment vs Price ──
    if df_sent is not None:
        st.subheader("Sentiment vs Price (Dual Axis)")
        st.caption(f"FinBERT sentiment smoothed over {ctrl['rolling_w']} days vs AAPL Close price.")

        sent_smooth = df_sent.copy()
        sent_smooth['Smoothed'] = sent_smooth['daily_sentiment_score'].rolling(ctrl['rolling_w']).mean()

        fig_sp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sp.add_trace(go.Scatter(x=df_feat['Date'], y=df_feat['Close'],
                                     name='AAPL Close', line=dict(color='white', width=1.5)),
                         secondary_y=False)
        fig_sp.add_trace(go.Scatter(x=sent_smooth['Date'], y=sent_smooth['Smoothed'],
                                     name='Sentiment', fill='tozeroy',
                                     line=dict(color=C_SENT, width=2)),
                         secondary_y=True)
        fig_sp.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig_sp.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1])
        fig_sp.update_layout(template=TEMPLATE, height=400, hovermode='x unified')
        st.plotly_chart(fig_sp, use_container_width=True)

    # ── VIX Fear Context ──
    if 'VIX_Close' in df_feat.columns:
        st.subheader("VIX Fear Index Context")
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(x=df_feat['Date'], y=df_feat['VIX_Close'],
                                      name='VIX', line=dict(color=C_NEG, width=2)))
        # Shade danger zone
        fig_vix.add_hrect(y0=30, y1=df_feat['VIX_Close'].max() + 5,
                           fillcolor=C_NEG, opacity=0.1, line_width=0,
                           annotation_text="HIGH FEAR", annotation_position="top left")
        fig_vix.add_hline(y=20, line_dash='dash', line_color='white', opacity=0.3,
                           annotation_text="Historical Average")
        fig_vix.update_layout(template=TEMPLATE, height=350, yaxis_title="VIX Level",
                               hovermode='x unified')
        st.plotly_chart(fig_vix, use_container_width=True)

    # ── Cross-Correlation Heatmap ──
    st.subheader("🔗 Feature Cross-Correlation Matrix")
    st.caption("If two features are >0.8 correlated, one is redundant. This proves you understand multicollinearity.")

    corr_cols = [c for c in ['RSI','MACD','MACD_Signal','Volume_Z','VIX_Close',
                              'daily_sentiment_score','Log_Return',
                              'SMA_20','SMA_50','Volume_Change'] if c in df_feat.columns]
    if len(corr_cols) >= 2:
        corr_matrix = df_feat[corr_cols].corr()
        fig_corr = px.imshow(
            corr_matrix.values,
            x=corr_cols, y=corr_cols,
            color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
            text_auto='.2f', aspect='equal',
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(template=TEMPLATE, height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough features to compute correlation matrix.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Model Internals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_internals(df_preds, optuna_params):
    st.header("⚙️ Model Internals")

    # ── Optuna Results ──
    st.subheader("🎯 Optuna Hyperparameter Optimizer Results")
    if optuna_params is not None:
        param_descriptions = {
            "d_model":    "Transformer embedding dimension — how much info each token carries",
            "nhead":      "Number of attention heads — parallel attention mechanisms",
            "num_layers": "Encoder depth — number of stacked transformer layers",
            "dropout":    "Regularization — fraction of neurons disabled during training",
            "lr":         "Learning rate — step size for gradient descent",
            "batch_size": "Batch size — samples processed per gradient update",
            "epochs":     "Total training epochs"
        }
        rows_data = []
        for k, v in optuna_params.items():
            desc = param_descriptions.get(k, "")
            display_v = f"{v:.6f}" if isinstance(v, float) else str(v)
            rows_data.append({"Parameter": k, "Optimal Value": display_v, "What It Controls": desc})
        st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)
    else:
        st.info("No Optuna results found. Run the training pipeline first.")

    st.divider()

    if df_preds is None:
        st.warning("⚠️ No prediction details found. Re-run the backtest: `.venv\\Scripts\\python.exe run.py --step backtest`")
        return

    # ── Prediction Confidence Distribution ──
    st.subheader("📊 Transformer Prediction Confidence Distribution")
    st.caption("If the histogram clusters near 0.5, the model is guessing. Spread toward 0 and 1 means strong conviction.")

    probs = df_preds['Transformer_Prob'].dropna()
    buy_probs  = probs[probs > 0.5]
    sell_probs = probs[probs <= 0.5]

    fig_conf = go.Figure()
    fig_conf.add_trace(go.Histogram(
        x=buy_probs, nbinsx=40, name='BUY signals',
        marker_color=C_POS, opacity=0.7
    ))
    fig_conf.add_trace(go.Histogram(
        x=sell_probs, nbinsx=40, name='SELL signals',
        marker_color=C_NEG, opacity=0.7
    ))
    fig_conf.add_vline(x=0.5, line_dash='dash', line_color='white',
                        annotation_text="Decision Boundary (0.5)")
    fig_conf.update_layout(
        template=TEMPLATE, barmode='overlay', height=400,
        xaxis_title="Model Confidence (Sigmoid Output)",
        yaxis_title="Number of Trading Days"
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # Confidence stats
    s1, s2, s3 = st.columns(3)
    s1.metric("Mean Confidence (BUY days)", f"{buy_probs.mean():.3f}" if len(buy_probs) > 0 else "N/A")
    s2.metric("Mean Confidence (SELL days)", f"{sell_probs.mean():.3f}" if len(sell_probs) > 0 else "N/A")
    high_conv = ((probs > 0.7) | (probs < 0.3)).sum()
    s3.metric("High-Conviction Days (>70%)", f"{high_conv} / {len(probs)} ({high_conv/max(len(probs),1)*100:.1f}%)")

    st.divider()

    # ── RL Agent Action Distribution ──
    st.subheader("🤖 RL Agent Action Distribution")
    st.caption("Shows what percentage of time the PPO agent chose each action. 'Always Long' = converged on buy-and-hold baseline.")

    rl_signals = df_preds['RL_Signal']
    action_counts = {
        'Short (-1)': (rl_signals == -1).sum(),
        'Hold (0)':   (rl_signals == 0).sum(),
        'Long (+1)':  (rl_signals == 1).sum()
    }
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(action_counts.keys()),
        values=list(action_counts.values()),
        marker_colors=[C_NEG, C_BENCH, C_POS],
        textinfo='label+percent', hole=0.4
    )])
    fig_pie.update_layout(template=TEMPLATE, height=380)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Signal Agreement Timeline ──
    st.subheader("🤝 Transformer vs RL — Signal Agreement")
    st.caption("Green = both agree. Red = they disagree. Agreement rate reveals model consensus.")

    agree = (df_preds['Transformer_Signal'] == df_preds['RL_Signal']).astype(int)
    agreement_pct = agree.mean() * 100

    fig_agree = go.Figure()
    colors = [C_POS if a == 1 else C_NEG for a in agree]
    fig_agree.add_trace(go.Bar(
        x=df_preds['Date'], y=agree.replace({1: 1, 0: -1}),
        marker_color=colors, showlegend=False
    ))
    fig_agree.update_layout(
        template=TEMPLATE, height=250,
        yaxis=dict(tickvals=[-1, 1], ticktext=["Disagree", "Agree"]),
        title=f"Overall Agreement: {agreement_pct:.1f}%"
    )
    st.plotly_chart(fig_agree, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Raw Data Explorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_explorer():
    st.header("🗃️ Raw Data Explorer")

    datasets = {
        "Backtest Returns (VBT)": "data/processed/backtest_results_vbt.csv",
        "Raw Features (Unscaled)": "data/processed/raw_features.csv",
        "Scaled Training Data": "data/processed/train_scaled.csv",
        "Sentiment Scores": "data/processed/daily_sentiment.csv",
        "Prediction Details": "data/processed/prediction_details.csv",
    }

    choice = st.selectbox("Select Dataset", list(datasets.keys()))
    path = datasets[choice]

    if not os.path.exists(path):
        st.warning(f"File not found: `{path}`. Run the pipeline to generate it.")
        return

    df = pd.read_csv(path)

    # Summary stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", str(len(df.columns)))
    file_size = os.path.getsize(path)
    c3.metric("File Size", f"{file_size / 1024:.1f} KB")

    st.dataframe(df, use_container_width=True, height=500)

    # Download
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_data,
        file_name=os.path.basename(path),
        mime="text/csv"
    )

    # Quick stats for numeric columns
    with st.expander("📋 Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.title("📈 AI Trading Terminal — Transformer vs Reinforcement Learning")

    # Load all data
    df_daily, df_cum = load_returns()
    df_feat  = load_features()
    df_sent  = load_sentiment()
    df_preds = load_predictions()
    optuna_p = load_optuna_params()

    # Sidebar
    ctrl = render_sidebar(df_feat)

    if df_daily is None or df_cum is None:
        st.error("No backtest data found. Run the pipeline first:")
        st.code(".venv\\Scripts\\python.exe run.py --step all", language="bash")
        return

    # Tabs
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Executive Summary",
        "🔬 Strategy Deep Dive",
        "🧠 Feature Intelligence",
        "⚙️ Model Internals",
        "🗃️ Data Explorer"
    ])

    with t1: tab_executive(df_daily, df_cum, ctrl)
    with t2: tab_deep_dive(df_daily, df_cum, ctrl)
    with t3: tab_features(df_feat, df_sent, ctrl)
    with t4: tab_internals(df_preds, optuna_p)
    with t5: tab_explorer()

if __name__ == "__main__":
    main()
