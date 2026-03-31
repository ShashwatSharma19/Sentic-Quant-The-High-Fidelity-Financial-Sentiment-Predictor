"""
backtest.py
-----------
VectorBT professional backtester for both Transformer and RL Agents.
"""
import pandas as pd
import numpy as np
import torch
import os
import json
import vectorbt as vbt
from stable_baselines3 import PPO

from src.model import TimeSeriesTransformer
from src.config import *

def run_backtest():
    print("Loading test data...")
    if not os.path.exists(DATA_PROCESSED_TRAIN) or not os.path.exists(DATA_PROCESSED_RAW_FEATURES):
        raise FileNotFoundError("Missing data files.")
        
    df_features = pd.read_csv(DATA_PROCESSED_TRAIN)
    df_raw = pd.read_csv(DATA_PROCESSED_RAW_FEATURES)
    
    # Extract Dates & Prices
    dates = pd.to_datetime(df_raw['Date'].values)
    close_prices = df_raw['Close'].values
    
    # We want features without Date/Target
    cols_to_drop = ['Date', 'Target'] if 'Date' in df_features.columns else ['Target']
    feat_cols = [c for c in df_features.columns if c not in cols_to_drop]
    feature_matrix = df_features[feat_cols].values
    
    train_size = int(len(feature_matrix) * TRAIN_SPLIT_RATIO)
    
    # ----------------------------------------------------
    # 1. Transformers Validation Set (Starts at train_size + SEQ_LEN)
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = feature_matrix.shape[1]
    
    # Load optimal Optuna dimensions
    d_model, nhead, num_layers, dropout = D_MODEL, N_HEAD, NUM_LAYERS, DROPOUT
    if os.path.exists("models/best_params.json"):
        with open("models/best_params.json", "r") as f:
            bp = json.load(f)
        d_model = bp.get("d_model", d_model)
        nhead = bp.get("nhead", nhead)
        num_layers = bp.get("num_layers", num_layers)
        dropout = bp.get("dropout", dropout)
    
    transformer_model = TimeSeriesTransformer(
        input_dim=input_dim, d_model=d_model, nhead=nhead, 
        num_layers=num_layers, dropout=dropout
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        transformer_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    transformer_model.eval()
    
    transformer_signals = []
    transformer_probs = []
    # Transformers needs a rolling window sequence
    for i in range(train_size, len(feature_matrix) - SEQ_LEN):
        seq = feature_matrix[i : i + SEQ_LEN]
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = transformer_model(X).item()
        transformer_probs.append(prob)
        transformer_signals.append(1 if prob > 0.5 else -1)
        
    transformer_signals = np.array(transformer_signals)
    transformer_probs = np.array(transformer_probs)
    
    # Align timelines: Transformers skip the first SEQ_LEN items
    val_start_idx = train_size + SEQ_LEN
    val_dates = dates[val_start_idx:]
    val_prices = close_prices[val_start_idx:]
    
    # ----------------------------------------------------
    # 2. RL Agent Validation Set
    # ----------------------------------------------------
    rl_signals = []
    rl_path = "models/ppo_trading_agent.zip"
    if os.path.exists(rl_path):
        rl_model = PPO.load(rl_path)
        position = 0.0
        # RL evaluates purely on the single current macro state
        for i in range(val_start_idx, len(feature_matrix)):
            obs = np.append(feature_matrix[i], position).astype(np.float32)
            action, _ = rl_model.predict(obs, deterministic=True)
            position = float(action) - 1.0 # map to -1, 0, 1
            rl_signals.append(position)
    else:
        rl_signals = np.zeros(len(val_dates))
        
    rl_signals = np.array(rl_signals)
    
    # Align lengths strictly
    min_len = min(len(val_prices), len(transformer_signals), len(rl_signals))
    val_prices = val_prices[:min_len]
    val_dates = val_dates[:min_len]
    transformer_signals = transformer_signals[:min_len]
    rl_signals = rl_signals[:min_len]
    
    # ----------------------------------------------------
    # 3. VectorBT Implementation & Reporting
    # ----------------------------------------------------
    price_series = pd.Series(val_prices, index=val_dates)
    
    # Convert signals [-1, 0, 1] into Entry/Exit bool arrays for VectorBT
    tf_entries = transformer_signals == 1
    tf_exits = transformer_signals == -1
    
    rl_entries = rl_signals == 1
    rl_exits = rl_signals == -1
    
    print("\n[Backtesting with professional VectorBT engine]")
    
    # 0.1% strict trading fees modeling real markets
    # Notice how vectorbt calculates EVERYTHING natively
    pf_tf = vbt.Portfolio.from_signals(price_series, entries=tf_entries, exits=tf_exits, fees=0.001, freq='d')
    pf_rl = vbt.Portfolio.from_signals(price_series, entries=rl_entries, exits=rl_exits, fees=0.001, freq='d')
    
    print("\n=== TRANSFORMER BACKTEST ===")
    print(f"Total Return: {pf_tf.total_return():.2%}")
    print(f"Sharpe Ratio: {pf_tf.sharpe_ratio():.2f}")
    print(f"Max Drawdown: {pf_tf.max_drawdown():.2%}")
    
    print("\n=== REINFORCEMENT LEARNING BACKTEST ===")
    print(f"Total Return: {pf_rl.total_return():.2%}")
    print(f"Sharpe Ratio: {pf_rl.sharpe_ratio():.2f}")
    print(f"Max Drawdown: {pf_rl.max_drawdown():.2%}")
    
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame({
        "Transformer_Return": pf_tf.returns(),
        "RL_Return": pf_rl.returns(),
        "Benchmark_Return": price_series.pct_change()
    }).to_csv("data/processed/backtest_results_vbt.csv")
    
    # Save prediction details for frontend diagnostics
    pd.DataFrame({
        "Date": val_dates[:min_len],
        "Transformer_Prob": transformer_probs[:min_len],
        "Transformer_Signal": transformer_signals[:min_len],
        "RL_Signal": rl_signals[:min_len]
    }).to_csv("data/processed/prediction_details.csv", index=False)
    
    print("\nBacktest curves saved successfully to 'backtest_results_vbt.csv'.")
    print("Prediction details saved to 'prediction_details.csv'.")

if __name__ == "__main__":
    run_backtest()
