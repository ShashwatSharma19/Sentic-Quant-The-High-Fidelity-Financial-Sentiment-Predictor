"""
rl_agent.py
-----------
Trains a Proximal Policy Optimization (PPO) agent against the custom Trading Environment.
"""
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import DATA_PROCESSED_TRAIN, DATA_PROCESSED_RAW_FEATURES, TRAIN_SPLIT_RATIO
from src.env import TradingEnv

def train_rl_agent():
    print("Loading data for Reinforcement Learning training...")
    
    if not os.path.exists(DATA_PROCESSED_TRAIN) or not os.path.exists(DATA_PROCESSED_RAW_FEATURES):
        raise FileNotFoundError("Missing processed training data. Run feature engineering first.")
        
    df_features = pd.read_csv(DATA_PROCESSED_TRAIN)
    df_raw = pd.read_csv(DATA_PROCESSED_RAW_FEATURES)
    
    # Exclude non-numerical components
    cols_to_drop = ['Date', 'Target'] if 'Date' in df_features.columns else ['Target']
    feat_cols = [c for c in df_features.columns if c not in cols_to_drop]
    
    features = df_features[feat_cols].values
    prices = df_raw['Close'].values
    
    # Train Split boundary
    train_size = int(len(features) * TRAIN_SPLIT_RATIO)
    
    train_features = features[:train_size]
    train_prices = prices[:train_size]
    
    print(f"Dataset split for RL: {len(train_features)} days of trading experience.")
    
    # Spin up vectorized environment requested by Stable Baselines
    env_fn = lambda: TradingEnv(train_features, train_prices)
    vec_env = DummyVecEnv([env_fn])
    
    print("Building fresh PPO Agent...")
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, ent_coef=0.01)
    
    print("Training RL Bot...")
    model.learn(total_timesteps=30_000)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_trading_agent")
    print("RL Agent successfully trained and saved to 'models/ppo_trading_agent.zip'.")

if __name__ == "__main__":
    train_rl_agent()
