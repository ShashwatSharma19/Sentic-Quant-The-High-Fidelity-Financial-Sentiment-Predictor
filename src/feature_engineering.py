import pandas as pd
import numpy as np
import os
import joblib
from datetime import timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from src.config import *

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features():
    print("Loading data...")
    if not os.path.exists(DATA_PIPELINE_OUTPUT):
        raise FileNotFoundError(f"Pipeline output not found at {DATA_PIPELINE_OUTPUT}. Run ingestion first.")
        
    df = pd.read_parquet(DATA_PIPELINE_OUTPUT)
    
    # Ensure date formats match (Parquet handles dates, but just to be sure for pandas)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # Fill missing sentiment just in case
    if 'daily_sentiment_score' not in df.columns:
        df['daily_sentiment_score'] = 0.0
    df['daily_sentiment_score'] = df['daily_sentiment_score'].fillna(0.0)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Feature Engineering
    print("Calculating technical indicators...")
    
    # 1. RSI
    df['RSI'] = compute_rsi(df['Close'])
    
    # 2. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 3. Log Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 4. Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # 5. MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 6. Volume Z-Score (vs 20-day mean)
    volume_mean = df['Volume'].rolling(window=20).mean()
    volume_std = df['Volume'].rolling(window=20).std()
    df['Volume_Z'] = (df['Volume'] - volume_mean) / volume_std
    
    # 7. VIX (Market Fear Index) Integration
    print("Fetching Macro VIX data...")
    # Add 1 day to end to ensure inclusive range from yfinance
    end_date_vix = df['Date'].max() + timedelta(days=1)
    vix_df = yf.Ticker("^VIX").history(start=df['Date'].min(), end=end_date_vix)
    if not vix_df.empty:
        # Convert index to simple date to match df['Date']
        vix_df.index = vix_df.index.tz_localize(None).date
        vix_df = vix_df[['Close']].rename(columns={'Close': 'VIX_Close'})
        df = df.merge(vix_df, left_on='Date', right_index=True, how='left')
        df['VIX_Close'] = df['VIX_Close'].ffill().bfill()
    else:
        df['VIX_Close'] = 20.0 # fallback mean if VIX fetching fails
    
    # 8. Target: Next Day Direction (1 if Next Close > Current Close, else 0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN values created by rolling windows
    df = df.dropna()
    
    # Select features for model
    feature_cols = [
        'Close', 'Volume', 'RSI', 'SMA_20', 'SMA_50', 
        'daily_sentiment_score', 'Log_Return', 'Volume_Change',
        'MACD', 'MACD_Signal', 'Volume_Z', 'VIX_Close'
    ]
    
    # Save processed dataframe with features (unscaled for inspection if needed)
    df.to_csv(DATA_PROCESSED_RAW_FEATURES, index=False)
    
    print("Normalizing features (Fit on Train only)...")
    
    # Split into Train/Test to fit scaler
    train_size = int(len(df) * TRAIN_SPLIT_RATIO)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    scaler = MinMaxScaler()
    
    # Fit ONLY on training data
    scaler.fit(train_df[feature_cols])
    
    # Transform both
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Combine back
    df_scaled = pd.concat([train_df, test_df])
    
    # Save scaler for inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    
    df_scaled.to_csv(DATA_PROCESSED_TRAIN, index=False)
    print(f"Feature engineering complete. Data saved to {DATA_PROCESSED_TRAIN}")
    print(f"Data shape: {df_scaled.shape}")
    
    return df_scaled

if __name__ == "__main__":
    create_features()
