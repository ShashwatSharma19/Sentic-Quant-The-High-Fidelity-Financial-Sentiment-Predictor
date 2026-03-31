"""
config.py
---------
Centralized configuration management for the Algorithmic Trading Pipeline.
Uses Pydantic BaseSettings to allow easy overrides via Environment Variables.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Pipeline Settings
    ticker: str = "AAPL"
    start_date: str = "2020-01-01"
    
    # Model Hyperparameters
    sequence_length: int = 30
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 20
    batch_size: int = 32
    
    # Backtest Settings
    train_split: float = 0.8

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

# File Paths
DATA_PIPELINE_OUTPUT = "data/processed/pipeline_output.parquet"
DATA_PROCESSED_RAW_FEATURES = "data/processed/raw_features.csv"
DATA_PROCESSED_TRAIN = "data/processed/train_scaled.csv"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/model.pth"

# Backward compatibility mapped to Pydantic Settings
TRAIN_SPLIT_RATIO = settings.train_split
SEQ_LEN = settings.sequence_length
BATCH_SIZE = settings.batch_size
D_MODEL = settings.d_model
N_HEAD = settings.nhead
NUM_LAYERS = settings.num_encoder_layers
DROPOUT = settings.dropout
LEARNING_RATE = settings.learning_rate
EPOCHS = settings.epochs
