import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import math
from src.config import *

class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Features: All columns except 'Date' and 'Target'
        # Assuming last column is Target and 'Date' is dropped or handled before
        x = self.data[idx : idx + self.seq_len, :-1] # all cols except target
        y = self.data[idx + self.seq_len - 1, -1]   # target at the end of sequence
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        
        # Linear projection to map input features to d_model dimensions
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, 1) # Binary classification (probability)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        src = self.input_proj(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # Take the output of the last time step
        output = output[:, -1, :] 
        output = self.decoder(output)
        return self.sigmoid(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

import optuna
import json

def get_dataloaders(seq_len, batch_size):
    if not os.path.exists(DATA_PROCESSED_TRAIN):
        raise FileNotFoundError(f"{DATA_PROCESSED_TRAIN} not found")
        
    df = pd.read_csv(DATA_PROCESSED_TRAIN)
    
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
        
    data = df.values
    feature_dim = data.shape[1] - 1 # exclude target
    
    train_size = int(len(data) * TRAIN_SPLIT_RATIO)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = StockDataset(train_data, seq_len)
    val_dataset = StockDataset(val_data, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, feature_dim

def train_model(params=None, save_model=True):
    if params is None:
        params = {
            "d_model": D_MODEL,
            "nhead": N_HEAD,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        }
        
    train_loader, val_loader, feature_dim = get_dataloaders(SEQ_LEN, params["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TimeSeriesTransformer(
        input_dim=feature_dim, 
        d_model=params["d_model"], 
        nhead=params["nhead"], 
        num_layers=params["num_layers"], 
        dropout=params["dropout"]
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
    
    best_val_loss = float('inf')
    
    for epoch in range(params["epochs"]):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y = y.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y = y.unsqueeze(1)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if save_model:
            print(f"Epoch {epoch+1}/{params['epochs']} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_model:
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
                
    if save_model:
        print("Training complete. Best model saved.")
        
    return best_val_loss

def objective(trial):
    params = {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": EPOCHS, # use setting
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64])
    }
    return train_model(params=params, save_model=False)

def run_optuna_search(n_trials=10):
    print("Starting Optuna Hyperparameter Search...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Val Loss: {trial.value}")
    print("  Params: ")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
        
    print("\nTraining final model with best parameters...")
    best_params = trial.params
    best_params["epochs"] = EPOCHS # Ensure we run the full epochs
    
    os.makedirs("models", exist_ok=True)
    with open("models/best_params.json", "w") as f:
        json.dump(best_params, f)
        
    train_model(params=best_params, save_model=True)

if __name__ == "__main__":
    # If the user runs the pipeline directly, just run optuna search for 5 trials.
    run_optuna_search(n_trials=5)
