# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import json
import os
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RUTAS RELATIVAS (Para que funcione en tu PC y en Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/model_universe_v1.pth")
METRICS_PATH = os.path.join(BASE_DIR, "../saved_models/smart_metrics.json")

model = None
smart_metrics = {}

@app.on_event("startup")
async def load_system():
    global model, smart_metrics
    print("🚀 INICIANDO SKIBOSS DSS...")
    
    # 1. Cargar Métricas (La Auditoría)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            smart_metrics = json.load(f)
        print(f"✅ Auditoría cargada: {len(smart_metrics)} activos auditados.")
    else:
        print("⚠️ ALERTA: No se encontró smart_metrics.json (Se operará a ciegas).")

    # 2. Cargar Cerebro
    try:
        env = gym.make("CartPole-v1")
        env.observation_space = spaces.Box(low=-5, high=5, shape=(6,), dtype=np.float32)
        env.action_space = spaces.Discrete(3)
        
        model = PPO("MlpPolicy", env, device="cpu") # Force CPU para inferencia
        if os.path.exists(MODEL_PATH):
            model.policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            print("🧠 Cerebro Institucional Cargado.")
        else:
            print(f"❌ ERROR: No encuentro el cerebro en {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

def analyze_ticker(ticker):
    try:
        # Descarga rápida (últimos 6 meses para indicadores)
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if len(df) < 60: return None
        
        df.columns = [c.lower() for c in df.columns]
        
        # INGENIERÍA DE FEATURES (Idéntica al entrenamiento)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['vol_20'] = df['log_ret'].rolling(20).std()
        df['vol_60'] = df['log_ret'].rolling(60).std()
        df['vol_regime'] = (df['vol_20'] / df['vol_60']).fillna(1.0)
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean()
        df['dist_sma50'] = (df['close'] - sma_50) / sma_50
        df['dist_sma200'] = (df['close'] - sma_200) / sma_200
        df['vol_rel'] = (df['volume'] / df['volume'].rolling(20).mean()).fillna(1.0)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi'] - 50) / 50 
        
        features = ['log_ret', 'vol_regime', 'dist_sma50', 'dist_sma200', 'vol_rel', 'rsi_norm']
        obs = df.iloc[-1][features].values.astype(np.float32)
        
        # PREDICCIÓN
        action, _ = model.predict(obs, deterministic=True)
        
        # DATOS DE SALIDA
        price = df.iloc[-1]['close']
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        
        signal = "NEUTRAL"
        sl, tp = 0, 0
        
        if action == 1:
            signal = "LONG"
            sl, tp = price - (2*atr), price + (3*atr)
        elif action == 2:
            signal = "SHORT"
            sl, tp = price + (2*atr), price - (3*atr)
            
        # RECUPERAR AUDITORÍA
        stats = smart_metrics.get(ticker, {"win_rate": 0, "edge": 0})
        
        return {
            "ticker": ticker,
            "signal": signal,
            "price": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "win_rate": stats['win_rate'],
            "edge": stats['edge'],
            "confidence": "ALTA" if stats['edge'] > 5.0 else ("MEDIA" if stats['edge'] > 0 else "BAJA")
        }
    except:
        return None

@app.get("/radar")
def get_radar():
    # LISTA DE VIGILANCIA (Puedes editarla)
    watchlist = ["SPY", "QQQ", "IWM", "BTC-USD", "ETH-USD", "GGAL.BA", "YPF.BA", "AAPL", "NVDA", "GLD", "HYG"]
    results = []
    print("📡 Escaneando mercado...")
    for t in watchlist:
        res = analyze_ticker(t)
        if res and res['signal'] != "NEUTRAL":
            results.append(res)
    return results