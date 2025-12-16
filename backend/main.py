# backend/main.py - DRL TRADER CLOUD V8 (AUTONOMOUS)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import json
import os
import smtplib
import asyncio
from email.mime.text import MIMEText
from datetime import datetime
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

# --- 📧 TUS DATOS (PONER AQUÍ) ---
EMAIL_SENDER = "labfitperformance@gmail.com"
EMAIL_PASSWORD = "Lamela55."  # Clave de Aplicación de 16 letras
EMAIL_RECEIVER = "labfitperformance@gmail.com"

# RUTAS RELATIVAS PARA LA NUBE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# En Render, a veces la estructura cambia, usamos rutas seguras
MODEL_PATH = os.path.join(BASE_DIR, "model_universe_v1.pth") 
METRICS_PATH = os.path.join(BASE_DIR, "smart_metrics.json")
HISTORY_PATH = "/tmp/history_log.json" # En la nube usamos /tmp (es volátil pero funciona)

model = None
smart_metrics = {}
cached_results = [] # Memoria RAM para guardar el último escaneo

SECTOR_MAP = {
    "SPY": "ETF Índices", "QQQ": "ETF Índices", "IWM": "ETF Índices",
    "AAPL": "Tecnología", "MSFT": "Tecnología", "NVDA": "Semiconductores",
    "GOOGL": "Tecnología", "TSLA": "Consumo", "GLD": "Commodities",
    "BTC-USD": "Cripto", "ETH-USD": "Cripto", "GGAL.BA": "Arg - Bancos",
    "YPF.BA": "Arg - Energía"
}

# --- EMAIL ---
def send_email_alert(data):
    try:
        subject = f"🚀 SEÑAL DRL: {data['ticker']} ({data['signal']})"
        body = f"""
        ALERTA 24/7 - DRL TRADER CLOUD
        ------------------------------
        ACTIVO:  {data['ticker']}
        SEÑAL:   {data['signal']}
        PRECIO:  ${data['price']}
        OBJETIVO: ${data['tp']}
        
        El bot está escaneando desde la nube.
        """
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"📧 Mail enviado: {data['ticker']}")
    except Exception as e:
        print(f"❌ Error mail: {e}")

# --- ANALISIS ---
def analyze_ticker(ticker):
    try:
        # En nube, descargamos menos data para ser más rápidos
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if len(df) < 60: return None
        df.columns = [c.lower() for c in df.columns]
        
        # Features express
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
        
        obs = df.iloc[-1][['log_ret', 'vol_regime', 'dist_sma50', 'dist_sma200', 'vol_rel', 'rsi_norm']].values.astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        
        price = df.iloc[-1]['close']
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        
        if action == 0: return None

        signal = "LONG" if action == 1 else "SHORT"
        sl = price - (2*atr) if action == 1 else price + (2*atr)
        tp = price + (3*atr) if action == 1 else price - (3*atr)
        
        stats = smart_metrics.get(ticker, {"win_rate": 0, "edge": 0})
        
        return {
            "ticker": ticker,
            "category": SECTOR_MAP.get(ticker, "General"),
            "signal": signal,
            "price": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "est_time": "1-3 Días",
            "win_rate": stats['win_rate'],
            "edge": stats['edge']
        }
    except: return None

# --- TAREA DE FONDO (EL CORAZÓN AUTÓNOMO) ---
async def background_scanner():
    global cached_results
    watchlist = ["SPY", "QQQ", "AAPL", "NVDA", "BTC-USD", "ETH-USD", "GGAL.BA", "YPF.BA", "MSFT", "TSLA"]
    
    # Historial en memoria RAM para no spamear mails
    sent_today = []

    print("🤖 INICIANDO ESCÁNER AUTÓNOMO 24/7...")
    
    while True:
        try:
            print(f"📡 Escaneando mercado... {datetime.now()}")
            new_results = []
            
            for t in watchlist:
                res = analyze_ticker(t)
                if res:
                    new_results.append(res)
                    # Lógica de Email
                    uid = f"{res['ticker']}_{datetime.now().strftime('%Y-%m-%d')}"
                    if uid not in sent_today:
                        send_email_alert(res)
                        sent_today.append(uid)
            
            cached_results = new_results # Actualizamos la memoria para el Dashboard
            
        except Exception as e:
            print(f"⚠️ Error en ciclo: {e}")
            
        # Esperar 5 minutos (300 segundos) para cuidar la IP de Render
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    global model, smart_metrics
    
    # 1. Cargar Archivos (Manejando rutas de Render)
    # En Render a veces hay que subir los modelos a la raíz
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f: smart_metrics = json.load(f)
    
    try:
        env = gym.make("CartPole-v1")
        env.observation_space = spaces.Box(low=-5, high=5, shape=(6,), dtype=np.float32)
        env.action_space = spaces.Discrete(3)
        model = PPO("MlpPolicy", env, device="cpu")
        if os.path.exists(MODEL_PATH):
            model.policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            print("🧠 Cerebro Cargado.")
        else:
            print(f"❌ NO ENCUENTRO EL MODELO EN: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error carga: {e}")

    # 2. Iniciar el Escáner en Segundo Plano
    asyncio.create_task(background_scanner())

@app.get("/radar")
def get_radar():
    # El Frontend ya no pide escanear, solo pide "qué tienes en memoria"
    return cached_results

@app.get("/")
def home():
    return {"status": "DRL Trader Running 24/7"}
