# backend/main.py - DRL TRADER ULTIMATE (FULL UNIVERSE + STEALTH + TEST SIGNAL)
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

# --- 📧 TUS DATOS (EDITAR AQUÍ) ---
EMAIL_SENDER = "TU_EMAIL@gmail.com"          # <--- PONER TU GMAIL
EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"       # <--- PONER TU CLAVE DE APLICACIÓN
EMAIL_RECEIVER = EMAIL_SENDER

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_universe_v1.pth")
METRICS_PATH = os.path.join(BASE_DIR, "smart_metrics.json")
HISTORY_PATH = "/tmp/history_log.json" if os.path.exists("/tmp") else os.path.join(BASE_DIR, "history_log.json")

model = None
smart_metrics = {}
cached_results = []

# --- 🌎 TU UNIVERSO COMPLETO (175 ACTIVOS DEL CSV) ---
SECTOR_MAP = {
    # ETFs USA
    "SPY": "ETF", "QQQ": "ETF", "DIA": "ETF", "IWM": "ETF", "VTI": "ETF",
    "VOO": "ETF", "IVV": "ETF", "XLK": "ETF", "XLF": "ETF", "XLV": "ETF",
    "XLE": "ETF", "XLI": "ETF", "XLP": "ETF", "XLY": "ETF", "XLU": "ETF",
    "XLRE": "ETF", "XLB": "ETF", "ARKK": "ETF", "SMH": "ETF", "SOXX": "ETF",
    "IBB": "ETF", "XBI": "ETF", "HACK": "ETF", "KWEB": "ETF", "EEM": "ETF",
    "EWZ": "ETF", "RSX": "ETF", "TLT": "ETF", "IEF": "ETF", "SHY": "BondETF",
    "HYG": "BondETF", "LQD": "BondETF", "BITO": "ETF", "XBTF": "ETF",
    
    # COMMODITIES
    "GLD": "CommodityETF", "SLV": "CommodityETF", "USO": "CommodityETF", "UNG": "CommodityETF",
    
    # CRYPTO
    "BTC-USD": "Crypto", "ETH-USD": "Crypto", 
    
    # FUTURES
    "ES=F": "Future", "NQ=F": "Future", "YM=F": "Future", "RTY=F": "Future", 
    "CL=F": "Future", "GC=F": "Future", "SI=F": "Future", "NG=F": "Future", 
    "ZB=F": "Future", "ZN=F": "Future", "ZF=F": "Future", "6E=F": "Future", "6B=F": "Future",
    
    # INDICES
    "DX-Y.NYB": "Index", "^GSPC": "Index", "^NDX": "Index", "^DJI": "Index", 
    "^RUT": "Index", "^VIX": "Index", "^GDAXI": "Index",
    
    # STOCKS (USA & ADRs)
    "AAPL": "Stock", "MSFT": "Stock", "GOOGL": "Stock", "AMZN": "Stock", "META": "Stock",
    "NVDA": "Stock", "TSLA": "Stock", "NFLX": "Stock", "AMD": "Stock", "INTC": "Stock",
    "IBM": "Stock", "CRM": "Stock", "ORCL": "Stock", "ADBE": "Stock", "PYPL": "Stock",
    "SQ": "Stock", "SHOP": "Stock", "AVGO": "Stock", "TSM": "Stock", "ASML": "Stock",
    "BABA": "Stock", "GE": "Stock", "BA": "Stock", "CAT": "Stock", "DE": "Stock",
    "MMM": "Stock", "NKE": "Stock", "SBUX": "Stock", "KO": "Stock", "PEP": "Stock",
    "MCD": "Stock", "WMT": "Stock", "COST": "Stock", "PG": "Stock", "JNJ": "Stock",
    "PFE": "Stock", "UNH": "Stock", "ABBV": "Stock", "LLY": "Stock", "XOM": "Stock",
    "CVX": "Stock", "COP": "Stock", "BP": "Stock", "TTE": "Stock", "JPM": "Stock",
    "GS": "Stock", "MS": "Stock", "BAC": "Stock", "WFC": "Stock", "C": "Stock",
    "BLK": "Stock", "V": "Stock", "MA": "Stock", "AXP": "Stock", "BRK-B": "Stock",
    "DIS": "Stock", "CMCSA": "Stock", "T": "Stock", "VZ": "Stock", "TMUS": "Stock",
    "LOW": "Stock", "HD": "Stock", "UPS": "Stock", "FDX": "Stock", "TSCO": "Stock",
    "PLTR": "Stock", "SNOW": "Stock", "NET": "Stock", "CRWD": "Stock", "ZS": "Stock",
    "OKTA": "Stock", "PANW": "Stock",

    # ARGENTINA (ADRS & LOCALES)
    "CEPU.BA": "ArgStock", "YPFD.BA": "ArgStock", "GGAL.BA": "ArgStock", "BMA.BA": "ArgStock",
    "PAMP.BA": "ArgStock", "TGSU2.BA": "ArgStock", "TRAN.BA": "ArgStock", "EDN.BA": "ArgStock",
    "MIRG.BA": "ArgStock", "CRES.BA": "ArgStock", "ALUA.BA": "ArgStock", "TXAR.BA": "ArgStock",
    "VALE.BA": "ArgStock", "LOMA.BA": "ArgStock", "TGNO4.BA": "ArgStock", "SUPV.BA": "ArgStock",
    "BYMA.BA": "ArgStock", "COME.BA": "ArgStock", "IRSA.BA": "ArgStock", "AGRO.BA": "ArgStock",
    "CADO.BA": "ArgStock", "SEMI.BA": "ArgStock", "MORI.BA": "ArgStock", "LEDE.BA": "ArgStock",
    "BBAR.BA": "ArgStock", "CTIO.BA": "ArgStock", "DGCU2.BA": "ArgStock",
    
    # BONOS ARGENTINOS
    "AL30": "ArgBond", "AL29": "ArgBond", "AL35": "ArgBond", "AL41": "ArgBond",
    "GD29": "ArgBond", "GD30": "ArgBond", "GD35": "ArgBond", "GD38": "ArgBond",
    "GD41": "ArgBond", "GD46": "ArgBond", "BONCER": "ArgBond", "T2X4": "ArgBond",
    "TV24": "ArgBond", "DICP": "ArgBond", "PARA": "ArgBond", "PARP": "ArgBond"
}

# --- FUNCIONES ---

def send_email_alert(data):
    if "xxxx" in EMAIL_PASSWORD: return 
    try:
        subject = f"🚀 DRL SIGNAL: {data['ticker']} ({data['signal']})"
        body = f"""
        ALERTA DRL TRADER (PURE AI)
        ---------------------------
        ACTIVO:   {data['ticker']} ({data['category']})
        SEÑAL:    {data['signal']}
        PRECIO:   ${data['price']}
        OBJETIVO: ${data['tp']}
        STOP:     ${data['sl']}
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

def get_time_estimate(price, tp, atr):
    if atr == 0: return "Indefinido"
    distance = abs(tp - price)
    daily_move = atr * 0.8
    days = int(distance / daily_move)
    if days < 1: return "Intradía"
    return f"{days}-{days+2} Días"

def analyze_ticker(ticker):
    try:
        # Descarga con reintentos para evitar bloqueos
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        
        # Validación de datos vacíos
        if df.empty or len(df) < 50: return None

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Features Engineering
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['vol_20'] = df['log_ret'].rolling(20).std()
        df['vol_60'] = df['log_ret'].rolling(60).std()
        df['vol_regime'] = (df['vol_20'] / df['vol_60']).fillna(1.0)
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean()
        df['dist_sma50'] = (df['close'] - sma_50) / sma_50
        df['dist_sma200'] = (df['close'] - sma_200) / sma_200
        df['vol_rel'] = (df['volume'] / df['volume'].rolling(20).mean()).fillna(1.0)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi'] - 50) / 50 
        
        # PREDICCIÓN PURE AI (Sin Híbrido)
        obs = df.iloc[-1][['log_ret', 'vol_regime', 'dist_sma50', 'dist_sma200', 'vol_rel', 'rsi_norm']].values.astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        
        price = df.iloc[-1]['close']
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        
        signal = "NEUTRAL"
        if action == 1: signal = "LONG"
        elif action == 2: signal = "SHORT"
        
        # Calculo TP/SL
        sl = price * 0.95; tp = price * 1.05
        if "LONG" in signal: sl=price-(2*atr); tp=price+(3*atr)
        elif "SHORT" in signal: sl=price+(2*atr); tp=price-(3*atr)
        
        stats = smart_metrics.get(ticker, {"win_rate": 0, "edge": 0})
        
        result = {
            "ticker": ticker,
            "category": SECTOR_MAP.get(ticker, "General"),
            "signal": signal,
            "price": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "est_time": get_time_estimate(price, tp, atr) if signal != "NEUTRAL" else "Monitoreando...",
            "win_rate": stats['win_rate'],
            "edge": stats['edge']
        }
        
        if signal != "NEUTRAL":
            save_and_notify(result)
            
        return result
    except: return None

def save_and_notify(signal_data):
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            try: history = json.load(f)
            except: pass
    
    today = datetime.now().strftime("%Y-%m-%d")
    exists = any(x['ticker'] == signal_data['ticker'] and x['date'] == today for x in history)
    
    if not exists:
        signal_data['date'] = today
        signal_data['status'] = "VIGENTE"
        history.insert(0, signal_data)
        with open(HISTORY_PATH, 'w') as f: json.dump(history[:100], f, indent=4)
        send_email_alert(signal_data)

# --- TAREA DE FONDO (ESCÁNER STEALTH) ---
async def background_scanner():
    global cached_results
    watchlist = list(SECTOR_MAP.keys())
    
    # --- 1. SEÑAL DE PRUEBA (VISUALIZACIÓN INMEDIATA) ---
    print("🧪 GENERANDO SEÑAL DE PRUEBA DE CONEXIÓN...")
    test_signal = {
        "ticker": "TEST-SYSTEM",
        "category": "System Check",
        "signal": "LONG",
        "price": 1000.00,
        "sl": 950.00, "tp": 1100.00,
        "est_time": "Online OK",
        "win_rate": 100, "edge": 100
    }
    cached_results = [test_signal] 
    
    print(f"🤖 ESCÁNER INICIADO: {len(watchlist)} ACTIVOS")
    
    while True:
        try:
            print(f"📡 Escaneando mercado (Modo Sigiloso)... {datetime.now().strftime('%H:%M')}")
            temp_results = []
            
            for t in watchlist:
                res = analyze_ticker(t)
                if res:
                    temp_results.append(res)
                
                # --- PAUSA ANTI-BLOQUEO (CRUCIAL PARA 175 ACTIVOS) ---
                await asyncio.sleep(3) 
            
            # Actualizamos la lista real solo al terminar la vuelta completa
            if len(temp_results) > 0:
                cached_results = temp_results
                print(f"✅ Vuelta completada. {len(temp_results)} activos procesados.")
            
        except Exception as e:
            print(f"⚠️ Error loop: {e}")
            
        await asyncio.sleep(600) # Espera 10 min entre vueltas

@app.on_event("startup")
async def startup_event():
    global model, smart_metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f: smart_metrics = json.load(f)
    try:
        env = gym.make("CartPole-v1")
        env.observation_space = spaces.Box(low=-5, high=5, shape=(6,), dtype=np.float32)
        env.action_space = spaces.Discrete(3)
        model = PPO("MlpPolicy", env, device="cpu")
        if os.path.exists(MODEL_PATH):
            model.policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            print("🧠 Cerebro IA Cargado.")
    except Exception as e: print(f"❌ Error carga: {e}")
    asyncio.create_task(background_scanner())

@app.get("/radar")
def get_radar(): return cached_results

@app.get("/history")
def get_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r') as f: return json.load(f)
        except: return []
    return []

@app.get("/")
def home(): return {"status": "Online"}
