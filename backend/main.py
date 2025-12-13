# backend/main.py (REEMPLAZAR TODO)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np
import pandas as pd
import os
import time
from typing import List, Dict, Any

# Importaciones del CORE del Robot
from .rl_agent import DRLAgent  # Clase que encapsula el cerebro .pth
from .tradingview_api import fetch_historic_data # Para obtener data en tiempo real
from .database import get_active_trades, add_active_trade, close_trade_and_save_to_history, get_trading_history, get_trade_by_id
from .auth import get_current_user # Autenticación


# --- CONFIGURACIÓN DE LA APLICACIÓN Y DRL ---
app = FastAPI(title="SKIBOSS AI DRL API", version="1.0")

# CORS para permitir que el Frontend (Web App) acceda a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todo para desarrollo (cambiar a URL específica en prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización del Agente DRL (Cerebro)
model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'model_v1.pth')
dRL_agent = DRLAgent(model_path) 


# Modelos Pydantic para el Frontend
class TradeEntry(BaseModel):
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_type: str # LONG o SHORT
    timeframe: str 
    # Añadimos campos obligatorios del Prompt Maestro
    asset_type: str
    duration: str
    comment: str = ""

class TradeClose(BaseModel):
    trade_id: str
    close_price: float
    close_time: str
    close_comment: str
    result_dollars: float
    result_percent: float


# --- ENDPOINTS GENERALES ---

@app.get("/", summary="Estado de la API (Requiere Auth)")
async def root(user: str = Depends(get_current_user)):
    return {"status": "Online", "model_loaded": dRL_agent.is_loaded, "user": user}

# --- ENDPOINT 1: DASHBOARD DE RECOMENDACIONES (RADAR) ---

@app.get("/radar_signals", summary="Obtiene el radar de señales de alta confianza", response_model=Dict[str, Any])
async def get_radar_signals(timeframe: str = "1h", min_confidence: float = 0.85, 
                            user: str = Depends(get_current_user)):
    """ 
    Itera sobre todos los activos (160+), analiza el contexto DRL y filtra 
    solo las entradas con alta seguridad y R/R > 2.0.
    """
    
    # NOTA: La lógica de iteración masiva y análisis DRL queda aquí
    # Por razones de performance en la simulación, DEBEMOS SIMULAR LA SALIDA:
    
    # 🚨 SIMULACIÓN DE LA SALIDA DEL RADAR (REEMPLAZAR CON LÓGICA DRL REAL)
    if not dRL_agent.is_loaded:
        raise HTTPException(status_code=503, detail="Cerebro DRL no cargado. Esperando entrenamiento.")
        
    simulated_signals = [
        {"symbol": "BTC-USD", "asset_type": "Cripto futuros", "signal": "LONG", "entry_price": 60120.50, "stop_loss": 58000.00, "take_profit": 64360.50, "rr_ratio": 2.0, "expected_roi": 7.0, "confidence": 92.5, "duration": "Corto (horas-días)", "comment": "Order Flow muestra fuerte acumulación institucional en el VWAP diario."},
        {"symbol": "SPY", "asset_type": "ETF", "signal": "SHORT", "entry_price": 505.50, "stop_loss": 508.00, "take_profit": 498.00, "rr_ratio": 3.0, "expected_roi": 1.5, "confidence": 88.0, "duration": "Intradía", "comment": "Rechazo en zona de alto volumen de perfil semanal. Posible reversión."}
    ]
    # FIN SIMULACIÓN
    
    return {
        "status": "OK",
        "total_signals": len(simulated_signals),
        "signals": simulated_signals
    }

# --- ENDPOINT 2: GESTIÓN DE ENTRADAS ACTIVAS ---

@app.post("/active_trades/open", summary="Confirma una nueva operación", response_model=Dict[str, str])
async def open_trade(trade: TradeEntry, user: str = Depends(get_current_user)):
    """ Registra una nueva entrada confirmada por el usuario (Pasa del Dashboard a Activas). """
    trade_dict = trade.model_dump()
    trade_dict["open_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    trade_id = add_active_trade(trade_dict)
    return {"status": "Trade abierto y registrado", "id": trade_id}

@app.get("/active_trades", summary="Obtiene la lista de operaciones abiertas", response_model=List[Dict[str, Any]])
async def get_active_trades_list(user: str = Depends(get_current_user)):
    """ Obtiene la lista y reanaliza cada trade con el DRL. """
    trades = get_active_trades()
    
    # 🚨 Lógica de Reanálisis DRL (Sugiere Mantener/Ajustar/Cerrar)
    for trade in trades:
        # Aquí debería ir: dRL_agent.reanalyze_trade(trade.symbol, trade.entry_price)
        trade['drl_advice'] = {
            "action": "Mantener",
            "reason": f"Contexto sigue alineado con la señal original. Volumen bajo. (Último análisis: {time.ctime()})"
        }
    return trades

# --- ENDPOINT 3: HISTORIAL / DIARIO DE TRADING ---

@app.get("/history", summary="Obtiene el historial de operaciones cerradas", response_model=List[Dict[str, Any]])
async def get_history(user: str = Depends(get_current_user)):
    return get_trading_history()

@app.post("/history/close", summary="Cierra una operación activa")
async def close_trade(close_data: TradeClose, user: str = Depends(get_current_user)):
    """ Mueve un trade de Activas a Historial. """
    success = close_trade_and_save_to_history(close_data.trade_id, close_data.model_dump())
    if not success:
        raise HTTPException(status_code=404, detail="Trade ID no encontrado en operaciones activas.")
    return {"status": "Trade cerrado y movido al historial con éxito."}

# --- ENDPOINT 4: GRÁFICOS (DETALLE DEL ACTIVO) ---

@app.get("/asset_details/{symbol}", summary="Obtiene contexto para gráficos")
async def get_asset_context(symbol: str, user: str = Depends(get_current_user)):
    # Lógica para obtener Earnings, Noticias, etc., para el gráfico
    return {
        "symbol": symbol,
        "news": ["Próximos earnings el 15/01.", "Gran evento macro: decisión de tipos el viernes."],
        "current_price": 500.00 # Obtener precio actual (ya cubierto por yfinance)
    }

# --- ENDPOINTS AUXILIARES ---

@app.get("/asset_types")
async def get_asset_types():
    # Simulación de tipos de activos para los filtros del frontend
    return ["Acción USA", "ETF", "Índice", "Cripto futuros", "Bonos"]