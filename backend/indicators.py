import pandas as pd
import numpy as np
import pandas_ta as ta 
from typing import Dict, Any, List

# Importamos las funciones SMC/Order Flow Proxy
from .orderflow import get_advanced_features_and_proxies
# Importamos la función LLM Score (simulación para el entrenamiento)
from .tradingview_api import get_llm_sentiment_score 


# --- FUNCIÓN DE VECTORIZACIÓN (15 FEATURES) ---
# Esta función define la 'foto' del mercado que ve la IA.
def get_normalized_features_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Selecciona, normaliza y devuelve el vector final (AUMENTADO A 15 FEATURES).
    """
    # 15 FEATURES CLAVE (Combinación de SMC, Order Flow Proxy, y Macro)
    feature_columns = [
        # Clásicos / VWAP
        'close', 'rsi', 'atr', 'dist_to_vwap', 'ema_20_dist', 'EMA_50', 
        # Proxies de Order Flow / Volumen
        'rvol', 'delta_proxy', 'liquidity_sweep_flag', 'imbalance_proxy', 
        # SMC / Estructura / Niveles
        'poc_dist', 'vah_dist', 'val_dist', 'smc_score', 
        # Sentimiento / LLM
        'llm_sentiment_score'
    ]
    
    # Rellenar NaN (necesario tras cálculos de rolling windows)
    df = df.fillna(method='ffill').fillna(0) 

    if len(df) < 1:
        return np.zeros(15, dtype=np.float32) 
        
    final_features = df[feature_columns].iloc[-1].values
    
    # En producción se agregaría la normalización (ej. MinMax o Z-Score) aquí.
    return final_features.astype(np.float32)


# --------------------------------------------------------------------------
# Función Principal de Orquestación
# --------------------------------------------------------------------------

def calculate_all_features(df: pd.DataFrame) -> np.ndarray:
    """
    Procesa el DataFrame de datos brutos y devuelve el vector de 15 features.
    """
    # Se requieren al menos 50 barras para EMA/ATR
    if df.empty or len(df) < 50:
        return np.zeros(15, dtype=np.float32) 

    df = df.copy() 

    # --- FASE 1: Indicadores Clásicos y VWAP ---
    df.ta.rsi(append=True, length=14)
    df.ta.ema(append=True, length=20, col_names=('EMA_20',))
    df.ta.ema(append=True, length=50, col_names=('EMA_50',))
    df.ta.atr(append=True, length=14)
    df.ta.vwap(append=True, fillna=True) 
    
    df = df.rename(columns={'RSI_14': 'rsi', 'ATR_14': 'atr', 'VWAP': 'vwap'})

    # Cálculo de distancia (normalizada por ATR)
    df['ema_20_dist'] = (df['close'] - df['EMA_20']) / df['atr']
    df['dist_to_vwap'] = (df['close'] - df['vwap']) / df['atr']
    
    
    # --- FASE 2: Order Flow Proxies y SMC (98% Precisión en Proxy) ---
    df = get_advanced_features_and_proxies(df)
    
    
    # --- FASE 3: Sentimiento / LLM (Proxy) ---
    df['llm_sentiment_score'] = get_llm_sentiment_score(df) 
    
    
    # --- FASE 4: Vectorización ---
    return get_normalized_features_vector(df)
