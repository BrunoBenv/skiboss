# skiboss/backend/indicators.py

import pandas as pd
import numpy as np
import pandas_ta as ta 
from typing import Dict, Any, List

# --- ASUMIMOS QUE ESTA FUNCIÓN ESTÁ EN .orderflow ---
from .orderflow import get_advanced_features 

# --- FUNCIÓN DE VECTORIZACIÓN (SIMPLIFICADA) ---
def get_normalized_features_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Función que selecciona, normaliza y devuelve el vector final de 12 features.
    
    NOTA: En un proyecto real se implementaría la normalización (ej. Z-Score).
    """
    
    # Lista de features necesarias para el vector final de 12 elementos.
    # El modelo DRL usa estas 12 para la toma de decisiones.
    feature_columns = ['close', 'rsi', 'atr', 'dist_to_vwap', 'ema_20_dist', 'EMA_50', 
                       'smc_score', 'fvg_strength', 'poc_dist', 'vah_dist', 'val_dist', 'volume']
    
    # Rellenar cualquier valor NaN que quede después de los cálculos
    df = df.fillna(method='ffill').fillna(0) 

    if len(df) < 1:
        return np.zeros(12, dtype=np.float32)
        
    # Extraer los últimos valores (la última vela)
    final_features = df[feature_columns].iloc[-1].values
    
    return final_features.astype(np.float32)


# --------------------------------------------------------------------------
# Función Principal de Orquestación
# --------------------------------------------------------------------------

def calculate_all_features(df: pd.DataFrame) -> np.ndarray:
    """
    Procesa el DataFrame de datos brutos y devuelve el vector de features para la IA.
    """
    
    if df.empty or len(df) < 50:
        # Devolvemos un vector de ceros si no hay suficientes datos para los indicadores
        return np.zeros(12, dtype=np.float32) 

    df = df.copy() # Trabajamos en una copia

    # --- FASE 1: Indicadores Clásicos y de Volumen con Pandas-TA ---
    
    # Añadir RSI, EMA, ATR
    df.ta.rsi(append=True, length=14)
    df.ta.ema(append=True, length=20, col_names=('EMA_20',))
    df.ta.ema(append=True, length=50, col_names=('EMA_50',))
    df.ta.atr(append=True, length=14)
    
    # AÑADIR VWAP (Volume Weighted Average Price) - Indicador clave de volumen
    df.ta.vwap(append=True) 
    
    # Renombrar columnas para consistencia
    df = df.rename(columns={'RSI_14': 'rsi', 'ATR_14': 'atr', 'VWAP': 'vwap'})


    # Cálculo de distancia (Normalización por ATR)
    df['ema_20_dist'] = (df['close'] - df['EMA_20']) / df['atr']
    
    # NUEVA FEATURE CLAVE: Distancia de Precio a VWAP (normalizada por ATR)
    df['dist_to_vwap'] = (df['close'] - df['vwap']) / df['atr']
    
    # --- FASE 2: Indicadores SMC/Order Flow ---
    # Asumimos que get_advanced_features añade 'poc_dist', 'vah_dist', 'val_dist', etc.
    df = get_advanced_features(df)
    
    # --- FASE 3: Normalización Final (Para la NN) ---
    
    return get_normalized_features_vector(df)
