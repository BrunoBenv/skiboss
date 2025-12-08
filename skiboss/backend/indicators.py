# skiboss/backend/indicators.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Importamos las funciones SMC que acabas de crear
try:
    from .orderflow import get_advanced_features
except ImportError:
    # Esto es una advertencia, pero permite que el archivo se defina
    print("ADVERTENCIA: No se pudo importar orderflow.get_advanced_features.")
    get_advanced_features = lambda df: df


# --------------------------------------------------------------------------
# Funciones de Indicadores Clásicos
# --------------------------------------------------------------------------

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula el Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calcula la Exponential Moving Average (EMA)."""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula el Average True Range (ATR) para medir la volatilidad."""
    # True Range (TR)
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)
    
    # ATR (EMA del TR)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr.fillna(0)

def get_normalized_features_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Toma el DataFrame final con todos los features y devuelve el vector 
    de entrada para la Red Neuronal (NN) en el punto de tiempo más reciente.
    
    Normaliza los valores para que estén entre -1 y 1 (o 0 y 1, según el tipo).
    """
    
    # 1. Seleccionar la última fila (el estado actual del mercado)
    current_state = df.iloc[-1]
    
    # 2. Definir las Features que Irán a la NN
    # La Red Neuronal solo necesita valores numéricos normalizados.
    feature_list = [
        # Clásicos (Normalizar)
        'rsi', 'ema_20_dist', 'atr',
        # SMC (Binarios/Normalizados)
        'fvg', 'order_block', 'market_structure',
        # Perfil de Volumen (Distancia Normalizada)
        'dist_to_poc_norm', 'dist_to_vah_norm', 'dist_to_val_norm',
        # Volumen (Normalizar)
        'volume_norm'
    ]
    
    # 3. Crear el Vector
    vector = current_state[feature_list].values.astype(np.float32)
    
    return vector

# --------------------------------------------------------------------------
# Función Principal de Orquestación
# --------------------------------------------------------------------------

def calculate_all_features(df: pd.DataFrame) -> np.ndarray:
    """
    Procesa el DataFrame de datos brutos y devuelve el vector de features para la IA.
    """
    
    if df.empty or len(df) < 50:
        raise ValueError("Datos insuficientes para el cálculo de indicadores (se requieren al menos 50 barras).")

    df = df.copy() # Trabajamos en una copia

    # --- FASE 1: Indicadores Clásicos ---
    df['rsi'] = calculate_rsi(df, period=14)
    df['ema_20'] = calculate_ema(df, period=20)
    df['atr'] = calculate_atr(df, period=14)

    # Cálculo de distancia (Normalización por ATR para hacerlo independiente de precios)
    df['ema_20_dist'] = (df['close'] - df['ema_20']) / df['atr']
    
    # --- FASE 2: Indicadores SMC/Order Flow ---
    df = get_advanced_features(df)
    
    # --- FASE 3: Normalización Final (Para la NN) ---
    
    # Normalizar Volumen (Z-score o Min-Max de las últimas 100 barras)
    # Usamos Min-Max simplificado:
    recent_volume = df['volume'].iloc[-100:].copy()
    min_vol = recent_volume.min()
    max_vol = recent_volume.max()
    df['volume_norm'] = (df['volume'] - min_vol) / (max_vol - min_vol) if (max_vol - min_vol) > 0 else 0.0

    # Normalizar distancias de Perfil de Volumen por el ATR actual para escalamiento
    latest_atr = df['atr'].iloc[-1]
    df['dist_to_poc_norm'] = df['dist_to_poc'] / latest_atr
    df['dist_to_vah_norm'] = df['dist_to_vah'] / latest_atr
    df['dist_to_val_norm'] = df['dist_to_val'] / latest_atr

    # --- FASE 4: Devolver el Vector ---
    return get_normalized_features_vector(df)