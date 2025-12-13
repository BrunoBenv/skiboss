# skiboss/backend/orderflow.py

import pandas as pd
import numpy as np
from typing import Dict, Any

# --------------------------------------------------------------------------
# NOTA: Estas funciones son simplificaciones basadas en lógica de velas (OHLCV)
# y no en datos de tick o nivel 2, que son necesarios para un Order Flow "puro".
# Son adecuadas para entrenar un modelo DRL en timeframes de 1h o mayores.
# --------------------------------------------------------------------------

def detect_fair_value_gaps(df: pd.DataFrame) -> pd.Series:
    """
    Detecta Fair Value Gaps (FVG) o Imbalances.
    Un FVG es un área de precio ineficiente. Se requiere un lookback de 3 velas.
    
    Retorna una Serie: 1.0 (FVG Alcista/Bullish), -1.0 (FVG Bajista/Bearish), 0.0 (Ninguno).
    """
    fvg = pd.Series(0.0, index=df.index)
    
    # Asegurarse de tener suficientes datos para el lookback (3 velas)
    if len(df) < 3:
        return fvg
        
    # Recorre desde la tercera vela
    for i in range(2, len(df)):
        # FVG Alcista (Bullish Imbalance): La sombra superior de la vela 3 (i) no toca la sombra inferior de la vela 1 (i-2)
        # Es decir: High[i-2] < Low[i]
        if df['high'].iloc[i-2] < df['low'].iloc[i]:
            # El FVG se forma en la vela central (i-1)
            fvg.iloc[i-1] = 1.0
        
        # FVG Bajista (Bearish Imbalance): La sombra inferior de la vela 3 (i) no toca la sombra superior de la vela 1 (i-2)
        # Es decir: Low[i-2] > High[i]
        elif df['low'].iloc[i-2] > df['high'].iloc[i]:
            fvg.iloc[i-1] = -1.0
            
    # Hacemos un shift para alinear el FVG con el momento en que está activo
    return fvg.shift(-1).fillna(0)

def detect_order_blocks(df: pd.DataFrame, lookback: int = 4) -> pd.Series:
    """
    Detecta Order Blocks (OB) simplificados: última vela bajista/alcista antes de un impulso fuerte.
    
    Retorna una Serie: 1.0 (OB Alcista), -1.0 (OB Bajista), 0.0 (Ninguno).
    """
    ob = pd.Series(0.0, index=df.index)
    
    # Determinamos si la vela es bajista (roja) o alcista (verde)
    is_bearish = df['close'] < df['open']
    
    for i in range(lookback, len(df)):
        # 1. OB Alcista (Bullish OB):
        # Es la última vela bajista (is_bearish) antes de 'lookback' velas consecutivas alcistas (impulso)
        is_bullish_impulse = all(df['close'].iloc[i - lookback + 1:i+1] > df['open'].iloc[i - lookback + 1:i+1])
        if is_bearish.iloc[i - lookback] and is_bullish_impulse:
            ob.iloc[i - lookback] = 1.0 # El OB es la vela bajista previa al impulso
            
        # 2. OB Bajista (Bearish OB):
        # Es la última vela alcista (NOT is_bearish) antes de 'lookback' velas consecutivas bajistas (impulso)
        is_bearish_impulse = all(df['close'].iloc[i - lookback + 1:i+1] < df['open'].iloc[i - lookback + 1:i+1])
        if not is_bearish.iloc[i - lookback] and is_bearish_impulse:
            ob.iloc[i - lookback] = -1.0 # El OB es la vela alcista previa al impulso
            
    return ob

def calc_market_structure(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Detecta BOS (Break of Structure) y CHoCH (Change of Character) simplificados.
    Identifica si el precio actual rompe un máximo (High) o un mínimo (Low) significativo (de los últimos 'window' períodos).
    
    Retorna Serie: 1.0 (Estructura Alcista/BOS), -1.0 (Estructura Bajista/BOS), 0.0 (Consolidación).
    """
    bos = pd.Series(0.0, index=df.index)
    
    # Máximo y Mínimo relevante (Watermarks)
    high_watermark = df['high'].rolling(window=window).max().shift(1)
    low_watermark = df['low'].rolling(window=window).min().shift(1)

    # Alcista (BOS/CHoCH): Cierre rompe el máximo anterior
    bos[(df['close'] > high_watermark)] = 1.0
    
    # Bajista (BOS/CHoCH): Cierre rompe el mínimo anterior
    bos[(df['close'] < low_watermark)] = -1.0
    
    return bos

def calculate_volume_profile(df: pd.DataFrame, last_bars: int = 100) -> Dict[str, float]:
    """
    Calcula el POC, VAH y VAL simplificados para el perfil de volumen de las últimas N barras.
    (Basado en OHLCV, no en datos por tick).
    
    Retorna un diccionario con los niveles de precio clave.
    """
    
    recent_df = df.iloc[-last_bars:]
    
    if recent_df.empty:
        # Devuelve un precio de referencia si no hay datos
        current_price = df['close'].iloc[-1] if not df.empty else 0.0
        return {"poc": current_price, "vah": current_price, "val": current_price}
        
    # POC (Point of Control): Precio con mayor volumen TOTAL en el rango
    # (Usamos el precio de cierre de la barra con mayor volumen)
    poc_price = recent_df.loc[recent_df['volume'].idxmax()]['close']
    
    # VAH y VAL (Value Area High/Low): Simplificado al máximo y mínimo de las últimas N barras.
    # Nota: El cálculo real implica acumular el 68% del volumen en torno al POC.
    vah = recent_df['high'].max() 
    val = recent_df['low'].min()
    
    return {
        "poc": poc_price,
        "vah": vah,
        "val": val
    }
    
def get_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas las features avanzadas de SMC y las combina en el DataFrame.
    """
    df = df.copy() # Trabajamos sobre una copia
    
    # 1. Indicadores SMC de la vela
    df['fvg'] = detect_fair_value_gaps(df)
    df['order_block'] = detect_order_blocks(df)
    df['market_structure'] = calc_market_structure(df)
    
    # 2. Volume Profile (Se calcula una vez para el rango, y se asigna al último punto)
    # Importante: Solo necesitamos los niveles de VAH/VAL/POC para el punto de predicción.
    vp_data = calculate_volume_profile(df)
    
    # 3. Creación de features de distancia (para la IA)
    # Estas features indican si el precio actual está cerca de un nivel clave
    df['dist_to_poc'] = df['close'] - vp_data['poc']
    df['dist_to_vah'] = df['close'] - vp_data['vah']
    df['dist_to_val'] = df['close'] - vp_data['val']
    
    return df