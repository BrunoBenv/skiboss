# skiboss/backend/orderflow.py

import pandas as pd
import numpy as np

# Constantes para Proxies
RVOL_PERIOD = 20 
ATR_FACTOR = 0.5 

def get_advanced_features_and_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de Order Flow Proxy, Microestructura y SMC.
    
    Retorna el DataFrame con las nuevas columnas añadidas.
    """
    # 1. Delta Proxy (CVD Reconstruido)
    df['up_volume'] = np.where(df['close'] >= df['open'], df['volume'], 0)
    df['down_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['delta_proxy'] = df['up_volume'] - df['down_volume']
    
    # 2. Volumen Relativo (RVOL)
    df['volume_avg_20'] = df['volume'].rolling(window=RVOL_PERIOD).mean()
    df['rvol'] = df['volume'] / df['volume_avg_20']
    df['rvol'] = df['rvol'].fillna(1.0) 

    # 3. Imbalance Proxy (Agresividad de la Vela)
    df['range'] = df['high'] - df['low']
    df['body_size'] = np.abs(df['close'] - df['open'])
    df['imbalance_proxy'] = df['body_size'] / df['range']
    df['imbalance_proxy'] = df['imbalance_proxy'].fillna(0)

    # 4. Liquidez Sweep Proxy (Mecha Larga + Volumen Alto)
    upper_wick = df['high'] - np.maximum(df['close'], df['open'])
    lower_wick = np.minimum(df['close'], df['open']) - df['low']
    
    df['liquidity_sweep_flag'] = np.where(
        ((upper_wick > ATR_FACTOR * df['atr']) | (lower_wick > ATR_FACTOR * df['atr'])) & 
        (df['rvol'] > 1.2), 
        1, # Señal de barrida de liquidez (institucional proxy)
        0
    )

    # 5. SMC Score / Market Profile Proxy (Basado en Clústeres de Precios)
    # Estos proxies se usarán como distancia en el vector de features
    
    # La implementación real de POC/VAL/VAH por clústeres OHLCV es compleja.
    # Usaremos una aproximación basada en la volatilidad.
    df['poc_dist'] = df['close'].rolling(window=20).mean() / df['atr'] # Proxy del POC
    df['vah_dist'] = df['poc_dist'] + 0.5 # Proxy del VAH
    df['val_dist'] = df['poc_dist'] - 0.5 # Proxy del VAL
    
    # SMC Score: Puntuación de estructura (BOS/CHOCH)
    df['smc_score'] = np.random.uniform(-1, 1, size=len(df)) # Placeholder
    
    # 6. FVG Strength (Proxy de ineficiencia)
    df['fvg_strength'] = (df['high'].shift(1) - df['low'].shift(2)).fillna(0) / df['atr']
    
    return df.drop(columns=['up_volume', 'down_volume', 'range', 'body_size', 'volume_avg_20'])
