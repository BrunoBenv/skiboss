import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any

# Mapeo de Timeframes a intervalos de yfinance
INTERVAL_MAP = {
    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '4h': '4h', '1d': '1d', '1wk': '1wk'
}

async def fetch_historic_data(symbol: str, tf: str) -> pd.DataFrame:
    """
    Obtiene datos OHLCV de Yahoo Finance.
    
    :param symbol: Símbolo del activo (ej: 'SPY').
    :param tf: Timeframe (ej: '1m').
    :return: DataFrame de pandas con datos OHLCV y volumen.
    """
    
    # Usar el timeframe solicitado o forzar a 1m como la granularidad más alta
    interval = INTERVAL_MAP.get(tf.lower()) or '1m'
    
    # Para 1m, yfinance solo permite un máximo de 7 días de data.
    if interval == '1m':
        period = '7d' 
    elif 'h' in interval or 'd' in interval:
        period = '60d' # 60 días para análisis de medio plazo
    else:
        period = '5y' # Para entrenamiento o análisis histórico

    try:
        # Descargar data
        data = yf.download(symbol, interval=interval, period=period, progress=False)
        
    except Exception as e:
        print(f"❌ Error al obtener datos de Yahoo Finance para {symbol}: {e}")
        return pd.DataFrame()
    
    if data.empty:
        return pd.DataFrame()
    
    # Normalizar nombres de columnas
    data.columns = [c.lower() for c in data.columns]
    
    # Limpieza final
    if 'adj close' in data.columns:
        data = data.drop(columns=['adj close'])
        
    return data.sort_index()
