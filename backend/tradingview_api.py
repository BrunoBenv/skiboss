import pandas as pd
import numpy as np
import yfinance as yf
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
    
    # Usar el timeframe solicitado o el más bajo (1m) como fallback
    interval = INTERVAL_MAP.get(tf.lower()) or '1m'
    
    # 1. Ajuste del período de descarga según la granularidad
    # Para 1m, yfinance solo permite un máximo de 7 días de data.
    if interval == '1m':
        period = '7d' 
    elif 'h' in interval or 'd' in interval:
        period = '60d' # 60 días para análisis de medio plazo
    else:
        period = '5y' # Para análisis histórico
        
    # 2. Ajuste de tickers de criptomonedas (yfinance usa el formato BTC-USD)
    if symbol in ['BTC', 'ETH']:
        yf_symbol = f"{symbol}-USD"
    else:
        yf_symbol = symbol

    try:
        # Descargar data
        data = yf.download(yf_symbol, interval=interval, period=period, progress=False)
        
    except Exception as e:
        print(f"❌ Error al obtener datos de Yahoo Finance para {yf_symbol}: {e}")
        return pd.DataFrame()
    
    if data.empty:
        return pd.DataFrame()
    
    # 3. Normalizar nombres de columnas a minúsculas
    data.columns = [c.lower() for c in data.columns]
    
    # Limpieza final
    if 'adj close' in data.columns:
        data = data.drop(columns=['adj close'])
        
    return data.sort_index()


# --- LLM Sentiment Score (Proxy para el Entrenamiento) ---
def get_llm_sentiment_score(df: pd.DataFrame) -> pd.Series:
    """
    SIMULACIÓN: Genera un score de sentimiento (proxy) para el entrenamiento.
    En producción, esta función llamaría a un modelo BERT/LLM de noticias gratuito
    (ej. usando APIs gratuitas de Finnhub o scraping de noticias + modelo de sentimiento).
    
    Retorna una serie de Pandas de scores de -1 a +1.
    """
    # Simula una ligera correlación con el momentum
    # Esto actúa como un proxy del factor fundamental que la IA debe aprender a usar
    sentiment = df['close'].diff().rolling(window=5).mean().fillna(0) * 5
    sentiment = np.clip(sentiment, -1.0, 1.0)
    return pd.Series(sentiment, index=df.index)
