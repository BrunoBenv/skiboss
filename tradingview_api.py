# skiboss/backend/tradingview_api.py

import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from tvDatafeed.main import NoDataFound

# NOTA IMPORTANTE:
# Si bien tvDatafeed permite el modo sin login, la estabilidad 
# y la obtención de datos de alto volumen (para perfiles) mejora 
# si usas un login. Aquí lo configuramos en modo anónimo por defecto.
try:
    # Se inicializa el objeto TvDatafeed
    tv = TvDatafeed()
    print("✅ TvDatafeed inicializado.")
except Exception as e:
    print(f"❌ Error al inicializar TvDatafeed: {e}")
    tv = None # Aseguramos que la variable sea None si falla

# Mapeo de Timeframes solicitados a objetos Interval de tvDatafeed
INTERVAL_MAP = {
    '1m': Interval.in_1_minute, '5m': Interval.in_5_minute, 
    '15m': Interval.in_15_minute, '30m': Interval.in_30_minute,
    '1h': Interval.in_1_hour, '4h': Interval.in_4_hour, 
    '1d': Interval.in_1_day, '1wk': Interval.in_1_week
}

async def fetch_historic_data(symbol: str, tf: str) -> pd.DataFrame:
    """
    Obtiene datos OHLCV, incluyendo volumen por barra de TradingView.
    
    :param symbol: Símbolo del activo (ej: 'SPY').
    :param tf: Timeframe (ej: '1h').
    :return: DataFrame de pandas con datos OHLCV y volumen.
    """
    if tv is None:
        print("❌ Error: TvDatafeed no está inicializado.")
        return pd.DataFrame()

    interval = INTERVAL_MAP.get(tf.lower())
    if not interval:
        print(f"❌ Error: Timeframe '{tf}' no soportado o inválido.")
        return pd.DataFrame()

    try:
        # Intentamos obtener 500 barras (suficiente para análisis de perfil/SMC)
        # Usamos 'NASDAQ' como exchange por defecto, pero se puede parametrizar.
        data = tv.get_hist(symbol=symbol, exchange='NASDAQ', interval=interval, n_bars=500)
        
    except NoDataFound:
        print(f"❌ Error: No se encontraron datos para {symbol} en {tf}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error al obtener datos de TradingView para {symbol}: {e}")
        return pd.DataFrame()
    
    if data is None or data.empty:
        return pd.DataFrame()
    
    # Normalizar nombres de columnas para que sean consistentes con los módulos de indicadores
    # Cambiamos 'open', 'high', 'low', 'close', 'volume' a minúsculas
    data.columns = [c.lower() for c in data.columns]
    
    # Asegurarse de que el índice sea temporal y esté ordenado
    return data.sort_index()

# -----------------------------------------------------------
# NOTA sobre yahoo_api.py:
# Este proyecto puede usar AlphaVantage o Yahoo (que es más simple para live pricing).
# Ya que tvDatafeed nos da datos históricos de alta calidad para SMC, solo
# incluimos el fetch principal aquí. Si se requiere live pricing exacto
# y muy rápido, crearemos un wrapper simple en yahoo_api.py más adelante.
# -----------------------------------------------------------