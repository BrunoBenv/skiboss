from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import json
import asyncio

# Importaciones de los módulos que crearemos después
# Nota: Si intentas ejecutar esto antes de crear esos archivos, dará error de importación.
try:
    from .rl_agent import RLAgent
    from .indicators import calculate_all_features
    from .tradingview_api import fetch_historic_data
except ImportError:
    # Esto permite que FastAPI se inicie incluso si los módulos no existen aún.
    RLAgent = None
    calculate_all_features = None
    fetch_historic_data = None
    print("ADVERTENCIA: Módulos internos (rl_agent, indicators, etc.) no encontrados. Ejecución de /signal fallará.")


# --- Definición del Esquema de Respuesta (Output Schema) ---
class SignalResponse(BaseModel):
    """Estructura de la respuesta que el modelo devolverá."""
    signal: str  # BUY, SELL, HOLD
    confidence: float  # Probabilidad (0.0 a 1.0)
    stop_loss: float  # Precio sugerido
    take_profit: float  # Precio sugerido
    explanation: str  # Razón de la señal


# --- Inicialización del App ---
app = FastAPI(
    title="SKIBOSS AI Trading System API", 
    version="1.0.0", 
    description="Backend para la generación de señales de trading basadas en DRL y SMC."
)

# --- Inicialización del Modelo DRL ---
SKIBOSS_AGENT = None
MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/model_v1.pth")

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al inicio del servidor (Render.com)."""
    global SKIBOSS_AGENT
    if RLAgent is not None:
        try:
            # Inicializa el agente DRL y carga el modelo PyTorch
            SKIBOSS_AGENT = RLAgent(model_path=MODEL_PATH)
            print(f"✅ Modelo DRL cargado exitosamente desde: {MODEL_PATH}")
        except FileNotFoundError:
            print(f"❌ ADVERTENCIA: Archivo del modelo no encontrado en {MODEL_PATH}. El endpoint /signal no funcionará.")
        except Exception as e:
            print(f"❌ ERROR CRÍTICO al cargar el modelo DRL: {e}")

# --- Endpoints ---

@app.get("/health", tags=["General"])
def health_check():
    """Verifica si el servidor está activo y si el modelo está cargado."""
    model_status = "Cargado" if SKIBOSS_AGENT else "Pendiente de carga o error"
    return {
        "status": "OK",
        "service": "SKIBOSS AI Backend",
        "model_status": model_status
    }

@app.get("/signal", response_model=SignalResponse, tags=["Trading"])
async def get_signal(
    symbol: str = Query(..., description="Símbolo del activo (ej: SPY)"),
    tf: str = Query("1h", description="Timeframe (ej: 1h, 4h, 1d)")
):
    """
    Genera una señal de trading (BUY/SELL/HOLD) utilizando el modelo DRL.
    """
    if SKIBOSS_AGENT is None or calculate_all_features is None or fetch_historic_data is None:
        raise HTTPException(
            status_code=503, 
            detail="Servicio no disponible. El modelo AI o los módulos de datos no están cargados/implementados."
        )

    try:
        # 1. Obtener datos (Usando tvDatafeed/Yahoo)
        raw_data = await fetch_historic_data(symbol, tf)
        if raw_data is None or raw_data.empty or len(raw_data) < 50: # Se requieren al menos 50 barras para indicadores
            raise ValueError("No se pudieron obtener suficientes datos históricos/actuales.")
        
        # 2. Calcular features avanzados (Paso 2 y 3 de la lógica)
        features_vector = calculate_all_features(raw_data)
        
        # 3. Ejecutar el modelo DRL (Paso 4 de la lógica)
        # El agente devuelve un diccionario que coincide con SignalResponse
        signal_output = SKIBOSS_AGENT.predict_signal(features_vector)
        
        # 4. Devolver JSON
        return signal_output

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error en la generación de señal para {symbol}/{tf}: {e}")
        # En producción, usar un mensaje de error más genérico
        raise HTTPException(status_code=500, detail=f"Error interno al calcular la señal.")

@app.get("/model-version", tags=["General"])
def get_model_version():
    """Devuelve la versión del modelo cargado."""
    return {"version": SKIBOSS_AGENT.version if SKIBOSS_AGENT else "N/A"}

# Endpoints de utilidad (implementar la lógica en rl_agent y utils si es necesario)
@app.post("/explain", tags=["Utility"])
def explain_last_signal():
    """Permite al usuario obtener una explicación más detallada de la última señal generada."""
    return {"detail": "Función pendiente de implementación en rl_agent/utils."}

@app.post("/feedback", tags=["Utility"])
def submit_feedback():
    """Permite enviar feedback sobre la calidad de la última señal para reentrenamiento futuro."""
    return {"detail": "Función pendiente de implementación en utils."}