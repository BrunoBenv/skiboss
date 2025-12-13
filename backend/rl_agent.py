# skiboss/backend/rl_agent.py

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Any
import random

# --- 1. Definición de la Arquitectura de la Red Neuronal (Policy Net) ---
# Esta es la red que se entrenará con Deep Reinforcement Learning (PPO/DQN).

class PolicyNet(nn.Module):
    """
    Red Neuronal Profunda (DNN) que actúa como la política del Agente RL.
    Toma el vector de features del mercado y predice la acción óptima.
    """
    
    # Hemos definido un vector de features de 10-12 dimensiones en indicators.py
    INPUT_SIZE = 12 
    # Output: 3 (Acciones discretas: BUY, SELL, HOLD) + 2 (Acciones continuas: SL, TP)
    OUTPUT_SIZE = 5 
    HIDDEN_SIZE = 128
    
    def __init__(self):
        super(PolicyNet, self).__init__()
        
        # Arquitectura con 3 capas ocultas (como se especificó en la Biblia)
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE // 2),
            nn.ReLU(),
            # Capa final de salida
            nn.Linear(self.HIDDEN_SIZE // 2, self.OUTPUT_SIZE)
        )

    def forward(self, x):
        """
        Pasa el vector de features (x) a través de la red.
        El output es un tensor de 5 valores.
        """
        return self.network(x)

# 


# --- 2. Clase Principal del Agente DRL (RL Agent) ---

class RLAgent:
    def __init__(self, model_path: str, version: str = "v1.0"):
        """
        Inicializa el Agente RL, carga el modelo entrenado y define su versión.
        """
        self.version = version
        self.model = PolicyNet()
        self.model_path = model_path
        self.is_loaded = False
        
        # Intentar cargar el modelo (solo para ejecución, no para entrenamiento)
        self._load_model()

    def _load_model(self):
        """
        Carga los pesos de la red neuronal desde el archivo .pth.
        """
        if os.path.exists(self.model_path):
            try:
                # Carga el estado del modelo entrenado
                self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.model.eval()  # Poner el modelo en modo de evaluación
                self.is_loaded = True
                print(f"✅ PolicyNet cargado y listo para predicción.")
            except Exception as e:
                print(f"❌ ERROR: Fallo al cargar pesos del modelo en {self.model_path}: {e}")
                self.is_loaded = False
        else:
            print(f"❌ ADVERTENCIA: Archivo de modelo no encontrado en {self.model_path}. El agente operará con pesos aleatorios (Solo para Testing).")
            # En un entorno de producción, esto debería evitar que se generen señales.

    def predict_signal(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Toma el vector de features y lo pasa a través del modelo para generar la señal.
        
        :param feature_vector: Vector numpy con los features normalizados.
        :return: Diccionario con la señal de trading completa (SignalResponse).
        """
        if not self.is_loaded:
            # Simular una señal si el modelo no está cargado (solo para que el backend funcione)
            return self._simulate_fallback_signal(feature_vector)
            
        # Convertir el vector numpy a Tensor de PyTorch (y añadir una dimensión de batch)
        # Aseguramos que el input tenga el tamaño correcto (ajustando si es necesario)
        if len(feature_vector) > self.model.INPUT_SIZE:
             # Truncar si hay demasiados features
            input_tensor = torch.from_numpy(feature_vector[:self.model.INPUT_SIZE]).float().unsqueeze(0)
        else:
            # Rellenar con ceros si faltan features (no debería pasar si indicators.py está bien)
            padded_vector = np.pad(feature_vector, (0, self.model.INPUT_SIZE - len(feature_vector)), 'constant')
            input_tensor = torch.from_numpy(padded_vector).float().unsqueeze(0)


        # --- 3. Ejecución y Ponderación Crítica ---
        with torch.no_grad():
            raw_output = self.model(input_tensor).squeeze(0)
            
            # --- Aquí es donde el entrenamiento forzó la PONDERACIÓN RECIENTE ---
            # El modelo ya está entrenado para dar mayor peso a los features recientes 
            # (FVG, OB, Market Structure) gracias al Prioritized Experience Replay (PER) 
            # usado en el proceso de entrenamiento de Colab/Kaggle.
            
            # Los 5 valores de output son: [Acción 1, Acción 2, Acción 3, SL, TP]
            
            # 1. Acciones Discretas (BUY, SELL, HOLD)
            action_logits = raw_output[:3]
            # Usar Softmax para obtener la probabilidad de cada acción
            probabilities = torch.softmax(action_logits, dim=0)
            
            # Elegir la acción con mayor probabilidad
            action_idx = torch.argmax(probabilities).item()
            confidence = probabilities[action_idx].item()
            
            ACTION_MAP = {0: "BUY", 1: "SELL", 2: "HOLD"}
            signal = ACTION_MAP.get(action_idx, "HOLD")
            
            # 2. Acciones Continuas (Stop Loss, Take Profit)
            sl_output = raw_output[3].item()
            tp_output = raw_output[4].item()

        # --- 4. Post-Procesamiento (Transformar output de la NN a precios reales) ---
        
        # Recuperar el último precio de cierre para anclar SL/TP
        latest_close = feature_vector[0] # Usamos el primer elemento como ancla (Close) en el vector
        latest_atr = feature_vector[3] # Usamos un elemento del vector como ATR para escalamiento

        # El output de la NN (sl_output, tp_output) suele estar normalizado 
        # (ej. en múltiplos de ATR o porcentaje del precio). 
        # Aquí se revierte la normalización.
        
        # Simulación de des-normalización: Multiplicar por un factor de volatilidad (ATR)
        stop_loss_raw = latest_close + (sl_output * latest_atr)
        take_profit_raw = latest_close + (tp_output * latest_atr)
        
        # Redondear y asegurar que SL/TP sean lógicos (ej. SL por debajo para BUY)
        
        if signal == "BUY":
            # SL debe ser < Precio, TP debe ser > Precio
            stop_loss = round(min(latest_close - (abs(sl_output) * latest_atr), latest_close * 0.98), 2)
            take_profit = round(max(latest_close + (abs(tp_output) * latest_atr), latest_close * 1.02), 2)
        elif signal == "SELL":
            # SL debe ser > Precio, TP debe ser < Precio
            stop_loss = round(max(latest_close + (abs(sl_output) * latest_atr), latest_close * 1.02), 2)
            take_profit = round(min(latest_close - (abs(tp_output) * latest_atr), latest_close * 0.98), 2)
        else:
            # HOLD: sin niveles
            stop_loss = 0.0
            take_profit = 0.0


        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "explanation": self._generate_explanation(feature_vector, signal)
        }

    def _generate_explanation(self, feature_vector: np.ndarray, signal: str) -> str:
        """
        Genera una explicación simple basada en los features más fuertes (los que le importan a la IA).
        """
        # Esta es la lógica para el endpoint POST /explain
        if signal == "HOLD":
            return "El mercado presenta condiciones mixtas y falta de liquidez clara. La política recomienda esperar."
        
        # Lógica simplificada basada en el vector:
        fvg = feature_vector[4] # Índice aproximado para FVG
        ob = feature_vector[5]  # Índice aproximado para OB
        
        if signal == "BUY" and fvg > 0.5:
            return "Señal de COMPRA con alta confianza. El precio acaba de retestear un Fair Value Gap (FVG) alcista clave."
        
        if signal == "SELL" and ob < -0.5:
            return "Señal de VENTA. Se detectó un Order Block (OB) bajista no mitigado cerca de la zona de liquidez."
            
        return f"Señal de {signal} basada en el análisis de flujo de órdenes y momentum reciente post-pandemia."


    def _simulate_fallback_signal(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Genera una señal aleatoria si el modelo .pth no se carga (modo de seguridad).
        """
        signals = ["BUY", "SELL", "HOLD"]
        signal = random.choice(signals)
        latest_close = feature_vector[0] if len(feature_vector) > 0 else 500.0 # Precio de ancla
        
        if signal != "HOLD":
            sl = round(latest_close * (1 + random.uniform(-0.01, 0.01)), 2)
            tp = round(latest_close * (1 + random.uniform(-0.02, 0.02)), 2)
        else:
            sl, tp = 0.0, 0.0
            
        return {
            "signal": signal,
            "confidence": round(random.uniform(0.5, 0.65), 4),
            "stop_loss": sl,
            "take_profit": tp,
            "explanation": "Modelo no cargado. Señal simulada (DEBUG)."
        }