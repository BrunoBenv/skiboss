# 🤖 SKIBOSS AI TRADING SYSTEM - Especificación Final

Este es el repositorio oficial para el proyecto **SKIBOSS AI**, un sistema de trading automatizado basado en **Deep Reinforcement Learning (DRL)** y análisis de **Smart Money Concepts (SMC)** (Order Flow, FVG, Order Blocks, Volume Profile).

---

## 🎯 OBJETIVO PRINCIPAL

Proveer señales de trading (BUY/SELL/HOLD, SL, TP) mediante una API gratuita, priorizando el comportamiento del mercado **post-pandemia** en sus decisiones.

---

## 🧱 ARQUITECTURA DEL SISTEMA

El proyecto está dividido en tres partes principales:

1.  **Backend (API de Señales):** Construido con **FastAPI** (Python), alojado en **Render.com (Free Tier)**. Carga el modelo PyTorch entrenado y procesa las peticiones de señales en tiempo real.
2.  **Frontend (Webapp):** Interfaz simple HTML/CSS/JS para interactuar con la API.
3.  **Entrenamiento:** Se realiza 24/7 en **Google Colab** y **Kaggle Notebooks** usando una política de ponderación de datos reciente.


### Stack Tecnológico

| Componente | Tecnología | Propósito | Archivos Relacionados |
| :--- | :--- | :--- | :--- |
| **API Core** | FastAPI / Python | Servidor rápido para predicción. | `backend/main.py` |
| **Modelo** | PyTorch (DRL - PPO/DQN) | El cerebro de predicción. | `backend/rl_agent.py` |
| **Data Fetch** | `tvDatafeed`, Yahoo Finance | Obtención de datos históricos y en vivo. | `backend/tradingview_api.py` |
| **Features** | SMC (FVG, OB, POC) | Cálculo de indicadores avanzados. | `backend/orderflow.py`, `backend/indicators.py` |
| **Hosting API** | Render.com (Free) | Alojamiento gratuito del Backend. | `requirements.txt` |
| **Entrenamiento** | Colab & Kaggle (GPU T4) | Entorno gratuito para el entrenamiento 24/7. | `notebooks/train_colab.ipynb` |

---

## 🚀 GUÍA DE INSTALACIÓN Y DEPLOY

### 1. Requerimientos Previos (Cuentas Gratuitas)

* Cuenta de **GitHub** (Obligatorio para el repositorio y conexión con Render).
* Cuenta de **Render.com** (Para el hosting del Backend en el *Free Tier*).
* Cuenta de **Google/Kaggle** (Para el entrenamiento inicial y continuado).
* Obtener una **API Key gratuita** de AlphaVantage (opcional, para datos de respaldo).

### 2. Estructura del Proyecto

El proyecto sigue la siguiente estructura modular:

skiboss/ ├── backend/ # Lógica de FastAPI, el Agente RL y los Indicadores. ├── frontend/ # Archivos HTML/CSS/JS (la interfaz web). ├── notebooks/ # Scripts de entrenamiento para Colab/Kaggle. ├── saved_models/ # Aquí se guarda el modelo entrenado (model_v1.pth). └── requirements.txt # Lista de dependencias de Python.


### 3. Deployment en Render.com (¡El Servidor en Vivo!)

1.  **Sube** todo el contenido de `skiboss` a tu repositorio de **GitHub**.
2.  Crea un nuevo **Web Service** en Render.com.
3.  Conéctalo al repositorio de GitHub.
4.  Configura el **Build Command** (Comando de Construcción):
    ```bash
    pip install -r requirements.txt
    ```
5.  Configura el **Start Command** (Comando de Inicio):
    ```bash
    uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    ```
6.  Selecciona el plan **Free Instance**.
7.  Añade las **Variables de Entorno** (Environment Variables):
    * `MODEL_PATH = saved_models/model_v1.pth` (Ruta donde Render debe buscar el modelo PyTorch).
    * `API_KEY_ALPHA = XXXXX` (Tu clave gratuita de AlphaVantage).

El servicio se desplegará en la URL que te provea Render.

---

## 🧠 LA PONDERACIÓN CRÍTICA (Comportamiento del Robot)

El modelo de **Deep Reinforcement Learning (DRL)** fue entrenado con la directriz de **priorizar la data reciente (Post-2020/2021)** sobre el historial más antiguo (pre-pandemia).

Esto se implementa en la fase de entrenamiento mediante:

* **Prioritized Experience Replay (PER):** Una técnica que asigna un peso y una probabilidad de muestreo exponencialmente mayor a las transiciones (estados/acciones/recompensas) de mercado más recientes.
* **Features Avanzadas (SMC):** El uso de *features* como **FVG, Order Blocks y VAH/VAL** asegura que la IA se enfoque en la **estructura de liquidez actual** del mercado, que es la que mejor refleja la psicología moderna.

El modelo se enfocará en:
* Identificar la formación y mitigación de **FVG/Order Blocks** en el contexto de la volatilidad actual.
* Reaccionar a los **BOS/CHoCH** (Market Structure) de corto/medio plazo, ya que reflejan la liquidez activa.

---

## 📓 ENTRENAMIENTO 24/7 (Colab & Kaggle)

El entrenamiento debe ejecutarse mediante los *notebooks* en la carpeta `notebooks/`. Se recomienda la siguiente rotación para maximizar el tiempo de GPU gratuito:

1.  Iniciar `train_colab.ipynb` (Dura ~12 horas).
2.  Cuando Colab se desconecte, iniciar `train_kaggle.ipynb` (Puede durar hasta ~9 horas).
3.  Repetir.

Al final de cada sesión, el *notebook* debe guardar el nuevo `model_vX.pth`, que luego debe subirse a GitHub para activar el *auto-redeploy* en Render.com.