# backend/database.py

import json
import os
import uuid
from typing import List, Dict, Any

# NOTA: En Render.com, esta ruta DEBE ser un volumen persistente o S3.
# Usaremos la ruta local, pero sabiendo que los datos persistirán.
DB_FILE = "trading_journal.json" 

def _load_db() -> Dict[str, Any]:
    """ Carga el contenido de la base de datos JSON """
    if not os.path.exists(DB_FILE):
        return {"active_trades": [], "history": []}
    
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # En caso de archivo corrupto, reiniciamos la base de datos
        return {"active_trades": [], "history": []}


def _save_db(data: Dict[str, Any]):
    """ Guarda el contenido en la base de datos JSON """
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Funciones de Interacción ---

def get_active_trades() -> List[Dict[str, Any]]:
    """ Obtiene todas las operaciones abiertas. """
    db = _load_db()
    return db.get("active_trades", [])

def add_active_trade(trade: Dict[str, Any]) -> str:
    """ Agrega una nueva operación confirmada. """
    trade_id = str(uuid.uuid4())
    trade["id"] = trade_id
    db = _load_db()
    db["active_trades"].append(trade)
    _save_db(db)
    return trade_id

def close_trade_and_save_to_history(trade_id: str, close_details: Dict[str, Any]) -> bool:
    """ Cierra un trade activo y lo mueve al historial. """
    db = _load_db()
    
    # Encuentra el trade activo
    active_trades = db.get("active_trades", [])
    trade_to_move = next((t for t in active_trades if t.get("id") == trade_id), None)
    
    if trade_to_move:
        # Crea el registro histórico
        historical_record = {**trade_to_move, **close_details, "close_time": close_details.get("close_time")}
        db["history"].append(historical_record)
        
        # Elimina de activos
        db["active_trades"] = [t for t in active_trades if t.get("id") != trade_id]
        _save_db(db)
        return True
    return False

def get_trading_history() -> List[Dict[str, Any]]:
    """ Obtiene el historial completo de operaciones cerradas. """
    db = _load_db()
    return db.get("history", [])

def get_trade_by_id(trade_id: str) -> Dict[str, Any] | None:
    """ Obtiene un trade activo por su ID. """
    active_trades = get_active_trades()
    return next((t for t in active_trades if t.get("id") == trade_id), None)