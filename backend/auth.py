# backend/auth.py

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
import os
import hmac # Usado para comparación segura de strings

load_dotenv()
security = HTTPBasic()

# Configuramos credenciales de Admin
# Usa variables de entorno o un fallback (para test local)
USER = os.getenv("APP_USERNAME", "admin")
PASS = os.getenv("APP_PASSWORD", "skiboss2025") 


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """ Verifica usuario y contraseña para evitar acceso público """
    
    # Compara el usuario y contraseña de forma segura
    correct_username = hmac.compare_digest(credentials.username, USER)
    correct_password = hmac.compare_digest(credentials.password, PASS)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales de acceso incorrectas. Acceso restringido.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username