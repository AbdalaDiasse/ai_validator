from fastapi import Header, HTTPException, status
from src.app.settings import settings

def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != settings.validator_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
