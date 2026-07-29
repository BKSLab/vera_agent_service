import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.core.settings import get_settings


async def verify_api_key(x_api_key: Annotated[str, Header()]) -> None:
    """Проверяет сервисный API-ключ BFF сайта."""
    expected = get_settings().app.api_key.get_secret_value()
    if not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Невалидный API-ключ.',
        )


VerifyApiKeyDep = Depends(verify_api_key)
