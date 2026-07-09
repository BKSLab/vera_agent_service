from typing import Annotated

import httpx
from fastapi import Depends

from app.clients.http_client import external_api_http_client


def get_http_session() -> httpx.AsyncClient:
    """Возвращает общий `httpx.AsyncClient` — не создаёт новый на каждый
    запрос. Жизненным циклом управляет `app.main.lifespan` (Этап 8)."""
    return external_api_http_client


HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_session)]
