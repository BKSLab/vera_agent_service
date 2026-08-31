from typing import Annotated

import httpx
from fastapi import Depends, Request


def get_http_session(request: Request) -> httpx.AsyncClient:
    """Возвращает shared-клиент, созданный и закрываемый lifespan."""
    client = getattr(request.app.state, 'external_api_http_client', None)
    if not isinstance(client, httpx.AsyncClient) or client.is_closed:
        raise RuntimeError('HTTP-клиент внешних API не инициализирован')
    return client


HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_session)]
