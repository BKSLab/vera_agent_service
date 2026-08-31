import pytest
from fastapi import FastAPI, Request

from app.clients.http_client import create_external_api_http_client
from app.dependencies.http_client import get_http_session


def _request_for(app: FastAPI) -> Request:
    return Request({'type': 'http', 'app': app})


async def test_http_session_returns_only_open_lifespan_client():
    app = FastAPI()
    request = _request_for(app)

    with pytest.raises(RuntimeError, match='не инициализирован'):
        get_http_session(request)

    async with create_external_api_http_client() as client:
        app.state.external_api_http_client = client

        assert get_http_session(request) is client

    with pytest.raises(RuntimeError, match='не инициализирован'):
        get_http_session(request)


async def test_http_client_factory_does_not_reuse_closed_pool():
    first = create_external_api_http_client()
    second = create_external_api_http_client()

    try:
        assert first is not second
        assert first.is_closed is False
        assert second.is_closed is False
    finally:
        await first.aclose()
        await second.aclose()
