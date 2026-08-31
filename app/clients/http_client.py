import httpx


def create_external_api_http_client() -> httpx.AsyncClient:
    """Создаёт общий HTTP-клиент для одного lifespan приложения.

    Владельцем и точкой закрытия остаётся ``app.main.lifespan``. Фабрика не
    хранит module-level singleton, поэтому повторный lifespan в том же процессе
    получает новый открытый connection pool.
    """
    return httpx.AsyncClient()
