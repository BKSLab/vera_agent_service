import httpx

# Общий `httpx.AsyncClient` — не пересоздаётся на каждый вызов LLM/MCP.
# Переиспользует соединение (TCP+TLS) между запросами графа. Жизненный
# цикл module-level singleton зарегистрирован на закрытие в `app.main.lifespan`.
external_api_http_client = httpx.AsyncClient()
