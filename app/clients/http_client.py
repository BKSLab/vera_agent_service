import httpx

# Общий `httpx.AsyncClient` — не пересоздаётся на каждый вызов LLM/MCP.
# Переиспользует соединение (TCP+TLS) между запросами графа. Жизненный
# цикл — module-level singleton, закрывается в app.main lifespan (Этап 8),
# по образцу vera_rag_service/app/clients/http_client.py.
external_api_http_client = httpx.AsyncClient()
