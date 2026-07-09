# vera_agent_service

Agent Service — оркестратор диалога AI-консультанта «Вера» (проект «Работа для всех»): принимает вопрос пользователя, решает, нужен ли поиск по базе знаний, и стримит ответ клиенту. Не хранит контент базы знаний и не знает деталей реализации инструментов — общается с ними только через внешние контракты (см. ниже).

## Роль в системе

Один из трёх сервисов архитектуры ассистента (`AGENT_VERA_ARCHITECTURE.md`): **Agent Service** (этот репозиторий, оркестратор) → **MCP Tools Server** (прослойка инструментов — отдельный трек, ещё не реализован) → **RAG Service** (`vera_rag_service`, семантический поиск по базе знаний, production-ready).

Итерация 1: единственный инструмент — `kb_search` (поиск по базе знаний), доступен без авторизации. Полная история решений, находок и статус по этапам — `AGENT_SERVICE_PLAN.md`.

## Как это работает

1. **Приём запроса** — consumer слушает очередь `agent.requests` (RabbitMQ). Payload не содержит истории диалога — только новое сообщение и `session_id` (`app/messaging/schemas.py`).
2. **Граф LangGraph** (`app/graph/`) — `analyze_intent` (короткий вызов LLM без стриминга: нужен ли `kb_search`) → при необходимости `call_kb_search` (вызов через MCP-клиент) → `generate_with_context`/`generate_direct` (стримингованная финальная генерация). Три ветки ответа при поиске: есть релевантные чанки / база честно не нашла ответ / поиск технически недоступен — модель не выдумывает факты ни в одном из случаев.
3. **История диалога** — Redis (LangGraph checkpointer, `app/checkpoint/`), ключ треда — `session_id`, TTL по умолчанию 24 часа неактивности. Единственный источник истории — не дублируется в payload RabbitMQ.
4. **MCP Tools Server ещё не существует** (отдельный трек) — `app/clients/mcp_client.py::build_kb_search_tool_proxy` собирает граф без сетевого обращения к нему; реальный удалённый тул резолвится лениво при первом фактическом вызове. До появления MCP Tools Server ответы по вопросам, требующим поиска, честно сообщают о временной недоступности (не ошибка, а спроектированная деградация).
5. **Доставка ответа** — `GET /sse/{session_id}` (`app/streaming/`), токены по мере генерации, терминальные события `done`/`error`. Один активный SSE-коннект на сессию (одна реплика сервиса — не масштабируется на несколько инстансов без перехода на Redis Pub/Sub).
6. **Наблюдаемость** — OpenTelemetry + `openinference-instrumentation-langchain` → Arize Phoenix (`app/observability/`). Автоинструментация покрывает вызовы LLM/графа, вручную добавлены spans на границах: `rabbitmq.consume`, `mcp.tool_call`, `sse.deliver`.

## Стек

FastAPI/`hypercorn` · LangGraph · `langchain-openai` (LLM-провайдер конфигурируется, не захардкожен) · `langchain-mcp-adapters` (MCP-клиент) · RabbitMQ/`aio-pika` (вход) · Redis Stack/`langgraph-checkpoint-redis` (состояние диалога — обычный Redis не подходит, нужен RediSearch) · SSE (выход) · OpenTelemetry/`openinference` → Arize Phoenix (наблюдаемость) · Docker Compose

## Контракты

Подробности, JSON-примеры и обоснования — `AGENT_SERVICE_PLAN.md`, раздел 3.

| Контракт | Кто использует | Кратко |
|---|---|---|
| `agent.requests` (RabbitMQ) | Next.js Proxy → Agent Service | `{session_id, user_id, message}`, без истории; retry только для системных сбоев (невалидный payload, сбой до начала стриминга), DLQ `agent.requests.dlq` |
| `GET /sse/{session_id}` | Клиент ← Agent Service | `data: {"type": "token"/"done"/"error", ...}` |
| Тул `kb_search` (MCP) | Agent Service → MCP Tools Server (будущий) | `kb_search(query: str, audience: "seeker"\|"employer"\|"both") -> {"chunks": [...]}` — формат чанков совпадает с `POST /api/v1/search` в `vera_rag_service` |
| `GET /health` | Оркестратор/мониторинг | `rabbitmq`/`redis` — жёсткий статус (влияет на код ответа); `mcp` — информационное поле, недоступность не даёт `503` (MCP ещё не существует) |

## Запуск локально

```bash
cp .env.example .env
# заполнить .env — минимум RABBITMQ_*, REDIS_*, LLM_* (LLM_API_KEY/LLM_API_URL/LLM_MODEL — любой OpenAI-совместимый провайдер)

docker compose up -d --build
```

| Сервис | Адрес |
|---|---|
| Agent Service | `http://localhost:8010` |
| `GET /health` | `http://localhost:8010/health` |
| RabbitMQ management UI | `http://localhost:15672` |
| Redis | `localhost:6379` (Redis Stack — не обычный Redis, см. `app/checkpoint/redis_saver.py`) |
| Arize Phoenix (трейсы) | `http://localhost:6006` |

Локально без Docker (venv):

```bash
python -m venv venv
venv\Scripts\activate                # Windows; source venv/bin/activate — Linux/macOS
pip install -r requirements-dev.txt

docker compose up -d rabbitmq redis  # только инфраструктура, приложение — из venv
hypercorn app.main:app --bind 0.0.0.0:8000 --reload
```

## Тестирование

```bash
pytest tests/                # юнит + интеграционные (требуют docker compose up -d rabbitmq redis)
ruff check .                 # линтер
```

Интеграционные тесты (маркер `integration`) используют реальные RabbitMQ и Redis Stack из `docker-compose.yml`, и настоящий мок MCP Tools Server (`tests/fixtures/mock_mcp_server.py`, поднимается тестами на свободном порту — не требует внешней инфраструктуры).

## Документация

- [`AGENT_SERVICE_PLAN.md`](AGENT_SERVICE_PLAN.md) — план реализации по этапам, зафиксированные технические решения, контракты, находки, соответствие WBS
- [`AGENT_VERA_ARCHITECTURE.md`](AGENT_VERA_ARCHITECTURE.md) — исходная архитектурная концепция трёх сервисов
- [`FASTAPI_PATTERNS.md`](FASTAPI_PATTERNS.md), [`LLM_CLIENT_REFERENCE.md`](LLM_CLIENT_REFERENCE.md) — эталонные паттерны кода проекта

## Статус

Итерация 1 реализована (`AGENT_SERVICE_PLAN.md`, этапы 0–12) и проверена: 81+ тестов (юнит + интеграционные на реальных RabbitMQ/Redis), полностью собранный production-образ поднят и отвечает `healthy`. MCP Tools Server — отдельный трек, ещё не реализован; до его появления вопросы, требующие `kb_search`, получают честный ответ о временной недоступности поиска, само приложение при этом полностью работоспособно.
