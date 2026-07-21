# vera_agent_service

Agent Service — оркестратор диалога AI-консультанта «Вера» (проект «Работа для всех»): принимает вопрос пользователя, решает, нужен ли поиск по базе знаний, и стримит ответ клиенту. Не хранит контент базы знаний и не знает деталей реализации инструментов — общается с ними только через внешние контракты (см. ниже).

## Роль в системе

Один из трёх сервисов архитектуры ассистента (`AGENT_VERA_ARCHITECTURE.md`): **Agent Service** (этот репозиторий, оркестратор) → **MCP Tools Server** (`vera_mcp_service`, прослойка инструментов) → **RAG Service** (`vera_rag_service`, семантический поиск по базе знаний).

Итерация 1: единственный инструмент — `vera_rag_kb` (поиск по базе знаний Vera RAG), доступен без авторизации. Полная история решений, находок и статус по этапам — `AGENT_SERVICE_PLAN.md`.

## Как это работает

1. **Приём запроса** — consumer слушает очередь `agent.requests` (RabbitMQ). Payload не содержит истории диалога — только новое сообщение и `session_id` (`app/messaging/schemas.py`).
2. **Граф LangGraph** (`app/graph/`) — `analyze_intent` (короткий вызов LLM без стриминга: нужен ли `vera_rag_kb`) → при необходимости `call_kb_search` (внутреннее имя узла, вызывающего MCP-инструмент) → `generate_with_context`/`generate_direct` (стримингованная финальная генерация). Три ветки ответа при поиске: есть релевантные чанки / база честно не нашла ответ / поиск технически недоступен — модель не выдумывает факты ни в одном из случаев.
3. **История диалога** — Redis (LangGraph checkpointer, `app/checkpoint/`), ключ треда — `session_id`, TTL по умолчанию 24 часа неактивности. Единственный источник истории — не дублируется в payload RabbitMQ.
4. **MCP Tools Server** — `app/clients/mcp_client.py::build_kb_search_tool_proxy` собирает граф без сетевого обращения к нему; реальный удалённый `vera_rag_kb` резолвится лениво при первом фактическом вызове. При недоступности MCP или RAG ответы по вопросам, требующим поиска, честно сообщают о временной недоступности.
5. **Доставка ответа** — `GET /sse/{session_id}` (`app/streaming/`), токены по мере генерации, терминальные события `done`/`error`. Один активный SSE-коннект на сессию (одна реплика сервиса — не масштабируется на несколько инстансов без перехода на Redis Pub/Sub).
6. **Наблюдаемость** — OpenTelemetry + `openinference-instrumentation-langchain` → Arize Phoenix (`app/observability/`). Один корневой span `vera.agent.request` охватывает обработку сообщения целиком; LLM/LangGraph остаются под автоинструментацией, а вызов поиска начинается с ручного `tool.vera_rag_kb`. SSE-токены учитываются счётчиками корневого span и не создают span на каждый токен.

## Стек

FastAPI/`hypercorn` · LangGraph · `langchain-openai` (LLM-провайдер конфигурируется, не захардкожен) · `langchain-mcp-adapters` (MCP-клиент) · RabbitMQ/`aio-pika` (вход) · Redis Stack/`langgraph-checkpoint-redis` (состояние диалога — обычный Redis не подходит, нужен RediSearch) · SSE (выход) · OpenTelemetry/`openinference` → Arize Phoenix (наблюдаемость) · Docker Compose

## Контракты

Подробности, JSON-примеры и обоснования — `AGENT_SERVICE_PLAN.md`, раздел 3.

| Контракт | Кто использует | Кратко |
|---|---|---|
| `agent.requests` (RabbitMQ) | Next.js Proxy → Agent Service | `{session_id, user_id, message}`, без истории; retry только для системных сбоев (невалидный payload, сбой до начала стриминга), DLQ `agent.requests.dlq` |
| `GET /sse/{session_id}` | Клиент ← Agent Service | `data: {"type": "token"/"done"/"error", ...}` |
| Тул `vera_rag_kb` (MCP) | Agent Service → MCP Tools Server | `vera_rag_kb(query: str, audience: "seeker"\|"employer"\|"both") -> {"chunks": [...]}` — формат чанков совпадает с `POST /api/v1/search` в `vera_rag_service` |
| `GET /health` | Оркестратор/мониторинг | `rabbitmq`/`redis` — жёсткий статус (влияет на код ответа); `mcp` — информационное поле, недоступность не даёт `503` |

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

### Трейс Agent → MCP → RAG

Все три сервиса экспортируют в один Phoenix project, заданный одинаковым
`PHOENIX_PROJECT_NAME`, но сохраняют разные `service.name`. W3C-контекст
`traceparent` передаётся динамически на каждом сетевом переходе, поэтому общий
project не просто складывает spans рядом, а позволяет собрать одно дерево:

```text
vera.agent.request                 vera_agent_service
├── LangChain/LLM spans            vera_agent_service
└── tool.vera_rag_kb               vera_agent_service
    └── mcp.execute.vera_rag_kb    vera_mcp_service
        └── rag.search             vera_rag_service
```

На корневом span видны `session.id`, результат маршрутизации, факт поиска,
число найденных чанков, итог обработки, число повторов и счётчики стриминга.
Текст входа/выхода добавляется только при `PHOENIX_CAPTURE_CONTENT=true`; по
умолчанию он отключён. MCP и RAG не экспортируют query, тексты чанков, промпты
или embedding-векторы. `PHOENIX_ENABLED=false` отключает экспорт без изменения
публичных функциональных контрактов. При штатной остановке накопленные spans
принудительно отправляются перед завершением процесса.

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

## Чеклист перед production-развёртыванием

Локально и функционально всё готово и проверено (см. «Статус» ниже) — но это не значит готовность к реальному прод-деплою. По приоритету, сверху вниз:

**P0 — перед production-деплоем:**
- Реальные LLM-, RabbitMQ- и Redis-credentials заданы в локальном `.env` и не коммитятся. Перед деплоем нужно безопасно перенести этот файл на сервер и проверить права доступа; `.env.example` намеренно содержит только плейсхолдеры.

**P1 — инфраструктура сейчас dev-уровня, не прод:**
- RabbitMQ и Redis вынесены в общие production-сервисы `rabbitmq_service_prod`/`redis_service_prod`; их резервирование, TLS/firewall и ротация credentials находятся в ответственности этих репозиториев.
- Нет Nginx/TLS перед `agent_service` — SSE-эндпоинт сейчас голый HTTP на `8010`; реверс-прокси явно вынесен «вне рамок плана» (раздел 3.2), как и у `vera_rag_service`.
- Phoenix (`6006`) — смотрит наружу тем же `docker-compose.yml`; в проде не должен быть публичным (та же оговорка, что и в README `vera_rag_service`).

**P2 — не верифицировано мной фактическим прогоном (честно, не «наверное сработает»):**
- CI (`.github/workflows/ci.yml`) написан и локально согласован с реальной инфраструктурой, но реальный прогон на GitHub Actions не проверялся — нет доступа к Actions из этой среды. Проверить на первом push/PR.
- Полный путь `Agent → MCP → RAG` с реальным контентом требует отдельного E2E-прогона и фиксации результата.

**Осознанно не блокер:** один инстанс (`HYPERCORN_WORKERS=1`, in-process SSE-очередь) — нормально для пилота, не масштабируется на несколько реплик без перехода на Redis Pub/Sub; уже задокументировано в `AGENT_SERVICE_PLAN.md` (раздел 0.1) как будущая задача, не забытый пробел.

## Статус

Итерация 1 реализована (`AGENT_SERVICE_PLAN.md`, этапы 0–12). Agent Service подключается к развёрнутому MCP Tools Server по streamable-http и использует публичный инструмент `vera_rag_kb`; при недоступности поиска граф переходит в спроектированную ветку деградации. Unit-набор и статические проверки проходят без локального поднятия инфраструктуры; полный integration/E2E-прогон остаётся шагом production-приёмки.
