# vera_agent_service

Agent Service — оркестратор диалога AI-консультанта «Вера» (проект «Работа для всех»): принимает вопрос пользователя, выбирает нужный инструмент и стримит ответ клиенту. Не хранит контент базы знаний и не знает деталей реализации инструментов — общается с ними только через внешние контракты (см. ниже).

## Роль в системе

Один из трёх сервисов архитектуры ассистента (`AGENT_VERA_ARCHITECTURE.md`): **Agent Service** (этот репозиторий, оркестратор) → **MCP Tools Server** (`vera_mcp_service`, прослойка инструментов) → **RAG Service** (`vera_rag_service`, семантический поиск по базе знаний).

Агенту доступны два общих инструмента без авторизации: `vera_rag_kb` для поиска по базе знаний и `send_consultation_email` для отправки уже состоявшейся консультации в доступном PDF. Полная история решений и статус по этапам — `AGENT_SERVICE_PLAN.md`.

## Как это работает

1. **Приём запроса** — consumer слушает очередь `agent.requests` (RabbitMQ). Payload не содержит истории диалога: `session_id` адресует историю, `request_id` — доставку ответа конкретного сообщения (`app/messaging/schemas.py`).
2. **Граф LangGraph** (`app/graph/`) — `analyze_intent` выбирает прямой ответ, поиск через `call_kb_search` или отправку через `call_consultation_email`; затем `generate_with_context`/`generate_direct` стримит финальный ответ. Поиск различает найденные данные, честное отсутствие ответа и техническую недоступность.
3. **История диалога** — Redis (LangGraph checkpointer, `app/checkpoint/`), ключ треда — `session_id`, TTL по умолчанию 24 часа неактивности. Единственный источник истории — не дублируется в payload RabbitMQ.
4. **MCP Tools Server** — локальные прокси-схемы собирают граф без сетевого обращения; оба удалённых инструмента резолвятся лениво при первом фактическом вызове. Идемпотентный поиск допускает ретраи. Мутирующая отправка письма выполняется строго один раз с отдельным увеличенным timeout: после сетевой неопределённости агент не повторяет ни тулу, ни весь граф.
5. **Доставка ответа** — `GET /sse/{request_id}` (`app/streaming/`), токены по мере генерации, терминальные события `done`/`error`. `request_id` изолирует late-connect буфер и SSE одного сообщения от других запросов той же сессии. Одна реплика сервиса по-прежнему не масштабируется без перехода на Redis Pub/Sub.
6. **Наблюдаемость** — OpenTelemetry + `openinference-instrumentation-langchain` → Arize Phoenix (`app/observability/`). Один корневой span `vera.agent.request` охватывает обработку сообщения целиком. Для `tool.send_consultation_email` сохраняются только безопасные агрегаты и статус — текст консультации, email и полный ответ тулы в этот span не записываются.
7. **Обратная связь и админка** — завершённые реплики сохраняются в PostgreSQL для оценок и экспертного разбора. Redis остаётся источником рабочей истории агента; PostgreSQL не участвует в построении контекста графа.

## Стек

FastAPI/`hypercorn` · LangGraph · `langchain-openai` (LLM-провайдер конфигурируется, не захардкожен) · `langchain-mcp-adapters` (MCP-клиент) · RabbitMQ/`aio-pika` (вход) · Redis Stack/`langgraph-checkpoint-redis` (состояние диалога — обычный Redis не подходит, нужен RediSearch) · PostgreSQL/SQLAlchemy/Alembic (аналитическая копия диалогов и обратная связь) · SQLAdmin (экспертная админка) · SSE (выход) · OpenTelemetry/`openinference` → Arize Phoenix (наблюдаемость) · Docker Compose

## Контракты

Подробности, JSON-примеры и обоснования — `AGENT_SERVICE_PLAN.md`, раздел 3.

| Контракт | Кто использует | Кратко |
|---|---|---|
| `agent.requests` (RabbitMQ) | Next.js Proxy → Agent Service | `{session_id, request_id, user_id, message}`, без истории; `request_id` обязателен, retry только для системных сбоев, DLQ `agent.requests.dlq` |
| `GET /sse/{request_id}` | Клиент ← Agent Service | `data: {"type": "token"/"done"/"error", ...}` только для одного пользовательского запроса |
| `PUT /api/v1/feedback/message` | Frontend → Agent Service | Создать или изменить оценку `up`/`down` для завершённого ответа |
| `POST /api/v1/feedback/session` | Frontend → Agent Service | Идемпотентно сохранить итоговую анкету по сессии |
| Тул `vera_rag_kb` (MCP) | Agent Service → MCP Tools Server | `vera_rag_kb(query: str) -> {"chunks": [...]}` — роль пользователя не передаётся в поиск; формат чанков совпадает с `POST /api/v1/search` в `vera_rag_service` |
| Тул `send_consultation_email` (MCP) | Agent Service → MCP Tools Server | `send_consultation_email(consultation_text: str, email: str) -> dict`; вызывается только по явной просьбе с указанным адресом, не ретраится |
| `GET /health` | Оркестратор/мониторинг | `rabbitmq`/`redis`/`database` — жёсткий статус (влияет на код ответа); `mcp` — информационное поле, недоступность не даёт `503` |

## Запуск локально

```bash
cp .env.example .env
# заполнить .env — RABBITMQ_*, REDIS_*, POSTGRES_*, LLM_*, API_KEY и ADMIN_*

docker compose up -d --build
```

| Сервис | Адрес |
|---|---|
| Agent Service | `http://localhost:8010` |
| `GET /health` | `http://localhost:8010/health` |
| Админка | `http://localhost:8010/admin` |
| PostgreSQL | `localhost:5438` |
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
├── tool.vera_rag_kb               vera_agent_service
│   └── mcp.execute.vera_rag_kb    vera_mcp_service
│       └── rag.search             vera_rag_service
└── tool.send_consultation_email   vera_agent_service
    └── mcp.execute.*              vera_mcp_service
```

На корневом span видны `session.id`, результат маршрутизации, факт поиска,
число найденных чанков, итог обработки, число повторов и счётчики стриминга.
Содержимое передаётся целиком: текст вопроса и ответа на корневом span,
сообщения и промпты LLM, аргументы и результаты обоих MCP-инструментов —
включая найденные чанки базы знаний. Без этого по трейсу нельзя понять, что
пришло на вход, какой инструмент вызвался, с чем и что вернул. Отключается
через `TRACE_CONTENT_ENABLED=false` — для окружений, где трейсы уходят
наружу; в остальном доступ к Phoenix ограничивается на уровне самого
сервиса, а не вырезанием данных из трейсов. `PHOENIX_ENABLED=false`
отключает экспорт без изменения публичных функциональных контрактов. При
штатной остановке накопленные spans принудительно отправляются перед
завершением процесса.

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

Интеграционные тесты (маркер `integration`) используют реальные RabbitMQ и Redis Stack из `docker-compose.yml`, PostgreSQL через Testcontainers и настоящий мок MCP Tools Server (`tests/fixtures/mock_mcp_server.py`, поднимается тестами на свободном порту — не требует внешней инфраструктуры).

## Документация

- [`AGENT_SERVICE_PLAN.md`](AGENT_SERVICE_PLAN.md) — план реализации по этапам, зафиксированные технические решения, контракты, находки, соответствие WBS
- [`AGENT_VERA_ARCHITECTURE.md`](AGENT_VERA_ARCHITECTURE.md) — исходная архитектурная концепция трёх сервисов
- [`ADMIN_GUIDE.md`](ADMIN_GUIDE.md) — работа с админкой и обратной связью
- [`SITE_FEEDBACK_API_CONTRACT.md`](SITE_FEEDBACK_API_CONTRACT.md) — контракт двух feedback-запросов для серверной части сайта
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

Базовая итерация и интеграция `send_consultation_email` реализованы. Agent Service подключается к MCP Tools Server по streamable-http, безопасно деградирует при недоступности поиска и не допускает автоматического дубля письма после неопределённого результата отправки. Полный production E2E с реальным SMTP остаётся шагом приёмки.
