# План упрощения и сквозной настройки Phoenix для сервисов «Веры»

> **Статус:** код, автоматические тесты и документация реализованы; остаётся живая приёмка на общей инфраструктуре.
> **Дата:** 2026-07-20.
> **Репозитории:** `vera_agent_service`, `vera_mcp_service`, `vera_rag_service`.
> **Главный принцип:** Phoenix должен сначала показывать понятный запуск агента, а технические детали сервисов — только там, где они помогают объяснить задержку, деградацию или ошибку.

## 1. Цель

Сделать один запрос пользователя читаемым в Phoenix как единое дерево от Agent Service до MCP и RAG Service:

```text
vera.agent.request                         service.name=vera_agent_service
├── LangChain/LLM: анализ намерения
├── tool.vera_rag_kb                      service.name=vera_agent_service
│   └── mcp.execute.vera_rag_kb           service.name=vera_mcp_service
│       └── rag.search                    service.name=vera_rag_service
└── LangChain/LLM: генерация ответа
```

Для прямого ответа без базы знаний дерево должно быть короче:

```text
vera.agent.request
├── LangChain/LLM: анализ намерения
└── LangChain/LLM: генерация ответа
```

### Definition of Done

- [x] Один пользовательский запрос создаёт ровно один корневой span `vera.agent.request`.
- [x] В корневом span понятны сессия, маршрут, итоговый статус и количество повторов; в разрешённом privacy-режиме также видны безопасные input/output.
- [x] Один ответ больше не создаёт span `sse.deliver` на каждый токен.
- [x] Agent, MCP и RAG отправляют spans в один Phoenix project текущего окружения.
- [ ] `trace_id` не меняется на границах Agent → MCP → RAG.
- [x] MCP и RAG не дублируют одинаковые по смыслу spans с одинаковыми именами.
- [x] RAG представлен одним компактным `rag.search` с агрегированной диагностикой, без дерева на каждый вариант запроса и каждый Qdrant-вызов.
- [x] При выключенном или недоступном Phoenix функциональность сервисов не ломается.
- [x] При штатной остановке каждый процесс выполняет `force_flush`, и завершённые spans не теряются.
- [ ] Юнит-, интеграционные и сквозные проверки из раздела 10 проходят.

## 2. Зафиксированные решения

### 2.1. Phoenix project и идентичность сервисов

Во всех трёх сервисах добавить одинаковую настройку:

```dotenv
PHOENIX_ENABLED=true
PHOENIX_OTLP_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_PROJECT_NAME=vera-local
PHOENIX_CAPTURE_CONTENT=false
PHOENIX_CONTENT_MAX_CHARS=12000
```

Рекомендуемые имена проектов:

| Окружение | `PHOENIX_PROJECT_NAME` |
|---|---|
| локальная разработка | `vera-local` |
| тестовый контур | `vera-testing` |
| production | `vera-production` |

Правила:

- `PHOENIX_PROJECT_NAME` одинаков для Agent, MCP и RAG в одном окружении;
- `service.name` остаётся разным: `vera_agent_service`, `vera_mcp_service`, `vera_rag_service`;
- project routing для OTLP/HTTP выполнять заголовком экспортёра `x-project-name`;
- не использовать имя сервиса как имя Phoenix project: это снова разделит один distributed trace между экранами;
- endpoint и project name задаются конфигурацией, не захардкоживаются в прикладном коде.

Phoenix 15.5.0+ поддерживает `x-project-name` для OTLP/HTTP; используемый образ `version-17.21.0` это поддерживает. Заголовок имеет приоритет над resource-атрибутом `openinference.project.name`.

### 2.2. Межсервисный контекст

Использовать стандартный W3C Trace Context:

- исходящий сервис вызывает `opentelemetry.propagate.inject(headers)`;
- входящий сервис вызывает `opentelemetry.propagate.extract(headers)`;
- передаются стандартные `traceparent` и, если присутствует, `tracestate`;
- не передавать `trace_id` отдельным самодельным HTTP-заголовком;
- значение контекста формируется на каждый вызов, а не один раз при старте процесса.

Последний пункт критичен для Agent Service: статические `headers` в конфигурации `MultiServerMCPClient` нельзя заполнять текущим trace-контекстом при создании клиента — клиент живёт дольше одного запроса. Нужен динамический `ToolCallInterceptor`, который добавляет заголовки непосредственно перед каждым MCP tool call.

### 2.3. Приватность input/output

В Agent Service добавить:

```dotenv
PHOENIX_CAPTURE_CONTENT=false
PHOENIX_CONTENT_MAX_CHARS=12000
```

Политика:

- `false` — production-safe режим по умолчанию: сохраняются длины, статусы, счётчики и технические атрибуты, но не тексты вопроса/ответа;
- `true` — разрешён только для контура, где доступ к self-hosted Phoenix ограничен и хранение пользовательского текста явно принято владельцем данных;
- для локальной/E2E-проверки читаемости Phoenix временно включать `true`; production оставлять `false`, пока владелец данных явно не утвердит хранение текстов;
- даже при `true` не экспортировать API-ключи, системные промпты, полную историю диалога, содержимое Redis state и внутренние сообщения инструментов;
- автоматические LangChain/OpenInference spans всегда создаются с `TraceConfig`, скрывающим inputs, outputs, prompts и messages; `PHOENIX_CAPTURE_CONTENT` разрешает текст только на ручном корневом `vera.agent.request`;
- `user_id` не экспортировать в открытом виде; достаточно `user.authenticated: bool`;
- `session.id` разрешить как span-атрибут для поиска конкретного диалога, но не делать resource-атрибутом;
- RAG Service не должен повторно сохранять полный query и тексты найденных чанков в Phoenix: при разрешённом capture они уже видны на корневом Agent-span, а подробный журнал поиска хранится в защищённой таблице `search_logs`.
- input/output обрезать до `PHOENIX_CONTENT_MAX_CHARS`; факт обрезки явно отмечать `input.truncated=true`/`output.truncated=true`, а исходную длину сохранять отдельным числовым атрибутом.

При `PHOENIX_CAPTURE_CONTENT=true` корневой span получает:

- `input.value` — только новое сообщение пользователя;
- `input.mime_type=text/plain`;
- `output.value` — только финальный текст ответа пользователю;
- `output.mime_type=text/plain`.

Настройка capture не должна менять функциональный ответ, SSE-поток или содержимое Redis: она управляет только экспортируемыми span attributes.

### 2.4. Статусы

| Ситуация | OTel status | Прикладной атрибут |
|---|---|---|
| Успешный прямой ответ | `UNSET`/успех | `agent.outcome=done` |
| Успешный ответ с RAG | `UNSET`/успех | `agent.outcome=done` |
| Query expansion/reranker ушёл в штатный fallback | не `ERROR` | `rag.outcome=degraded` |
| MCP/RAG недоступен, но агент честно ответил о деградации | не `ERROR` | `agent.outcome=degraded` |
| Пустая база/нет релевантных чанков | не `ERROR` | `rag.outcome=empty` |
| Сообщение ушло в DLQ или ответ оборвался ошибкой | `ERROR` | `agent.outcome=error` |
| Необработанное исключение конкретного сервиса | `ERROR`, `record_exception` | `error.type=<тип>` |

Fallback не должен окрашивать весь trace как аварийный, если пользователь получил предусмотренный продуктом ответ.

### 2.5. Что сознательно не делаем

- Не переносим `PhoenixThinkingSpanProcessor` и перевод reasoning из Bitrix-проекта.
- Не добавляем Eval Worker и автоматические оценки в рамках этой задачи.
- Не меняем версию Phoenix без отдельной причины.
- Не делаем Phoenix жёсткой startup-зависимостью сервисов.
- Не включаем тотальную автоинструментацию FastAPI, HTTPX, SQLAlchemy, Redis, Qdrant и RabbitMQ.
- Не создаём spans для каждого SSE-токена, варианта query expansion, category lane, Qdrant-вектора или кандидата reranker.
- Не добавляем миграцию `trace_id` в `search_logs` на первом этапе.
- Не распространяем trace-контекст от внешнего RabbitMQ publisher до Agent Service: продуктовый вход текущего scope начинается в Agent Service.

## 3. Единый контракт span-атрибутов

### 3.1. `vera.agent.request`

Обязательные атрибуты:

| Атрибут | Значение |
|---|---|
| `openinference.span.kind` | `AGENT` |
| `session.id` | `session_id` сообщения |
| `user.authenticated` | `user_id is not None` |
| `messaging.system` | `rabbitmq` |
| `messaging.destination.name` | `agent.requests` |
| `agent.input.char_count` | исходная длина нового сообщения |
| `agent.route` | `direct` или `knowledge_base` |
| `agent.search.required` | bool |
| `agent.search.unavailable` | bool |
| `agent.search.chunk_count` | число итоговых чанков MCP |
| `agent.tool_call_count` | число логических вызовов инструментов |
| `agent.retry.count` | число повторных попыток полной обработки RabbitMQ message |
| `agent.response.chunk_count` | число стриминговых фрагментов |
| `agent.response.char_count` | длина собранного ответа |
| `agent.outcome` | `done`, `degraded`, `error`, `invalid_payload`, `dlq` |

Условные атрибуты:

- `input.value`/`output.value` — только при `PHOENIX_CAPTURE_CONTENT=true`;
- `input.truncated`/`output.truncated` — только если сработал лимит `PHOENIX_CONTENT_MAX_CHARS`;
- `error.type` и exception event — только при реальной ошибке;
- `agent.mcp.retry_count` — если MCP пришлось повторять;
- `agent.streaming.started` — помогает отличить retry до стриминга от ошибки после начала стриминга.

### 3.2. `tool.vera_rag_kb` в Agent Service

| Атрибут | Значение |
|---|---|
| `openinference.span.kind` | `TOOL` |
| `tool.name` | `vera_rag_kb` |
| `tool.retry.count` | реально выполненные дополнительные попытки |
| `tool.outcome` | `ok`, `empty`, `unavailable`, `error` |
| `tool.result.chunk_count` | количество чанков |

Span представляет логический вызов инструмента с учётом retry и является родителем удалённого MCP-span.

### 3.3. `mcp.execute.vera_rag_kb` в MCP Service

| Атрибут | Значение |
|---|---|
| `openinference.span.kind` | `TOOL` |
| `mcp.server.name` | `vera-tools` |
| `mcp.tool.name` | `vera_rag_kb` |
| `mcp.tool.audience` | `seeker`, `employer` или `both` |
| `mcp.tool.query_length` | длина query без текста |
| `mcp.tool.result_chunk_count` | количество чанков |
| `mcp.tool.outcome` | `ok`, `empty`, `rag_unavailable`, `error` |

### 3.4. `rag.search` в RAG Service

| Атрибут | Значение |
|---|---|
| `openinference.span.kind` | `RETRIEVER` |
| `request.id` | существующий `X-Request-ID`/request context RAG |
| `rag.query.length` | длина query |
| `rag.audience` | фильтр audience либо пустая строка |
| `rag.topic` | topic либо пустая строка |
| `rag.category` | category либо пустая строка |
| `rag.top_k` | запрошенный top-k |
| `rag.query_variant_count` | число вариантов после expansion |
| `rag.dense_candidate_count` | агрегированное число dense-кандидатов |
| `rag.sparse_candidate_count` | агрегированное число sparse-кандидатов |
| `rag.rrf_candidate_count` | число кандидатов после fusion |
| `rag.result_chunk_count` | число возвращённых чанков |
| `rag.query_expansion.status` | текущий статус pipeline |
| `rag.reranker.status` | текущий статус pipeline |
| `rag.latency.query_expansion_ms` | latency существующего замера |
| `rag.latency.embed_query_ms` | latency существующего замера |
| `rag.latency.hybrid_search_ms` | latency существующего замера |
| `rag.latency.rerank_ms` | latency существующего замера |
| `rag.outcome` | `ok`, `empty`, `degraded`, `error` |

Не добавлять query, промпты query expansion/reranker, тексты чанков и полные payload Qdrant.

## 4. Agent Service — подробный план

Репозиторий: `D:\BKS.Lab\python\my_projects\vera_agent_service`.

### 4.1. Настройки и зависимости

Файлы:

- `app/core/settings.py`;
- `.env.example`;
- `requirements.txt` при необходимости прямой зависимости от semantic conventions;
- `docker-compose.yml`.

Правки:

- [x] Расширить `ObservabilitySettings` полями `phoenix_enabled`, `phoenix_project_name`, `phoenix_capture_content`, `phoenix_content_max_chars`.
- [x] Сохранить существующий `phoenix_otlp_endpoint`.
- [x] В `.env.example` описать одинаковые project names для трёх сервисов и privacy-режим.
- [x] В Compose оставить endpoint `http://phoenix:6006/v1/traces`, добавить project name и capture mode через environment/env_file.
- [x] Не добавлять `depends_on: phoenix` к Agent Service.

### 4.2. Инициализация и завершение tracing

Файл: `app/observability/tracing.py`.

Правки:

- [x] Если `phoenix_enabled=false`, возвращать no-op поведение без OTLP exporter.
- [x] Создавать `OTLPSpanExporter(endpoint=..., headers={'x-project-name': ...})`.
- [x] Сохранить `Resource({'service.name': 'vera_agent_service'})`.
- [x] Сохранить `LangChainInstrumentor`, но не добавлять вторую автоинструментацию тех же LangChain-вызовов.
- [x] Добавить `force_flush_tracing(timeout_millis=10_000)`.
- [x] Добавить `shutdown_tracing()` либо один идемпотентный helper, который flush-ит и завершает provider.
- [x] Не пытаться повторно вызывать `trace.set_tracer_provider()` в одном процессе; сохранить существующий тестовый паттерн с одним provider.

Файл: `app/main.py`.

- [x] Поместить flush/shutdown в `finally` lifespan после остановки consumer и закрытия рабочих ресурсов.
- [x] Ошибка exporter/flush логируется, но не мешает остановке приложения.

### 4.3. Контекст телеметрии одного запроса

Новый файл: `app/observability/request_trace.py`.

Назначение: хранить агрегаты одного RabbitMQ message независимо от внутренних LangGraph spans.

Предлагаемая структура:

```python
@dataclass
class AgentRequestTraceData:
    route: str = 'unknown'
    search_required: bool = False
    search_unavailable: bool = False
    search_chunk_count: int = 0
    tool_call_count: int = 0
    request_retry_count: int = 0
    mcp_retry_count: int = 0
    response_chunk_count: int = 0
    response_char_count: int = 0
    outcome: str = 'unknown'
```

- [x] Хранить объект в `ContextVar`.
- [x] Consumer устанавливает context перед запуском графа и обязательно сбрасывает token в `finally`.
- [x] Узлы графа обновляют только агрегаты, не создают новые ручные корневые spans.
- [x] Не хранить изменяемый объект как глобальное/default значение `ContextVar`: для каждого RabbitMQ message создавать новый экземпляр.
- [x] Не помещать в объект секреты или полную историю.
- [x] Написать тест на отсутствие утечки данных между двумя последовательными запросами и двумя параллельными async-задачами.

### 4.4. Корневой span и сбор ответа

Файл: `app/messaging/consumer.py`.

- [x] Заменить ручной `rabbitmq.consume` на `vera.agent.request`.
- [x] Начинать span до разбора payload, чтобы invalid payload/DLQ тоже были наблюдаемы.
- [x] Завершать root span только после публикации терминального SSE-события (`done`/`error`) и выполнения `ack`/`nack`; это и есть «полная обработка сообщения» в текущей архитектуре.
- [x] После успешного разбора установить `session.id`, `user.authenticated`, queue name и длину input.
- [x] При разрешённом capture установить `input.value` только из `payload.message`.
- [x] Во время `_stream_answer` считать число непустых chunks и символов.
- [x] Для `output.value` при разрешённом capture собрать финальный ответ из стриминговых chunks.
- [x] Буфер текста и счётчики ответа создавать заново для каждой retry-попытки; не переносить данные неуспешной попытки в следующий запуск графа.
- [x] В root attributes сохранять только фактически отданный пользователю финальный/частичный поток, а не внутренние ответы неуспешных попыток до начала стриминга.
- [x] Не включать `done`/`error` SSE event в текст ответа.
- [x] На каждом повторе полной обработки обновлять `agent.retry.count`; первая попытка не считается retry.
- [x] Перед завершением span перенести агрегаты `AgentRequestTraceData` в атрибуты.
- [x] При ошибке после начала стриминга записать `agent.streaming.started=true`, exception и `agent.outcome=error`.
- [x] При исчерпании retry и `nack(requeue=False)` записать `agent.outcome=dlq`.
- [x] При штатной деградации MCP не ставить OTel `ERROR`.
- [x] Исправить документацию: текущий `rabbitmq.consume` измеряет обработку после получения, а не фактическое время ожидания в очереди.

### 4.5. Маршрут и результат поиска

Файлы:

- `app/graph/nodes/analyze_intent.py`;
- `app/graph/nodes/call_kb_search.py`;
- при необходимости `app/graph/nodes/generate_direct.py` и `generate_with_context.py`.

Правки:

- [x] `analyze_intent`: записывать `route=knowledge_base`, `search_required=true`, если модель вернула tool call; иначе `route=direct`.
- [x] `call_kb_search`: увеличивать `tool_call_count` один раз на логический вызов, а не на каждую retry-попытку.
- [x] После MCP-ответа записывать `search_chunk_count`.
- [x] При `McpUnavailableError` записывать `search_unavailable=true` и outcome запроса `degraded`.
- [x] Не добавлять тексты чанков в span attributes.

### 4.6. Логический tool span и динамический `traceparent`

Файл: `app/clients/mcp_client.py`.

- [x] Переименовать ручной `mcp.tool_call` в `tool.vera_rag_kb` либо формировать имя как `tool.<tool.name>`.
- [x] Span должен охватывать все retry-попытки одного логического вызова.
- [x] Записывать `tool.retry.count`, outcome и result chunk count.
- [x] Реализовать `ToolCallInterceptor` для `MultiServerMCPClient`.
- [x] Зафиксировать контракт тестом для установленного и pinned `langchain-mcp-adapters==0.3.0`: interceptor получает изменяемое поле `request.headers` именно на каждый tool call.
- [x] В interceptor копировать существующие request headers, вызывать `opentelemetry.propagate.inject(headers)` и возвращать `request.override(headers=headers)`.
- [x] Не заменять MCP protocol/auth headers и не мутировать общий словарь connection config.
- [x] Убедиться тестом, что два параллельных tool call получают разные `traceparent` от своих активных spans.
- [x] Не использовать статический `headers` в `get_mcp_client()` для динамического trace-контекста.

### 4.7. Удаление токенных spans

Файл: `app/streaming/session_bus.py`.

- [x] Удалить `tracer` и обёртку `sse.deliver` из `publish()`.
- [x] Оставить `publish()` обычной доставкой события в очередь/буфер.
- [x] Число chunks и символов считать в consumer, где уже активен `vera.agent.request`.
- [x] Не создавать отдельный `sse.stream` на первом этапе: HTTP SSE connection живёт независимо от RabbitMQ processing context и потребовал бы дополнительного механизма linking.
- [x] Удалить/переписать тест, проверяющий наличие `sse.deliver`; заменить тестом, что публикация N токенов не создаёт N spans.

### 4.8. Тесты Agent Service

Файлы:

- `tests/unit/observability/test_tracing.py`;
- новый `tests/unit/observability/test_request_trace.py`;
- `tests/unit/messaging/test_consumer.py`;
- `tests/unit/clients/test_mcp_client.py`;
- `tests/unit/graph/test_analyze_intent.py`;
- `tests/unit/graph/test_call_kb_search.py`;
- `tests/integration/test_graph.py`;
- `tests/integration/test_mcp_client.py`;
- `tests/integration/test_consumer_sse_pipeline.py`.

Проверки:

- [x] Один message → один `vera.agent.request`.
- [x] Direct path: route `direct`, tool count `0`, нет MCP-span.
- [x] Search path: route `knowledge_base`, корректный chunk count.
- [x] Empty search: успех, chunk count `0`, не ERROR.
- [x] MCP unavailable: outcome `degraded`, а не terminal ERROR.
- [x] Retry до стриминга отражён в `agent.retry.count`.
- [x] Ошибка после первого токена не запускает повтор и помечает root ERROR.
- [x] При capture=false отсутствуют `input.value` и `output.value`.
- [x] При capture=true они содержат только новое сообщение и финальный ответ.
- [x] Контент длиннее лимита обрезается, а `*.truncated` и исходная длина заполнены корректно.
- [x] Повтор до начала стриминга не смешивает буферы output двух попыток.
- [x] N токенов не создают N `sse.deliver` spans.
- [x] MCP interceptor добавляет валидный `traceparent`.
- [x] `tool.vera_rag_kb` и root имеют один trace id и корректный parent id.
- [x] Project name передан exporter через `x-project-name`.
- [x] Shutdown вызывает force flush один раз.

## 5. MCP Service — подробный план

Репозиторий: `D:\BKS.Lab\python\my_projects\vera_mcp_service`.

### 5.1. Настройки tracing

Файлы:

- `app/core/settings.py`;
- `.env.example`;
- `app/observability/tracing.py`;
- `requirements.txt`;
- `docker-compose.yml`.

Правки:

- [x] Добавить `phoenix_enabled` и `phoenix_project_name` в `ObservabilitySettings`.
- [x] Настроить exporter с `x-project-name`.
- [x] Сохранить `service.name=vera_mcp_service`.
- [x] Добавить helpers для `extract`, `inject`, `force_flush` и shutdown без глобальной переинициализации provider.
- [x] Направить endpoint в тот же Phoenix, что и Agent/RAG.
- [x] Не добавлять отдельный контейнер Phoenix.

### 5.2. Извлечение контекста в FastMCP tool

Файл: `app/tools/vera_rag_kb.py`.

- [x] Добавить специальный параметр FastMCP `Context` в функцию инструмента; он не должен попадать в публичную JSON schema тула.
- [x] Получить входящий HTTP request из `ctx.request_context.request` и его headers.
- [x] Вызвать `propagate.extract(headers)`.
- [x] Создать `mcp.execute.vera_rag_kb` с извлечённым remote context как parent.
- [x] Сделать этот span текущим на всё время `rag_client.search(...)`, чтобы исходящий вызов RAG наследовал его.
- [x] Записать audience, query length, result chunk count и outcome.
- [x] На `RagUnavailableError` записать exception/ERROR и пробросить исключение без изменения существующего MCP-контракта.
- [x] Если request/context недоступен в прямом юнит-вызове `mcp.call_tool`, корректно начинать новый trace, а не падать.

Отдельно проверить реальным streamable-http тестом, что в используемой версии `mcp==1.28.1` `Context.request_context.request` содержит Starlette request и что извлечённый context сохраняется до выполнения тела тула. Если SDK меняет этот контракт, fallback — тонкий ASGI middleware вокруг `mcp.streamable_http_app()`; middleware должен отдельным тестом доказать перенос context в создаваемую SDK async-задачу. Код SDK в `site-packages` не патчить.

### 5.3. Передача контекста в RAG

Файл: `app/clients/rag_client.py`.

- [x] Перед `httpx_client.post` создать новый словарь headers с `X-API-Key`.
- [x] Вызвать `propagate.inject(headers)` при активном `mcp.execute.vera_rag_kb`.
- [x] Убедиться, что `traceparent` не сохраняется в module-level/shared headers между запросами.
- [x] Удалить или переименовать текущий ручной `rag.search` в MCP Service: фактический `rag.search` должен принадлежать RAG Service.
- [x] Не добавлять отдельный HTTP client span на первом этапе; latency внешнего вызова уже покрывается `mcp.execute.vera_rag_kb`, а детализация живёт в RAG span.
- [x] Health-check RAG не должен присоединяться к пользовательскому trace и не должен создавать прикладной `rag.search`.

### 5.4. Завершение процесса

Файл: `app/main.py`.

- [x] Обернуть blocking `mcp.run(transport='streamable-http')` в `try/finally`.
- [x] В `finally` выполнить `force_flush`/shutdown provider.
- [x] Корректно закрыть `httpx_client`, если выбранный жизненный цикл FastMCP позволяет это сделать без создания второго event loop; иначе вынести закрытие в поддерживаемый FastMCP lifespan и отдельно проверить регистрацию тулов до первого MCP-запроса.
- [x] Не возвращать прежнюю ошибочную схему, в которой регистрация инструментов выполнялась слишком поздно внутри session lifespan.

### 5.5. Тесты MCP Service

Файлы:

- `tests/unit/observability/test_tracing.py`;
- `tests/unit/tools/test_vera_rag_kb.py`;
- `tests/unit/clients/test_rag_client.py`;
- `tests/integration/test_protocol_compatibility.py`;
- `tests/integration/test_rag_contract.py`.

Проверки:

- [x] Известный входящий `traceparent` → `mcp.execute.vera_rag_kb` имеет тот же trace id и remote parent id.
- [x] Исходящий запрос в `httpx.MockTransport` содержит новый `traceparent`, чей parent — MCP span.
- [x] Параллельные MCP calls не смешивают headers/context.
- [x] Успех, пустой результат и RAG unavailable имеют разные outcome.
- [x] Текст query не попадает в span attributes.
- [x] Публичная schema `vera_rag_kb(query, audience)` не изменилась из-за параметра `Context`.
- [x] Реальный `MultiServerMCPClient` по streamable-http сохраняет один trace id.
- [x] Shutdown делает force flush.

## 6. RAG Service — минимальный план

Репозиторий: `D:\BKS.Lab\python\my_projects\vera_rag_service`.

RAG подключается к Phoenix не для показа отдельного пользовательского диалога, а для объяснения общей задержки и результата retrieval. Детальные стадии остаются в `search_logs`; в Phoenix создаётся один компактный span.

### 6.1. Зависимости и настройки

Файлы:

- `requirements.txt`;
- `app/core/settings.py`;
- `.env.example`;
- `docker-compose.yml`.

Правки:

- [x] Добавить согласованные версии `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`.
- [x] Добавить `ObservabilitySettings` с endpoint, enabled и project name.
- [x] Не добавлять `PHOENIX_CAPTURE_CONTENT`: RAG по принятой политике не экспортирует пользовательский query/chunk text независимо от Agent-настройки.
- [x] Настроить доступ контейнера к общему Phoenix через существующую production/local топологию.
- [x] Не поднимать собственный Phoenix.

### 6.2. Модуль tracing

Новый файл: `app/observability/tracing.py`.

- [x] Реализовать тот же базовый provider/exporter contract, что в Agent/MCP.
- [x] `service.name=vera_rag_service`.
- [x] Exporter header `x-project-name`.
- [x] `get_tracer()`, `force_flush_tracing()`, `shutdown_tracing()`.
- [x] Сохранить Phoenix мягкой зависимостью.

### 6.3. Извлечение входящего `traceparent`

Файл: `app/main.py`.

- [x] Добавить небольшой HTTP middleware, который вызывает `propagate.extract(request.headers)` и активирует извлечённый context только на время `call_next`.
- [x] Обязательно сбрасывать/отсоединять context token в `finally`, чтобы контекст одного запроса не утёк в следующий.
- [x] Не создавать автоматические spans для `/health`, `/metrics`, `/admin`, `/static`.
- [x] Не включать тотальную `FastAPIInstrumentor`, если единственная цель — propagation для `POST /api/v1/search`.
- [x] Существующий `X-Request-ID` middleware оставить; `request_id` и OTel `trace_id` выполняют разные задачи.
- [x] В lifespan настроить tracing до приёма запросов и flush/shutdown при остановке после завершения активной работы.

### 6.4. Один `rag.search`

Файл: `app/services/search.py`.

- [x] Обернуть `search_with_diagnostics()` одним span `rag.search`.
- [x] Если метод вызван через MCP/HTTP, span наследует извлечённый remote context.
- [x] Если метод вызван из админки/скрипта без входящего контекста, span может стать новым root в том же project.
- [x] Использовать существующие `perf_counter` latency, не дублировать таймеры.
- [x] После query expansion записать status и variant count.
- [x] После embedding записать latency, но не vector values.
- [x] После hybrid search записать dense/sparse/RRF counts.
- [x] После reranker записать status, latency и result count.
- [x] Централизовать финальное заполнение span attributes в helper/`finally`, чтобы ранний возврат при `no_candidates` и исключение на промежуточной стадии не оставляли противоречивые атрибуты.
- [x] Сформировать `rag.outcome`: `empty`, если нет результата; `degraded`, если хотя бы один штатный fallback; иначе `ok`.
- [x] Ошибки Embedding API/неперехваченные pipeline errors записывать как exception/ERROR и пробрасывать по существующему контракту.
- [x] Ошибка записи `search_logs`, которую сервис уже умеет переживать, не должна делать `rag.search` ошибочным; записать `rag.search_log.status=unavailable`.
- [x] Не добавлять четыре дочерних stage spans на первом этапе; latency хранить атрибутами одного `rag.search`.

### 6.5. Корреляция с `search_logs`

- [x] Записать текущий `request_id` в span attribute `request.id`.
- [x] Убедиться, что то же значение сохраняется существующим `SearchLog`.
- [x] Не добавлять миграцию БД для OTel trace id в этом этапе.
- [x] Если позже понадобится ссылка из админки прямо в Phoenix, вынести `trace_id`-колонку и миграцию в отдельную задачу.

### 6.6. Тесты RAG Service

Файлы:

- новый `tests/unit/observability/test_tracing.py`;
- `tests/unit/services/test_search_service.py`;
- `tests/api/endpoints/test_search.py`;
- `tests/api/endpoints/test_request_id.py`;
- новый интеграционный тест propagation при необходимости.

Проверки:

- [x] Входящий известный `traceparent` → `rag.search` сохраняет тот же trace id и remote parent.
- [x] Без `traceparent` поиск работает и создаёт самостоятельный trace.
- [x] Success: counts, latencies и statuses заполнены.
- [x] No candidates/no relevant: outcome `empty`, не ERROR.
- [x] Query expansion/reranker fallback: outcome `degraded`, не ERROR.
- [x] Embedding terminal failure: ERROR + recorded exception.
- [x] Search log repository failure: поиск успешен, span не ERROR.
- [x] В spans отсутствуют query text, embedding vector, prompts и chunk text.
- [x] Два параллельных HTTP-запроса не смешивают trace ids/request ids.
- [x] Shutdown делает force flush.

## 7. Изменения документации

### Agent Service

- [x] Обновить `README.md`: показать новое минимальное дерево и privacy flag.
- [x] Обновить `AGENT_SERVICE_PLAN.md`: прежняя проверка искусственного дерева не считается доказательством distributed trace.
- [x] Обновить `AGENT_VERA_ARCHITECTURE.md`: убрать утверждение, что `rabbitmq.consume` измеряет время ожидания в очереди, если publisher timestamp не передаётся.
- [x] Описать единый Phoenix project и имена spans.

### MCP Service

- [x] Обновить `README.md` и `MCP_SERVICE_PLAN.md`: зафиксировать W3C propagation и роль MCP span.
- [x] Удалить утверждение, что общий Phoenix instance сам по себе гарантирует единое дерево: общий collector без context propagation этого не делает.

### RAG Service

- [x] Обновить `README.md` и `RAG_SERVICE_PLAN.md`: прежний отказ от локального standalone tracing заменён минимальным участием в продуктовом distributed trace.
- [x] Подчеркнуть, что `search_logs` остаётся детальным журналом retrieval, Phoenix его не заменяет.

## 8. Порядок реализации

### Этап 1 — читаемый Agent trace

- [x] Project name и privacy settings в Agent.
- [x] Корневой `vera.agent.request`.
- [x] Агрегаты route/search/retry/response.
- [x] Удаление per-token `sse.deliver`.
- [x] Force flush.
- [x] Юнит- и observability/contract-интеграционные тесты Agent.

Результат: Phoenix уже становится заметно понятнее даже до изменений соседних сервисов.

### Этап 2 — Agent → MCP

- [x] Динамический interceptor injection в Agent.
- [x] Context extraction и `mcp.execute.vera_rag_kb` в MCP.
- [x] Единый project name.
- [x] Force flush MCP.
- [x] Протокольный интеграционный тест.

Результат: Agent и MCP видны одним trace.

### Этап 3 — минимальный RAG

- [x] Injection MCP → RAG.
- [x] Extraction в RAG middleware.
- [x] Один `rag.search` с агрегатами.
- [x] Корреляция `request.id` с `search_logs`.
- [x] Тесты RAG.

Результат: один trace проходит все три сервиса, но остаётся компактным.

### Этап 4 — сквозная проверка и документация

- [ ] Поднять Phoenix, Agent, MCP, RAG, RabbitMQ и Redis.
- [ ] Выполнить сценарии раздела 10.4.
- [ ] Проверить trace через Phoenix UI/API.
- [x] Обновить три README и архитектурные планы.
- [ ] Снять фактическое число spans на один запрос до/после и записать результат.

## 9. Команды локальной проверки

Запускать из соответствующего репозитория и его venv.

### Agent Service

```powershell
.\venv\Scripts\python.exe -m pytest tests\unit
.\venv\Scripts\python.exe -m pytest tests\integration\test_mcp_client.py tests\integration\test_graph.py tests\integration\test_consumer_sse_pipeline.py
.\venv\Scripts\python.exe -m ruff check .
```

### MCP Service

```powershell
.\venv\Scripts\python.exe -m pytest tests\unit
.\venv\Scripts\python.exe -m pytest tests\integration\test_protocol_compatibility.py tests\integration\test_rag_contract.py
.\venv\Scripts\python.exe -m ruff check .
```

### RAG Service

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit
.\.venv\Scripts\python.exe -m pytest tests\api\endpoints\test_search.py tests\api\endpoints\test_request_id.py
.\.venv\Scripts\python.exe -m ruff check .
```

Полные интеграционные тесты RAG запускать только с нужной инфраструктурой Postgres/Qdrant согласно его README.

## 10. Матрица проверок

### 10.1. Юнит-уровень

| Сервис | Проверка | Критерий |
|---|---|---|
| Agent | root span | один `vera.agent.request` |
| Agent | content policy | текст есть только при capture=true |
| Agent | SSE | N chunks не создают N spans |
| Agent | MCP injection | валидный динамический `traceparent` |
| MCP | extraction | trace id совпал с переданным |
| MCP | RAG injection | исходящий parent соответствует MCP span |
| RAG | extraction | `rag.search` — потомок remote span |
| RAG | privacy | query/chunks отсутствуют в attributes |
| Все | shutdown | `force_flush` вызван один раз |

### 10.2. Контрактный уровень

- [x] Публичная MCP schema не изменилась.
- [x] Формат `{'chunks': [...]}` не изменился.
- [x] RabbitMQ payload и SSE payload не изменились.
- [x] RAG `POST /api/v1/search` и авторизация `X-API-Key` не изменились.
- [ ] Phoenix disabled не меняет функциональные ответы сервисов.

### 10.3. Интеграционный trace-контракт

На автоматическом уровне проверить каждый сетевой hop отдельно с известным parent context:

- Agent test подтверждает injection в MCP headers;
- MCP test подтверждает extraction входящего parent и injection нового parent в RAG headers;
- RAG test подтверждает extraction и parent relation `rag.search`.

In-memory exporters разных процессов не объединяются. Полное дерево трёх реальных процессов подтверждается только живым E2E через общий OTLP/Phoenix.

Ожидаемая цепочка:

```text
parent span
└── tool.vera_rag_kb                 Agent
    └── mcp.execute.vera_rag_kb      MCP
        └── rag.search               RAG
```

Для каждого дочернего span проверить:

- одинаковый `trace_id`;
- ожидаемый `parent.span_id`;
- корректный `service.name`;
- одинаковый Phoenix project routing config;
- отсутствие пользовательского текста вне разрешённого Agent root.

### 10.4. Живые E2E-сценарии

1. **Прямой вопрос без RAG**
   - один root;
   - нет MCP/RAG spans;
   - output завершён;
   - нет `sse.deliver` на токены.

2. **Вопрос по базе знаний с результатом**
   - один trace через три service names;
   - `rag.result_chunk_count > 0`;
   - Agent outcome `done`.

3. **Поиск без релевантных документов**
   - `rag.outcome=empty`;
   - trace не ERROR;
   - Agent даёт предусмотренный ответ «в базе нет ответа».

4. **Query expansion или reranker fallback**
   - `rag.outcome=degraded`;
   - причина видна в status attributes;
   - пользователь получает ответ.

5. **RAG недоступен**
   - MCP span ERROR;
   - Agent root `degraded`, если деградация обработана;
   - один trace не распадается на независимые trace ids до точки отказа.

6. **Ошибка после начала SSE**
   - нет повторной генерации;
   - Agent root ERROR;
   - `agent.streaming.started=true`.

7. **Параллельные запросы двух сессий**
   - разные trace ids;
   - session ids и ContextVar-агрегаты не смешиваются;
   - MCP headers соответствуют своим активным spans.

8. **Штатная остановка процессов**
   - последние завершённые traces появляются в Phoenix после stop;
   - в логах нет необработанной ошибки flush.

Для сценария проверки input/output локальный/E2E-контур запускается с `PHOENIX_CAPTURE_CONTENT=true` и тестовыми данными без персональной информации. Отдельно повторить запрос с `false` и подтвердить отсутствие текстов.

### 10.5. Проверка в Phoenix

- [ ] Открыть project текущего окружения (`vera-local`/`vera-testing`/`vera-production`).
- [ ] Найти запрос по `session.id`.
- [ ] Убедиться, что сверху виден `vera.agent.request`, а не технический span.
- [ ] Убедиться, что Agent/MCP/RAG находятся в одном trace и различаются по `service.name`.
- [ ] Убедиться, что на один ответ отсутствуют десятки `sse.deliver`.
- [ ] Проверить input/output в разрешённом окружении и их отсутствие при capture=false.
- [ ] Проверить, что `/health` и `/metrics` не засоряют AI project.
- [ ] Сравнить количество spans до/после на одинаковом запросе.

## 11. Риски и меры

| Риск | Мера |
|---|---|
| Глобальный `TracerProvider` нельзя безопасно заменить повторно | Идемпотентная инициализация; один provider на тестовый процесс |
| `shutdown()` остановит общий provider посреди набора тестов | В production вызывать shutdown только при завершении процесса; тесты проверяют helper на mock/provider и не завершают общий provider между кейсами |
| Статический MCP header привяжет все calls к неправильному trace | Только динамический `ToolCallInterceptor` |
| ContextVar/attach context утечёт между async requests | Всегда reset/detach в `finally`; параллельные тесты |
| Retry создаст несколько несвязанных roots | Один логический span вокруг цикла retry, попытки — счётчик/events |
| Полные тексты попадут в Phoenix | Capture flag; RAG всегда без query/chunk content |
| Автоинструментация снова создаст шум | Ручные границы; LangChain auto оставить только в Agent |
| MCP SDK не отдаст HTTP request в tool `Context` | Сначала протокольный spike/test; fallback на ASGI middleware без патча SDK |
| BatchSpanProcessor потеряет хвост при stop | `force_flush` + shutdown во всех трёх сервисах |
| Phoenix недоступен | Неблокирующий exporter; приложение продолжает работать |
| Высокая кардинальность `session.id` | Только span attribute, не resource attribute |
| Разные project names разделят trace в UI | Одинаковая переменная и E2E-проверка конфигурации |

## 12. Файлы, которые предположительно изменятся

### `vera_agent_service`

```text
.env.example
README.md
AGENT_SERVICE_PLAN.md
AGENT_VERA_ARCHITECTURE.md
app/core/settings.py
app/main.py
app/observability/tracing.py
app/observability/request_trace.py        # новый
app/messaging/consumer.py
app/clients/mcp_client.py
app/streaming/session_bus.py
app/graph/nodes/analyze_intent.py
app/graph/nodes/call_kb_search.py
tests/unit/observability/test_tracing.py
tests/unit/observability/test_request_trace.py  # новый
tests/unit/messaging/test_consumer.py
tests/unit/clients/test_mcp_client.py
tests/unit/graph/test_analyze_intent.py
tests/unit/graph/test_call_kb_search.py
tests/integration/test_mcp_client.py
tests/integration/test_graph.py
tests/integration/test_consumer_sse_pipeline.py
```

### `vera_mcp_service`

```text
.env.example
README.md
MCP_SERVICE_PLAN.md
requirements.txt
docker-compose.yml
app/core/settings.py
app/main.py
app/observability/tracing.py
app/tools/vera_rag_kb.py
app/clients/rag_client.py
tests/unit/observability/test_tracing.py
tests/unit/tools/test_vera_rag_kb.py
tests/unit/clients/test_rag_client.py
tests/integration/test_protocol_compatibility.py
tests/integration/test_rag_contract.py
```

### `vera_rag_service`

```text
.env.example
README.md
RAG_SERVICE_PLAN.md
requirements.txt
docker-compose.yml
app/core/settings.py
app/main.py
app/observability/__init__.py              # новый
app/observability/tracing.py               # новый
app/services/search.py
tests/unit/observability/__init__.py       # новый
tests/unit/observability/test_tracing.py   # новый
tests/unit/services/test_search_service.py
tests/api/endpoints/test_search.py
tests/api/endpoints/test_request_id.py
```

Список уточняется во время реализации, но публичные API-контракты менять не планируется.

## 13. Состояние рабочих деревьев перед реализацией

На момент создания плана уже есть пользовательские незакоммиченные изменения:

- `vera_agent_service` — изменён ряд файлов синхронизации Agent/MCP;
- `vera_mcp_service` — рабочее дерево было чистым при последней проверке;
- `vera_rag_service` — изменены `.env.example`, `.gitignore`, присутствует незатреканный `count_articles.py`.

Перед началом каждого этапа обязательно:

- [x] повторно выполнить `git status --short` в нужном репозитории;
- [x] не перезаписывать и не откатывать пользовательские изменения;
- [x] сверить пересекающиеся файлы перед `apply_patch`;
- [ ] коммитить три репозитория раздельно, если пользователь позже попросит commit/push.

## 14. Справочные источники

- Phoenix: настройка проектов и `x-project-name` — <https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-projects>
- Phoenix: OTLP project routing, доступно с 15.5.0 — <https://arize.com/docs/phoenix/release-notes/05-2026/05-08-2026-otlp-project-header>
- OpenTelemetry Python: W3C context propagation и `inject`/`extract` — <https://opentelemetry.io/docs/languages/python/propagation/>
- OpenTelemetry Python: default propagators `tracecontext,baggage` — <https://opentelemetry.io/docs/languages/python/instrumentation/>

## 15. Следующая сессия — живая приёмка

Кодовых пунктов без живой инфраструктуры не осталось. Следующая сессия нужна не
для продолжения реализации «по предположению», а для проверки реального дерева и
фиксации измеряемого результата. Если приёмка найдёт дефект, только тогда добавить
точечную правку и regression-тест в соответствующий сервис.

### 15.0. Состояние передачи между сессиями

- [x] Agent Service: `ruff check app tests` — чисто; 87 автоматических тестов прошли. Единственный включённый в целевой прогон live-RabbitMQ тест не смог открыть AMQP-соединение из sandbox (`WinError 5`), а не упал на assertion кода.
- [x] MCP Service: 49 тестов прошли; `ruff check --no-cache .` — чисто.
- [x] RAG Service: 173 unit/API теста прошли; `ruff check --no-cache .` — чисто.
- [x] `git diff --check` прошёл во всех трёх репозиториях; существующие пользовательские изменения RAG (`.gitignore`, `count_articles.py` и прежняя часть `.env.example`) не откатывались.

### 15.1. Предварительные условия

- [ ] Убедиться, что доступны реальные LLM credentials Agent Service.
- [ ] Убедиться, что RAG Service штатно поднимает Postgres/Qdrant, применяет миграции и содержит хотя бы один тестовый документ. Если снова возникает `InvalidCatalogNameError`, сначала восстановить провижининг БД — без этого сценарий с результатом поиска непроверяем.
- [ ] Проверить общую Docker-сеть `vera_network` и DNS-имена контейнеров Agent/MCP/RAG/Phoenix.
- [ ] Задать во всех трёх процессах одинаковые `PHOENIX_OTLP_ENDPOINT` и `PHOENIX_PROJECT_NAME=vera-local`; оставить разные `service.name`.
- [ ] Для основного privacy-прогона установить `PHOENIX_CAPTURE_CONTENT=false`; для отдельного content-прогона использовать `true` только с синтетическим текстом без персональных данных.

### 15.2. Регрессия на реальной инфраструктуре

- [ ] Поднять Phoenix, RabbitMQ, Redis, Agent, MCP и RAG из их штатных compose-конфигураций.
- [ ] Выполнить полный `pytest tests/` Agent Service с доступными RabbitMQ/Redis и зафиксировать итог. В финальном целевом прогоне текущей сессии 87 тестов прошли, а live-RabbitMQ pipeline test был заблокирован сетевой политикой sandbox.
- [ ] Повторить полные тесты MCP и unit/API-набор RAG командами раздела 9 после старта окружения.
- [ ] Проверить `GET /health` всех сервисов до E2E, чтобы не смешивать сетевой/БД-блокер с ошибкой tracing.

### 15.3. E2E и Phoenix

- [ ] Отправить реальное сообщение через RabbitMQ и полностью прочитать SSE до `done`: один прямой вопрос без RAG и один вопрос с успешным поиском.
- [ ] Выполнить восемь сценариев раздела 10.4, включая empty, degraded, RAG unavailable, ошибку после начала SSE и две параллельные сессии.
- [ ] В Phoenix найти запрос по `session.id` и зафиксировать `trace_id`, span ids и `service.name` дерева `vera.agent.request → tool.vera_rag_kb → mcp.execute.vera_rag_kb → rag.search`.
- [ ] Подтвердить, что прямой ответ не содержит MCP/RAG spans, а SSE не создаёт per-token `sse.deliver`.
- [ ] Подтвердить privacy в двух режимах: при capture=false нет input/output; при capture=true синтетические input/output есть только на разрешённом Agent root. В MCP/RAG query/chunk text отсутствует в обоих режимах.
- [ ] Подтвердить, что `/health` и `/metrics` не засоряют AI project.
- [ ] Штатно остановить три процесса и убедиться, что последние завершённые spans появились в Phoenix, а в логах нет ошибки flush/shutdown.

### 15.4. Закрытие плана

- [ ] Записать в этот файл дату прогона, окружение, проверенный `trace_id`, число spans для прямого и RAG-запроса и фактический результат каждого сценария.
- [ ] Если доступен сохранённый trace старой реализации, записать число spans до/после на одном сценарии. Если старого trace нет, явно отметить отсутствие корректной baseline вместо выдуманного сравнения.
- [ ] После доказанного E2E отметить Definition of Done про неизменный `trace_id`, Этап 4 и пункты 10.5; до этого они намеренно остаются открытыми.
- [ ] Коммитить Agent, MCP и RAG раздельно только после просмотра diff и по отдельному подтверждению пользователя.
