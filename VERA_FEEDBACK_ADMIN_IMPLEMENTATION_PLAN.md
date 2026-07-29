# План реализации хранения диалогов, обратной связи и админки

Основание: `VERA_FEEDBACK_ADMIN_TZ.md`, `FASTAPI_PATTERNS.md` и реализация
БД/SQLAdmin в `vera_rag_service`.

## 1. Зависимости и настройки БД

1. Добавить `sqlalchemy[asyncio]`, `asyncpg`, `alembic`, `sqladmin`,
   `WTForms` и `itsdangerous`.
2. Добавить `DBSettings` и настройки входа в админку.
3. Добавить `app/db/session.py` с async engine и session factory.
4. Добавить `app/dependencies/db_session.py`.
5. Подключить уже созданную PostgreSQL из `.env`.
6. В `entrypoint.sh` выполнять `alembic upgrade head` перед запуском сервиса.

## 2. Четыре таблицы

Добавить ORM-модели и Alembic-миграции:

1. `vera_chat_sessions`:
   `id`, `session_id`, `user_id`, `created_at`, `last_activity_at`.
2. `vera_chat_turns`:
   `id`, FK на сессию, уникальный `request_id`, порядковый номер,
   `user_id`, вопрос, ответ, sources JSONB, технические метаданные JSONB,
   статус, безопасная ошибка, timestamps и latency.
3. `vera_message_feedback`:
   `id`, уникальный FK на реплику, `up/down`, экспертный статус,
   заметка, теги и timestamps.
4. `vera_session_feedback`:
   `id`, FK на сессию, уникальный `submission_id`, audience, usefulness,
   trust, comment, contact email, экспертный статус, заметка, теги и
   timestamps.

Для каждой модели:

- отдельный файл в `app/db/models`;
- отдельная миграция;
- constraints и индексы из ТЗ;
- timestamps на стороне PostgreSQL.

## 3. Репозитории и сервисы

Добавить repositories:

- `ChatSessionRepository`;
- `ChatTurnRepository`;
- `MessageFeedbackRepository`;
- `SessionFeedbackRepository`.

Добавить services:

- `ChatPersistenceService` — идемпотентно сохраняет сессию и реплику;
- `MessageFeedbackService` — создаёт или изменяет оценку ответа;
- `SessionFeedbackService` — сохраняет развёрнутый отзыв.

Зависимости собрать по существующему паттерну:

```text
Endpoint → Service → Repository → PostgreSQL
```

Отдельный Unit of Work и дополнительные абстракции не нужны.

## 4. Сохранение диалогов

Точечно дополнить текущий `AgentRequestConsumer`:

1. После разбора RabbitMQ payload создать сессию и реплику со статусом
   `processing`.
2. По завершении генерации сохранить фактически отправленный ответ,
   sources, маршрут обработки, latency и статус `completed`.
3. При ошибке сохранить `failed` или `delivery_unconfirmed`.
4. По уникальному `request_id` не создавать дубль и не запускать повторную
   генерацию уже завершённой реплики.

Redis, RabbitMQ, LangGraph и SSE не переделываются.

## 5. Два endpoint

### Оценка ответа

Один endpoint создаёт или изменяет `up/down` по `session_id` и
`request_id`.

Проверки:

- реплика существует;
- относится к переданной сессии;
- имеет статус `completed`;
- повторный запрос обновляет существующую оценку, а не создаёт новую.

### Развёрнутый отзыв

Один endpoint сохраняет анкету по `session_id` и `submission_id`.

Проверки:

- сессия существует;
- рейтинги находятся в диапазоне 1–5;
- строки имеют ограничения длины;
- повторный `submission_id` не создаёт дубль.

Оба endpoint:

- используют Pydantic-схемы;
- получают готовый service через DI;
- не принимают вопрос и ответ от сайта;
- документируются в OpenAPI.

## 6. Админка

Перенести минимальный каркас SQLAdmin из `vera_rag_service`:

- `app/admin/__init__.py`;
- `app/admin/auth.py`;
- `app/admin/views.py`;
- шаблон `sqladmin/base.html`;
- `admin-theme.css`.

Добавить:

1. Список сессий.
2. Список реплик.
3. Список оценок.
4. Список развёрнутых отзывов.
5. Детальный просмотр реплики.
6. Детальный просмотр сессии с хронологией вопросов и ответов.
7. Переход из оценки/отзыва к связанной реплике или сессии.

Исходные вопросы, ответы и отзывы через админку не редактируются. Для
экспертной обработки разрешается менять только статус, заметку и теги.

## 7. Тесты

Минимальный набор:

1. Integration-тесты repositories на PostgreSQL через Testcontainers.
2. Unit-тесты трёх services.
3. API-тесты двух endpoint через `dependency_overrides`.
4. Тест идемпотентности `request_id` и `submission_id`.
5. Тест изменения `up → down` без второй строки.
6. Тест сохранения завершённой и ошибочной реплики.
7. Тест запрета доступа к `/admin` без авторизации.
8. Тест открытия полной сессии из оценки.

## 8. Порядок реализации

1. Настройки БД, async session и Alembic.
2. Четыре модели и миграции.
3. Четыре repositories и DI.
4. Три services.
5. Сохранение реплик в consumer.
6. Два endpoint.
7. SQLAdmin.
8. Тесты и краткое обновление `ADMIN_GUIDE.md`.

Вне объёма: email, фоновые задачи, новый broker, отдельный frontend,
экспорт, BI, автоматическая оценка ответов и другие дополнительные
механизмы.
