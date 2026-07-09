#!/bin/bash

set -e

# Добавляем в PYTHONPATH и корневую директорию проекта (/app),
# и директорию с исходным кодом (/app/app).
# Это позволяет и команде запуска найти app.main, и самому приложению
# найти свои внутренние модули (api, core и т.д.).
export PYTHONPATH="/app:/app/app:${PYTHONPATH}"

# Нет реляционной БД/миграций в этом сервисе (AGENT_SERVICE_PLAN.md,
# раздел 0.1 — Redis используется только как checkpointer с TTL, не как
# хранилище со схемой) — шаг alembic upgrade head здесь не нужен.

# ВАЖНО: HYPERCORN_WORKERS должен оставаться 1 (см. .env.example) —
# per-session SSE-очередь и RabbitMQ-consumer живут в памяти одного
# процесса (AGENT_SERVICE_PLAN.md, раздел 0.1, допущение "один инстанс").
echo "Starting in production mode..."
exec hypercorn app.main:app --bind 0.0.0.0:8000 --workers "${HYPERCORN_WORKERS:-1}"
