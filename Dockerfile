FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# В отличие от vera_rag_service здесь нет asyncpg/зависимостей с C-расширениями
# без готовых wheel-пакетов под python:3.12-slim (fastapi/langgraph/aio-pika/
# httpx/redis и т.д. — чистый Python или manylinux-wheels), поэтому build-тулчейн
# (gcc/libpq-dev) в этом Dockerfile не нужен вовсе — не только в runtime-образе.

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Только runtime-зависимости (tzdata, curl для HEALTHCHECK) — без
# dev-зависимостей (pytest/ruff/testcontainers, см. requirements-dev.txt) —
# меньше образ, меньше поверхность атаки при компрометации процесса.
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata curl && \
    ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    echo "Europe/Moscow" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# Непривилегированный пользователь — при компрометации процесса атакующий
# не получает root внутри контейнера.
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
