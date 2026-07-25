import asyncio
import logging
import time

logger = logging.getLogger('vera_agent_service')

LATE_CONNECT_BUFFER_SECONDS: float = 60.0
"""Буфер для событий, опубликованных раньше, чем клиент подключился к SSE
(AGENT_SERVICE_PLAN.md, Этап 7.3/раздел 0.1) — предложение по умолчанию,
подлежит подтверждению (раздел 6 плана, открытый вопрос)."""


class SessionBus:
    """Реестр per-request `asyncio.Queue` для доставки токенов в SSE
    (Этап 7).

    Наполняется RabbitMQ-consumer'ом (Этап 6) через метод `publish` —
    сигнатура совпадает с `TokenSink`-протоколом
    (`app/messaging/consumer.py`), читается SSE-эндпоинтом
    (`app/streaming/sse.py`).

    Один активный `asyncio.Queue` на `request_id`. `session_id` при этом
    остаётся ключом истории LangGraph и сюда не попадает.

    Просроченные буферы запросов без подписчика очищаются при следующей
    публикации. Это ограничивает память окном late-connect без отдельной
    фоновой задачи.
    """

    def __init__(self, buffer_seconds: float = LATE_CONNECT_BUFFER_SECONDS):
        self._buffer_seconds = buffer_seconds
        self._queues: dict[str, asyncio.Queue] = {}
        self._buffers: dict[str, list[tuple[float, dict]]] = {}

    def subscribe(self, request_id: str) -> asyncio.Queue:
        """Регистрирует очередь для запроса и сразу отдаёт в неё события,
        буферизованные до подключения (Этап 7.3) — если они не истекли."""
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[request_id] = queue

        buffered = self._buffers.pop(request_id, [])
        now = time.monotonic()
        for buffered_at, event in buffered:
            if now - buffered_at <= self._buffer_seconds:
                queue.put_nowait(event)
            else:
                logger.warning(
                    '⚠️ Буферизованное событие запроса %s отброшено — истёк буфер позднего подключения',
                    request_id,
                )
        return queue

    def _prune_expired_buffers(self, now: float) -> None:
        expired_request_ids = [
            buffered_request_id
            for buffered_request_id, events in self._buffers.items()
            if events and now - events[-1][0] > self._buffer_seconds
        ]
        for expired_request_id in expired_request_ids:
            del self._buffers[expired_request_id]
            logger.warning(
                '⚠️ Просроченный буфер запроса %s очищен без подключения клиента',
                expired_request_id,
            )

    def unsubscribe(self, request_id: str, queue: asyncio.Queue) -> None:
        """Отписывает очередь, только если она всё ещё текущая — не была
        замещена новой подпиской (например второй вкладкой)."""
        if self._queues.get(request_id) is queue:
            del self._queues[request_id]

    async def publish(self, request_id: str, event: dict) -> None:
        """`TokenSink`-совместимый метод (см. `app/messaging/consumer.py`).

        Публикует событие в очередь активного подписчика; если подписчика
        ещё нет (consumer начал стриминг раньше, чем клиент открыл SSE) —
        буферизует до `LATE_CONNECT_BUFFER_SECONDS`.
        """
        now = time.monotonic()
        self._prune_expired_buffers(now)

        queue = self._queues.get(request_id)
        if queue is not None:
            await queue.put(event)
            return
        self._buffers.setdefault(request_id, []).append((now, event))
