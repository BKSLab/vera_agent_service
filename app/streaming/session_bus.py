import asyncio
import logging
import time

logger = logging.getLogger('vera_agent_service')

LATE_CONNECT_BUFFER_SECONDS: float = 60.0
"""Буфер для событий, опубликованных раньше, чем клиент подключился к SSE
(AGENT_SERVICE_PLAN.md, Этап 7.3/раздел 0.1) — предложение по умолчанию,
подлежит подтверждению (раздел 6 плана, открытый вопрос)."""


class SessionBus:
    """Реестр per-session `asyncio.Queue` для доставки токенов в SSE
    (Этап 7).

    Наполняется RabbitMQ-consumer'ом (Этап 6) через метод `publish` —
    сигнатура совпадает с `TokenSink`-протоколом
    (`app/messaging/consumer.py`), читается SSE-эндпоинтом
    (`app/streaming/sse.py`).

    Один активный `asyncio.Queue` на `session_id` (раздел 0.1: допущение
    "один инстанс" — не масштабируется на несколько реплик без перехода на
    Redis Pub/Sub, отдельная будущая задача). Повторная подписка
    (например вторая вкладка) замещает предыдущую.

    Известное ограничение: буфер незабранных событий сессии, к которой
    никто ни разу не подключился, не очищается сам по себе (только когда
    кто-то подписывается — тогда просроченные записи отбрасываются). Для
    итерации 1 это не критично (короткий буфер, единичный инстанс,
    невысокая нагрузка) — при необходимости ужесточить, потребуется
    отдельная фоновая задача-уборщик.
    """

    def __init__(self, buffer_seconds: float = LATE_CONNECT_BUFFER_SECONDS):
        self._buffer_seconds = buffer_seconds
        self._queues: dict[str, asyncio.Queue] = {}
        self._buffers: dict[str, list[tuple[float, dict]]] = {}

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Регистрирует очередь для сессии и сразу отдаёт в неё события,
        буферизованные до подключения (Этап 7.3) — если они не истекли."""
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[session_id] = queue

        buffered = self._buffers.pop(session_id, [])
        now = time.monotonic()
        for buffered_at, event in buffered:
            if now - buffered_at <= self._buffer_seconds:
                queue.put_nowait(event)
            else:
                logger.warning(
                    '⚠️ Буферизованное событие сессии %s отброшено — истёк буфер позднего подключения',
                    session_id,
                )
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """Отписывает очередь, только если она всё ещё текущая — не была
        замещена новой подпиской (например второй вкладкой)."""
        if self._queues.get(session_id) is queue:
            del self._queues[session_id]

    async def publish(self, session_id: str, event: dict) -> None:
        """`TokenSink`-совместимый метод (см. `app/messaging/consumer.py`).

        Публикует событие в очередь активного подписчика; если подписчика
        ещё нет (consumer начал стриминг раньше, чем клиент открыл SSE) —
        буферизует до `LATE_CONNECT_BUFFER_SECONDS`.
        """
        queue = self._queues.get(session_id)
        if queue is not None:
            await queue.put(event)
            return
        self._buffers.setdefault(session_id, []).append((time.monotonic(), event))
