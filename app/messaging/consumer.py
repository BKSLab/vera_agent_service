import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from os import getpid
from socket import gethostname

import aio_pika
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import ValidationError

from app.db.models.chat_turn import (
    STATUS_COMPLETED,
    STATUS_DELIVERY_UNCONFIRMED,
    STATUS_GENERATION_FAILED,
    STATUS_STREAM_INTERRUPTED,
)
from app.exceptions.chat_session import ChatSessionAccessDeniedError
from app.exceptions.chat_turn import (
    ChatPersistenceServiceError,
    ChatTurnSessionMismatchError,
)
from app.exceptions.llm import EmptyLlmStreamError
from app.exceptions.messaging import InvalidAgentRequestError
from app.graph.policy import UNSAFE_TOOL_CALL_RESPONSE, contains_pseudo_tool_call
from app.messaging.schemas import AgentRequestMessage
from app.observability.request_trace import (
    AgentRequestTraceData,
    reset_request_trace,
    set_request_trace,
)
from app.observability.tracing import get_tracer
from app.services.chat_persistence import (
    START_DUPLICATE_IN_PROGRESS,
    START_DUPLICATE_TERMINAL,
    ChatPersistenceService,
    TurnStartResult,
)

logger = logging.getLogger('vera_agent_service')

DEFAULT_RETRIES: int = 3
DEFAULT_RETRY_DELAY: float = 1.0
DEFAULT_MAX_RETRY_DELAY: float = 30.0
JITTER_RATIO: float = 0.1

DEFAULT_PERSISTENCE_RETRIES: int = 3
"""Повторы только для сохранения результата: они не переигрывают граф и
поэтому безопасны даже после мутирующего инструмента."""

DEFAULT_LEASE_SECONDS: float = 900.0
"""Срок аренды реплики. Должен с запасом превышать самый долгий инструмент
(отправка консультации — до 360 секунд), иначе живую обработку перехватит
повторная доставка того же `request_id`."""

GENERATION_FAILED_MESSAGE = 'Сервис временно недоступен, попробуйте позже.'
PERSISTENCE_UNAVAILABLE_MESSAGE = 'Сервис временно недоступен, попробуйте позже.'
STREAM_INTERRUPTED_MESSAGE = 'Произошла ошибка при формировании ответа.'
DUPLICATE_IN_PROGRESS_MESSAGE = 'Этот запрос уже обрабатывается.'
COMMIT_FAILED_MESSAGE = (
    'Ответ подготовлен, но его не удалось сохранить. '
    'Проверьте историю диалога позже.'
)
SHUTDOWN_MESSAGE = (
    'Обработка прервана перезапуском сервиса. Попробуйте повторить запрос.'
)
STALE_TURN_MESSAGE = (
    'Обработка запроса была прервана. Попробуйте повторить вопрос.'
)
"""Текст для реплик, брошенных упавшим процессом и закрытых при старте."""

MUTATING_TOOL_UNCONFIRMED_MESSAGE = (
    'Не удалось подтвердить результат отправки консультации. '
    'Проверьте почту перед новой попыткой.'
)

FINAL_RESPONSE_NODES = frozenset({'generate_direct', 'generate_with_context'})
"""Только эти узлы формируют пользовательский ответ.

`analyze_intent` тоже вызывает chat-модель и может породить
`on_chat_model_stream`, но его текст является внутренним результатом маршрутизации
и не должен попадать в SSE.
"""

TokenSink = Callable[[str, dict], Awaitable[None]]
"""Принимает `(request_id, событие)`. Событие — SSE-контракт (раздел 3.2
плана): `{"type": "token", "content": ...}` / `{"type": "done"}` /
`{"type": "error", "detail": ...}`. Конкретная реализация — `session_bus`
(Этап 7); здесь используется только через этот интерфейс, чтобы Этап 6
оставался тестируемым независимо от Этапа 7 (раздел 0 подхода к плану)."""

PersistenceServiceFactory = Callable[
    [],
    AbstractAsyncContextManager[ChatPersistenceService],
]


@dataclass
class _Delivery:
    """Гарантирует инварианты одной доставки RabbitMQ.

    Ровно один `ack`/`nack` и ровно одно терминальное SSE-событие на запрос:
    обе гарантии раньше держались на дисциплине каждой ветки кода, из-за чего
    неожиданное исключение оставляло delivery неподтверждённой, а клиента —
    без завершения потока (VERA-015).
    """

    message: aio_pika.abc.AbstractIncomingMessage
    token_sink: TokenSink
    request_id: str | None = None
    partial_answer: str | None = None
    _terminal_sent: bool = False
    _settled: bool = False
    _started_at: float = field(default_factory=time.monotonic)

    def bind(self, request_id: str) -> None:
        """Связывает доставку с адресуемым запросом."""
        self.request_id = request_id

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started_at) * 1000)

    async def send_terminal(self, event: dict) -> None:
        """Отправляет терминальное событие, если оно ещё не отправлялось."""
        if self._terminal_sent or self.request_id is None:
            return
        self._terminal_sent = True
        await self.token_sink(self.request_id, event)

    async def ack(self) -> None:
        if self._settled:
            return
        self._settled = True
        await self.message.ack()

    async def nack(self) -> None:
        """Отклоняет доставку без повторной постановки — в DLQ."""
        if self._settled:
            return
        self._settled = True
        await self.message.nack(requeue=False)

    async def settle_if_pending(self) -> None:
        """Страховка инварианта: неподтверждённых доставок не остаётся."""
        if self._settled:
            return
        logger.error(
            '❌ Доставка запроса %s осталась неподтверждённой — отправляем в DLQ',
            self.request_id,
        )
        await self.nack()


@dataclass
class TurnPersistenceData:
    """Данные ответа для постоянного хранения, отдельно от telemetry."""

    sources: list = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


def _get_backoff_delay(attempt: int) -> float:
    base_delay = min(DEFAULT_MAX_RETRY_DELAY, DEFAULT_RETRY_DELAY * (2 ** (attempt - 1)))
    jitter = base_delay * JITTER_RATIO * random.random()
    return base_delay + jitter


def _initial_state(payload: AgentRequestMessage) -> dict:
    return {
        'session_id': payload.session_id,
        'user_id': payload.user_id,
        'messages': [HumanMessage(content=payload.message, id=payload.request_id)],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


class AgentRequestConsumer:
    """Consumer очереди `agent.requests` (Этап 6, AGENT_VERA_ARCHITECTURE.md
    раздел "Интеграция с RabbitMQ").

    Retry-политика — **только для системных сбоев обработки сообщения**,
    до начала стриминга ответа клиенту (раздел 0.1 плана): реализована как
    вызов графа внутри одного и того же message delivery `retries` раз с
    экспоненциальным backoff, а не через broker-level повторную доставку с
    задержкой — plain RabbitMQ без дополнительных плагинов (`x-delayed-message`)
    не умеет отложенный requeue, а поднимать для этого отдельный плагин
    ради 3 попыток избыточно. `nack(requeue=False)` после исчерпания
    попыток уходит в `agent.requests.dlq` через `x-dead-letter-exchange`,
    объявленный на очереди.

    Ошибка **после** того как хотя бы один токен уже отдан в `token_sink` —
    не ретраится вообще (ни в рамках одной доставки, ни через DLQ):
    `ack`-ается как обработанное, SSE получает `error`-событие. Requeue
    сообщения, часть которого уже видел пользователь, создал бы
    дублирование/рассинхронизацию потока (раздел 0.1).
    """

    def __init__(
        self,
        connection_url: str,
        queue_name: str,
        dlq_name: str,
        graph: CompiledStateGraph,
        token_sink: TokenSink,
        persistence_service_factory: PersistenceServiceFactory | None = None,
        retries: int = DEFAULT_RETRIES,
        prefetch_count: int = 1,
        persistence_retries: int = DEFAULT_PERSISTENCE_RETRIES,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        worker_id: str | None = None,
        trace_content_enabled: bool = False,
    ):
        self._connection_url = connection_url
        self._queue_name = queue_name
        self._dlq_name = dlq_name
        self._graph = graph
        self._token_sink = token_sink
        self._persistence_service_factory = persistence_service_factory
        self._retries = retries
        self._prefetch_count = prefetch_count
        self._persistence_retries = persistence_retries
        self._lease_seconds = lease_seconds
        self._worker_id = worker_id or f'{gethostname()}:{getpid()}'
        self._trace_content_enabled = trace_content_enabled
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._queue: aio_pika.abc.AbstractQueue | None = None
        self._consumer_tag: str | None = None

    @property
    def is_connected(self) -> bool:
        """Для `GET /health` (Этап 8) — жёсткий статус RabbitMQ."""
        return self._connection is not None and not self._connection.is_closed

    async def start(self) -> None:
        """Подключается к RabbitMQ, объявляет очередь + DLQ (через
        dead-letter-exchange) и начинает потребление сообщений."""
        self._connection = await aio_pika.connect_robust(self._connection_url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._prefetch_count)

        dlx_name = f'{self._queue_name}.dlx'
        dlx = await self._channel.declare_exchange(dlx_name, aio_pika.ExchangeType.FANOUT, durable=True)
        dlq = await self._channel.declare_queue(self._dlq_name, durable=True)
        await dlq.bind(dlx)

        self._queue = await self._channel.declare_queue(
            self._queue_name,
            durable=True,
            arguments={'x-dead-letter-exchange': dlx_name},
        )
        consumer_tag = await self._queue.consume(self._handle_message)
        self._consumer_tag = consumer_tag
        logger.info('🚀 Consumer очереди %s запущен', self._queue_name)

    async def stop(self) -> None:
        if self._queue is not None and self._consumer_tag is not None:
            await self._queue.cancel(self._consumer_tag)
        if self._connection is not None:
            await self._connection.close()
        logger.info('✅ Consumer очереди %s остановлен', self._queue_name)

    async def _handle_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        """Один root span охватывает обработку delivery до терминального SSE и ack/nack."""
        trace_data = AgentRequestTraceData()
        context_token = set_request_trace(trace_data)
        delivery = _Delivery(message=message, token_sink=self._token_sink)
        try:
            with get_tracer().start_as_current_span(
                'vera.agent.request',
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                    'messaging.system': 'rabbitmq',
                    'messaging.destination.name': self._queue_name,
                },
            ) as span:
                try:
                    await self._handle_message_body(delivery, span, trace_data)
                except asyncio.CancelledError as error:
                    # Остановка сервиса во время обработки. Исход запроса
                    # действительно неизвестен, поэтому он и записывается как
                    # неопределённый, а не как ошибка генерации. Отмену не
                    # проглатываем — она должна дойти до задачи consumer.
                    trace_data.outcome = 'cancelled'
                    _mark_span_error(span, error)
                    await self._settle_shutdown(delivery)
                    raise
                except Exception as error:
                    # Сюда попадает только то, что не предусмотрено ветками
                    # ниже. Клиент всё равно обязан получить терминальное
                    # событие, а delivery — ровно одно подтверждение.
                    trace_data.outcome = 'error'
                    _mark_span_error(span, error)
                    logger.exception(
                        '❌ Непредвиденная ошибка обработки delivery очереди %s',
                        self._queue_name,
                    )
                    await delivery.send_terminal(
                        {'type': 'error', 'detail': GENERATION_FAILED_MESSAGE}
                    )
                    await delivery.nack()
                finally:
                    self._finalize_root_span(span, trace_data)
        finally:
            reset_request_trace(context_token)
            # Последняя страховка инварианта «ровно один ack/nack»: ни один
            # путь не имеет права оставить delivery неподтверждённым — при
            # prefetch_count=1 это остановило бы весь чат до разрыва
            # соединения с брокером.
            await delivery.settle_if_pending()

    async def _settle_shutdown(self, delivery: '_Delivery') -> None:
        """Фиксирует неопределённый исход при остановке сервиса."""
        if delivery.request_id is None:
            await delivery.nack()
            return
        await delivery.send_terminal({'type': 'error', 'detail': SHUTDOWN_MESSAGE})
        await self._fail_persistence(
            request_id=delivery.request_id,
            status=STATUS_DELIVERY_UNCONFIRMED,
            safe_error='ServiceShutdown',
            answer=delivery.partial_answer,
            latency_ms=delivery.elapsed_ms(),
            terminal_detail=SHUTDOWN_MESSAGE,
        )
        await delivery.nack()

    async def _handle_message_body(
        self,
        delivery: '_Delivery',
        span: Span,
        trace_data: AgentRequestTraceData,
    ) -> None:
        try:
            payload = _parse_payload(delivery.message.body)
        except InvalidAgentRequestError as error:
            # Терминальное SSE намеренно не отправляется: `request_id` из
            # невалидного сообщения не подтверждён ничем, и отправка по нему
            # позволила бы завершить чужой поток (VERA-016).
            logger.error('❌ Невалидный payload %s: %s', self._queue_name, error)
            trace_data.outcome = 'invalid_payload'
            _mark_span_error(span, error)
            await delivery.nack()
            return

        delivery.bind(payload.request_id)
        trace_data.request_id = payload.request_id
        span.set_attribute('session.id', payload.session_id)
        span.set_attribute('request.id', payload.request_id)
        span.set_attribute('user.authenticated', payload.user_id is not None)
        span.set_attribute('agent.input.char_count', len(payload.message))
        self._set_content(
            span,
            SpanAttributes.INPUT_VALUE,
            SpanAttributes.INPUT_MIME_TYPE,
            payload.message,
        )

        try:
            persistence_start = await self._start_persistence_with_retry(payload)
        except (
            ChatSessionAccessDeniedError,
            ChatTurnSessionMismatchError,
        ) as error:
            logger.error(
                '❌ Запрос не относится к доступной сессии. session_id=%s, request_id=%s.',
                payload.session_id,
                payload.request_id,
            )
            await delivery.send_terminal({'type': 'error', 'detail': error.detail})
            await delivery.nack()
            trace_data.outcome = 'invalid_payload'
            _mark_span_error(span, error)
            return
        except ChatPersistenceServiceError as error:
            logger.error(
                '❌ Не удалось зарегистрировать реплику в PostgreSQL после %d попыток. '
                'session_id=%s, request_id=%s.',
                self._persistence_retries,
                payload.session_id,
                payload.request_id,
            )
            await delivery.send_terminal(
                {'type': 'error', 'detail': PERSISTENCE_UNAVAILABLE_MESSAGE}
            )
            await delivery.nack()
            trace_data.outcome = 'persistence_error'
            _mark_span_error(span, error)
            return

        if persistence_start is not None:
            if persistence_start.outcome == START_DUPLICATE_TERMINAL:
                await self._replay_terminal_outcome(
                    delivery, persistence_start, span, trace_data
                )
                return
            if persistence_start.outcome == START_DUPLICATE_IN_PROGRESS:
                await delivery.send_terminal(
                    {'type': 'error', 'detail': DUPLICATE_IN_PROGRESS_MESSAGE}
                )
                await delivery.ack()
                trace_data.outcome = 'duplicate_processing'
                return

        await self._process_claimed_turn(delivery, payload, span, trace_data)

    async def _replay_terminal_outcome(
        self,
        delivery: '_Delivery',
        persistence_start: TurnStartResult,
        span: Span,
        trace_data: AgentRequestTraceData,
    ) -> None:
        """Повторяет сохранённый исход, а не выдаёт любой ответ за успех.

        Прежняя реализация воспроизводила сохранённый текст как `token` + `done`
        и для `delivery_unconfirmed`, превращая зафиксированную ошибку в успех
        (VERA-034). Успехом считается только `completed`.
        """
        if persistence_start.status == STATUS_COMPLETED:
            answer = persistence_start.answer or ''
            if answer:
                await self._token_sink(
                    delivery.request_id, {'type': 'token', 'content': answer}
                )
            await delivery.send_terminal(
                {
                    'type': 'done',
                    'used_knowledge_base': persistence_start.used_knowledge_base,
                }
            )
            trace_data.outcome = 'done'
            trace_data.streaming_started = bool(answer)
            trace_data.response_chunk_count = 1 if answer else 0
            trace_data.response_char_count = len(answer)
            self._set_content(
                span,
                SpanAttributes.OUTPUT_VALUE,
                SpanAttributes.OUTPUT_MIME_TYPE,
                answer,
            )
        else:
            await delivery.send_terminal(
                {
                    'type': 'error',
                    'detail': persistence_start.terminal_detail or GENERATION_FAILED_MESSAGE,
                }
            )
            trace_data.outcome = 'error'
        await delivery.ack()

    async def _process_claimed_turn(
        self,
        delivery: '_Delivery',
        payload: AgentRequestMessage,
        span: Span,
        trace_data: AgentRequestTraceData,
    ) -> None:
        """Выполняет граф и приводит delivery к ровно одному исходу."""
        last_error: Exception | None = None
        for attempt in range(1, self._retries + 1):
            streaming_started = False
            response_chunks: list[str] = []
            persistence_data = TurnPersistenceData()
            trace_data.request_retry_count = attempt - 1
            trace_data.streaming_started = False
            trace_data.response_chunk_count = 0
            trace_data.response_char_count = 0
            try:
                async for content in self._stream_answer(payload, persistence_data):
                    await self._token_sink(payload.request_id, {'type': 'token', 'content': content})
                    streaming_started = True
                    trace_data.streaming_started = True
                    response_chunks.append(content)
                    delivery.partial_answer = ''.join(response_chunks)
                    trace_data.response_chunk_count += 1
                    trace_data.response_char_count += len(content)
                answer = ''.join(response_chunks)
                if not answer:
                    # Пустой поток — это отсутствие ответа, а не успех
                    # (VERA-018). До первого токена повтор безопасен.
                    raise EmptyLlmStreamError

                committed = await self._complete_persistence(
                    request_id=payload.request_id,
                    answer=answer,
                    persistence_data=persistence_data,
                    trace_data=trace_data,
                    latency_ms=delivery.elapsed_ms(),
                )
                if not committed:
                    # Токены уже у пользователя, но durable-записи нет.
                    # `done` в этой ситуации соврал бы: история осталась бы
                    # незавершённой, а оценка ответа была бы недоступна
                    # (VERA-004).
                    await delivery.send_terminal(
                        {'type': 'error', 'detail': COMMIT_FAILED_MESSAGE}
                    )
                    await self._fail_persistence(
                        request_id=payload.request_id,
                        status=STATUS_DELIVERY_UNCONFIRMED,
                        safe_error='CompletePersistenceFailed',
                        answer=answer,
                        latency_ms=delivery.elapsed_ms(),
                        terminal_detail=COMMIT_FAILED_MESSAGE,
                    )
                    await delivery.ack()
                    trace_data.outcome = 'error'
                    return

                await delivery.send_terminal(
                    {
                        'type': 'done',
                        # Непустые чанки — единственный признак того, что
                        # ответ действительно опирается на базу знаний.
                        # Честное «не нашлось» и техническая недоступность
                        # поиска дают пустой список и здесь неотличимы от
                        # прямого ответа — это и требуется: предлагать
                        # упростить нечего.
                        'used_knowledge_base': bool(persistence_data.sources),
                    }
                )
                await delivery.ack()
                trace_data.outcome = (
                    'degraded'
                    if (
                        trace_data.search_unavailable
                        or trace_data.consultation_email_status == 'error'
                    )
                    else 'done'
                )
                self._set_content(
                    span,
                    SpanAttributes.OUTPUT_VALUE,
                    SpanAttributes.OUTPUT_MIME_TYPE,
                    answer,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - сбой графа тоже должен попасть сюда
                last_error = error
                partial_answer = ''.join(response_chunks) or None

                if isinstance(error, EmptyLlmStreamError) and not trace_data.mutating_tool_called:
                    # ``astream_tokens`` уже сделал bounded retries самой
                    # финальной LLM-операции. Повтор всего графа здесь только
                    # добавлял бы пустые AIMessage в checkpoint и повторял бы
                    # поиск/маршрутизацию без пользы.
                    logger.error(
                        '❌ LLM завершила финальный поток без видимого текста '
                        '(session_id=%s, request_id=%s)',
                        payload.session_id,
                        payload.request_id,
                    )
                    await self._finish_with_error(
                        delivery=delivery,
                        span=span,
                        trace_data=trace_data,
                        error=error,
                        status=STATUS_GENERATION_FAILED,
                        detail=GENERATION_FAILED_MESSAGE,
                        answer=None,
                    )
                    return

                if trace_data.mutating_tool_called:
                    logger.error(
                        '❌ Ошибка после начала мутирующего MCP-вызова; повтор графа запрещён '
                        '(session_id=%s, request_id=%s): %s',
                        payload.session_id,
                        payload.request_id,
                        type(error).__name__,
                    )
                    await self._finish_with_error(
                        delivery=delivery,
                        span=span,
                        trace_data=trace_data,
                        error=error,
                        status=STATUS_DELIVERY_UNCONFIRMED,
                        detail=MUTATING_TOOL_UNCONFIRMED_MESSAGE,
                        answer=partial_answer,
                    )
                    return

                if streaming_started:
                    logger.error(
                        '❌ Ошибка после начала стриминга (session_id=%s, request_id=%s): %s',
                        payload.session_id,
                        payload.request_id,
                        error,
                    )
                    await self._finish_with_error(
                        delivery=delivery,
                        span=span,
                        trace_data=trace_data,
                        error=error,
                        status=STATUS_STREAM_INTERRUPTED,
                        detail=STREAM_INTERRUPTED_MESSAGE,
                        answer=partial_answer,
                    )
                    return

                span.add_event(
                    'agent.retry',
                    attributes={'retry.attempt': attempt, 'error.type': type(error).__name__},
                )
                logger.warning(
                    '⚠️ Ошибка обработки сообщения до начала стриминга '
                    '(попытка %d/%d, session_id=%s, request_id=%s): %s',
                    attempt,
                    self._retries,
                    payload.session_id,
                    payload.request_id,
                    error,
                )
                if attempt < self._retries:
                    await asyncio.sleep(_get_backoff_delay(attempt))

        logger.error(
            '❌ Не удалось обработать сообщение после %d попыток '
            '(session_id=%s, request_id=%s): %s',
            self._retries,
            payload.session_id,
            payload.request_id,
            last_error,
        )
        await self._finish_with_error(
            delivery=delivery,
            span=span,
            trace_data=trace_data,
            error=last_error,
            status=STATUS_GENERATION_FAILED,
            detail=GENERATION_FAILED_MESSAGE,
            answer=None,
            requeue_to_dlq=True,
        )

    async def _finish_with_error(
        self,
        delivery: '_Delivery',
        span: Span,
        trace_data: AgentRequestTraceData,
        error: Exception | None,
        status: str,
        detail: str,
        answer: str | None,
        requeue_to_dlq: bool = False,
    ) -> None:
        """Единая точка терминального неуспеха: SSE, статус в БД и ack/nack."""
        await delivery.send_terminal({'type': 'error', 'detail': detail})
        await self._fail_persistence(
            request_id=delivery.request_id,
            status=status,
            safe_error=type(error).__name__ if error is not None else 'UnknownError',
            answer=answer,
            latency_ms=delivery.elapsed_ms(),
            terminal_detail=detail,
        )
        if requeue_to_dlq:
            await delivery.nack()
            trace_data.outcome = 'dlq'
        else:
            # Часть ответа уже у пользователя либо мутирующий инструмент уже
            # начал работу — повторная обработка навредит сильнее потери.
            await delivery.ack()
            trace_data.outcome = 'error'
        self._set_content(
            span,
            SpanAttributes.OUTPUT_VALUE,
            SpanAttributes.OUTPUT_MIME_TYPE,
            answer or '',
        )
        if error is not None:
            _mark_span_error(span, error)

    def _set_content(
        self,
        span: Span,
        value_attribute: str,
        mime_type_attribute: str,
        content: str,
    ) -> None:
        if not self._trace_content_enabled:
            return
        span.set_attribute(value_attribute, content)
        span.set_attribute(mime_type_attribute, 'text/plain')

    @staticmethod
    def _finalize_root_span(span: Span, trace_data: AgentRequestTraceData) -> None:
        span.set_attribute('agent.route', trace_data.route)
        span.set_attribute('agent.search.required', trace_data.search_required)
        span.set_attribute('agent.search.unavailable', trace_data.search_unavailable)
        span.set_attribute('agent.search.chunk_count', trace_data.search_chunk_count)
        span.set_attribute('agent.tool_call_count', trace_data.tool_call_count)
        span.set_attribute('agent.retry.count', trace_data.request_retry_count)
        span.set_attribute('agent.mcp.retry_count', trace_data.mcp_retry_count)
        span.set_attribute('agent.response.chunk_count', trace_data.response_chunk_count)
        span.set_attribute('agent.response.char_count', trace_data.response_char_count)
        span.set_attribute('agent.streaming.started', trace_data.streaming_started)
        span.set_attribute(
            'agent.mutating_tool.called',
            trace_data.mutating_tool_called,
        )
        if trace_data.consultation_email_status is not None:
            span.set_attribute(
                'agent.consultation_email.status',
                trace_data.consultation_email_status,
            )
        if trace_data.consultation_email_error_code is not None:
            span.set_attribute(
                'agent.consultation_email.error_code',
                trace_data.consultation_email_error_code,
            )
        span.set_attribute('agent.outcome', trace_data.outcome)

    async def _start_persistence(
        self,
        payload: AgentRequestMessage,
    ) -> TurnStartResult | None:
        """Регистрирует или перезахватывает реплику до запуска графа."""
        if self._persistence_service_factory is None:
            return None
        async with self._persistence_service_factory() as service:
            return await service.start_turn(
                session_id=payload.session_id,
                request_id=payload.request_id,
                user_id=payload.user_id,
                anonymous_token_hash=payload.anonymous_token_hash,
                question=payload.message,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )

    async def _start_persistence_with_retry(
        self,
        payload: AgentRequestMessage,
    ) -> TurnStartResult | None:
        """Повторяет регистрацию при временном сбое БД.

        Мгновенный уход в DLQ по первой же ошибке PostgreSQL терял запрос на
        коротком недоступности базы (VERA-015). Граф здесь ещё не запускался,
        поэтому повтор безопасен и ничего не дублирует. Ошибки владения не
        ретраятся — они не временные.
        """
        last_error: ChatPersistenceServiceError | None = None
        for attempt in range(1, self._persistence_retries + 1):
            try:
                return await self._start_persistence(payload)
            except ChatPersistenceServiceError as error:
                last_error = error
                logger.warning(
                    '⚠️ Не удалось зарегистрировать реплику '
                    '(попытка %d/%d, request_id=%s): %s',
                    attempt,
                    self._persistence_retries,
                    payload.request_id,
                    type(error).__name__,
                )
                if attempt < self._persistence_retries:
                    await asyncio.sleep(_get_backoff_delay(attempt))
        raise last_error if last_error is not None else ChatPersistenceServiceError

    async def _complete_persistence(
        self,
        request_id: str,
        answer: str,
        persistence_data: TurnPersistenceData,
        trace_data: AgentRequestTraceData,
        latency_ms: int,
    ) -> bool:
        """Durable-сохраняет завершённый ответ.

        Возвращает признак успеха: `done` разрешено отправлять только после
        подтверждённого commit, иначе живой UI покажет успешный ответ, а
        история навсегда останется в `processing` (VERA-004). Повтор здесь
        сохраняет уже сформированный ответ и графа не переигрывает.
        """
        if self._persistence_service_factory is None:
            return True
        technical_metadata = {
            'route': trace_data.route,
            'tool_calls': persistence_data.tool_calls,
            'search_unavailable': trace_data.search_unavailable,
        }
        for attempt in range(1, self._persistence_retries + 1):
            try:
                async with self._persistence_service_factory() as service:
                    await service.complete_turn(
                        request_id=request_id,
                        answer=answer,
                        sources=persistence_data.sources,
                        technical_metadata=technical_metadata,
                        latency_ms=latency_ms,
                    )
                return True
            except ChatPersistenceServiceError:
                logger.warning(
                    '⚠️ Ответ сформирован, но не сохранён '
                    '(попытка %d/%d, request_id=%s)',
                    attempt,
                    self._persistence_retries,
                    request_id,
                )
                if attempt < self._persistence_retries:
                    await asyncio.sleep(_get_backoff_delay(attempt))
        logger.error(
            '❌ Ответ сформирован, но не сохранён в PostgreSQL после %d попыток. request_id=%s.',
            self._persistence_retries,
            request_id,
        )
        return False

    async def _fail_persistence(
        self,
        request_id: str,
        status: str,
        safe_error: str,
        answer: str | None,
        latency_ms: int,
        terminal_detail: str | None = None,
    ) -> None:
        """Сохраняет ошибочный статус, не меняя текущую RabbitMQ/SSE-ветку."""
        if self._persistence_service_factory is None:
            return
        try:
            async with self._persistence_service_factory() as service:
                await service.fail_turn(
                    request_id=request_id,
                    status=status,
                    safe_error=safe_error,
                    answer=answer,
                    latency_ms=latency_ms,
                    terminal_detail=terminal_detail,
                )
        except ChatPersistenceServiceError:
            logger.exception(
                '❌ Не удалось сохранить ошибочный статус реплики. request_id=%s.',
                request_id,
            )

    async def _stream_answer(
        self,
        payload: AgentRequestMessage,
        persistence_data: TurnPersistenceData,
    ) -> AsyncIterator[str]:
        config = {'configurable': {'thread_id': payload.session_id}}
        streamed_content = False
        deferred_answer: str | None = None
        blocked_pseudo_output = False
        async for event in self._graph.astream_events(_initial_state(payload), config=config, version='v2'):
            # Node-level outputs повторяют `tool_calls`/`retrieved_chunks`.
            # Единственный authoritative snapshot — корневой graph output.
            event_is_chain_end = event.get('event') == 'on_chain_end'
            event_node = event.get('metadata', {}).get('langgraph_node') or event.get('name')
            event_is_root = event_is_chain_end and not event.get('parent_ids')
            event_is_final_node = event_is_chain_end and event_node in FINAL_RESPONSE_NODES
            if event_is_root or event_is_final_node:
                output = event.get('data', {}).get('output')
                if isinstance(output, dict):
                    if event_is_root:
                        sources = output.get('retrieved_chunks')
                        if isinstance(sources, list):
                            persistence_data.sources = sources
                        tool_calls = output.get('tool_calls')
                        if isinstance(tool_calls, list):
                            persistence_data.tool_calls = list(
                                dict.fromkeys(
                                    tool_name
                                    for tool_name in tool_calls
                                    if isinstance(tool_name, str)
                                )
                            )
                    # Детерминированный результат email-тулы формируется
                    # AIMessage без model-stream события. Не
                    # считать их пустым ответом: отложим текст до завершения
                    # графа и отправим его как один SSE-token. В production
                    # LangGraph иногда маркирует final-node event parent_ids;
                    # поэтому учитываем и сам final node, а не только root.
                    if not streamed_content and (event_is_root or event_is_final_node):
                        messages = output.get('messages')
                        if isinstance(messages, list):
                            for message in reversed(messages):
                                # Ищем AIMessage только после текущей
                                # HumanMessage. Иначе при аварии финального
                                # узла можно случайно повторно отправить старый
                                # ответ из истории как ответ на новый запрос.
                                if isinstance(message, HumanMessage):
                                    break
                                if not isinstance(message, AIMessage):
                                    continue
                                content = message.content
                                if isinstance(content, str) and content:
                                    if contains_pseudo_tool_call(content):
                                        logger.error(
                                            'Заблокирован псевдовызов инструмента в финальном сообщении'
                                        )
                                    else:
                                        deferred_answer = content
                                break
            if event.get('event') != 'on_chat_model_stream':
                continue
            if event.get('metadata', {}).get('langgraph_node') not in FINAL_RESPONSE_NODES:
                continue
            content = event['data']['chunk'].content
            if not isinstance(content, str) or not content.strip():
                continue
            if contains_pseudo_tool_call(content):
                blocked_pseudo_output = True
                logger.error('Заблокирован псевдовызов инструмента в SSE-потоке')
                continue
            streamed_content = True
            yield content
        if not streamed_content and deferred_answer:
            yield deferred_answer
        elif not streamed_content and blocked_pseudo_output:
            # Даже если конкретная версия LangGraph не прислала финальный
            # snapshot, клиент не должен получить пустую реплику после
            # фильтрации служебного текста.
            yield UNSAFE_TOOL_CALL_RESPONSE


def _parse_payload(body: bytes) -> AgentRequestMessage:
    try:
        return AgentRequestMessage.model_validate_json(body)
    except ValidationError as error:
        raise InvalidAgentRequestError(str(error)) from error


def _mark_span_error(span: Span, error: Exception) -> None:
    span.set_attribute('error.type', type(error).__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))
