import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable

import aio_pika
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import ValidationError

from app.core.settings import ObservabilitySettings
from app.exceptions.messaging import InvalidAgentRequestError
from app.messaging.schemas import AgentRequestMessage
from app.observability.request_trace import (
    AgentRequestTraceData,
    reset_request_trace,
    set_request_trace,
)
from app.observability.tracing import get_tracer

logger = logging.getLogger('vera_agent_service')
tracer = get_tracer()

DEFAULT_RETRIES: int = 3
DEFAULT_RETRY_DELAY: float = 1.0
DEFAULT_MAX_RETRY_DELAY: float = 30.0
JITTER_RATIO: float = 0.1

TokenSink = Callable[[str, dict], Awaitable[None]]
"""Принимает `(session_id, событие)`. Событие — SSE-контракт (раздел 3.2
плана): `{"type": "token", "content": ...}` / `{"type": "done"}` /
`{"type": "error", "detail": ...}`. Конкретная реализация — `session_bus`
(Этап 7); здесь используется только через этот интерфейс, чтобы Этап 6
оставался тестируемым независимо от Этапа 7 (раздел 0 подхода к плану)."""


def _get_backoff_delay(attempt: int) -> float:
    base_delay = min(DEFAULT_MAX_RETRY_DELAY, DEFAULT_RETRY_DELAY * (2 ** (attempt - 1)))
    jitter = base_delay * JITTER_RATIO * random.random()
    return base_delay + jitter


def _initial_state(payload: AgentRequestMessage) -> dict:
    return {
        'session_id': payload.session_id,
        'user_id': payload.user_id,
        'messages': [HumanMessage(content=payload.message)],
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
        retries: int = DEFAULT_RETRIES,
        prefetch_count: int = 1,
        observability_settings: ObservabilitySettings | None = None,
    ):
        self._connection_url = connection_url
        self._queue_name = queue_name
        self._dlq_name = dlq_name
        self._graph = graph
        self._token_sink = token_sink
        self._retries = retries
        self._prefetch_count = prefetch_count
        self._observability_settings = observability_settings or ObservabilitySettings()
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
        try:
            with tracer.start_as_current_span(
                'vera.agent.request',
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                    'messaging.system': 'rabbitmq',
                    'messaging.destination.name': self._queue_name,
                },
            ) as span:
                try:
                    await self._handle_message_body(message, span, trace_data)
                except Exception as error:
                    trace_data.outcome = 'error'
                    _mark_span_error(span, error)
                    raise
                finally:
                    self._finalize_root_span(span, trace_data)
        finally:
            reset_request_trace(context_token)

    async def _handle_message_body(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
        span: Span,
        trace_data: AgentRequestTraceData,
    ) -> None:
        try:
            payload = _parse_payload(message.body)
        except InvalidAgentRequestError as error:
            logger.error('❌ Невалидный payload %s: %s', self._queue_name, error)
            trace_data.outcome = 'invalid_payload'
            _mark_span_error(span, error)
            await message.nack(requeue=False)
            return

        span.set_attribute('session.id', payload.session_id)
        span.set_attribute('user.authenticated', payload.user_id is not None)
        span.set_attribute('agent.input.char_count', len(payload.message))
        self._set_captured_content(
            span,
            SpanAttributes.INPUT_VALUE,
            SpanAttributes.INPUT_MIME_TYPE,
            'input.truncated',
            payload.message,
        )

        last_error: Exception | None = None
        for attempt in range(1, self._retries + 1):
            streaming_started = False
            response_chunks: list[str] = []
            trace_data.request_retry_count = attempt - 1
            trace_data.streaming_started = False
            trace_data.response_chunk_count = 0
            trace_data.response_char_count = 0
            try:
                async for content in self._stream_answer(payload):
                    await self._token_sink(payload.session_id, {'type': 'token', 'content': content})
                    streaming_started = True
                    trace_data.streaming_started = True
                    response_chunks.append(content)
                    trace_data.response_chunk_count += 1
                    trace_data.response_char_count += len(content)
                await self._token_sink(payload.session_id, {'type': 'done'})
                await message.ack()
                trace_data.outcome = 'degraded' if trace_data.search_unavailable else 'done'
                self._set_captured_content(
                    span,
                    SpanAttributes.OUTPUT_VALUE,
                    SpanAttributes.OUTPUT_MIME_TYPE,
                    'output.truncated',
                    ''.join(response_chunks),
                )
                return
            except Exception as error:  # noqa: BLE001 - сбой графа тоже должен попасть сюда
                last_error = error
                if streaming_started:
                    logger.error(
                        '❌ Ошибка после начала стриминга (session_id=%s): %s', payload.session_id, error
                    )
                    await self._token_sink(
                        payload.session_id,
                        {'type': 'error', 'detail': 'Произошла ошибка при формировании ответа.'},
                    )
                    await message.ack()
                    trace_data.outcome = 'error'
                    self._set_captured_content(
                        span,
                        SpanAttributes.OUTPUT_VALUE,
                        SpanAttributes.OUTPUT_MIME_TYPE,
                        'output.truncated',
                        ''.join(response_chunks),
                    )
                    _mark_span_error(span, error)
                    return
                span.add_event(
                    'agent.retry',
                    attributes={'retry.attempt': attempt, 'error.type': type(error).__name__},
                )
                logger.warning(
                    '⚠️ Ошибка обработки сообщения до начала стриминга (попытка %d/%d, session_id=%s): %s',
                    attempt,
                    self._retries,
                    payload.session_id,
                    error,
                )
                if attempt < self._retries:
                    await asyncio.sleep(_get_backoff_delay(attempt))

        logger.error(
            '❌ Не удалось обработать сообщение после %d попыток (session_id=%s): %s',
            self._retries,
            payload.session_id,
            last_error,
        )
        await self._token_sink(
            payload.session_id, {'type': 'error', 'detail': 'Сервис временно недоступен, попробуйте позже.'}
        )
        await message.nack(requeue=False)
        trace_data.outcome = 'dlq'
        if last_error is not None:
            _mark_span_error(span, last_error)

    def _set_captured_content(
        self,
        span: Span,
        value_attribute: str,
        mime_type_attribute: str,
        truncated_attribute: str,
        content: str,
    ) -> None:
        if not self._observability_settings.phoenix_capture_content:
            return
        max_chars = self._observability_settings.phoenix_content_max_chars
        span.set_attribute(value_attribute, content[:max_chars])
        span.set_attribute(mime_type_attribute, 'text/plain')
        if len(content) > max_chars:
            span.set_attribute(truncated_attribute, True)

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
        span.set_attribute('agent.outcome', trace_data.outcome)

    async def _stream_answer(self, payload: AgentRequestMessage) -> AsyncIterator[str]:
        config = {'configurable': {'thread_id': payload.session_id}}
        async for event in self._graph.astream_events(_initial_state(payload), config=config, version='v2'):
            if event['event'] == 'on_chat_model_stream':
                content = event['data']['chunk'].content
                if content:
                    yield content


def _parse_payload(body: bytes) -> AgentRequestMessage:
    try:
        return AgentRequestMessage.model_validate_json(body)
    except ValidationError as error:
        raise InvalidAgentRequestError(str(error)) from error


def _mark_span_error(span: Span, error: Exception) -> None:
    span.set_attribute('error.type', type(error).__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))
