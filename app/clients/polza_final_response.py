"""Прямая безопасная граница финальной генерации через Polza Chat Completions.

LangChain остаётся для intent/tool-routing, но не используется на последнем
модельном шаге: текущий адаптер ``ChatOpenAI`` отбрасывает нестандартные поля
``reasoning``/``reasoning_details``. Здесь оба канала читаются из сырого SSE,
а наружу возвращается только полностью проверенное поле ``answer``.
"""

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from langchain_core.messages import BaseMessage, SystemMessage, convert_to_openai_messages
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.trace import Span, Status, StatusCode

from app.core.settings import LlmSettings
from app.exceptions.llm import LlmApiRequestError
from app.graph.output_guard import OutputGuardDecision, OutputGuardReason, validate_structured_final_answer
from app.graph.policy import UNSAFE_TOOL_CALL_RESPONSE
from app.graph.prompts.system import FINAL_RESPONSE_RETRY_PROMPT
from app.observability.request_trace import get_request_trace
from app.observability.tracing import get_tracer

logger = logging.getLogger('vera_agent_service')

DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_REQUEST_RETRIES = 3
DEFAULT_OUTPUT_RETRIES = 1
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_MAX_RETRY_DELAY = 30.0
JITTER_RATIO = 0.1
MAX_FINAL_CONTENT_CHARS = 1_000_000
MAX_REASONING_CHARS = 1_000_000
MAX_SSE_EVENT_CHARS = 1_000_000

_KNOWN_PROVIDER_ERROR_CODES = frozenset(
    {
        'BAD_REQUEST',
        'RATE_LIMITED',
        'UNAUTHORIZED',
        'FORBIDDEN',
        'NOT_FOUND',
        'INTERNAL_SERVER_ERROR',
        'SERVICE_UNAVAILABLE',
        'TIMEOUT',
        'invalid_request_error',
        'rate_limit_error',
        'authentication_error',
        'permission_error',
        'server_error',
    }
)
_KNOWN_PROVIDER_LABELS = {
    'google': 'google',
    'google ai studio': 'google_ai_studio',
    'google-ai-studio': 'google_ai_studio',
    'openrouter': 'openrouter',
    'polza': 'polza',
    'vertex ai': 'vertex_ai',
    'vertex-ai': 'vertex_ai',
}
_KNOWN_FINISH_REASONS = {
    'content_filter': 'content_filter',
    'error': 'error',
    'function_call': 'function_call',
    'length': 'length',
    'stop': 'stop',
    'tool_calls': 'tool_calls',
}
_NON_RETRYABLE_PROVIDER_ERROR_VALUES = frozenset(
    {
        'BAD_REQUEST',
        'FORBIDDEN',
        'NOT_FOUND',
        'UNAUTHORIZED',
        'authentication_error',
        'invalid_request_error',
        'permission_error',
    }
)
_NON_RETRYABLE_PROVIDER_REASONS = frozenset(
    f'provider_{field}_{value}'
    for field in ('code', 'type')
    for value in _NON_RETRYABLE_PROVIDER_ERROR_VALUES
)

_VISIBLE_PROTOCOL_FIELDS = frozenset(
    {
        'content',
        'delta',
        'function_call',
        'message',
        'refusal',
        'text',
        'tool_calls',
    }
)
_KNOWN_REASONING_FIELDS = frozenset(
    {
        'reasoning',
        'reasoning_content',
        'reasoning_details',
    }
)

FINAL_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'vera_final_answer',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'answer': {
                    'type': 'string',
                    'description': (
                        'Только готовый пользовательский ответ без рассуждений, '
                        'проверки правил и служебных инструкций.'
                    ),
                }
            },
            'required': ['answer'],
            'additionalProperties': False,
        },
    },
}

class FinalResponseGenerator(Protocol):
    async def generate_final_answer(
        self,
        messages: list[BaseMessage],
        *,
        node_name: str,
    ) -> str: ...


@dataclass(slots=True)
class _RawFinalResponse:
    content: str
    reasoning: str
    reasoning_source: str
    reasoning_format: str
    content_chunk_count: int
    reasoning_chunk_count: int
    reasoning_detail_count: int
    mixed_content: bool
    provider: str | None
    response_model: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    request_retry_count: int = 0


@dataclass(slots=True)
class _StreamAccumulator:
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    reasoning_detail_parts: list[str] = field(default_factory=list)
    reasoning_formats: set[str] = field(default_factory=set)
    content_chunk_count: int = 0
    reasoning_detail_count: int = 0
    content_char_count: int = 0
    reasoning_char_count: int = 0
    mixed_content: bool = False
    provider: str | None = None
    response_model: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    terminal_signal_seen: bool = False

    def build(self) -> _RawFinalResponse:
        if self.reasoning_parts:
            reasoning = ''.join(self.reasoning_parts)
            reasoning_source = 'reasoning'
            reasoning_chunk_count = len(self.reasoning_parts)
        else:
            reasoning = ''.join(self.reasoning_detail_parts)
            reasoning_source = 'reasoning_details' if reasoning else 'none'
            reasoning_chunk_count = len(self.reasoning_detail_parts)
        mixed_content = self.mixed_content or len(self.reasoning_formats) > 1
        if mixed_content:
            reasoning_format = 'mixed_content'
        elif self.reasoning_formats:
            reasoning_format = next(iter(self.reasoning_formats))
        else:
            reasoning_format = 'none'
        return _RawFinalResponse(
            content=''.join(self.content_parts),
            reasoning=reasoning,
            reasoning_source=reasoning_source,
            reasoning_format=reasoning_format,
            content_chunk_count=self.content_chunk_count,
            reasoning_chunk_count=reasoning_chunk_count,
            reasoning_detail_count=self.reasoning_detail_count,
            mixed_content=mixed_content,
            provider=self.provider,
            response_model=self.response_model,
            finish_reason=self.finish_reason,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
            total_tokens=self.total_tokens,
        )


class _PolzaStreamError(RuntimeError):
    def __init__(self, reason_code: str, *, status_code: int | None = None):
        self.reason_code = reason_code
        self.status_code = status_code
        super().__init__(reason_code)


class _DuplicateSseJsonKeyError(ValueError):
    pass


class PolzaFinalResponseClient:
    """Буферизует, разделяет каналы и fail-closed проверяет финальный ответ."""

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        settings: LlmSettings,
        *,
        trace_content_enabled: bool = False,
        request_retries: int = DEFAULT_REQUEST_RETRIES,
        output_retries: int = DEFAULT_OUTPUT_RETRIES,
    ) -> None:
        if request_retries < 1:
            raise ValueError('request_retries должен быть не меньше 1')
        if output_retries not in (0, 1):
            raise ValueError('output_retries может быть только 0 или 1')
        self._httpx_client = httpx_client
        self._api_url = f"{settings.llm_api_url.rstrip('/')}/chat/completions"
        self._api_key = settings.llm_api_key.get_secret_value()
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._reasoning_effort = settings.llm_reasoning_effort
        self._trace_content_enabled = trace_content_enabled
        self._request_retries = request_retries
        self._output_retries = output_retries

    async def generate_final_answer(
        self,
        messages: list[BaseMessage],
        *,
        node_name: str,
    ) -> str:
        """Возвращает только разрешённый ответ или детерминированный fallback.

        Повторяется лишь этот финальный вызов; отклонённый output не добавляется
        в retry prompt и никогда не становится ``AIMessage``.
        """
        total_output_attempts = self._output_retries + 1
        total_raw_char_count = 0
        last_decision = OutputGuardDecision(False, OutputGuardReason.EMPTY_OUTPUT)
        last_rejection_reason = OutputGuardReason.ACCEPTED

        for output_attempt in range(1, total_output_attempts + 1):
            attempt_messages = messages
            if output_attempt > 1:
                attempt_messages = [
                    *messages,
                    SystemMessage(content=FINAL_RESPONSE_RETRY_PROMPT),
                ]

            logger.info(
                '🚀 Финальная LLM-генерация начата '
                '(node=%s, model=%s, output_attempt=%d/%d)',
                node_name,
                self._model,
                output_attempt,
                total_output_attempts,
            )
            started_at = time.perf_counter()
            with get_tracer().start_as_current_span(
                'llm.polza.final_response',
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                    SpanAttributes.LLM_MODEL_NAME: self._model,
                    SpanAttributes.LLM_REQUEST_MODEL_NAME: self._model,
                    'vera.llm.node': node_name,
                    'vera.llm.output_attempt': output_attempt,
                },
            ) as span:
                try:
                    openai_messages = _convert_messages(attempt_messages)
                except Exception as error:  # noqa: BLE001 - content не должен попасть в общий error path
                    safe_error = LlmApiRequestError('message_serialization_error')
                    span.record_exception(safe_error)
                    span.set_status(Status(StatusCode.ERROR, 'message_serialization_error'))
                    logger.error(
                        '❌ Не удалось сериализовать сообщения финальной LLM-генерации '
                        '(node=%s, error_type=%s)',
                        node_name,
                        _safe_label(type(error).__name__),
                    )
                    raise safe_error from None
                self._set_trace_input(span, openai_messages)
                try:
                    raw = await self._request_with_retry(openai_messages, node_name=node_name)
                except LlmApiRequestError as error:
                    span.record_exception(error)
                    span.set_status(Status(StatusCode.ERROR, 'llm_api_request_failed'))
                    span.set_attribute('vera.llm.outcome', 'request_failed')
                    raise

                latency_ms = int((time.perf_counter() - started_at) * 1000)
                total_raw_char_count += len(raw.content)
                last_decision = validate_structured_final_answer(
                    raw.content,
                    mixed_content=raw.mixed_content,
                )
                self._set_response_trace_attributes(
                    span,
                    raw,
                    last_decision,
                    latency_ms=latency_ms,
                    output_retry_count=output_attempt - 1,
                )

            logger.info(
                '📥 Финальный LLM-поток получен '
                '(node=%s, model=%s, provider=%s, output_attempt=%d/%d, '
                'content_chunks=%d, content_chars=%d, reasoning_chunks=%d, '
                'reasoning_chars=%d, reasoning_tokens=%s, finish_reason=%s, latency_ms=%d)',
                node_name,
                self._model,
                raw.provider or 'unknown',
                output_attempt,
                total_output_attempts,
                raw.content_chunk_count,
                len(raw.content),
                raw.reasoning_chunk_count,
                len(raw.reasoning),
                raw.reasoning_tokens,
                raw.finish_reason or 'unknown',
                latency_ms,
            )

            if last_decision.accepted and last_decision.answer is not None:
                status = 'accepted' if output_attempt == 1 else 'retried'
                self._update_request_trace(
                    status=status,
                    reason=(
                        OutputGuardReason.ACCEPTED
                        if output_attempt == 1
                        else last_rejection_reason
                    ),
                    retry_count=output_attempt - 1,
                    raw_char_count=total_raw_char_count,
                    final_char_count=len(last_decision.answer),
                )
                logger.info(
                    '✅ Финальный ответ разрешён output guard '
                    '(node=%s, status=%s, output_attempt=%d/%d, final_chars=%d)',
                    node_name,
                    status,
                    output_attempt,
                    total_output_attempts,
                    len(last_decision.answer),
                )
                return last_decision.answer

            last_rejection_reason = last_decision.reason
            if output_attempt < total_output_attempts:
                logger.warning(
                    '🛡️ Финальный ответ отклонён; повторяется только финальная генерация '
                    '(node=%s, reason=%s, output_attempt=%d/%d)',
                    node_name,
                    last_decision.reason,
                    output_attempt,
                    total_output_attempts,
                )

        self._update_request_trace(
            status='blocked',
            reason=last_decision.reason,
            retry_count=self._output_retries,
            raw_char_count=total_raw_char_count,
            final_char_count=len(UNSAFE_TOOL_CALL_RESPONSE),
        )
        logger.error(
            '❌ Финальные ответы отклонены output guard; возвращён безопасный fallback '
            '(node=%s, reason=%s, retry_count=%d)',
            node_name,
            last_decision.reason,
            self._output_retries,
        )
        return UNSAFE_TOOL_CALL_RESPONSE

    async def _request_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        node_name: str,
    ) -> _RawFinalResponse:
        last_error = _PolzaStreamError('unknown_request_error')
        attempts_made = 0
        for request_attempt in range(1, self._request_retries + 1):
            attempts_made = request_attempt
            try:
                async with asyncio.timeout(DEFAULT_TIMEOUT_SECONDS):
                    result = await self._stream_once(messages)
            except TimeoutError:
                last_error = _PolzaStreamError('request_timeout')
            except (httpx.TimeoutException, httpx.RequestError) as error:
                last_error = _PolzaStreamError(type(error).__name__)
            except _PolzaStreamError as error:
                last_error = error
            except Exception as error:  # noqa: BLE001 - наружу только безопасный код
                last_error = _PolzaStreamError(
                    f'unexpected_{_safe_label(type(error).__name__)}'
                )
            else:
                result.request_retry_count = request_attempt - 1
                if request_attempt > 1:
                    logger.info(
                        '✅ Запрос финальной LLM-генерации выполнен после повтора '
                        '(node=%s, request_attempt=%d/%d)',
                        node_name,
                        request_attempt,
                        self._request_retries,
                    )
                return result

            logger.warning(
                '⚠️ Ошибка запроса финальной LLM-генерации '
                '(node=%s, request_attempt=%d/%d, reason=%s, http_status=%s)',
                node_name,
                request_attempt,
                self._request_retries,
                last_error.reason_code,
                last_error.status_code,
            )
            if not _is_retryable_stream_error(last_error):
                logger.info(
                    '⛔ Ошибка финального LLM-запроса не подлежит повтору '
                    '(node=%s, reason=%s, http_status=%s)',
                    node_name,
                    last_error.reason_code,
                    last_error.status_code,
                )
                break
            if request_attempt < self._request_retries:
                delay = _get_backoff_delay(request_attempt)
                logger.info(
                    '🔄 Повтор финального LLM-запроса через %.1fс '
                    '(node=%s, next_attempt=%d/%d)',
                    delay,
                    node_name,
                    request_attempt + 1,
                    self._request_retries,
                )
                await asyncio.sleep(delay)

        logger.error(
            '❌ Запросы финальной LLM-генерации исчерпаны '
            '(node=%s, attempts=%d, reason=%s, http_status=%s)',
            node_name,
            attempts_made,
            last_error.reason_code,
            last_error.status_code,
        )
        raise LlmApiRequestError(last_error.reason_code)

    async def _stream_once(self, messages: list[dict[str, Any]]) -> _RawFinalResponse:
        payload: dict[str, Any] = {
            'model': self._model,
            'messages': messages,
            'stream': True,
            'stream_options': {'include_usage': True},
            'response_format': FINAL_RESPONSE_JSON_SCHEMA,
        }
        if self._temperature is not None:
            payload['temperature'] = self._temperature
        if self._reasoning_effort is not None:
            # ``exclude=false`` фиксирует выбранную политику явно: reasoning
            # сохраняется для защищённого Phoenix, но никогда не становится
            # пользовательским content. Это не отключает reasoning и не
            # снижает качество ответа.
            payload['reasoning'] = {
                'effort': self._reasoning_effort,
                'exclude': False,
            }

        accumulator = _StreamAccumulator()
        async with self._httpx_client.stream(
            'POST',
            self._api_url,
            headers={
                'Authorization': f'Bearer {self._api_key}',
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            json=payload,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        ) as response:
            if response.status_code < 200 or response.status_code >= 300:
                raise _PolzaStreamError(
                    'http_error',
                    status_code=response.status_code,
                )
            content_type = response.headers.get('content-type', '').partition(';')[0]
            if content_type.strip().casefold() != 'text/event-stream':
                raise _PolzaStreamError('invalid_sse_content_type')
            async for event_data in _iter_sse_data(response):
                if event_data.strip() == '[DONE]':
                    accumulator.terminal_signal_seen = True
                    break
                try:
                    event = json.loads(
                        event_data,
                        object_pairs_hook=_reject_duplicate_sse_keys,
                        parse_constant=_reject_non_finite_json_number,
                    )
                except _DuplicateSseJsonKeyError as error:
                    raise _PolzaStreamError('duplicate_sse_json_key') from error
                except (json.JSONDecodeError, ValueError) as error:
                    raise _PolzaStreamError('invalid_sse_json') from error
                if not isinstance(event, dict):
                    raise _PolzaStreamError('invalid_sse_event')
                _consume_event(event, accumulator)
        if not accumulator.terminal_signal_seen:
            raise _PolzaStreamError('incomplete_sse_stream')
        result = accumulator.build()
        if (
            result.response_model is not None
            and result.response_model not in {self._model, 'multiple'}
        ):
            # Ответ провайдера — недоверенный ввод. Не экспортируем
            # произвольную строку как telemetry label.
            result.response_model = 'other'
        return result

    def _set_trace_input(self, span: Span, messages: list[dict[str, Any]]) -> None:
        if not self._trace_content_enabled:
            return
        span.set_attribute(
            SpanAttributes.INPUT_VALUE,
            json.dumps(
                messages,
                ensure_ascii=False,
                default=str,
            ),
        )
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, 'application/json')

    def _set_response_trace_attributes(
        self,
        span: Span,
        raw: _RawFinalResponse,
        decision: OutputGuardDecision,
        *,
        latency_ms: int,
        output_retry_count: int,
    ) -> None:
        span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL_NAME, raw.response_model or self._model)
        span.set_attribute(SpanAttributes.LLM_PROVIDER, raw.provider or 'unknown')
        span.set_attribute('vera.llm.content.chunk_count', raw.content_chunk_count)
        span.set_attribute('vera.llm.content.char_count', len(raw.content))
        span.set_attribute('vera.llm.reasoning.chunk_count', raw.reasoning_chunk_count)
        span.set_attribute('vera.llm.reasoning.detail_count', raw.reasoning_detail_count)
        span.set_attribute('vera.llm.reasoning.char_count', len(raw.reasoning))
        span.set_attribute('vera.llm.reasoning.source', raw.reasoning_source)
        span.set_attribute('vera.llm.reasoning.format', raw.reasoning_format)
        span.set_attribute('vera.llm.request.retry_count', raw.request_retry_count)
        span.set_attribute('vera.llm.latency_ms', latency_ms)
        guard_status = 'blocked'
        if decision.accepted:
            guard_status = 'retried' if output_retry_count else 'accepted'
        span.set_attribute('vera.llm.output_guard.status', guard_status)
        span.set_attribute('vera.llm.output_guard.reason', decision.reason)
        span.set_attribute('vera.llm.output_guard.retry_count', output_retry_count)
        span.set_attribute('vera.llm.output_guard.raw_char_count', len(raw.content))
        span.set_attribute(
            'vera.llm.output_guard.final_char_count',
            len(decision.answer) if decision.answer is not None else 0,
        )
        if raw.finish_reason is not None:
            span.set_attribute(SpanAttributes.LLM_FINISH_REASON, raw.finish_reason)
        _set_optional_int_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, raw.prompt_tokens)
        _set_optional_int_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, raw.completion_tokens)
        _set_optional_int_attribute(span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, raw.total_tokens)
        _set_optional_int_attribute(
            span,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
            raw.reasoning_tokens,
        )
        if self._trace_content_enabled:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, raw.content)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, 'application/json')
            if raw.reasoning:
                span.set_attribute('vera.llm.reasoning.content', raw.reasoning)

    @staticmethod
    def _update_request_trace(
        *,
        status: str,
        reason: OutputGuardReason,
        retry_count: int,
        raw_char_count: int,
        final_char_count: int,
    ) -> None:
        trace_data = get_request_trace()
        if trace_data is None:
            return
        trace_data.output_guard_status = status
        trace_data.output_guard_reason = reason
        trace_data.output_guard_retry_count = retry_count
        trace_data.output_guard_raw_char_count = raw_char_count
        trace_data.output_guard_final_char_count = final_char_count


async def _iter_sse_data(response: httpx.Response):
    """Выделяет ``data`` событий SSE независимо от сетевых границ строк."""
    data_lines: list[str] = []
    data_char_count = 0
    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip('\r')
        if not line:
            if data_lines:
                yield '\n'.join(data_lines)
                data_lines = []
                data_char_count = 0
            continue
        if line.startswith(':'):
            continue
        if line.startswith('data:'):
            data_line = line[5:].lstrip(' ')
            data_char_count += len(data_line) + bool(data_lines)
            if data_char_count > MAX_SSE_EVENT_CHARS:
                raise _PolzaStreamError('sse_event_too_large')
            data_lines.append(data_line)
    if data_lines:
        yield '\n'.join(data_lines)


def _consume_event(event: dict[str, Any], accumulator: _StreamAccumulator) -> None:
    error = event.get('error')
    if error is not None:
        raise _PolzaStreamError(_safe_error_code(error))

    if any(
        event.get(protocol_field) is not None
        for protocol_field in _VISIBLE_PROTOCOL_FIELDS
    ):
        # В Chat Completions эти поля допустимы только внутри choice/delta.
        # Игнорировать второй видимый канал было бы неоднозначным разбором.
        accumulator.mixed_content = True
    if _has_unknown_reasoning_field(event):
        accumulator.mixed_content = True

    provider = event.get('provider') or event.get('provider_name')
    if isinstance(provider, str) and provider:
        safe_provider = _known_protocol_label(
            provider,
            _KNOWN_PROVIDER_LABELS,
        )
        if accumulator.provider is None:
            accumulator.provider = safe_provider
        elif accumulator.provider != safe_provider:
            # Polza может менять routing-метку между промежуточными и финальным
            # usage event одного запроса (например, google -> openrouter).
            # Это telemetry, а не второй output-канал.
            accumulator.provider = 'multiple'
    response_model = event.get('model')
    if isinstance(response_model, str) and response_model:
        if accumulator.response_model is None:
            accumulator.response_model = response_model
        elif accumulator.response_model != response_model:
            accumulator.response_model = 'multiple'
    _consume_reasoning_aliases(event, accumulator, format_name='field')
    _consume_reasoning_details(
        event.get('reasoning_details'),
        accumulator,
        format_name='field',
    )
    _consume_usage(event.get('usage'), accumulator)

    choices = event.get('choices')
    if choices is None:
        return
    if not isinstance(choices, list):
        raise _PolzaStreamError('invalid_choices')
    if len(choices) > 1:
        accumulator.mixed_content = True

    for choice in choices:
        if not isinstance(choice, dict):
            accumulator.mixed_content = True
            continue
        if accumulator.terminal_signal_seen and _choice_contains_payload(choice):
            # После finish_reason новые completion-данные недопустимы. Usage
            # event без payload остаётся разрешённым.
            accumulator.mixed_content = True
        index = choice.get('index', 0)
        if index is not None and (
            not isinstance(index, int) or isinstance(index, bool) or index != 0
        ):
            accumulator.mixed_content = True
            continue
        if _has_unknown_reasoning_field(choice):
            accumulator.mixed_content = True
        if any(
            choice.get(protocol_field) is not None
            for protocol_field in ('content', 'text')
        ):
            accumulator.mixed_content = True
        finish_reason = choice.get('finish_reason')
        if isinstance(finish_reason, str) and finish_reason:
            safe_finish_reason = _known_protocol_label(
                finish_reason,
                _KNOWN_FINISH_REASONS,
            )
            if (
                accumulator.finish_reason is not None
                and accumulator.finish_reason != safe_finish_reason
            ):
                accumulator.mixed_content = True
            accumulator.finish_reason = safe_finish_reason
            accumulator.terminal_signal_seen = True
            if accumulator.finish_reason != 'stop':
                # Усечённый, отфильтрованный либо неизвестно завершившийся
                # ответ нельзя считать полным даже при синтаксически валидном
                # JSON envelope.
                accumulator.mixed_content = True
        elif finish_reason is not None:
            accumulator.mixed_content = True
        if choice.get('tool_calls') or choice.get('function_call') or choice.get('refusal'):
            accumulator.mixed_content = True
        _consume_reasoning_aliases(choice, accumulator, format_name='field')
        _consume_reasoning_details(
            choice.get('reasoning_details'),
            accumulator,
            format_name='field',
        )

        delta = choice.get('delta')
        message = choice.get('message')
        if delta is not None and message is not None:
            accumulator.mixed_content = True
        if delta is None:
            delta = message
        if not isinstance(delta, dict):
            if delta is not None:
                accumulator.mixed_content = True
            continue
        if _has_unknown_reasoning_field(delta):
            accumulator.mixed_content = True
        role = delta.get('role')
        if role is not None and role != 'assistant':
            accumulator.mixed_content = True
        if any(
            delta.get(protocol_field) not in (None, '', [], {})
            for protocol_field in ('audio', 'image', 'images', 'output_audio')
        ):
            accumulator.mixed_content = True
        if delta.get('tool_calls') or delta.get('function_call') or delta.get('refusal'):
            accumulator.mixed_content = True
        _consume_content(delta.get('content'), accumulator)
        _consume_reasoning_aliases(delta, accumulator, format_name='delta')
        _consume_reasoning_details(
            delta.get('reasoning_details'),
            accumulator,
            format_name='delta',
        )


def _consume_content(value: Any, accumulator: _StreamAccumulator) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value:
            _append_content(value, accumulator)
        return
    blocks = value if isinstance(value, list) else [value]
    for block in blocks:
        if isinstance(block, str):
            if block:
                _append_content(block, accumulator)
            continue
        if not isinstance(block, dict):
            accumulator.mixed_content = True
            continue
        if _has_unknown_reasoning_field(block):
            accumulator.mixed_content = True
        block_type = block.get('type')
        text = _text_value(block.get('text'))
        if block_type in ('text', 'output_text') and text is not None:
            if text:
                _append_content(text, accumulator)
            continue
        if isinstance(block_type, str) and ('reasoning' in block_type or block_type in {'think', 'thinking'}):
            accumulator.reasoning_formats.add('content_block')
            if text:
                _append_reasoning(text, accumulator.reasoning_parts, accumulator)
            continue
        accumulator.mixed_content = True


def _consume_reasoning_aliases(
    container: dict[str, Any],
    accumulator: _StreamAccumulator,
    *,
    format_name: str,
) -> None:
    reasoning = container.get('reasoning')
    reasoning_content = container.get('reasoning_content')
    if reasoning is not None and reasoning_content is not None:
        accumulator.mixed_content = True
    _consume_reasoning(
        reasoning if reasoning is not None else reasoning_content,
        accumulator,
        format_name=format_name,
    )


def _consume_reasoning(
    value: Any,
    accumulator: _StreamAccumulator,
    *,
    format_name: str,
) -> None:
    if value is None:
        return
    accumulator.reasoning_formats.add(format_name)
    if isinstance(value, str):
        if value:
            _append_reasoning(value, accumulator.reasoning_parts, accumulator)
        return
    if isinstance(value, (list, dict)):
        _consume_reasoning_details(
            value,
            accumulator,
            format_name=format_name,
        )
        return
    # Неизвестное представление отдельного reasoning-канала не копируется в
    # content, но помечает формат неоднозначным для fail-closed решения.
    accumulator.mixed_content = True


def _consume_reasoning_details(
    value: Any,
    accumulator: _StreamAccumulator,
    *,
    format_name: str,
) -> None:
    if value is None:
        return
    accumulator.reasoning_formats.add(format_name)
    details = value if isinstance(value, list) else [value]
    for detail in details:
        if not isinstance(detail, dict):
            accumulator.mixed_content = True
            continue
        accumulator.reasoning_detail_count += 1
        text = _text_value(detail.get('text'))
        if text is None:
            text = _text_value(detail.get('content'))
        if text:
            _append_reasoning(
                text,
                accumulator.reasoning_detail_parts,
                accumulator,
            )


def _append_content(value: str, accumulator: _StreamAccumulator) -> None:
    next_char_count = accumulator.content_char_count + len(value)
    if next_char_count > MAX_FINAL_CONTENT_CHARS:
        raise _PolzaStreamError('content_too_large')
    accumulator.content_parts.append(value)
    accumulator.content_chunk_count += 1
    accumulator.content_char_count = next_char_count


def _append_reasoning(
    value: str,
    target: list[str],
    accumulator: _StreamAccumulator,
) -> None:
    next_char_count = accumulator.reasoning_char_count + len(value)
    if next_char_count > MAX_REASONING_CHARS:
        raise _PolzaStreamError('reasoning_too_large')
    target.append(value)
    accumulator.reasoning_char_count = next_char_count


def _consume_usage(value: Any, accumulator: _StreamAccumulator) -> None:
    if not isinstance(value, dict):
        return
    prompt_tokens = _optional_int(value.get('prompt_tokens'))
    completion_tokens = _optional_int(value.get('completion_tokens'))
    total_tokens = _optional_int(value.get('total_tokens'))
    if prompt_tokens is not None:
        accumulator.prompt_tokens = prompt_tokens
    if completion_tokens is not None:
        accumulator.completion_tokens = completion_tokens
    if total_tokens is not None:
        accumulator.total_tokens = total_tokens
    completion_details = value.get('completion_tokens_details')
    if isinstance(completion_details, dict):
        reasoning_tokens = _optional_int(completion_details.get('reasoning_tokens'))
        if reasoning_tokens is not None:
            accumulator.reasoning_tokens = reasoning_tokens
    if accumulator.reasoning_tokens is None:
        accumulator.reasoning_tokens = _optional_int(value.get('reasoning_tokens'))


def _text_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get('value')
        return nested if isinstance(nested, str) else None
    return None


def _safe_error_code(error: Any) -> str:
    if isinstance(error, dict):
        for key in ('code', 'type'):
            value = error.get(key)
            if isinstance(value, str) and value in _KNOWN_PROVIDER_ERROR_CODES:
                return f'provider_{key}_{value}'
    return 'provider_error'


def _has_unknown_reasoning_field(container: dict[str, Any]) -> bool:
    for key in container:
        normalized = key.strip().casefold().replace('-', '_')
        if normalized in _KNOWN_REASONING_FIELDS:
            continue
        if normalized.startswith('reasoning') or normalized in {
            'analysis',
            'think',
            'thinking',
            'thought',
            'thoughts',
        }:
            return True
    return False


def _choice_contains_payload(choice: dict[str, Any]) -> bool:
    for protocol_field in (
        'content',
        'function_call',
        'reasoning',
        'reasoning_content',
        'reasoning_details',
        'refusal',
        'text',
        'tool_calls',
    ):
        if choice.get(protocol_field) not in (None, '', [], {}):
            return True
    for protocol_field in ('delta', 'message'):
        container = choice.get(protocol_field)
        if container in (None, '', [], {}):
            continue
        if not isinstance(container, dict):
            return True
        for channel_field in (
            'audio',
            'content',
            'function_call',
            'image',
            'images',
            'output_audio',
            'reasoning',
            'reasoning_content',
            'reasoning_details',
            'refusal',
            'tool_calls',
        ):
            if container.get(channel_field) not in (None, '', [], {}):
                return True
    return False


def _reject_duplicate_sse_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateSseJsonKeyError(key)
        result[key] = value
    return result


def _reject_non_finite_json_number(value: str) -> None:
    raise ValueError(f'non_finite_json_number:{value}')


def _safe_label(value: str) -> str:
    """Не допускает переносов и произвольного provider-текста в логи."""
    sanitized = re.sub(r'[^a-zA-Z0-9_./:-]+', '_', value.strip())
    return (sanitized or 'unknown')[:120]


def _known_protocol_label(value: str, labels: dict[str, str]) -> str:
    """Возвращает только известную константу, не произвольный provider input."""
    return labels.get(value.strip().casefold(), 'other')


def _is_retryable_stream_error(error: _PolzaStreamError) -> bool:
    if error.reason_code in _NON_RETRYABLE_PROVIDER_REASONS:
        return False
    status_code = error.status_code
    if status_code is None:
        return True
    return status_code in {408, 409, 425, 429} or status_code >= 500


def _convert_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    converted = convert_to_openai_messages(messages)
    if not isinstance(converted, list) or not all(
        isinstance(message, dict) for message in converted
    ):
        raise TypeError('unsupported_messages_shape')
    # Валидируем до request-retry: несерилизуемая история не исправится
    # повторным HTTP-вызовом и не должна отдавать исходный TypeError в span.
    json.dumps(converted, ensure_ascii=False, allow_nan=False)
    return converted


def _optional_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _set_optional_int_attribute(span: Span, name: str, value: int | None) -> None:
    if value is not None:
        span.set_attribute(name, value)


def _get_backoff_delay(attempt: int) -> float:
    base_delay = min(DEFAULT_MAX_RETRY_DELAY, DEFAULT_RETRY_DELAY * (2 ** (attempt - 1)))
    return base_delay + base_delay * JITTER_RATIO * random.random()
