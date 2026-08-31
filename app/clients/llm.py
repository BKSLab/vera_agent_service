import asyncio
import logging
import random
from collections.abc import AsyncIterator

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError

from app.core.settings import LlmSettings
from app.exceptions.llm import EmptyLlmStreamError, LlmApiRequestError

logger = logging.getLogger('vera_agent_service')

DEFAULT_TIMEOUT_SECONDS: float = 90.0
DEFAULT_RETRIES: int = 3
DEFAULT_RETRY_DELAY: float = 1.0
DEFAULT_MAX_RETRY_DELAY: float = 30.0
JITTER_RATIO: float = 0.1

# Ошибки одной попытки запроса (сеть/таймаут) — уходят в retry.
# Ошибки контента (пустой ответ) обрабатываются отдельно в ainvoke_with_retry,
# не через исключение — см. LLM_CLIENT_REFERENCE.md, различие "ошибка
# запроса" / "ошибка контента".
_REQUEST_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    httpx.TimeoutException,
    httpx.RequestError,
)


def _visible_text_from_content(content: object) -> str:
    """Извлекает только текст, предназначенный пользователю.

    OpenAI-совместимые провайдеры могут прислать content-блоки списком:
    reasoning/tool metadata в таком списке не является ответом и не должен
    взводить флаг начала стриминга или попасть в SSE.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get('type') not in (None, 'text'):
            continue
        text = block.get('text')
        if isinstance(text, str):
            parts.append(text)
    return ''.join(parts)


def _visible_text_from_chunk(chunk: AIMessageChunk) -> str:
    return _visible_text_from_content(chunk.content)


def get_chat_model(httpx_client: httpx.AsyncClient, settings: LlmSettings) -> ChatOpenAI:
    """Создаёт LangChain chat-модель поверх OpenAI-совместимого API.

    Провайдер конфигурируется через `settings` (AGENT_VERA_ARCHITECTURE.md) —
    класс не завязан на конкретного поставщика.

    Возвращает "сырой" `ChatOpenAI`, не обёрнутый ретраями: `.bind_tools()`
    (Этап 4.1) нужно вызывать на этом объекте до применения ретраев —
    `Runnable.with_retry()` не сохраняет метод `bind_tools` у обёрнутого
    объекта. Для intent/tool-routing ретраи применяются явно через
    `ainvoke_with_retry`. Финальные ответы обрабатывает отдельный прямой
    Polza-клиент с собственной retry/output-safety политикой.

    `max_retries=0` — повторы делает не сам openai SDK, а вызывающая граница,
    чтобы не задваивать retry-политику и логи двух разных механизмов.
    """
    chat_model_kwargs: dict[str, object] = {
        'model': settings.llm_model,
        'base_url': settings.llm_api_url,
        'api_key': settings.llm_api_key.get_secret_value(),
        'http_async_client': httpx_client,
        'timeout': DEFAULT_TIMEOUT_SECONDS,
        'max_retries': 0,
    }
    if settings.llm_temperature is not None:
        chat_model_kwargs['temperature'] = settings.llm_temperature
    if settings.llm_reasoning_effort is not None:
        # `reasoning` — расширение Polza для Chat Completions. Передаём его
        # через `extra_body`: openai SDK не знает этого именованного аргумента,
        # а поле ChatOpenAI.reasoning автоматически переключило бы запрос на
        # Responses API и изменило бы существующий контракт агента.
        chat_model_kwargs['extra_body'] = {
            'reasoning': {
                'effort': settings.llm_reasoning_effort,
            }
        }
    return ChatOpenAI(**chat_model_kwargs)


def _get_backoff_delay(attempt: int) -> float:
    """Attempt 1 → ~1s, attempt 2 → ~2s, attempt 3 → ~4s (до max_delay),
    джиттер ±10% — по образцу `LLM_CLIENT_REFERENCE.md`."""
    base_delay = min(DEFAULT_MAX_RETRY_DELAY, DEFAULT_RETRY_DELAY * (2 ** (attempt - 1)))
    jitter = base_delay * JITTER_RATIO * random.random()
    return base_delay + jitter


async def ainvoke_with_retry(
    model: BaseChatModel,
    messages: list[BaseMessage],
    retries: int = DEFAULT_RETRIES,
) -> AIMessage:
    """Нестримингованный вызов модели с ретраями (используется
    `analyze_intent`, Этап 4.1 — раздел 0.1 плана: короткий вызов без
    стриминга).

    Различает в логах ошибку запроса (сеть/таймаут) и ошибку контента
    (пустой ответ без `tool_calls`) — по духу `LLM_CLIENT_REFERENCE.md`,
    хотя механизм отличается (LangChain `Runnable`, не собственный
    HTTP-клиент).

    Raises:
        LlmApiRequestError: если все попытки исчерпаны без успеха.
    """
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            result = await model.ainvoke(messages)
        except _REQUEST_ERRORS as error:
            last_error = error
            logger.warning('⚠️ Ошибка запроса к LLM (попытка %d/%d): %s', attempt, retries, error)
        else:
            if not _visible_text_from_content(result.content) and not result.tool_calls:
                last_error = ValueError('LLM вернул пустой ответ без tool_calls')
                logger.warning(
                    '📭 Некорректный контент от LLM (попытка %d/%d): %s', attempt, retries, last_error
                )
            else:
                if attempt > 1:
                    logger.info('✅ Ответ от LLM получен с %d-й попытки', attempt)
                return result

        if attempt < retries:
            delay = _get_backoff_delay(attempt)
            logger.info('🔄 Повтор через %.1fс (следующая попытка: %d/%d)', delay, attempt + 1, retries)
            await asyncio.sleep(delay)

    logger.error(
        '❌ Не удалось получить ответ от LLM после %d попыток. Последняя ошибка: %s', retries, last_error
    )
    raise LlmApiRequestError(str(last_error))


async def astream_tokens(
    model: BaseChatModel,
    messages: list[BaseMessage],
    retries: int = DEFAULT_RETRIES,
) -> AsyncIterator[str]:
    """Низкоуровневый compatibility-helper для текстового model stream.

    Финальные пользовательские узлы его больше не используют: их прямая
    Polza-граница полностью буферизует и проверяет structured output, а
    consumer безусловно игнорирует ``on_chat_model_stream``.

    Ретраится только получение **первого видимого текстового токена** —
    служебные role/reasoning/tool-call чанки и финальный DONE не считаются
    началом ответа. Как только текст отдан вызывающему коду, повтор всего
    вызова прекращается. Если поток завершился без текста, повторяется только
    этот финальный LLM-вызов, а не весь граф.

    Raises:
        LlmApiRequestError: если запросы к API не удалось выполнить.
        EmptyLlmStreamError: если все потоки завершились без текста.
    """
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        visible_text_emitted = False
        try:
            async for chunk in model.astream(messages):
                content = _visible_text_from_chunk(chunk)
                if not content.strip():
                    continue
                visible_text_emitted = True
                yield content
        except _REQUEST_ERRORS as error:
            if visible_text_emitted:
                raise
            last_error = error
            logger.warning(
                '⚠️ Ошибка запроса к LLM при старте стриминга (попытка %d/%d): %s', attempt, retries, error
            )
        else:
            if visible_text_emitted:
                if attempt > 1:
                    logger.info('✅ Видимый текстовый стриминг начат с %d-й попытки', attempt)
                return
            last_error = EmptyLlmStreamError()
            logger.warning(
                '📭 LLM завершила поток без видимого текста (попытка %d/%d)',
                attempt,
                retries,
            )

        if attempt < retries:
            delay = _get_backoff_delay(attempt)
            logger.info('🔄 Повтор через %.1fс (следующая попытка: %d/%d)', delay, attempt + 1, retries)
            await asyncio.sleep(delay)

    if isinstance(last_error, EmptyLlmStreamError):
        logger.error(
            '❌ Не удалось получить видимый текстовый ответ LLM после %d попыток',
            retries,
        )
        raise last_error

    logger.error(
        '❌ Не удалось начать стриминг ответа LLM после %d попыток. Последняя ошибка: %s',
        retries,
        last_error,
    )
    raise LlmApiRequestError(str(last_error))
