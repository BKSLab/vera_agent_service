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
from app.exceptions.llm import LlmApiRequestError

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


def get_chat_model(httpx_client: httpx.AsyncClient, settings: LlmSettings) -> ChatOpenAI:
    """Создаёт LangChain chat-модель поверх OpenAI-совместимого API.

    Провайдер конфигурируется через `settings` (AGENT_VERA_ARCHITECTURE.md) —
    класс не завязан на конкретного поставщика.

    Возвращает "сырой" `ChatOpenAI`, не обёрнутый ретраями: `.bind_tools()`
    (Этап 4.1) нужно вызывать на этом объекте до применения ретраев —
    `Runnable.with_retry()` не сохраняет метод `bind_tools` у обёрнутого
    объекта. Ретраи применяются явно через `ainvoke_with_retry`/
    `astream_tokens` ниже, в месте фактического вызова (граф, Этап 4).

    `max_retries=0` — повторы делает не сам openai SDK, а ainvoke_with_retry/
    astream_tokens, чтобы не задваивать retry-политику и логи двух разных
    механизмов.
    """
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_api_url,
        api_key=settings.llm_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        http_async_client=httpx_client,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=0,
    )


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
            if not result.content and not result.tool_calls:
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
    """Стримингованный вызов модели (генерация ответа, Этап 4.3/4.4).

    Ретраится только получение **первого** чанка — как только хотя бы один
    токен отдан вызывающему коду, повтор всего вызова прекращается и любая
    следующая ошибка всплывает немедленно. Согласуется с решением раздела
    0.1 плана: RabbitMQ-consumer (Этап 6.3) не переигрывает сообщение после
    того как стриминг клиенту уже начался — тот же принцип применяется и
    здесь, на уровне отдельного вызова LLM.

    Raises:
        LlmApiRequestError: если не удалось получить даже первый чанк после
            исчерпания попыток.
    """
    last_error: Exception | None = None
    stream: AsyncIterator[AIMessageChunk] | None = None
    first_chunk: AIMessageChunk | None = None

    for attempt in range(1, retries + 1):
        stream = model.astream(messages)
        try:
            first_chunk = await anext(stream, None)
        except _REQUEST_ERRORS as error:
            last_error = error
            logger.warning(
                '⚠️ Ошибка запроса к LLM при старте стриминга (попытка %d/%d): %s', attempt, retries, error
            )
            if attempt < retries:
                delay = _get_backoff_delay(attempt)
                logger.info('🔄 Повтор через %.1fс (следующая попытка: %d/%d)', delay, attempt + 1, retries)
                await asyncio.sleep(delay)
            continue
        else:
            if attempt > 1:
                logger.info('✅ Стриминг начат с %d-й попытки', attempt)
            break
    else:
        logger.error(
            '❌ Не удалось начать стриминг ответа LLM после %d попыток. Последняя ошибка: %s',
            retries,
            last_error,
        )
        raise LlmApiRequestError(str(last_error))

    if first_chunk is not None and first_chunk.content:
        yield first_chunk.content
    if stream is not None:
        async for chunk in stream:
            if chunk.content:
                yield chunk.content
