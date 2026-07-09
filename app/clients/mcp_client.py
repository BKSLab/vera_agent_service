import asyncio
import json
import logging
import random
from typing import Any

from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.settings import McpSettings
from app.exceptions.mcp import McpUnavailableError

logger = logging.getLogger('vera_agent_service')

DEFAULT_RETRY_DELAY: float = 0.5
DEFAULT_MAX_RETRY_DELAY: float = 5.0
JITTER_RATIO: float = 0.1

MCP_SERVER_NAME = 'vera-tools'
"""Имя сервера в конфигурации `MultiServerMCPClient` — единственный сервер
на итерацию 1 (см. AGENT_SERVICE_PLAN.md, раздел 0 — MCP Tools Server один,
инструментов пока тоже один: `kb_search`)."""


def get_mcp_client(settings: McpSettings) -> MultiServerMCPClient:
    """Создаёт клиент MCP Tools Server.

    Сервиса пока не существует (AGENT_SERVICE_PLAN.md, раздел 0) — клиент
    собирается против конфигурируемого `MCP_SERVER_URL`, но в тестах и
    локальной разработке подменяется мок-сервером
    (`tests/fixtures/mock_mcp_server.py`).

    `handle_tool_errors=False` — ошибка выполнения тула на стороне
    MCP-сервера должна прийти как исключение, а не как текстовый
    content-блок вида `"Error executing tool ..."` — иначе
    `call_tool_with_retry` не сможет отличить ошибку от валидного ответа.
    """
    return MultiServerMCPClient(
        {
            MCP_SERVER_NAME: {
                'url': settings.mcp_server_url,
                'transport': 'sse',
                'timeout': settings.mcp_call_timeout_seconds,
                'sse_read_timeout': settings.mcp_call_timeout_seconds,
            }
        },
        handle_tool_errors=False,
    )


def _get_backoff_delay(attempt: int) -> float:
    base_delay = min(DEFAULT_MAX_RETRY_DELAY, DEFAULT_RETRY_DELAY * (2 ** (attempt - 1)))
    jitter = base_delay * JITTER_RATIO * random.random()
    return base_delay + jitter


async def get_tools_with_retry(
    client: MultiServerMCPClient,
    retries: int,
    timeout_seconds: float,
) -> list[BaseTool]:
    """Список тулов MCP Tools Server с ретраями на уровне клиента (Этап 3.2
    плана) — независимо от retry-политики самого RabbitMQ-сообщения
    (раздел 0.1, Этап 6.3): здесь устойчивость одного вызова, там —
    устойчивость доставки сообщения.

    Raises:
        McpUnavailableError: если все попытки исчерпаны.
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await asyncio.wait_for(client.get_tools(), timeout=timeout_seconds)
        except Exception as error:
            # Соединение с MCP-сервером падает через anyio TaskGroup —
            # реальная ошибка приходит завёрнутой в ExceptionGroup, а не как
            # конкретный сетевой тип исключения (проверено эмпирически) —
            # поэтому ловим широко, тип ошибки не имеет значения для
            # решения "поиск сейчас недоступен".
            last_error = error
            logger.warning('⚠️ MCP Tools Server недоступен (попытка %d/%d): %s', attempt, retries, error)
            if attempt < retries:
                delay = _get_backoff_delay(attempt)
                logger.info('🔄 Повтор через %.1fс (следующая попытка: %d/%d)', delay, attempt + 1, retries)
                await asyncio.sleep(delay)

    logger.error('❌ MCP Tools Server недоступен после %d попыток. Последняя ошибка: %s', retries, last_error)
    raise McpUnavailableError(str(last_error))


async def call_tool_with_retry(
    tool: BaseTool,
    arguments: dict,
    retries: int,
    timeout_seconds: float,
) -> dict:
    """Вызывает MCP-тул с ретраями и разбирает content-блоки ответа в
    обычный `dict` (см. `_parse_tool_result`).

    Raises:
        McpUnavailableError: если все попытки исчерпаны — сеть, таймаут,
            ошибка выполнения тула на MCP-сервере или неожиданный формат
            ответа (раздел 0.1: для вызывающего кода все три равнозначны).
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw_result = await asyncio.wait_for(tool.ainvoke(arguments), timeout=timeout_seconds)
            return _parse_tool_result(raw_result)
        except Exception as error:
            last_error = error
            logger.warning(
                '⚠️ Ошибка вызова тула %s (попытка %d/%d): %s', tool.name, attempt, retries, error
            )
            if attempt < retries:
                delay = _get_backoff_delay(attempt)
                logger.info('🔄 Повтор через %.1fс (следующая попытка: %d/%d)', delay, attempt + 1, retries)
                await asyncio.sleep(delay)

    logger.error(
        '❌ Не удалось вызвать тул %s после %d попыток. Последняя ошибка: %s', tool.name, retries, last_error
    )
    raise McpUnavailableError(str(last_error))


def build_kb_search_tool_proxy(client: MultiServerMCPClient) -> BaseTool:
    """Локальный тул с той же сигнатурой и описанием, что и удалённый
    `kb_search` на MCP Tools Server (контракт — раздел 3.3 плана).

    Нужен, чтобы `app/graph/build.py` (Этап 4) мог собрать граф и
    вызвать `.bind_tools([kb_search_tool])` (`analyze_intent`, Этап 4.1)
    **без сетевого обращения к MCP Tools Server** — сервиса ещё не
    существует (раздел 0 плана), а `.bind_tools()` нужна только схема
    аргументов, не факт доступности сервера. Реальный удалённый тул
    резолвится лениво при **первом фактическом вызове** (не при создании
    графа) и кешируется на весь срок жизни процесса — повторные вызовы не
    делают лишний `get_tools()`.

    Проксирование — без собственного retry: `call_kb_search` (Этап 4.2)
    вызывает этот тул через `call_tool_with_retry`, которая уже
    оборачивает вызов ретраями/таймаутом — дублировать эту логику здесь
    означало бы ретраи в квадрате (N попыток снаружи × N попыток внутри).
    """
    resolved_tool: list[BaseTool] = []

    @tool
    async def kb_search(query: str, audience: str = 'both') -> dict:
        """Поиск по базе знаний о правах людей с инвалидностью в сфере
        трудоустройства и трудовой деятельности.

        audience: 'seeker' | 'employer' | 'both'
        """
        if not resolved_tool:
            tools = await client.get_tools()
            resolved_tool.append(next(candidate for candidate in tools if candidate.name == 'kb_search'))
        return await resolved_tool[0].ainvoke({'query': query, 'audience': audience})

    return kb_search


def _parse_tool_result(raw_result: Any) -> dict:
    """Разбирает ответ MCP-тула в обычный `dict`.

    MCP-протокол возвращает результат тула как список content-блоков
    (`[{"type": "text", "text": "<json>"}]`), не сырой `dict` — проверено
    эмпирически с `langchain-mcp-adapters==0.3.0`. Ожидаемый формат
    text-блока для `kb_search` — JSON, совпадающий по структуре с ответом
    `POST /api/v1/search` из `vera_rag_service` (раздел 3.3 плана).
    """
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, list) and raw_result:
        first_block = raw_result[0]
        text = first_block.get('text') if isinstance(first_block, dict) else None
        if text:
            return json.loads(text)
    raise McpUnavailableError(f'Неожиданный формат ответа MCP-тула: {raw_result!r}')
