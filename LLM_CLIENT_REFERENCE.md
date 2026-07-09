# Эталонная реализация LLM-клиента

> Сопровождающий документ к `FASTAPI_PATTERNS.md`, раздел 13 («Клиенты внешних API/LLM»).
> В отличие от основного документа, здесь код приводится **целиком**, готовый к копированию
> в `clients/llm.py` нового проекта — паттерны retry/backoff и извлечения контента у LLM-клиентов
> достаточно стабильны, чтобы не абстрагировать их в псевдокод.

## Что исправлено относительно исходной версии

Код адаптирован из рабочего проекта и почищен от нескольких проблем, которые не стоит переносить дальше:

1. **Убран `pprint(response)` в `_extract_content`** — debug-остаток, который печатал в stdout полный ответ LLM на каждый вызов (потенциально большой и чувствительный текст). Заменён на `logger.debug(...)` — диагностика осталась, но не засоряет вывод в проде (где `DEBUG` обычно выключен).
2. **Реализован, а не закомментирован, `response_format`** — было `# if schema: payload["response_format"] = {"type": "json_object"}`. Закомментированный код — мёртвый груз; либо реализуем, либо убираем. Реализовали: если передана Pydantic-схема, клиент сам просит у LLM JSON-режим.
3. **`self.timeout` теперь реально используется** — в исходной версии атрибут выставлялся в конструкторе, но не передавался в `httpx_client.post(...)`; реальный таймаут брался из настроек самого `httpx.AsyncClient`. Это два независимых источника правды с одинаковым значением `90` — ловушка для будущего рефакторинга. Теперь `timeout` явно передаётся в каждый запрос.
4. **`lambda` заменена на `functools.partial`** — присвоение lambda переменной (`extractor = lambda response: ...`) ловится линтерами (`ruff`/`flake8: E731`) и менее читаемо, чем частичное применение функции.
5. **Провайдер-специфичные поля payload вынесены из жёсткого кода** — в исходной версии в payload на каждый запрос добавлялся блок `"reasoning": {...}`, специфичный для конкретного LLM-провайдера/прокси, хотя класс называется «клиент для OpenAI-совместимых API». Стандартный контракт Chat Completions этого поля не знает. Вынесено в параметр конструктора `extra_payload: dict | None` — провайдер-специфичные расширения передаются явно при создании клиента, а не зашиваются в общий код.
6. Магические числа retry-стратегии (`timeout`, `retries`, `delay`, `max_delay`) подняты из тела `__init__` в **константы класса** с возможностью переопределить через конструктор — соответствует правилу «константы поведения — `UPPER_CASE` атрибуты класса» (`FASTAPI_PATTERNS.md`, раздел 9).

Что оставлено как было (это и так хорошие решения):
- Разделение retry на «ошибка запроса» (сеть/HTTP/таймаут) и «ошибка контента» (невалидный JSON/пустой ответ/не прошла валидация схемы) — оба типа уходят в retry, но логируются разными сообщениями.
- Экспоненциальный backoff с джиттером — `delay = min(max_delay, base_delay * 2**(attempt-1))`, плюс случайные ±10%, чтобы параллельные клиенты не повторяли запрос синхронно.
- Schema-валидация ответа через `schema.model_validate_json(content)` сразу после извлечения текста.

## `exceptions/llm.py`

```python
from fastapi import status


class LlmClientRequestError(Exception):
    """Ошибка одной попытки запроса к LLM API (сеть, HTTP, таймаут).

    Перехватывается и retry'ится внутри LlmClient — никогда не пересекает
    границу клиента, поэтому у неё намеренно нет status_code/detail
    (см. FASTAPI_PATTERNS.md, раздел 8 — "внутренние" исключения).
    """


class LlmClientContentError(Exception):
    """Ошибка контента одной попытки: пустой ответ, невалидный JSON,
    ответ не прошёл валидацию по Pydantic-схеме.

    Как и LlmClientRequestError — внутреннее исключение одной попытки,
    обрабатывается retry-циклом, до сервиса/эндпоинта не доходит.
    """


class LlmApiRequestError(Exception):
    """Финальная ошибка: все попытки запроса к LLM исчерпаны.

    Это единственное исключение модуля, которое пересекает границу
    клиента — у него есть status_code/detail, как у любого исключения,
    которое может всплыть до эндпоинта.
    """
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, error_details: str, request_url: str):
        self.error_details = error_details
        self.request_url = request_url
        super().__init__(self.error_details, self.request_url)

    def __str__(self) -> str:
        return (
            f"Ошибка запроса к LLM API. URL: {self.request_url}. "
            f"Подробности: {self.error_details}"
        )

    @property
    def detail(self) -> str:
        return f"Ошибка при запросе к LLM API. Подробности: {self.error_details}"
```

## `clients/llm.py`

```python
import asyncio
import functools
import json
import random
from collections.abc import Callable
from pprint import pformat
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from core.config_logger import logger
from exceptions.llm import (
    LlmApiRequestError,
    LlmClientContentError,
    LlmClientRequestError,
)

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


class LlmClient:
    """Клиент для LLM API, совместимых с контрактом OpenAI Chat Completions.

    Провайдер-специфичные расширения payload (нестандартные поля, которые
    понимает конкретный провайдер/прокси, но не описывает базовый контракт)
    передаются через extra_payload в конструкторе, а не зашиваются в код
    клиента — это позволяет переиспользовать класс для любого совместимого
    провайдера без правок логики.
    """

    DEFAULT_TIMEOUT_SECONDS: int = 90
    DEFAULT_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: float = 1.0
    DEFAULT_MAX_RETRY_DELAY: float = 30.0
    JITTER_RATIO: float = 0.1

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        model: str,
        url: str,
        headers: dict,
        temperature: float = 0.3,
        stream: bool = False,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
        delay: float = DEFAULT_RETRY_DELAY,
        max_delay: float = DEFAULT_MAX_RETRY_DELAY,
        extra_payload: dict | None = None,
    ):
        """Инициализирует клиент.

        Args:
            httpx_client: Готовый httpx.AsyncClient (жизненным циклом управляет
                вызывающий код через DI — см. FASTAPI_PATTERNS.md, раздел 6).
            model: Имя модели по умолчанию для запросов.
            url: URL эндпоинта Chat Completions провайдера.
            headers: Заголовки запроса (Authorization, Content-Type и т.п.).
            temperature: Температура генерации по умолчанию.
            stream: Признак потокового ответа.
            timeout: Таймаут одного HTTP-запроса в секундах.
            retries: Максимальное количество попыток на один вызов.
            delay: Базовая задержка перед повтором (секунды).
            max_delay: Верхняя граница задержки между повторами.
            extra_payload: Провайдер-специфичные поля, добавляемые в payload
                каждого запроса (например, нестандартный режим reasoning
                у конкретного провайдера). None — без расширений.
        """
        self.httpx_client = httpx_client
        self.model = model
        self.url = url
        self.headers = headers
        self.temperature = temperature
        self.stream = stream
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.max_delay = max_delay
        self.extra_payload = extra_payload or {}

    def _get_backoff_delay(self, attempt: int) -> float:
        """Вычисляет задержку по экспоненциальному алгоритму с джиттером.

        Attempt 1 → ~1s, attempt 2 → ~2s, attempt 3 → ~4s (до max_delay).
        Джиттер ±10% предотвращает одновременные повторы нескольких клиентов.
        """
        base_delay = min(self.max_delay, self.delay * (2 ** (attempt - 1)))
        jitter = base_delay * self.JITTER_RATIO * random.random()
        return base_delay + jitter

    async def _send_request_to_llm(self, payload: dict) -> dict:
        """Отправляет один запрос к LLM и возвращает сырой ответ.

        Raises:
            LlmClientRequestError: При HTTP-ошибке, таймауте или сетевой ошибке.
                Перехватывается retry-циклом, до сервиса не доходит.
        """
        data_json = json.dumps(payload, ensure_ascii=False)
        try:
            logger.info("📤 Отправка запроса к LLM, модель: %s", payload.get("model"))
            response = await self.httpx_client.post(
                url=self.url,
                headers=self.headers,
                data=data_json,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            logger.error(
                "🌐 HTTP %s от LLM: %s", error.response.status_code, error.response.text
            )
            raise LlmClientRequestError(f"HTTP {error.response.status_code}") from error
        except httpx.TimeoutException as error:
            logger.error("⏱️ Таймаут при запросе к LLM (%ss): %s", self.timeout, error)
            raise LlmClientRequestError("Таймаут запроса к LLM") from error
        except httpx.RequestError as error:
            logger.error(
                "🌐 Сетевая ошибка при запросе к LLM: %s: %s", type(error).__name__, error
            )
            raise LlmClientRequestError(f"Сетевая ошибка: {type(error).__name__}") from error

    def _extract_content(self, response: dict) -> str:
        """Извлекает и валидирует текстовый контент из ответа LLM.

        Raises:
            LlmClientContentError: Если структура ответа не соответствует
                ожидаемому контракту, или контент пустой.
        """
        try:
            content = response.get("choices")[0].get("message").get("content")
        except (KeyError, IndexError, TypeError) as error:
            logger.debug("Полный ответ LLM с невалидной структурой: %s", pformat(response))
            raise LlmClientContentError(
                f"Невалидная структура ответа LLM: {type(error).__name__}"
            ) from error

        if not content or not content.strip():
            raise LlmClientContentError("LLM вернул пустой ответ")

        return content

    def _extract_validated(self, response: dict, schema: type[PydanticModel]) -> PydanticModel:
        """Извлекает контент из ответа LLM и валидирует его по Pydantic-схеме.

        Raises:
            LlmClientContentError: Если контент не прошёл валидацию схемы.
        """
        content = self._extract_content(response)
        try:
            return schema.model_validate_json(content)
        except ValidationError as error:
            logger.warning(
                "📋 Ответ LLM не прошёл валидацию схемы %s: %s", schema.__name__, error
            )
            raise LlmClientContentError(
                f"Ответ не соответствует схеме {schema.__name__}"
            ) from error

    async def _fetch_with_retries(
        self, payload: dict, extractor: Callable[[dict], Any]
    ) -> Any:
        """Выполняет запрос к LLM с экспоненциальными повторами при любых ошибках.

        Различает ошибку самого запроса (сеть/HTTP/таймаут) и ошибку контента
        ответа (невалидный JSON, не прошёл валидацию схемы) — обе уходят
        в retry, но логируются разными сообщениями для диагностики.

        Raises:
            LlmApiRequestError: Если все попытки исчерпаны без успеха.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                response = await self._send_request_to_llm(payload)
                content = extractor(response)
                if attempt > 1:
                    logger.info("✅ Ответ от LLM получен с %s-й попытки", attempt)
                return content
            except LlmClientContentError as error:
                last_error = error
                logger.warning(
                    "📭 Некорректный контент от LLM (попытка %d/%d): %s",
                    attempt, self.retries, error,
                )
            except LlmClientRequestError as error:
                last_error = error
                logger.warning(
                    "⚠️ Ошибка запроса к LLM (попытка %d/%d): %s",
                    attempt, self.retries, error,
                )

            if attempt < self.retries:
                delay = self._get_backoff_delay(attempt)
                logger.info(
                    "🔄 Повтор через %.1fс (следующая попытка: %d/%d)",
                    delay, attempt + 1, self.retries,
                )
                await asyncio.sleep(delay)

        logger.error(
            "❌ Не удалось получить ответ от LLM после %d попыток. Последняя ошибка: %s",
            self.retries, last_error,
        )
        raise LlmApiRequestError(error_details=str(last_error), request_url=self.url)

    async def get_llm_response(
        self,
        content: str,
        prompt: str,
        model: str | None = None,
        schema: type[PydanticModel] | None = None,
        max_completion_tokens: int = 6000,
    ) -> str | PydanticModel:
        """Получает ответ от LLM.

        Args:
            content: Пользовательское сообщение (роль "user").
            prompt: Системный промпт (роль "system") — см. FASTAPI_PATTERNS.md,
                раздел 13: системные промпты хранятся отдельным модулем
                строковых констант, не собираются инлайн в коде сервиса.
            model: Модель для этого конкретного вызова, если отличается от
                модели по умолчанию, заданной в конструкторе.
            schema: Pydantic-схема для structured output. Если передана,
                клиент включает JSON-режим и валидирует ответ по схеме —
                ошибка валидации уходит в retry, а не сразу в финальный отказ.
            max_completion_tokens: Лимит токенов ответа.

        Returns:
            Текст ответа (если schema не передана) или валидированный
            экземпляр Pydantic-модели (если передана).

        Raises:
            LlmApiRequestError: Если все попытки запроса исчерпаны.
        """
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": max_completion_tokens,
            "stream": self.stream,
        }
        if schema:
            payload["response_format"] = {"type": "json_object"}
        if self.extra_payload:
            payload.update(self.extra_payload)

        extractor = (
            functools.partial(self._extract_validated, schema=schema)
            if schema
            else self._extract_content
        )
        return await self._fetch_with_retries(payload, extractor=extractor)
```

## Подключение через DI

Согласуется с `FASTAPI_PATTERNS.md`, раздел 6 — фабрика в `dependencies/clients.py`, провайдер-специфичные поля (если они есть у конкретного провайдера) передаются здесь, а не внутри `LlmClient`:

```python
def get_llm_client(httpx_client: HTTPClientDep) -> LlmClient:
    return LlmClient(
        httpx_client=httpx_client,
        model=settings.llm.llm_model.get_secret_value(),
        url=settings.llm.llm_api_url.get_secret_value(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm.llm_api_key.get_secret_value()}",
        },
        # extra_payload={"reasoning": {"effort": "high"}},  # только если провайдер это поддерживает
    )


LlmClientDep = Annotated[LlmClient, Depends(get_llm_client)]
```
