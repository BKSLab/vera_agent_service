class LlmApiRequestError(Exception):
    """Финальная ошибка: все попытки нестримингованного вызова LLM
    (`ainvoke_with_retry`, `app/clients/llm.py`) исчерпаны.

    Пересекает границу клиента — аналог `LlmApiRequestError` из
    `LLM_CLIENT_REFERENCE.md`, хотя механизм ретраев здесь другой
    (обёртка над LangChain `Runnable`, не собственный HTTP-клиент).
    """

    def __init__(self, error_details: str):
        self.error_details = error_details
        super().__init__(self.error_details)

    def __str__(self) -> str:
        return f'Ошибка запроса к LLM API. Подробности: {self.error_details}'


class EmptyLlmStreamError(RuntimeError):
    """LLM завершила стрим без видимого текстового ответа."""

    def __init__(self) -> None:
        super().__init__('LLM не вернула ни одного токена ответа')
