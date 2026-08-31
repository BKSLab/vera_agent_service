class LlmApiRequestError(Exception):
    """Финальная ошибка: все допустимые попытки вызова LLM исчерпаны.

    Используется и LangChain-вызовами маршрутизации, и прямой безопасной
    границей финального ответа. До consumer доходит только безопасный код
    причины, без тела ответа провайдера и исходного exception message.
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
