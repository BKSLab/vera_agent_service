class McpUnavailableError(Exception):
    """MCP Tools Server недоступен — все попытки подключения/вызова тула
    исчерпаны (`app/clients/mcp_client.py`).

    Единое исключение для трёх сценариев одной попытки: сеть/таймаут,
    ошибка выполнения тула на стороне MCP-сервера, неожиданный формат
    ответа. Для узла графа `call_kb_search` (Этап 4.2) все три означают
    одно и то же — "результат недоступен, нужно честно сообщить
    пользователю, не выдумывать ответ" (AGENT_SERVICE_PLAN.md, раздел 0.1)
    — различать их дальше по стеку не требуется.

    Перехватывается внутри узла графа `call_kb_search` и не пробрасывается
    дальше как необработанное исключение — граф должен деградировать, а не
    падать.
    """

    def __init__(self, error_details: str):
        self.error_details = error_details
        super().__init__(self.error_details)

    def __str__(self) -> str:
        return f'MCP Tools Server недоступен. Подробности: {self.error_details}'
