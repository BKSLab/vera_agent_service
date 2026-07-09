class InvalidAgentRequestError(Exception):
    """Payload сообщения `agent.requests` не прошёл валидацию
    `app/messaging/schemas.py::AgentRequestMessage` — системный сбой,
    сообщение уходит в `agent.requests.dlq` без ретраев (не имеет смысла
    повторять заведомо невалидное сообщение)."""

    def __init__(self, error_details: str):
        self.error_details = error_details
        super().__init__(self.error_details)
