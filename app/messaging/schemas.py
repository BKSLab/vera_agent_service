from typing import Self

from pydantic import BaseModel, Field, model_validator

MAX_MESSAGE_LENGTH: int = 4000
"""Лимит длины пользовательского сообщения — защита от аномально больших
сообщений (AGENT_SERVICE_PLAN.md, раздел 6, открытый вопрос: конкретное
значение подлежит подтверждению; 4000 символов — предложение по
умолчанию, с большим запасом по сравнению с типичным вопросом)."""


class AgentRequestMessage(BaseModel):
    """Payload очереди `agent.requests` (контракт — AGENT_SERVICE_PLAN.md,
    раздел 3.1).

    Поле `history` **сознательно отсутствует** — единственный источник
    истории диалога это Redis-checkpointer (Этап 5), ключ треда —
    `session_id`. `request_id` адресует доставку ответа конкретного сообщения
    и не участвует в накоплении истории.
    """

    session_id: str = Field(min_length=1, max_length=100)
    request_id: str = Field(min_length=1, max_length=100)
    user_id: str | None = Field(default=None, min_length=1, max_length=255)
    anonymous_token_hash: str | None = Field(default=None, min_length=64, max_length=64)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @model_validator(mode='after')
    def validate_owner(self) -> Self:
        """Запрещает создание сессии без механизма владения."""
        if self.user_id is None and self.anonymous_token_hash is None:
            raise ValueError('Должен быть указан владелец запроса')
        return self
