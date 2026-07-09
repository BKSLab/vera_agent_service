from pydantic import BaseModel, Field

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
    `session_id`.
    """

    session_id: str = Field(min_length=1)
    user_id: str | None = None
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
