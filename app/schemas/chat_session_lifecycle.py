from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResolveChatSessionRequest(BaseModel):
    """Тело синхронного определения границы диалога."""

    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'session_id': 'current-conversation-uuid',
                'replacement_session_id': 'replacement-conversation-uuid',
            }
        }
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description='Текущий идентификатор сессии диалога.',
    )
    replacement_session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description='Заранее созданный идентификатор на случай истечения.',
    )

    @model_validator(mode='after')
    def validate_distinct_session_ids(self) -> 'ResolveChatSessionRequest':
        """Не позволяет заменить истёкшую сессию ею же."""
        if self.session_id == self.replacement_session_id:
            raise ValueError(
                'replacement_session_id должен отличаться от session_id'
            )
        return self


class ResolveChatSessionResponse(BaseModel):
    """Результат определения серверной границы диалога."""

    session_id: str = Field(
        ...,
        description='Фактический идентификатор активной сессии.',
    )
    previous_session_id: str | None = Field(
        None,
        description='Закрытая предыдущая сессия при boundary=expired.',
    )
    boundary: Literal['created', 'retained', 'expired'] = Field(
        ...,
        description='Серверное решение о границе контекста.',
    )
    session_ttl_seconds: int = Field(
        ...,
        gt=0,
        description='Единый TTL неактивности PostgreSQL и Redis.',
    )
