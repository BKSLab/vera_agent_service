from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageFeedbackRequest(BaseModel):
    """Запрос на создание или изменение оценки ответа."""

    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'session_id': 'conversation-uuid',
                'request_id': 'message-uuid',
                'value': 'down',
            }
        }
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description='Идентификатор сессии диалога.',
        examples=['conversation-uuid'],
    )
    request_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description='Идентификатор оцениваемого ответа.',
        examples=['message-uuid'],
    )
    value: Literal['up', 'down'] = Field(
        ...,
        description='Положительная или отрицательная оценка ответа.',
        examples=['down'],
    )


class MessageFeedbackResponse(BaseModel):
    """Текущее состояние оценки конкретного ответа."""

    id: UUID = Field(..., description='Внутренний идентификатор оценки.')
    session_id: str = Field(..., description='Идентификатор сессии диалога.')
    request_id: str = Field(..., description='Идентификатор оценённого ответа.')
    value: Literal['up', 'down'] = Field(..., description='Текущее значение оценки.')
    review_status: str = Field(..., description='Статус экспертной проверки.')
    created_at: datetime = Field(..., description='Дата создания оценки.')
    updated_at: datetime = Field(..., description='Дата последнего изменения оценки.')
