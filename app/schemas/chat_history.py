from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CurrentChatSessionResponse(BaseModel):
    """Текущая сессия авторизованного пользователя."""

    session_id: str | None = Field(
        None,
        description='Идентификатор последней активной сессии или null.',
    )


class ChatHistoryTurnResponse(BaseModel):
    """Одна пользовательская реплика и ответ Веры."""

    request_id: str = Field(..., description='Идентификатор запроса и ответа.')
    sequence_number: int = Field(..., ge=1, description='Позиция реплики в диалоге.')
    question: str = Field(..., description='Исходный вопрос пользователя.')
    answer: str | None = Field(None, description='Сохранённый ответ Веры.')
    status: str = Field(..., description='Статус обработки реплики.')
    feedback_value: Literal['up', 'down'] | None = Field(
        None,
        description='Сохранённая пользователем оценка ответа.',
    )
    created_at: datetime = Field(..., description='Дата создания реплики.')
    completed_at: datetime | None = Field(
        None,
        description='Дата завершения обработки реплики.',
    )


class ChatHistoryResponse(BaseModel):
    """История одной сессии диалога с Верой."""

    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'session_id': 'conversation-uuid',
                'turns': [
                    {
                        'request_id': 'message-uuid',
                        'sequence_number': 1,
                        'question': 'Какая квота действует для работодателей?',
                        'answer': 'Размер квоты зависит от численности работников.',
                        'status': 'completed',
                        'feedback_value': 'up',
                        'created_at': '2026-07-29T12:00:00Z',
                        'completed_at': '2026-07-29T12:00:05Z',
                    }
                ],
            }
        }
    )

    session_id: str = Field(..., description='Идентификатор сессии диалога.')
    turns: list[ChatHistoryTurnResponse] = Field(
        default_factory=list,
        description='Реплики в порядке их появления.',
    )
    next_before_sequence: int | None = Field(
        None,
        description='Курсор следующей страницы более старых реплик.',
    )
