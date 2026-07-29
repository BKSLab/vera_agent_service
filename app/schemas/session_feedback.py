from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class SessionFeedbackRequest(BaseModel):
    """Запрос на сохранение развёрнутого отзыва по сессии."""

    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'session_id': 'conversation-uuid',
                'submission_id': 'feedback-submission-uuid',
                'audience': 'employer',
                'usefulness': 3,
                'trust': 2,
                'comment': 'Не хватило пояснения по источнику',
                'contact_email': 'user@example.ru',
            }
        }
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description='Идентификатор существующей сессии.',
        examples=['conversation-uuid'],
    )
    submission_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description='Ключ идемпотентности одной отправки анкеты.',
        examples=['feedback-submission-uuid'],
    )
    audience: Literal['seeker', 'employer', 'other'] | None = Field(
        None,
        description='Аудитория пользователя.',
        examples=['employer'],
    )
    usefulness: int | None = Field(
        None,
        ge=1,
        le=5,
        description='Полезность консультации от 1 до 5.',
        examples=[3],
    )
    trust: int | None = Field(
        None,
        ge=1,
        le=5,
        description='Доверие к ответу от 1 до 5.',
        examples=[2],
    )
    comment: str | None = Field(
        None,
        max_length=4000,
        description='Свободный комментарий пользователя.',
        examples=['Не хватило пояснения по источнику'],
    )
    contact_email: EmailStr | None = Field(
        None,
        max_length=320,
        description='Контактный email пользователя.',
        examples=['user@example.ru'],
    )


class SessionFeedbackResponse(BaseModel):
    """Сохранённый развёрнутый отзыв."""

    id: UUID = Field(..., description='Внутренний идентификатор отзыва.')
    session_id: str = Field(..., description='Идентификатор сессии.')
    submission_id: str = Field(..., description='Ключ идемпотентности отправки.')
    review_status: str = Field(..., description='Статус экспертной проверки.')
    created_at: datetime = Field(..., description='Дата сохранения отзыва.')
