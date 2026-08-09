import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import MessageFeedbackServiceDep
from app.exceptions.chat_session import ChatSessionAccessDeniedError
from app.exceptions.chat_turn import (
    ChatTurnNotCompletedError,
    ChatTurnNotFoundError,
    ChatTurnSessionMismatchError,
)
from app.exceptions.message_feedback import MessageFeedbackServiceError
from app.schemas.message_feedback import (
    MessageFeedbackRequest,
    MessageFeedbackResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/feedback', tags=['Обратная связь'], dependencies=[VerifyApiKeyDep])


@router.put(
    path='/message',
    status_code=status.HTTP_200_OK,
    summary='Сохранить оценку ответа',
    description='Создаёт или изменяет оценку up/down для одного завершённого ответа Веры.',
    operation_id='upsertMessageFeedback',
    response_description='Текущее состояние оценки ответа.',
    response_model=MessageFeedbackResponse,
    responses={
        401: {'description': 'Невалидный сервисный API-ключ.'},
        404: {'description': 'Реплика не найдена.'},
        409: {'description': 'Реплика не относится к сессии или ещё не завершена.'},
        422: {'description': 'Ошибка валидации запроса.'},
        429: {'description': 'Превышен лимит запросов.'},
        500: {'description': 'Ошибка сохранения оценки.'},
    },
)
@limiter.limit('60/minute')
async def upsert_message_feedback(
    request: Request,
    data: MessageFeedbackRequest,
    service: MessageFeedbackServiceDep,
    user_id: Annotated[
        str | None,
        Header(alias='X-Vera-User-ID', max_length=255),
    ] = None,
    anonymous_token_hash: Annotated[
        str | None,
        Header(alias='X-Vera-Anonymous-Token-Hash', min_length=64, max_length=64),
    ] = None,
) -> MessageFeedbackResponse:
    """Создаёт или изменяет оценку конкретного ответа.

    Args:
        request: HTTP-запрос для rate limit.
        data: Идентификаторы и значение оценки.
        service: Сервис оценок ответов.

    Returns:
        Текущее состояние оценки.
    """
    logger.info(
        '🚀 Сохранение оценки ответа. session_id=%s, request_id=%s.',
        data.session_id,
        data.request_id,
    )
    try:
        feedback = await service.upsert_feedback(
            session_id=data.session_id,
            request_id=data.request_id,
            value=data.value,
            user_id=user_id,
            anonymous_token_hash=anonymous_token_hash,
        )
        logger.info('✅ Оценка ответа сохранена. request_id=%s.', data.request_id)
        return MessageFeedbackResponse(
            id=feedback.id,
            session_id=data.session_id,
            request_id=data.request_id,
            value=feedback.value,
            review_status=feedback.review_status,
            created_at=feedback.created_at,
            updated_at=feedback.updated_at,
        )
    except (
        ChatSessionAccessDeniedError,
        ChatTurnNotFoundError,
        ChatTurnSessionMismatchError,
        ChatTurnNotCompletedError,
        MessageFeedbackServiceError,
    ) as error:
        logger.exception(
            '❌ Не удалось сохранить оценку. request_id=%s. Тип=%s.',
            data.request_id,
            type(error).__name__,
        )
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error
