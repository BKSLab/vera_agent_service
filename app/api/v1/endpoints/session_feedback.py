import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import SessionFeedbackServiceDep
from app.exceptions.chat_session import ChatSessionNotFoundError
from app.exceptions.session_feedback import (
    SessionFeedbackServiceError,
    SessionFeedbackSubmissionMismatchError,
)
from app.schemas.session_feedback import (
    SessionFeedbackRequest,
    SessionFeedbackResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/feedback', tags=['Обратная связь'], dependencies=[VerifyApiKeyDep])


@router.post(
    path='/session',
    status_code=status.HTTP_201_CREATED,
    summary='Сохранить развёрнутый отзыв',
    description='Идемпотентно сохраняет анкету обратной связи по существующей сессии.',
    operation_id='createSessionFeedback',
    response_description='Сохранённый развёрнутый отзыв.',
    response_model=SessionFeedbackResponse,
    responses={
        401: {'description': 'Невалидный сервисный API-ключ.'},
        404: {'description': 'Сессия не найдена.'},
        409: {'description': 'submission_id уже относится к другой сессии.'},
        422: {'description': 'Ошибка валидации запроса.'},
        429: {'description': 'Превышен лимит запросов.'},
        500: {'description': 'Ошибка сохранения отзыва.'},
    },
)
@limiter.limit('10/minute')
async def create_session_feedback(
    request: Request,
    data: SessionFeedbackRequest,
    service: SessionFeedbackServiceDep,
) -> SessionFeedbackResponse:
    """Сохраняет развёрнутый отзыв по сессии.

    Args:
        request: HTTP-запрос для rate limit.
        data: Поля анкеты и ключ идемпотентности.
        service: Сервис развёрнутых отзывов.

    Returns:
        Идентификаторы сохранённого отзыва.
    """
    logger.info(
        '🚀 Сохранение отзыва по сессии. session_id=%s, submission_id=%s.',
        data.session_id,
        data.submission_id,
    )
    try:
        feedback = await service.create_feedback(
            session_id=data.session_id,
            submission_id=data.submission_id,
            audience=data.audience,
            usefulness=data.usefulness,
            trust=data.trust,
            comment=data.comment,
            contact_email=str(data.contact_email) if data.contact_email is not None else None,
        )
        logger.info(
            '✅ Отзыв по сессии сохранён. submission_id=%s.',
            data.submission_id,
        )
        return SessionFeedbackResponse(
            id=feedback.id,
            session_id=data.session_id,
            submission_id=feedback.submission_id,
            review_status=feedback.review_status,
            created_at=feedback.created_at,
        )
    except (
        ChatSessionNotFoundError,
        SessionFeedbackSubmissionMismatchError,
        SessionFeedbackServiceError,
    ) as error:
        logger.exception(
            '❌ Не удалось сохранить отзыв. submission_id=%s. Тип=%s.',
            data.submission_id,
            type(error).__name__,
        )
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error
