import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Request, status

from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import ChatHistoryServiceDep
from app.exceptions.chat_session import (
    ChatSessionNotFoundError,
    ChatSessionServiceError,
)
from app.schemas.chat_history import (
    ChatHistoryResponse,
    ChatHistoryTurnResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix='/chat',
    tags=['История диалога'],
    dependencies=[VerifyApiKeyDep],
)


@router.get(
    path='/sessions/{session_id}/history',
    status_code=status.HTTP_200_OK,
    summary='Получить историю диалога',
    description=(
        'Возвращает сохранённые вопросы, ответы Веры и оценки '
        'в порядке их появления в сессии.'
    ),
    operation_id='getChatHistory',
    response_description='Сохранённая история диалога.',
    response_model=ChatHistoryResponse,
    responses={
        401: {'description': 'Невалидный сервисный API-ключ.'},
        404: {'description': 'Сессия не найдена.'},
        422: {'description': 'Ошибка валидации session_id.'},
        429: {'description': 'Превышен лимит запросов.'},
        500: {'description': 'Ошибка чтения истории.'},
    },
)
@limiter.limit('60/minute')
async def get_chat_history(
    request: Request,
    session_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=100,
            description='Идентификатор сессии диалога.',
        ),
    ],
    service: ChatHistoryServiceDep,
) -> ChatHistoryResponse:
    """Возвращает пользовательскую историю существующей сессии."""
    logger.info('🚀 Загрузка истории диалога. session_id=%s.', session_id)
    try:
        turns = await service.get_history(session_id)
        logger.info(
            '✅ История диалога загружена. session_id=%s, turns=%s.',
            session_id,
            len(turns),
        )
        return ChatHistoryResponse(
            session_id=session_id,
            turns=[
                ChatHistoryTurnResponse(
                    request_id=turn.request_id,
                    sequence_number=turn.sequence_number,
                    question=turn.question,
                    answer=turn.answer,
                    status=turn.status,
                    feedback_value=(
                        turn.feedback.value
                        if turn.feedback is not None
                        else None
                    ),
                    created_at=turn.created_at,
                    completed_at=turn.completed_at,
                )
                for turn in turns
            ],
        )
    except (
        ChatSessionNotFoundError,
        ChatSessionServiceError,
    ) as error:
        logger.exception(
            '❌ Не удалось загрузить историю. session_id=%s. Тип=%s.',
            session_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error
