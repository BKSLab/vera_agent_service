import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Path, Query, Request, status

from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import (
    ChatHistoryServiceDep,
    ChatSessionLifecycleServiceDep,
)
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionNotFoundError,
    ChatSessionServiceError,
)
from app.schemas.chat_history import (
    ChatHistoryResponse,
    ChatHistoryTurnResponse,
    CurrentChatSessionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix='/chat',
    tags=['История диалога'],
    dependencies=[VerifyApiKeyDep],
)


@router.get(
    path='/sessions/current',
    status_code=status.HTTP_200_OK,
    summary='Получить текущую сессию пользователя',
    description=(
        'Возвращает последнюю открытую сессию авторизованного пользователя '
        'как кандидата для обязательного resolve. Если пользователь ещё не '
        'общался с Верой, session_id равен null.'
    ),
    operation_id='getCurrentChatSession',
    response_description='Текущая сессия пользователя.',
    response_model=CurrentChatSessionResponse,
    responses={
        401: {'description': 'Невалидный сервисный API-ключ.'},
        422: {'description': 'Не передан идентификатор пользователя.'},
        429: {'description': 'Превышен лимит запросов.'},
        500: {'description': 'Ошибка чтения сессии.'},
    },
)
@limiter.limit('60/minute')
async def get_current_chat_session(
    request: Request,
    service: ChatSessionLifecycleServiceDep,
    user_id: Annotated[
        str,
        Header(alias='X-Vera-User-ID', min_length=1, max_length=255),
    ],
) -> CurrentChatSessionResponse:
    """Возвращает predecessor candidate для последующего resolve."""
    logger.info('🚀 Поиск текущей сессии пользователя.')
    try:
        chat_session = await service.get_current_user_session(user_id)
        return CurrentChatSessionResponse(
            session_id=(
                chat_session.session_id if chat_session is not None else None
            )
        )
    except ChatSessionServiceError as error:
        logger.exception(
            '❌ Не удалось найти текущую сессию пользователя.'
        )
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error


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
        403: {'description': 'Сессия принадлежит другому владельцу.'},
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
    user_id: Annotated[
        str | None,
        Header(alias='X-Vera-User-ID', max_length=255),
    ] = None,
    anonymous_token_hash: Annotated[
        str | None,
        Header(alias='X-Vera-Anonymous-Token-Hash', min_length=64, max_length=64),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    before_sequence: Annotated[int | None, Query(ge=1)] = None,
) -> ChatHistoryResponse:
    """Возвращает пользовательскую историю существующей сессии."""
    logger.info('🚀 Загрузка истории диалога. session_id=%s.', session_id)
    try:
        page = await service.get_history(
            session_id,
            user_id=user_id,
            anonymous_token_hash=anonymous_token_hash,
            limit=limit,
            before_sequence=before_sequence,
        )
        logger.info(
            '✅ История диалога загружена. session_id=%s, turns=%s.',
            session_id,
            len(page.turns),
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
                    used_knowledge_base=bool(turn.sources),
                    created_at=turn.created_at,
                    completed_at=turn.completed_at,
                )
                for turn in page.turns
            ],
            next_before_sequence=page.next_before_sequence,
        )
    except (
        ChatSessionAccessDeniedError,
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
