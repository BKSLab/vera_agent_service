import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import ChatSessionLifecycleServiceDep
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionResolutionConflictError,
    ChatSessionServiceError,
)
from app.schemas.chat_session_lifecycle import (
    ResolveChatSessionRequest,
    ResolveChatSessionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix='/chat',
    tags=['Жизненный цикл диалога'],
    dependencies=[VerifyApiKeyDep],
)


@router.post(
    path='/sessions/resolve',
    status_code=status.HTTP_200_OK,
    summary='Определить активную границу диалога',
    description=(
        'Синхронно создаёт первую сессию, сохраняет живую либо закрывает '
        'истёкшую и создаёт переданную замену.'
    ),
    operation_id='resolveChatSession',
    response_description='Фактическая сессия и серверная граница контекста.',
    response_model=ResolveChatSessionResponse,
    responses={
        401: {'description': 'Невалидный сервисный API-ключ.'},
        403: {'description': 'Сессия принадлежит другому владельцу.'},
        409: {'description': 'Для сессии уже создан другой successor.'},
        422: {'description': 'Ошибка валидации тела или owner-заголовков.'},
        429: {'description': 'Превышен лимит запросов.'},
        500: {'description': 'Ошибка PostgreSQL или Redis.'},
    },
)
@limiter.limit('60/minute')
async def resolve_chat_session(
    request: Request,
    data: ResolveChatSessionRequest,
    service: ChatSessionLifecycleServiceDep,
    user_id: Annotated[
        str | None,
        Header(alias='X-Vera-User-ID', max_length=255),
    ] = None,
    anonymous_token_hash: Annotated[
        str | None,
        Header(
            alias='X-Vera-Anonymous-Token-Hash',
            min_length=64,
            max_length=64,
        ),
    ] = None,
    refreshed_anonymous_token_hash: Annotated[
        str | None,
        Header(
            alias='X-Vera-Refreshed-Anonymous-Token-Hash',
            min_length=64,
            max_length=64,
        ),
    ] = None,
    replacement_anonymous_token_hash: Annotated[
        str | None,
        Header(
            alias='X-Vera-Replacement-Anonymous-Token-Hash',
            min_length=64,
            max_length=64,
        ),
    ] = None,
) -> ResolveChatSessionResponse:
    """Возвращает сессию, которую BFF должен использовать до publish.

    Args:
        request: HTTP-запрос для rate limit.
        data: Текущий и запасной идентификаторы сессии.
        service: Сервис единого жизненного цикла.
        user_id: Авторизованный владелец, если есть.
        anonymous_token_hash: Доказательство текущего anonymous-владельца.
        refreshed_anonymous_token_hash: Новый hash retained-сессии.
        replacement_anonymous_token_hash: Hash заранее созданной замены.

    Returns:
        Серверное решение о границе контекста.
    """
    logger.info(
        '🚀 Определение границы диалога. session_id=%s.',
        data.session_id,
    )
    try:
        resolution = await service.resolve_session(
            session_id=data.session_id,
            replacement_session_id=data.replacement_session_id,
            user_id=user_id,
            anonymous_token_hash=anonymous_token_hash,
            refreshed_anonymous_token_hash=(
                refreshed_anonymous_token_hash
            ),
            replacement_anonymous_token_hash=(
                replacement_anonymous_token_hash
            ),
        )
        logger.info(
            '✅ Граница диалога определена. session_id=%s, boundary=%s.',
            resolution.session_id,
            resolution.boundary,
        )
        return ResolveChatSessionResponse(
            session_id=resolution.session_id,
            previous_session_id=resolution.previous_session_id,
            boundary=resolution.boundary,
            session_ttl_seconds=resolution.session_ttl_seconds,
        )
    except (
        ChatSessionAccessDeniedError,
        ChatSessionResolutionConflictError,
        ChatSessionServiceError,
    ) as error:
        logger.exception(
            '❌ Не удалось определить границу диалога. session_id=%s. Тип=%s.',
            data.session_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error
