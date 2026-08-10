import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Path, Request, status

from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import ChatSessionLifecycleServiceDep
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionAlreadyClosedError,
    ChatSessionNotFoundError,
    ChatSessionResolutionConflictError,
    ChatSessionServiceError,
)
from app.schemas.chat_session_lifecycle import (
    CloseChatSessionResponse,
    CreateChatSessionRequest,
    CreateChatSessionResponse,
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
    path='/sessions',
    status_code=status.HTTP_200_OK,
    summary='Создать новую сессию диалога',
    description=(
        'Создаёт сессию с переданным клиентским идентификатором. Повтор для '
        'той же открытой owned-сессии возвращает тот же успешный результат.'
    ),
    operation_id='createChatSession',
    response_description='Созданная открытая сессия и серверный TTL.',
    response_model=CreateChatSessionResponse,
    responses={
        401: {'description': 'Невалидный сервисный API-ключ.'},
        403: {'description': 'Идентификатор занят чужой сессией.'},
        409: {'description': 'Owned-сессия с этим ID уже закрыта.'},
        422: {'description': 'Ошибка валидации тела или owner-заголовков.'},
        500: {'description': 'Ошибка PostgreSQL.'},
    },
)
async def create_chat_session(
    data: CreateChatSessionRequest,
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
) -> CreateChatSessionResponse:
    """Создаёт явно выбранную границу нового диалога.

    Args:
        data: Клиентский идентификатор новой сессии.
        service: Сервис единого жизненного цикла.
        user_id: Авторизованный владелец, если есть.
        anonymous_token_hash: Доказательство anonymous-владельца.

    Returns:
        Созданная сессия и единый серверный TTL.
    """
    logger.info('🚀 Явное создание сессии. session_id=%s.', data.session_id)
    try:
        created = await service.create_session(
            session_id=data.session_id,
            user_id=user_id,
            anonymous_token_hash=anonymous_token_hash,
        )
        logger.info(
            '✅ Сессия явно создана. session_id=%s.',
            created.session_id,
        )
        return CreateChatSessionResponse(
            session_id=created.session_id,
            session_ttl_seconds=created.session_ttl_seconds,
        )
    except (
        ChatSessionAccessDeniedError,
        ChatSessionAlreadyClosedError,
        ChatSessionServiceError,
    ) as error:
        logger.exception(
            '❌ Не удалось явно создать сессию. session_id=%s. Тип=%s.',
            data.session_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error


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


@router.post(
    path='/sessions/{session_id}/close',
    status_code=status.HTTP_200_OK,
    summary='Закрыть сессию диалога',
    description=(
        'Выставляет closed_at для owned-сессии. Повторное закрытие '
        'идемпотентно возвращает ранее сохранённый момент.'
    ),
    operation_id='closeChatSession',
    response_description='Закрытая сессия и сохранённый момент закрытия.',
    response_model=CloseChatSessionResponse,
    responses={
        401: {'description': 'Невалидный сервисный API-ключ.'},
        403: {'description': 'Сессия принадлежит другому владельцу.'},
        404: {'description': 'Сессия не найдена.'},
        422: {'description': 'Ошибка валидации пути или owner-заголовков.'},
        500: {'description': 'Ошибка PostgreSQL.'},
    },
)
async def close_chat_session(
    session_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=100,
            description='Идентификатор закрываемой сессии.',
        ),
    ],
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
) -> CloseChatSessionResponse:
    """Явно закрывает owned-сессию диалога.

    Args:
        session_id: Идентификатор закрываемой сессии.
        service: Сервис единого жизненного цикла.
        user_id: Авторизованный владелец, если есть.
        anonymous_token_hash: Доказательство anonymous-владельца.

    Returns:
        Сессия и сохранённый момент её закрытия.
    """
    logger.info('🚀 Явное закрытие сессии. session_id=%s.', session_id)
    try:
        closed = await service.close_session(
            session_id=session_id,
            user_id=user_id,
            anonymous_token_hash=anonymous_token_hash,
        )
        logger.info(
            '✅ Сессия явно закрыта. session_id=%s.',
            closed.session_id,
        )
        return CloseChatSessionResponse(
            session_id=closed.session_id,
            closed_at=closed.closed_at,
        )
    except (
        ChatSessionAccessDeniedError,
        ChatSessionNotFoundError,
        ChatSessionServiceError,
    ) as error:
        logger.exception(
            '❌ Не удалось явно закрыть сессию. session_id=%s. Тип=%s.',
            session_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error
