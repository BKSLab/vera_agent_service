import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.exceptions.chat_turn import ChatTurnAlreadyExistsError, ChatTurnRepositoryError
from app.messaging.schemas import AgentRequestMessage
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_persistence import START_CLAIMED, ChatPersistenceService

pytestmark = pytest.mark.integration


class _SequenceBarrier:
    """Сводит конкурентные транзакции у чтения sequence без моков БД."""

    def __init__(self) -> None:
        self._arrived = 0
        self._lock = asyncio.Lock()
        self._both_arrived = asyncio.Event()

    async def wait(self) -> None:
        async with self._lock:
            self._arrived += 1
            if self._arrived == 2:
                self._both_arrived.set()
        try:
            await asyncio.wait_for(self._both_arrived.wait(), timeout=0.5)
        except TimeoutError:
            # При корректном FOR UPDATE вторая транзакция дойдёт сюда только
            # после commit первой; таймаут не снимает и не подменяет DB-lock.
            return


class _CoordinatedChatTurnRepository(ChatTurnRepository):
    def __init__(self, db_session, sequence_barrier: _SequenceBarrier):
        super().__init__(db_session)
        self._sequence_barrier = sequence_barrier

    async def get_next_sequence_number(self, chat_session_id):
        await self._sequence_barrier.wait()
        return await super().get_next_sequence_number(chat_session_id)


@pytest.mark.parametrize(
    (
        'user_id',
        'anonymous_token_hash',
        'expected_user_id',
        'expected_anonymous_token_hash',
    ),
    [
        ('user-1', None, 'user-1', None),
        (None, 'a' * 64, None, 'a' * 64),
        ('user-1', 'a' * 64, 'user-1', None),
    ],
)
async def test_validated_owner_is_persisted_on_real_postgresql(
    db_session,
    user_id: str | None,
    anonymous_token_hash: str | None,
    expected_user_id: str | None,
    expected_anonymous_token_hash: str | None,
) -> None:
    payload = AgentRequestMessage(
        session_id=f'session-{expected_user_id or "anonymous"}',
        request_id=f'request-{expected_user_id or "anonymous"}',
        user_id=user_id,
        anonymous_token_hash=anonymous_token_hash,
        message='Вопрос',
    )
    session_repository = ChatSessionRepository(db_session)
    service = ChatPersistenceService(
        session_repository,
        ChatTurnRepository(db_session),
    )

    result = await service.start_turn(
        session_id=payload.session_id,
        request_id=payload.request_id,
        user_id=payload.user_id,
        anonymous_token_hash=payload.anonymous_token_hash,
        question=payload.message,
        worker_id='worker-1',
        lease_seconds=900.0,
    )
    saved_session = await session_repository.get_by_session_id(payload.session_id)

    assert result.outcome == START_CLAIMED
    assert saved_session is not None
    assert saved_session.user_id == expected_user_id
    assert saved_session.anonymous_token_hash == expected_anonymous_token_hash


async def test_concurrent_turns_get_distinct_sequence_numbers_on_real_postgresql(
    db_engine,
    db_session,
) -> None:
    session_repository = ChatSessionRepository(db_session)
    await session_repository.save(
        ChatSession(session_id='session-concurrent', anonymous_token_hash='a' * 64)
    )
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    sequence_barrier = _SequenceBarrier()

    async def start_turn(request_id: str, worker_id: str):
        async with session_factory() as worker_session:
            service = ChatPersistenceService(
                ChatSessionRepository(worker_session),
                _CoordinatedChatTurnRepository(worker_session, sequence_barrier),
            )
            return await service.start_turn(
                session_id='session-concurrent',
                request_id=request_id,
                user_id=None,
                anonymous_token_hash='a' * 64,
                question='Вопрос',
                worker_id=worker_id,
                lease_seconds=900.0,
            )

    results = await asyncio.gather(
        start_turn('request-concurrent-1', 'worker-1'),
        start_turn('request-concurrent-2', 'worker-2'),
    )

    async with session_factory() as verification_session:
        turns = list(
            (
                await verification_session.execute(
                    select(ChatTurn)
                    .join(ChatTurn.chat_session)
                    .where(ChatSession.session_id == 'session-concurrent')
                )
            )
            .unique()
            .scalars()
            .all()
        )

    assert [result.outcome for result in results] == [START_CLAIMED, START_CLAIMED]
    assert len(turns) == 2
    assert {turn.sequence_number for turn in turns} == {1, 2}


async def test_turn_unique_constraints_are_distinguished_on_real_postgresql(
    db_session,
) -> None:
    session_repository = ChatSessionRepository(db_session)
    turn_repository = ChatTurnRepository(db_session)
    chat_session = await session_repository.save(
        ChatSession(session_id='session-constraints', anonymous_token_hash='a' * 64)
    )
    chat_session_id = chat_session.id
    await turn_repository.save(
        ChatTurn(
            request_id='request-constraint-1',
            chat_session_id=chat_session_id,
            sequence_number=1,
            question='Вопрос',
        )
    )

    with pytest.raises(ChatTurnAlreadyExistsError):
        await turn_repository.save(
            ChatTurn(
                request_id='request-constraint-1',
                chat_session_id=chat_session_id,
                sequence_number=2,
                question='Повтор request_id',
            )
        )

    with pytest.raises(ChatTurnRepositoryError) as exc_info:
        await turn_repository.save(
            ChatTurn(
                request_id='request-constraint-2',
                chat_session_id=chat_session_id,
                sequence_number=1,
                question='Повтор sequence',
            )
        )

    assert 'uq_vera_chat_turns_session_sequence' in str(exc_info.value)
