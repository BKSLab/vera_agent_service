from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Предоставляет отдельную сессию PostgreSQL на HTTP-запрос."""
    async with async_session_factory() as db_session:
        yield db_session


DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
