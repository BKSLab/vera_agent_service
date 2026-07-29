import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from app.db.models.base import Base


@pytest_asyncio.fixture(scope='session')
def postgres_container():
    """Поднимает один реальный PostgreSQL на весь тестовый прогон."""
    with PostgresContainer('postgres:16-alpine') as container:
        yield container


@pytest_asyncio.fixture
async def db_engine(postgres_container):
    """Создаёт AsyncEngine на event loop конкретного теста."""
    url = postgres_container.get_connection_url().replace('psycopg2', 'asyncpg')
    engine = create_async_engine(url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Предоставляет изолированную сессию и очищает таблицы после теста."""
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()
    async with db_engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            await connection.execute(table.delete())
