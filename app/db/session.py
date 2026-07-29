from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.settings import get_settings

engine: AsyncEngine = create_async_engine(get_settings().db.url_connect)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)
