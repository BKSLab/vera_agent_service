import httpx

from app.core.settings import get_settings
from app.dependencies.clients import get_chat_model_dependency


async def test_chat_model_dependency_uses_llm_settings():
    settings = get_settings().llm
    async with httpx.AsyncClient() as httpx_client:
        model = get_chat_model_dependency(httpx_client)

    assert model.model_name == settings.llm_model
    assert model.openai_api_base == settings.llm_api_url
    assert model.temperature == settings.llm_temperature
    # Ретраи применяются явно в app/clients/llm.py (ainvoke_with_retry/
    # astream_tokens), не на уровне openai SDK — см. docstring get_chat_model.
    assert model.max_retries == 0
