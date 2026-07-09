from typing import Annotated

from fastapi import Depends
from langchain_openai import ChatOpenAI

from app.clients.llm import get_chat_model
from app.core.settings import get_settings
from app.dependencies.http_client import HttpClientDep


def get_chat_model_dependency(httpx_client: HttpClientDep) -> ChatOpenAI:
    return get_chat_model(httpx_client=httpx_client, settings=get_settings().llm)


ChatModelDep = Annotated[ChatOpenAI, Depends(get_chat_model_dependency)]
