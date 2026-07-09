from typing import Annotated

from fastapi import Depends
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from app.clients.llm import get_chat_model
from app.clients.mcp_client import get_mcp_client
from app.core.settings import get_settings
from app.dependencies.http_client import HttpClientDep


def get_chat_model_dependency(httpx_client: HttpClientDep) -> ChatOpenAI:
    return get_chat_model(httpx_client=httpx_client, settings=get_settings().llm)


def get_mcp_client_dependency() -> MultiServerMCPClient:
    return get_mcp_client(settings=get_settings().mcp)


ChatModelDep = Annotated[ChatOpenAI, Depends(get_chat_model_dependency)]
McpClientDep = Annotated[MultiServerMCPClient, Depends(get_mcp_client_dependency)]
