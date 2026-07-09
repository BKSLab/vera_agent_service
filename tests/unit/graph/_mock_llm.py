"""Общие помощники для мока chat-модели в тестах узлов графа (не тестовый
модуль сам по себе — pytest его не собирает, имя не начинается с test_).
"""

import json

import httpx
from langchain_openai import ChatOpenAI


def chat_model_with_handler(handler, *, streaming: bool = True) -> ChatOpenAI:
    """`streaming=False` — для `analyze_intent` (Этап 4.1, нестримингованный
    вызов): при `streaming=True` `ChatOpenAI` шлёт `"stream": true` и ждёт
    SSE-ответ даже от `.ainvoke()`, что не совпадает с форматом ответа
    обычного `chat.completion` в моках ниже."""
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    return ChatOpenAI(
        model='test-model',
        base_url='http://mock/v1',
        api_key='test-key',
        http_async_client=async_client,
        max_retries=0,
        streaming=streaming,
    )


def stream_response(pieces: list[str]) -> httpx.Response:
    chunks = [
        {
            'id': 'x',
            'object': 'chat.completion.chunk',
            'created': 1,
            'model': 'test-model',
            'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}],
        }
        for piece in pieces
    ]
    body = ''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks) + 'data: [DONE]\n\n'
    return httpx.Response(200, content=body.encode('utf-8'), headers={'content-type': 'text/event-stream'})
