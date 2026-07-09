import json
from collections.abc import AsyncIterator

import httpx
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI

from app.clients.llm import ainvoke_with_retry, astream_tokens
from app.exceptions.llm import LlmApiRequestError


def _chat_model(handler) -> ChatOpenAI:
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    return ChatOpenAI(
        model='test-model',
        base_url='http://mock/v1',
        api_key='test-key',
        http_async_client=async_client,
        max_retries=0,
    )


def _completion_response(content: str = 'Ответ', tool_calls: list[dict] | None = None) -> httpx.Response:
    message: dict = {'role': 'assistant', 'content': content}
    if tool_calls:
        message['tool_calls'] = tool_calls
    return httpx.Response(
        200,
        json={
            'id': 'x',
            'object': 'chat.completion',
            'created': 1,
            'model': 'test-model',
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        },
    )


def _stream_response(pieces: list[str]) -> httpx.Response:
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


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Ретраи в тестах не должны реально ждать секунды backoff'а."""

    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr('app.clients.llm.asyncio.sleep', _instant_sleep)


async def test_ainvoke_with_retry_returns_content_on_success():
    model = _chat_model(lambda request: _completion_response('Привет'))
    result = await ainvoke_with_retry(model, [HumanMessage(content='Привет')])
    assert result.content == 'Привет'


async def test_ainvoke_with_retry_accepts_tool_calls_without_content():
    tool_calls = [
        {
            'id': 'call_1',
            'type': 'function',
            'function': {'name': 'kb_search', 'arguments': '{"query": "квота"}'},
        }
    ]
    model = _chat_model(lambda request: _completion_response('', tool_calls))
    result = await ainvoke_with_retry(model, [HumanMessage(content='квота?')])
    assert result.tool_calls
    assert result.tool_calls[0]['name'] == 'kb_search'


async def test_ainvoke_with_retry_retries_on_network_error_then_succeeds():
    attempts = {'n': 0}

    def handler(request):
        attempts['n'] += 1
        if attempts['n'] < 3:
            raise httpx.ConnectError('boom', request=request)
        return _completion_response('Успех')

    model = _chat_model(handler)
    result = await ainvoke_with_retry(model, [HumanMessage(content='hi')], retries=3)
    assert result.content == 'Успех'
    assert attempts['n'] == 3


async def test_ainvoke_with_retry_raises_after_exhausting_network_retries():
    def handler(request):
        raise httpx.ConnectError('boom', request=request)

    model = _chat_model(handler)
    with pytest.raises(LlmApiRequestError):
        await ainvoke_with_retry(model, [HumanMessage(content='hi')], retries=2)


async def test_ainvoke_with_retry_retries_on_empty_content_then_succeeds():
    attempts = {'n': 0}

    def handler(request):
        attempts['n'] += 1
        if attempts['n'] < 2:
            return _completion_response('')
        return _completion_response('Наконец-то ответ')

    model = _chat_model(handler)
    result = await ainvoke_with_retry(model, [HumanMessage(content='hi')], retries=3)
    assert result.content == 'Наконец-то ответ'
    assert attempts['n'] == 2


async def test_astream_tokens_yields_chunks_in_order():
    model = _chat_model(lambda request: _stream_response(['При', 'вет', '!']))
    collected = [chunk async for chunk in astream_tokens(model, [HumanMessage(content='hi')])]
    assert collected == ['При', 'вет', '!']


async def test_astream_tokens_retries_before_first_chunk_on_network_error():
    attempts = {'n': 0}

    def handler(request):
        attempts['n'] += 1
        if attempts['n'] < 2:
            raise httpx.ConnectError('boom', request=request)
        return _stream_response(['Ок'])

    model = _chat_model(handler)
    collected = [chunk async for chunk in astream_tokens(model, [HumanMessage(content='hi')], retries=3)]
    assert collected == ['Ок']
    assert attempts['n'] == 2


async def test_astream_tokens_raises_if_first_chunk_never_succeeds():
    def handler(request):
        raise httpx.ConnectError('boom', request=request)

    model = _chat_model(handler)
    with pytest.raises(LlmApiRequestError):
        async for _ in astream_tokens(model, [HumanMessage(content='hi')], retries=2):
            pass


class _FakeMidStreamFailureModel:
    """Дублирует интерфейс `BaseChatModel.astream` ровно настолько,
    насколько нужно `astream_tokens` — без реального HTTP, чтобы
    детерминированно проверить, что после первого выданного чанка
    повторных попыток нет."""

    def __init__(self) -> None:
        self.call_count = 0

    async def astream(self, messages) -> AsyncIterator[AIMessageChunk]:
        self.call_count += 1
        yield AIMessageChunk(content='первый-чанк')
        raise httpx.ConnectError('обрыв соединения после первого чанка')


async def test_astream_tokens_does_not_retry_after_first_chunk_was_yielded():
    """Ключевой инвариант раздела 0.1 плана: если стриминг клиенту уже
    начался, повторный вызов всей генерации не производится — ошибка
    всплывает немедленно, а не "тихо" переигрывается."""
    fake_model = _FakeMidStreamFailureModel()

    collected = []
    with pytest.raises(httpx.ConnectError):
        async for chunk in astream_tokens(fake_model, [HumanMessage(content='hi')], retries=3):
            collected.append(chunk)

    assert collected == ['первый-чанк']
    assert fake_model.call_count == 1
