"""Интеграционные тесты полного графа (Этап 4) — LLM замокан на уровне
HTTP-транспорта (`httpx.MockTransport`), MCP Tools Server — настоящий
поднятый мок-сервер (`tests/fixtures/mock_mcp_server.py`, как в
`tests/integration/test_mcp_client.py`, Этап 3). Проверяет сквозную
маршрутизацию графа, а не только отдельные узлы.
"""

import json
import time

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.clients.mcp_client import get_mcp_client, get_tools_with_retry
from app.core.settings import McpSettings
from app.graph.build import build_graph
from tests.fixtures.mock_mcp_server import create_mock_mcp_app, run_mock_mcp_server

pytestmark = pytest.mark.integration


def _initial_state(text: str) -> dict:
    return {
        'session_id': 's',
        'user_id': None,
        'messages': [HumanMessage(content=text)],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


def _completion(message: dict, finish_reason: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            'id': 'x',
            'object': 'chat.completion',
            'created': 1,
            'model': 'test-model',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        },
    )


def _tool_call_completion(query: str, audience: str = 'both') -> httpx.Response:
    return _completion(
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {
                        'name': 'kb_search',
                        'arguments': json.dumps({'query': query, 'audience': audience}),
                    },
                }
            ],
        },
        finish_reason='tool_calls',
    )


def _direct_completion(content: str) -> httpx.Response:
    return _completion({'role': 'assistant', 'content': content}, finish_reason='stop')


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


def _build_chat_model(handler) -> ChatOpenAI:
    """Одна модель на весь граф — как в `build_graph` (Этап 4.5).

    `payload.get('stream')` различает нестримингованный вызов
    `analyze_intent` (`.ainvoke()`) от стримингованного вызова генерации
    (`.astream()`) — оба используют один и тот же объект модели.
    """
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    return ChatOpenAI(
        model='test-model', base_url='http://mock/v1', api_key='test-key', http_async_client=async_client, max_retries=0
    )


def _conversational_handler(*, needs_tool: bool, tool_query: str = 'квота', final_pieces: list[str] | None = None):
    final_pieces = final_pieces or ['Квота ', 'составляет ', '2%.']

    def handler(request):
        payload = json.loads(request.content)
        if not payload.get('stream'):
            # analyze_intent: нестримингованный вызов с забинженными тулами
            return _tool_call_completion(tool_query) if needs_tool else _direct_completion('ok')
        # generate_with_context / generate_direct: стримингованный вызов
        return _stream_response(final_pieces)

    return handler


async def test_question_needing_kb_search_reaches_generate_with_context():
    chunks = [{'chunk_id': 'c1', 'source_title': 'ФЗ-181, Статья 21', 'text': 'Квота составляет 2 процента'}]
    app = create_mock_mcp_app(chunks=chunks)
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        chat_model = _build_chat_model(
            _conversational_handler(needs_tool=True, tool_query='квота', final_pieces=['Квота ', '2%, источник ФЗ-181.'])
        )
        graph = build_graph(chat_model, tools[0], mcp_settings).compile()

        result = await graph.ainvoke(_initial_state('Какая квота на трудоустройство инвалидов?'))

        assert result['retrieved_chunks'] == chunks
        assert result['search_unavailable'] is False
        assert result['tool_calls'] == ['kb_search']
        final_message = result['messages'][-1]
        assert isinstance(final_message, AIMessage)
        assert final_message.content == 'Квота 2%, источник ФЗ-181.'


async def test_greeting_goes_directly_to_generate_direct_without_mcp_call():
    app = create_mock_mcp_app(fail_message='kb_search не должен вызываться для этого вопроса')
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        chat_model = _build_chat_model(
            _conversational_handler(needs_tool=False, final_pieces=['Здравствуйте', '! Чем могу помочь?'])
        )
        graph = build_graph(chat_model, tools[0], mcp_settings).compile()

        result = await graph.ainvoke(_initial_state('привет'))

        assert result['retrieved_chunks'] == []
        assert result['tool_calls'] == []
        final_message = result['messages'][-1]
        assert final_message.content == 'Здравствуйте! Чем могу помочь?'


async def test_mcp_unavailable_degrades_gracefully_without_raising():
    app = create_mock_mcp_app(fail_message='RAG Service недоступен (мок)')
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=1.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        chat_model = _build_chat_model(
            _conversational_handler(
                needs_tool=True, tool_query='квота', final_pieces=['Поиск сейчас недоступен, попробуйте позже.']
            )
        )
        graph = build_graph(chat_model, tools[0], mcp_settings).compile()

        result = await graph.ainvoke(_initial_state('Какая квота на трудоустройство инвалидов?'))

        assert result['search_unavailable'] is True
        assert result['retrieved_chunks'] == []
        final_message = result['messages'][-1]
        assert 'недоступен' in final_message.content


async def test_empty_chunks_produces_honest_refusal_not_error():
    app = create_mock_mcp_app(chunks=[])
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        chat_model = _build_chat_model(
            _conversational_handler(
                needs_tool=True,
                tool_query='вопрос вне тематики',
                final_pieces=['В базе знаний нет ответа на этот вопрос.'],
            )
        )
        graph = build_graph(chat_model, tools[0], mcp_settings).compile()

        result = await graph.ainvoke(_initial_state('Вопрос вне тематики базы знаний'))

        assert result['retrieved_chunks'] == []
        assert result['search_unavailable'] is False
        final_message = result['messages'][-1]
        assert 'нет ответа' in final_message.content


async def test_first_token_arrives_quickly_against_mocked_llm():
    """Ранняя проверка плотности графа (Этап 4.7) — не финальный замер TTFT
    (провайдер и MCP настоящие только в проде, полноценный замер — Этап
    10), только подтверждение, что архитектура (2 вызова LLM) не вносит
    искусственных задержек на уровне самого графа."""
    app = create_mock_mcp_app(chunks=[{'chunk_id': 'c1', 'text': 'квота 2%'}])
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        chat_model = _build_chat_model(_conversational_handler(needs_tool=True, tool_query='квота'))
        graph = build_graph(chat_model, tools[0], mcp_settings).compile()

        start = time.perf_counter()
        first_token_at: float | None = None
        async for event in graph.astream_events(_initial_state('Какая квота?'), version='v2'):
            if event['event'] == 'on_chat_model_stream' and event['data']['chunk'].content:
                first_token_at = time.perf_counter()
                break

        assert first_token_at is not None
        assert (first_token_at - start) < 2.0
