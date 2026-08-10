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
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.clients.mcp_client import get_mcp_client, get_tools_with_retry
from app.core.settings import GraphContextSettings, McpSettings
from app.graph.build import build_graph
from app.messaging.consumer import AgentRequestConsumer, TurnPersistenceData
from app.messaging.schemas import AgentRequestMessage
from app.observability.tracing import reset_for_tests
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
        'consultation_email_guard_notice': None,
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


def _tool_call_completion(name: str, arguments: dict) -> httpx.Response:
    return _completion(
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {
                        'name': name,
                        'arguments': json.dumps(arguments),
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


def _stream_tool_call_response(name: str, arguments: dict) -> httpx.Response:
    chunk = {
        'id': 'x',
        'object': 'chat.completion.chunk',
        'created': 1,
        'model': 'test-model',
        'choices': [
            {
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'index': 0,
                            'id': 'call_1',
                            'type': 'function',
                            'function': {
                                'name': name,
                                'arguments': json.dumps(arguments),
                            },
                        }
                    ],
                },
                'finish_reason': 'tool_calls',
            }
        ],
    }
    body = f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'
    return httpx.Response(
        200,
        content=body.encode('utf-8'),
        headers={'content-type': 'text/event-stream'},
    )


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


def _conversational_handler(
    *,
    tool_name: str | None = None,
    tool_arguments: dict | None = None,
    final_pieces: list[str] | None = None,
):
    final_pieces = final_pieces or ['Квота ', 'составляет ', '2%.']

    def handler(request):
        payload = json.loads(request.content)
        if not payload.get('stream'):
            # analyze_intent: нестримингованный вызов с забинженными тулами
            if tool_name is not None:
                return _tool_call_completion(tool_name, tool_arguments or {})
            return _direct_completion('ok')
        # generate_with_context / generate_direct: стримингованный вызов
        return _stream_response(final_pieces)

    return handler


def _tools_by_name(tools) -> dict:
    return {tool.name: tool for tool in tools}


def _compile_graph(chat_model, tools, settings, context_settings=None):
    by_name = _tools_by_name(tools)
    return build_graph(
        chat_model,
        by_name['vera_rag_kb'],
        by_name['send_consultation_email'],
        settings,
        context_settings or GraphContextSettings(),
    ).compile()


async def test_question_needing_kb_search_reaches_generate_with_context():
    chunks = [{'chunk_id': 'c1', 'source_title': 'ФЗ-181, Статья 21', 'text': 'Квота составляет 2 процента'}]
    app = create_mock_mcp_app(chunks=chunks)
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        chat_model = _build_chat_model(
            _conversational_handler(
                tool_name='vera_rag_kb',
                tool_arguments={'query': 'квота'},
                final_pieces=['Квота ', '2%, источник ФЗ-181.'],
            )
        )
        graph = _compile_graph(chat_model, tools, mcp_settings)

        result = await graph.ainvoke(_initial_state('Какая квота на трудоустройство инвалидов?'))

        assert result['retrieved_chunks'] == chunks
        assert result['search_unavailable'] is False
        assert result['tool_calls'] == ['vera_rag_kb']
        final_message = result['messages'][-1]
        assert isinstance(final_message, AIMessage)
        assert final_message.content == 'Квота 2%, источник ФЗ-181.'


async def test_factual_question_reaches_kb_search_even_without_model_tool_call():
    """VERA-021: модель ошибочно решает ответить напрямую на вопрос из
    предметной области базы знаний — код принудительно направляет его в
    поиск вместо прямого ответа мимо RAG."""
    chunks = [{'chunk_id': 'c1', 'source_title': 'ФЗ-181, Статья 21', 'text': 'Квота составляет 2 процента'}]
    app = create_mock_mcp_app(chunks=chunks)
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        chat_model = _build_chat_model(
            _conversational_handler(final_pieces=['Квота ', '2%, источник ФЗ-181.'])
        )
        graph = _compile_graph(chat_model, tools, mcp_settings)

        result = await graph.ainvoke(_initial_state('Какая квота на трудоустройство инвалидов?'))

        assert result['tool_calls'] == ['vera_rag_kb']
        assert result['retrieved_chunks'] == chunks


async def test_context_budget_trims_old_turns_without_losing_current_answer():
    """VERA-020: превышение token budget не роняет запрос и не теряет
    последний turn — старые реплики схлопываются в одну сводку, а вопрос
    текущей реплики по-прежнему доходит до модели и получает ответ."""
    app = create_mock_mcp_app(fail_message='vera_rag_kb не должен вызываться для этого вопроса')
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        captured: dict = {}

        def handler(request):
            payload = json.loads(request.content)
            captured['message_count'] = len(payload['messages'])
            if not payload.get('stream'):
                return _direct_completion('ok')
            return _stream_response(['Ответ на последний вопрос.'])

        chat_model = _build_chat_model(handler)
        context_settings = GraphContextSettings(
            context_max_turns=1, context_older_turns_summary_max_chars=100
        )
        graph = _compile_graph(chat_model, tools, mcp_settings, context_settings=context_settings)

        history: list = []
        for i in range(1, 6):
            history.append(HumanMessage(content=f'Вопрос {i}'))
            history.append(AIMessage(content=f'Ответ {i}'))
        history.append(HumanMessage(content='Последний вопрос'))
        state = {
            'session_id': 's',
            'user_id': None,
            'messages': history,
            'retrieved_chunks': [],
            'tool_calls': [],
            'search_unavailable': False,
            'consultation_email_guard_notice': None,
        }

        result = await graph.ainvoke(state)

        assert result['messages'][-1].content == 'Ответ на последний вопрос.'
        # SYSTEM_PROMPT + одна сводка старых реплик + последний вопрос — вместо
        # SYSTEM_PROMPT + 11 сообщений полной истории без бюджета.
        assert captured['message_count'] == 3


async def test_greeting_goes_directly_to_generate_direct_without_mcp_call():
    app = create_mock_mcp_app(fail_message='vera_rag_kb не должен вызываться для этого вопроса')
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=5.0)

        chat_model = _build_chat_model(
            _conversational_handler(final_pieces=['Здравствуйте', '! Чем могу помочь?'])
        )
        graph = _compile_graph(chat_model, tools, mcp_settings)

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
                tool_name='vera_rag_kb',
                tool_arguments={'query': 'квота'},
                final_pieces=['Поиск сейчас недоступен, попробуйте позже.'],
            )
        )
        graph = _compile_graph(chat_model, tools, mcp_settings)

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
                tool_name='vera_rag_kb',
                tool_arguments={'query': 'вопрос вне тематики'},
                final_pieces=['В базе знаний нет ответа на этот вопрос.'],
            )
        )
        graph = _compile_graph(chat_model, tools, mcp_settings)

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

        chat_model = _build_chat_model(
            _conversational_handler(
                tool_name='vera_rag_kb',
                tool_arguments={'query': 'квота'},
            )
        )
        graph = _compile_graph(chat_model, tools, mcp_settings)

        start = time.perf_counter()
        first_token_at: float | None = None
        async for event in graph.astream_events(_initial_state('Какая квота?'), version='v2'):
            if event['event'] == 'on_chat_model_stream' and event['data']['chunk'].content:
                first_token_at = time.perf_counter()
                break

        assert first_token_at is not None
        assert (first_token_at - start) < 2.0


async def test_consultation_email_requires_current_turn_confirmation_before_sending():
    """VERA-021: мутирующая отправка не выполняется по одному лишь
    намерению модели, распознанному за один шаг. Первая попытка (без
    предшествующего запроса подтверждения) блокируется кодом; тул
    вызывается только после явного подтверждения адреса в следующей
    реплике пользователя."""
    requests: list[dict] = []
    app = create_mock_mcp_app(consultation_requests=requests)
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(
            mcp_server_url=url,
            mcp_call_timeout_seconds=5.0,
            mcp_call_retries=1,
            mcp_consultation_email_timeout_seconds=5.0,
        )
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(
            mcp_client,
            retries=1,
            timeout_seconds=5.0,
        )
        consultation_text = (
            'Итоговая консультация.\n\nРаботодатель обязан учитывать '
            'требования законодательства.'
        )
        email = 'user@example.com'

        def handler(request):
            payload = json.loads(request.content)
            if not payload.get('stream'):
                # Модель каждый раз пытается вызвать мутирующий тул — код
                # должен заблокировать попытку без адреса в ТЕКУЩЕМ
                # сообщении пользователя (VERA-021).
                return _tool_call_completion(
                    'send_consultation_email',
                    {'consultation_text': consultation_text, 'email': email},
                )
            last_user_message = [m for m in payload['messages'] if m['role'] == 'user'][-1]['content']
            if email in last_user_message:
                return _stream_response(['Документ отправлен на ', email, '.'])
            return _stream_response(['Подтвердите, пожалуйста, ваш email?'])

        chat_model = _build_chat_model(handler)
        graph = _compile_graph(chat_model, tools, mcp_settings)

        turn1 = await graph.ainvoke(_initial_state('Отправь итоговую консультацию мне на почту'))

        assert requests == []
        assert turn1['tool_calls'] == []
        assert turn1['messages'][-1].content == 'Подтвердите, пожалуйста, ваш email?'

        turn2_state = {
            'session_id': 's',
            'user_id': None,
            'messages': [*turn1['messages'], HumanMessage(content=f'Да, {email}')],
            'retrieved_chunks': [],
            'tool_calls': [],
            'search_unavailable': False,
            'consultation_email_guard_notice': None,
        }
        turn2 = await graph.ainvoke(turn2_state)

        assert requests == [{'consultation_text': consultation_text, 'email': email}]
        assert turn2['tool_calls'] == ['send_consultation_email']
        assert turn2['messages'][-1].content == f'Документ отправлен на {email}.'


async def test_automatic_email_spans_redact_consultation_and_recipient():
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter)
    requests: list[dict] = []
    app = create_mock_mcp_app(consultation_requests=requests)
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(
            mcp_server_url=url,
            mcp_call_timeout_seconds=5.0,
            mcp_call_retries=1,
            mcp_consultation_email_timeout_seconds=5.0,
        )
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(
            mcp_client,
            retries=1,
            timeout_seconds=5.0,
        )
        consultation_text = 'Уникальный секретный текст консультации VERA-022'
        recipient = 'vera-022-private@example.test'

        def handler(request):
            payload = json.loads(request.content)
            if not payload.get('stream'):
                return _tool_call_completion(
                    'send_consultation_email',
                    {'consultation_text': consultation_text, 'email': recipient},
                )
            last_user_message = [m for m in payload['messages'] if m['role'] == 'user'][-1]['content']
            if recipient in last_user_message:
                return _stream_response(['Документ отправлен.'])
            return _stream_response(['Подтвердите, пожалуйста, ваш email?'])

        chat_model = _build_chat_model(handler)
        graph = _compile_graph(chat_model, tools, mcp_settings)

        # Первая попытка без предшествующего запроса подтверждения
        # блокируется кодом (VERA-021) — тул ещё не вызывается.
        turn1 = await graph.ainvoke(_initial_state('Отправь итоговую консультацию на почту'))
        turn2_state = {
            'session_id': 's',
            'user_id': None,
            'messages': [*turn1['messages'], HumanMessage(content=f'Да, {recipient}')],
            'retrieved_chunks': [],
            'tool_calls': [],
            'search_unavailable': False,
            'consultation_email_guard_notice': None,
        }
        await graph.ainvoke(turn2_state)

    assert requests == [
        {
            'consultation_text': consultation_text,
            'email': recipient,
        }
    ]
    spans = exporter.get_finished_spans()
    automatic_email_spans = [
        span for span in spans if span.name == 'send_consultation_email'
    ]
    assert automatic_email_spans
    serialized_attributes = str([dict(span.attributes) for span in spans])
    assert consultation_text not in serialized_attributes
    assert recipient not in serialized_attributes


async def test_consumer_collects_authoritative_metadata_from_real_graph_events():
    chunks = [
        {
            'chunk_id': 'c1',
            'source_title': 'ФЗ-181, Статья 21',
            'text': 'Квота составляет 2 процента',
        }
    ]
    app = create_mock_mcp_app(chunks=chunks)
    async with run_mock_mcp_server(app) as url:
        mcp_settings = McpSettings(
            mcp_server_url=url,
            mcp_call_timeout_seconds=5.0,
            mcp_call_retries=1,
        )
        mcp_client = get_mcp_client(mcp_settings)
        tools = await get_tools_with_retry(
            mcp_client,
            retries=1,
            timeout_seconds=5.0,
        )

        def handler(request):
            request_payload = json.loads(request.content)
            if request_payload.get('tools'):
                return _stream_tool_call_response(
                    'vera_rag_kb',
                    {'query': 'квота'},
                )
            return _stream_response(['Квота ', '2%, источник ФЗ-181.'])

        chat_model = _build_chat_model(handler)
        graph = _compile_graph(chat_model, tools, mcp_settings)

        async def unused_token_sink(_request_id: str, _event: dict) -> None:
            pass

        consumer = AgentRequestConsumer(
            connection_url='amqp://unused',
            queue_name='agent.requests',
            dlq_name='agent.requests.dlq',
            graph=graph,
            token_sink=unused_token_sink,
        )
        payload = AgentRequestMessage(
            session_id='real-events-session',
            request_id='real-events-request',
            user_id='user-1',
            message='Какая квота на трудоустройство инвалидов?',
        )
        persistence_data = TurnPersistenceData()

        answer = ''.join(
            [
                chunk
                async for chunk in consumer._stream_answer(
                    payload,
                    persistence_data,
                )
            ]
        )

        assert answer == 'Квота 2%, источник ФЗ-181.'
        assert persistence_data.tool_calls == ['vera_rag_kb']
        assert persistence_data.sources == chunks
