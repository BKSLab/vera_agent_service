import asyncio
import json
import logging

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.clients.polza_final_response import (
    FINAL_RESPONSE_JSON_SCHEMA,
    PolzaFinalResponseClient,
)
from app.core.settings import LlmSettings
from app.exceptions.llm import LlmApiRequestError
from app.graph.policy import UNSAFE_TOOL_CALL_RESPONSE
from app.observability.request_trace import (
    AgentRequestTraceData,
    reset_request_trace,
    set_request_trace,
)
from app.observability.tracing import reset_for_tests


def _settings(*, api_key: str = 'unit-test-api-key') -> LlmSettings:
    return LlmSettings(
        llm_api_key=api_key,
        llm_api_url='https://polza.test/api/v1/',
        llm_model='google/gemini-3.7-flash',
        llm_temperature=0.3,
        llm_reasoning_effort='low',
    )


def _sse_response(events: list[dict], *, status_code: int = 200) -> httpx.Response:
    body = ''.join(
        f'data: {json.dumps(event, ensure_ascii=False)}\n\n'
        for event in events
    )
    body += 'data: [DONE]\n\n'
    return httpx.Response(
        status_code,
        content=body.encode('utf-8'),
        headers={'content-type': 'text/event-stream'},
    )


def _literal_sse_response(body: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=body.encode('utf-8'),
        headers={'content-type': 'text/event-stream'},
    )


def _answer_events(
    answer: str,
    *,
    reasoning: str | None = None,
    reasoning_details: list[dict] | None = None,
) -> list[dict]:
    raw_content = json.dumps({'answer': answer}, ensure_ascii=False)
    split_at = max(1, len(raw_content) // 2)
    first_delta: dict = {'role': 'assistant'}
    if reasoning is not None:
        first_delta['reasoning'] = reasoning
    if reasoning_details is not None:
        first_delta['reasoning_details'] = reasoning_details
    return [
        {
            'id': 'completion-1',
            'model': 'google/gemini-3.7-flash',
            'provider': 'openrouter',
            'choices': [
                {'index': 0, 'delta': first_delta, 'finish_reason': None}
            ],
        },
        {
            'id': 'completion-1',
            'model': 'google/gemini-3.7-flash',
            'provider': 'openrouter',
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': raw_content[:split_at]},
                    'finish_reason': None,
                }
            ],
        },
        {
            'id': 'completion-1',
            'model': 'google/gemini-3.7-flash',
            'provider': 'openrouter',
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': raw_content[split_at:]},
                    'finish_reason': None,
                }
            ],
        },
        {
            'id': 'completion-1',
            'model': 'google/gemini-3.7-flash',
            'provider': 'openrouter',
            'choices': [
                {'index': 0, 'delta': {}, 'finish_reason': 'stop'}
            ],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30,
                'completion_tokens_details': {'reasoning_tokens': 7},
            },
        },
    ]


def _build_client(
    handler,
    *,
    trace_content_enabled: bool = False,
    request_retries: int = 1,
    output_retries: int = 1,
    api_key: str = 'unit-test-api-key',
) -> PolzaFinalResponseClient:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PolzaFinalResponseClient(
        http_client,
        _settings(api_key=api_key),
        trace_content_enabled=trace_content_enabled,
        request_retries=request_retries,
        output_retries=output_retries,
    )


@pytest.mark.parametrize(
    ('request_retries', 'output_retries'),
    [(0, 1), (1, -1), (1, 2)],
)
def test_rejects_retry_configuration_outside_safety_contract(
    request_retries,
    output_retries,
):
    with pytest.raises(ValueError):
        _build_client(
            lambda _request: pytest.fail('HTTP не должен вызываться'),
            request_retries=request_retries,
            output_retries=output_retries,
        )


async def test_sends_strict_schema_and_returns_only_answer_not_reasoning():
    payloads: list[dict] = []
    secret_reasoning = 'Скрытая цепочка рассуждений, которой не должно быть в ответе.'

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _sse_response(
            _answer_events(
                'Проверенный ответ пользователю.',
                reasoning=secret_reasoning,
                reasoning_details=[
                    {
                        'type': 'reasoning.text',
                        'format': 'google-gemini-v1',
                        'text': secret_reasoning,
                    }
                ],
            )
        )

    client = _build_client(handler)

    answer = await client.generate_final_answer(
        [SystemMessage(content='Системная инструкция'), HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Проверенный ответ пользователю.'
    assert secret_reasoning not in answer
    assert len(payloads) == 1
    assert payloads[0]['stream'] is True
    assert payloads[0]['stream_options'] == {'include_usage': True}
    assert payloads[0]['response_format'] == FINAL_RESPONSE_JSON_SCHEMA
    assert payloads[0]['reasoning'] == {'effort': 'low', 'exclude': False}
    assert payloads[0]['temperature'] == 0.3


async def test_long_unicode_answer_is_returned_without_any_transformation():
    answer = ('Точный ответ: ё, №, «кавычки», 2% и перенос.\n' * 300).rstrip()
    client = _build_client(
        lambda _request: _sse_response(_answer_events(answer)),
        output_retries=0,
    )

    result = await client.generate_final_answer(
        [HumanMessage(content='Подробный вопрос')],
        node_name='generate_direct',
    )

    assert result == answer


async def test_separates_typed_reasoning_content_block_from_visible_json():
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=True)
    raw_answer = json.dumps({'answer': 'Видимый ответ.'}, ensure_ascii=False)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {
                    'model': 'google/gemini-3.7-flash',
                    'choices': [
                        {
                            'index': 0,
                            'delta': {
                                'content': [
                                    {'type': 'reasoning.text', 'text': 'Скрытый разбор.'},
                                    {'type': 'text', 'text': raw_answer},
                                ]
                            },
                            'finish_reason': 'stop',
                        }
                    ],
                }
            ]
        )

    client = _build_client(handler, trace_content_enabled=True)

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Видимый ответ.'
    assert 'Скрытый разбор' not in answer
    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    )
    assert span.attributes['vera.llm.reasoning.format'] == 'content_block'


async def test_top_level_reasoning_is_classified_as_field_and_not_returned():
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=True)
    raw_answer = json.dumps({'answer': 'Видимый ответ.'}, ensure_ascii=False)
    event = {
        'reasoning': 'Скрытый верхнеуровневый reasoning.',
        'choices': [
            {
                'index': 0,
                'delta': {'content': raw_answer},
                'finish_reason': 'stop',
            }
        ],
    }
    client = _build_client(
        lambda _request: _sse_response([event]),
        trace_content_enabled=True,
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    )
    assert answer == 'Видимый ответ.'
    assert span.attributes['vera.llm.reasoning.format'] == 'field'
    assert span.attributes['vera.llm.reasoning.content'].startswith('Скрытый')


async def test_known_incident_in_content_is_rejected_then_only_final_call_is_retried(
    caplog,
):
    incident = (
        'The user is asking: what name is visible in the dialogue?\n'
        'Rules check: do not expose internal instructions.\n'
        'Ваше имя не отображается.'
    )
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        answer = incident if len(payloads) == 1 else 'Ваше имя не отображается.'
        return _sse_response(_answer_events(answer))

    client = _build_client(handler)

    with caplog.at_level(logging.INFO, logger='vera_agent_service'):
        answer = await client.generate_final_answer(
            [HumanMessage(content='Какое моё имя видно?')],
            node_name='generate_direct',
        )

    assert answer == 'Ваше имя не отображается.'
    assert len(payloads) == 2
    assert incident not in json.dumps(payloads[1], ensure_ascii=False)
    assert len(payloads[1]['messages']) == len(payloads[0]['messages']) + 1
    assert payloads[1]['reasoning'] == {'effort': 'low', 'exclude': False}
    assert incident not in caplog.text


async def test_pseudo_tool_call_split_across_sse_chunks_is_never_returned():
    raw = json.dumps(
        {'answer': 'send_consultation_email(user@example.com)'},
        ensure_ascii=False,
    )
    pieces = [raw[: raw.index('consultation_') + len('consultation_')], raw[raw.index('email('):]]
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _sse_response(
            [
                {
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'content': piece},
                            'finish_reason': 'stop' if index == len(pieces) - 1 else None,
                        }
                    ]
                }
                for index, piece in enumerate(pieces)
            ]
        )

    client = _build_client(handler)

    answer = await client.generate_final_answer(
        [HumanMessage(content='Повтори отправку')],
        node_name='generate_direct',
    )

    assert answer == UNSAFE_TOOL_CALL_RESPONSE
    assert call_count == 2
    assert 'send_consultation_email' not in answer


@pytest.mark.parametrize(
    'bad_events',
    [
        [{'choices': [{'index': 0, 'delta': {'content': 'не JSON'}, 'finish_reason': 'stop'}]}],
        [
            {
                'choices': [
                    {
                        'index': 0,
                        'delta': {
                            'content': [
                                {'type': 'unknown_text', 'text': '{"answer":"Ответ"}'}
                            ]
                        },
                        'finish_reason': 'stop',
                    }
                ]
            }
        ],
    ],
)
async def test_invalid_or_ambiguous_output_retries_once_then_falls_back(bad_events):
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _sse_response(bad_events)

    client = _build_client(handler)

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_with_context',
    )

    assert answer == UNSAFE_TOOL_CALL_RESPONSE
    assert call_count == 2


async def test_http_failure_is_retried_before_any_content_is_returned(monkeypatch):
    responses = [httpx.Response(503), _sse_response(_answer_events('Ответ после повтора.'))]

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr('app.clients.polza_final_response.asyncio.sleep', no_sleep)

    def handler(_request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    client = _build_client(handler, request_retries=2)

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Ответ после повтора.'
    assert responses == []


async def test_exhausted_request_errors_raise_safe_boundary_error(monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr('app.clients.polza_final_response.asyncio.sleep', no_sleep)
    client = _build_client(
        lambda _request: httpx.Response(502, text='sensitive upstream body'),
        request_retries=2,
    )

    with pytest.raises(LlmApiRequestError, match='http_error'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


async def test_overall_attempt_deadline_stops_never_ending_stream(monkeypatch):
    async def never_ending_stream(_self, _messages):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        'app.clients.polza_final_response.DEFAULT_TIMEOUT_SECONDS',
        0.01,
    )
    monkeypatch.setattr(
        PolzaFinalResponseClient,
        '_stream_once',
        never_ending_stream,
    )
    client = _build_client(
        lambda _request: pytest.fail('HTTP transport не должен вызываться'),
        request_retries=1,
        output_retries=0,
    )

    with pytest.raises(LlmApiRequestError, match='request_timeout'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


@pytest.mark.parametrize(
    ('limit_name', 'events', 'safe_reason'),
    [
        (
            'MAX_FINAL_CONTENT_CHARS',
            [
                {
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'content': '{"answer":"слишком длинно"}'},
                            'finish_reason': 'stop',
                        }
                    ]
                }
            ],
            'content_too_large',
        ),
        (
            'MAX_REASONING_CHARS',
            [
                {
                    'choices': [
                        {
                            'index': 0,
                            'delta': {
                                'reasoning': 'слишком длинный reasoning',
                                'content': '{"answer":"Ответ"}',
                            },
                            'finish_reason': 'stop',
                        }
                    ]
                }
            ],
            'reasoning_too_large',
        ),
    ],
)
async def test_provider_channels_have_bounded_memory(
    monkeypatch,
    limit_name,
    events,
    safe_reason,
):
    monkeypatch.setattr(
        f'app.clients.polza_final_response.{limit_name}',
        10,
    )
    client = _build_client(
        lambda _request: _sse_response(events),
        request_retries=1,
        output_retries=0,
    )

    with pytest.raises(LlmApiRequestError, match=safe_reason):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


async def test_single_sse_event_has_bounded_memory(monkeypatch):
    monkeypatch.setattr(
        'app.clients.polza_final_response.MAX_SSE_EVENT_CHARS',
        10,
    )
    client = _build_client(
        lambda _request: _sse_response(_answer_events('Ответ')),
        request_retries=1,
        output_retries=0,
    )

    with pytest.raises(LlmApiRequestError, match='sse_event_too_large'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


@pytest.mark.parametrize('status_code', [302, 400, 401, 403, 404, 422])
async def test_non_retryable_http_error_fails_after_one_request(
    monkeypatch,
    status_code,
):
    calls = 0

    async def unexpected_sleep(_seconds):
        pytest.fail('Терминальная 401 не должна ждать retry backoff')

    monkeypatch.setattr(
        'app.clients.polza_final_response.asyncio.sleep',
        unexpected_sleep,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, text='untrusted provider body')

    client = _build_client(
        handler,
        request_retries=3,
        output_retries=0,
    )

    with pytest.raises(LlmApiRequestError, match='http_error'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )

    assert calls == 1


@pytest.mark.parametrize(
    ('body', 'safe_reason'),
    [
        ('data: not-json\n\n', 'invalid_sse_json'),
        ('data: []\n\n', 'invalid_sse_event'),
        ('data: {"choices": {}}\n\n', 'invalid_choices'),
    ],
)
async def test_malformed_sse_contract_raises_only_safe_protocol_code(body, safe_reason):
    client = _build_client(
        lambda _request: _literal_sse_response(body),
        request_retries=1,
        output_retries=0,
    )

    with pytest.raises(LlmApiRequestError, match=safe_reason):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


@pytest.mark.parametrize('error_kind', ['httpx', 'unexpected'])
async def test_transport_and_unexpected_errors_are_mapped_without_message_leak(
    error_kind,
    caplog,
):
    sensitive = 'private-person@example.test'

    def handler(request: httpx.Request) -> httpx.Response:
        if error_kind == 'httpx':
            raise httpx.ConnectError(sensitive, request=request)
        raise RuntimeError(sensitive)

    client = _build_client(handler, request_retries=1, output_retries=0)

    with caplog.at_level(logging.WARNING, logger='vera_agent_service'):
        with pytest.raises(LlmApiRequestError):
            await client.generate_final_answer(
                [HumanMessage(content='Вопрос')],
                node_name='generate_direct',
            )

    assert sensitive not in caplog.text


async def test_sse_comment_and_final_event_without_blank_line_are_supported():
    raw = json.dumps({'answer': 'Ответ без завершающей пустой строки.'}, ensure_ascii=False)
    body = (
        ': keep-alive\n\n'
        f'data: {json.dumps({"choices": []})}\n\n'
        f'data: {json.dumps({"choices": [{"index": 0, "message": {"content": raw}, "finish_reason": "stop"}]}, ensure_ascii=False)}'
    )
    client = _build_client(
        lambda _request: _literal_sse_response(body),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Ответ без завершающей пустой строки.'


async def test_done_is_terminal_signal_when_finish_reason_is_absent():
    event = {
        'choices': [
            {
                'index': 0,
                'delta': {'content': '{"answer":"Полный ответ"}'},
                'finish_reason': None,
            }
        ]
    }
    client = _build_client(
        lambda _request: _sse_response([event]),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Полный ответ'


async def test_complete_json_without_done_or_finish_reason_retries_request(
    monkeypatch,
):
    calls = 0

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr('app.clients.polza_final_response.asyncio.sleep', no_sleep)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _literal_sse_response(
                'data: {"choices":[{"index":0,"delta":'
                '{"content":"{\\"answer\\":\\"Ответ\\"}"},'
                '"finish_reason":null}]}\n\n'
            )
        return _sse_response(_answer_events('Ответ после повтора.'))

    client = _build_client(
        handler,
        request_retries=2,
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Ответ после повтора.'
    assert calls == 2


@pytest.mark.parametrize(
    ('body', 'content_type', 'safe_reason'),
    [
        (
            'data: {"choices":[],"choices":[]}\n\ndata: [DONE]\n\n',
            'text/event-stream',
            'duplicate_sse_json_key',
        ),
        (
            'data: {"choices":[]}\n\ndata: [DONE]\n\n',
            'application/json',
            'invalid_sse_content_type',
        ),
        (
            'data: {"choices":[],"created":NaN}\n\ndata: [DONE]\n\n',
            'text/event-stream',
            'invalid_sse_json',
        ),
    ],
)
async def test_noncanonical_sse_contract_is_rejected(
    body,
    content_type,
    safe_reason,
):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body.encode(),
            headers={'content-type': content_type},
        )

    client = _build_client(
        handler,
        request_retries=1,
        output_retries=0,
    )

    with pytest.raises(LlmApiRequestError, match=safe_reason):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


@pytest.mark.parametrize(
    'event',
    [
        {
            'choices': [
                {'index': 0, 'delta': {'content': '{"answer":"Ответ"}'}, 'finish_reason': 'stop'},
                {'index': 1, 'delta': {'content': 'второй вариант'}, 'finish_reason': 'stop'},
            ]
        },
        {'choices': ['не объект']},
        {'choices': [{'index': 1, 'delta': {'content': '{"answer":"Ответ"}'}}]},
        {'choices': [{'index': False, 'delta': {'content': '{"answer":"Ответ"}'}}]},
        {'choices': [{'index': 0, 'delta': 'не объект'}]},
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': '{"answer":"Ответ"}',
                        'tool_calls': [{'function': {'name': 'unexpected'}}],
                    },
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'role': 'tool',
                        'content': '{"answer":"Ответ"}',
                    },
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': '{"answer":"Ответ"}',
                        'audio': {'data': 'другой output-канал'},
                    },
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'content': '{"answer":"Второй верхнеуровневый канал"}',
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': '{"answer":"Ответ"}'},
                    'finish_reason': 'stop',
                }
            ],
        },
        {
            'analysis': 'Неизвестный верхнеуровневый reasoning-канал',
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': '{"answer":"Ответ"}'},
                    'finish_reason': 'stop',
                }
            ],
        },
        {
            'choices': [
                {
                    'index': 0,
                    'analysis': 'Неизвестный choice reasoning-канал',
                    'delta': {'content': '{"answer":"Ответ"}'},
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': '{"answer":"Ответ"}'},
                    'message': {'content': '{"answer":"Другой ответ"}'},
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'text': '{"answer":"Другой канал"}',
                    'delta': {'content': '{"answer":"Ответ"}'},
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': '{"answer":"Ответ"}',
                        'analysis': 'Неизвестный reasoning-канал',
                    },
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': [
                            {
                                'type': 'text',
                                'text': '{"answer":"Ответ"}',
                                'reasoning_summary': 'Скрытый разбор',
                            }
                        ]
                    },
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': '{"answer":"Ответ"}',
                        'reasoning': 'Первый reasoning-канал',
                        'reasoning_content': 'Второй reasoning-канал',
                    },
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': '{"answer":"Ответ"}'},
                    'finish_reason': 7,
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': '{"answer":"Усечённый ответ"}'},
                    'finish_reason': 'length',
                }
            ]
        },
        {'choices': [{'index': 0, 'delta': {'reasoning': 42, 'content': '{"answer":"Ответ"}'}}]},
        {'choices': [{'index': 0, 'delta': {'reasoning_details': ['unknown'], 'content': '{"answer":"Ответ"}'}}]},
        {'choices': [{'index': 0, 'delta': {'content': ['{"answer":"Ответ"}', 42]}}]},
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': [
                            {'text': '{"answer":"Ответ"}'},
                        ]
                    },
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'tool_calls': [{'function': {'name': 'unexpected'}}],
                    'delta': {'content': '{"answer":"Ответ"}'},
                }
            ]
        },
    ],
)
async def test_ambiguous_provider_variants_fail_closed(event):
    client = _build_client(
        lambda _request: _sse_response([event]),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == UNSAFE_TOOL_CALL_RESPONSE


async def test_multiple_reasoning_transport_formats_fail_closed():
    event = {
        'reasoning': 'Верхнеуровневый reasoning.',
        'choices': [
            {
                'index': 0,
                'delta': {
                    'reasoning': 'Delta reasoning.',
                    'content': '{"answer":"Ответ"}',
                },
                'finish_reason': 'stop',
            }
        ],
    }
    client = _build_client(
        lambda _request: _sse_response([event]),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == UNSAFE_TOOL_CALL_RESPONSE


@pytest.mark.parametrize(
    ('metadata_field', 'first_value', 'second_value', 'span_attribute'),
    [
        (
            'model',
            'google/gemini-3.7-flash',
            'unexpected/model',
            SpanAttributes.LLM_RESPONSE_MODEL_NAME,
        ),
        ('provider', 'google', 'openrouter', SpanAttributes.LLM_PROVIDER),
    ],
)
async def test_changing_routing_metadata_is_normalized_without_blocking_output(
    metadata_field,
    first_value,
    second_value,
    span_attribute,
):
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=False)
    events = [
        {
            metadata_field: first_value,
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': '{"answer":"Ответ"}'},
                    'finish_reason': None,
                }
            ],
        },
        {
            metadata_field: second_value,
            'choices': [
                {'index': 0, 'delta': {}, 'finish_reason': 'stop'}
            ],
        },
    ]
    client = _build_client(
        lambda _request: _sse_response(events),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    )
    assert answer == 'Ответ'
    assert span.attributes[span_attribute] == 'multiple'


@pytest.mark.parametrize(
    'events',
    [
        [
            {
                'choices': [
                    {
                        'index': 0,
                        'delta': {'content': '{"answer":"Ответ"}'},
                        'finish_reason': 'stop',
                    }
                ]
            },
            {
                'choices': [
                    {
                        'index': 0,
                        'delta': {'content': ' после завершения'},
                        'finish_reason': None,
                    }
                ]
            },
        ],
        [
            {
                'choices': [
                    {
                        'index': 0,
                        'delta': {'content': '{"answer":"Ответ"}'},
                        'finish_reason': 'stop',
                    }
                ]
            },
            {
                'choices': [
                    {
                        'index': 0,
                        'delta': {},
                        'finish_reason': 'length',
                    }
                ]
            },
        ],
        [
            {
                'choices': [
                    {
                        'index': 0,
                        'delta': {'content': '{"answer":"Ответ"}'},
                        'finish_reason': 'stop',
                    }
                ]
            },
            {
                'choices': [
                    {
                        'index': 0,
                        'text': 'данные после завершения',
                        'finish_reason': None,
                    }
                ]
            },
        ],
        [
            {
                'choices': [
                    {
                        'index': 0,
                        'delta': {'content': '{"answer":"Ответ"}'},
                        'finish_reason': 'stop',
                    }
                ]
            },
            {
                'choices': [
                    {
                        'index': 0,
                        'delta': 'данные после завершения',
                        'finish_reason': None,
                    }
                ]
            },
        ],
    ],
)
async def test_content_after_finish_or_conflicting_finish_reason_fails_closed(
    events,
):
    client = _build_client(
        lambda _request: _sse_response(events),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == UNSAFE_TOOL_CALL_RESPONSE


async def test_nested_text_block_and_legacy_reasoning_usage_are_parsed():
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=True)
    raw = json.dumps({'answer': 'Ответ из typed text.'}, ensure_ascii=False)
    events = [
        {
            'provider_name': 'openrouter',
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': [{'type': 'text', 'text': {'value': raw}}],
                        'reasoning': [
                            {'type': 'reasoning.text', 'content': {'value': 'Скрытый разбор.'}}
                        ],
                    },
                    'finish_reason': 'stop',
                }
            ],
            'usage': {'reasoning_tokens': 11},
        }
    ]
    client = _build_client(
        lambda _request: _sse_response(events),
        trace_content_enabled=True,
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    )
    assert answer == 'Ответ из typed text.'
    assert span.attributes['llm.token_count.completion_details.reasoning'] == 11
    assert span.attributes['vera.llm.reasoning.content'] == 'Скрытый разбор.'
    assert span.attributes['vera.llm.reasoning.format'] == 'delta'


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('code', 'BAD_REQUEST'),
        ('type', 'BAD_REQUEST'),
        ('code', 'authentication_error'),
        ('type', 'authentication_error'),
        ('code', 'invalid_request_error'),
        ('type', 'permission_error'),
    ],
)
async def test_terminal_provider_error_is_preserved_and_not_retried(
    field,
    value,
):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _sse_response([{'error': {field: value}}])

    client = _build_client(
        handler,
        request_retries=3,
        output_retries=0,
    )

    with pytest.raises(LlmApiRequestError, match=f'provider_{field}_{value}'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )
    assert calls == 1


async def test_empty_provider_error_object_is_not_ignored():
    client = _build_client(
        lambda _request: _sse_response([{'error': {}}]),
        request_retries=1,
        output_retries=0,
    )

    with pytest.raises(LlmApiRequestError, match='provider_error'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


async def test_usage_only_event_without_choices_is_supported():
    events = [
        {'usage': {'prompt_tokens': 3}},
        *_answer_events('Ответ после usage event.'),
    ]
    client = _build_client(
        lambda _request: _sse_response(events),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Ответ после usage event.'


async def test_empty_choice_event_after_finish_does_not_create_second_channel():
    events = [
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': '{"answer":"Ответ"}'},
                    'finish_reason': 'stop',
                }
            ]
        },
        {
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': None, 'reasoning': None},
                    'finish_reason': 'stop',
                    'logprobs': None,
                    'native_finish_reason': 'STOP',
                }
            ],
            'usage': {'total_tokens': 3},
        },
    ]
    client = _build_client(
        lambda _request: _sse_response(events),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Ответ'


async def test_real_polza_terminal_usage_event_shape_is_supported():
    events = [
        {
            'provider': 'google',
            'model': 'google/gemini-3.7-flash',
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': '{"answer":"Ответ"}',
                        'reasoning': 'Скрытый reasoning',
                    },
                    'finish_reason': None,
                    'logprobs': None,
                    'native_finish_reason': None,
                }
            ],
        },
        {
            'provider': 'openrouter',
            'model': 'google/gemini-3.7-flash',
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': None, 'reasoning': None},
                    'finish_reason': 'stop',
                    'logprobs': None,
                    'native_finish_reason': 'STOP',
                }
            ],
            'usage': {'total_tokens': 10},
        },
    ]
    client = _build_client(
        lambda _request: _sse_response(events),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    assert answer == 'Ответ'
    assert 'reasoning' not in answer.casefold()


async def test_negative_or_boolean_usage_values_are_not_exported():
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=False)
    event = {
        'choices': [
            {
                'index': 0,
                'delta': {'content': '{"answer":"Ответ"}'},
                'finish_reason': 'stop',
            }
        ],
        'usage': {
            'prompt_tokens': -1,
            'completion_tokens': True,
            'total_tokens': -3,
            'completion_tokens_details': {'reasoning_tokens': -2},
        },
    }
    client = _build_client(
        lambda _request: _sse_response([event]),
        output_retries=0,
    )

    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    )
    assert answer == 'Ответ'
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in span.attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in span.attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in span.attributes
    assert (
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
        not in span.attributes
    )


async def test_application_logs_contain_only_aggregates_not_content_or_credentials(caplog):
    api_key = 'never-log-this-api-key'
    reasoning = 'never log internal reasoning'
    answer = 'Ответ для private-person@example.test'
    client = _build_client(
        lambda _request: _sse_response(_answer_events(answer, reasoning=reasoning)),
        api_key=api_key,
    )

    with caplog.at_level(logging.INFO, logger='vera_agent_service'):
        result = await client.generate_final_answer(
            [HumanMessage(content='Секретный вопрос')],
            node_name='generate_direct',
        )

    assert result == answer
    serialized_logs = caplog.text
    assert api_key not in serialized_logs
    assert reasoning not in serialized_logs
    assert answer not in serialized_logs
    assert 'Секретный вопрос' not in serialized_logs
    assert 'content_chars=' in serialized_logs
    assert 'reasoning_chars=' in serialized_logs


async def test_untrusted_response_metadata_is_reduced_to_safe_enums(caplog):
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=False)
    sensitive = 'private-person@example.test'
    events = _answer_events('Безопасный ответ.')
    for event in events:
        event['provider'] = sensitive
        event['model'] = sensitive
    events[-1]['choices'][0]['finish_reason'] = sensitive
    client = _build_client(
        lambda _request: _sse_response(events),
        output_retries=0,
    )

    with caplog.at_level(logging.INFO, logger='vera_agent_service'):
        answer = await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )

    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    )
    serialized_telemetry = f'{caplog.text} {dict(span.attributes)}'
    assert answer == UNSAFE_TOOL_CALL_RESPONSE
    assert sensitive not in serialized_telemetry
    assert span.attributes[SpanAttributes.LLM_PROVIDER] == 'other'
    assert span.attributes[SpanAttributes.LLM_RESPONSE_MODEL_NAME] == 'other'
    assert span.attributes[SpanAttributes.LLM_FINISH_REASON] == 'other'


@pytest.mark.parametrize(
    ('reasoning_delta', 'expected_source'),
    [
        ({'reasoning_content': 'Скрытый reasoning_content.'}, 'reasoning'),
        (
            {
                'reasoning_details': [
                    {
                        'type': 'reasoning.text',
                        'format': 'google-gemini-v1',
                        'text': 'Скрытый reasoning_details.',
                    }
                ]
            },
            'reasoning_details',
        ),
    ],
)
async def test_supported_reasoning_variants_are_kept_only_in_content_tracing(
    reasoning_delta,
    expected_source,
):
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=True)
    raw_answer = json.dumps({'answer': 'Пользовательский ответ.'}, ensure_ascii=False)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {
                    'choices': [
                        {
                            'index': 0,
                            'delta': reasoning_delta,
                            'finish_reason': None,
                        }
                    ]
                },
                {
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'content': raw_answer},
                            'finish_reason': 'stop',
                        }
                    ]
                },
            ]
        )

    client = _build_client(handler, trace_content_enabled=True)
    answer = await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    )
    assert answer == 'Пользовательский ответ.'
    assert 'Скрытый' not in answer
    assert span.attributes['vera.llm.reasoning.source'] == expected_source
    assert span.attributes['vera.llm.reasoning.format'] == 'delta'
    assert span.attributes['vera.llm.reasoning.content'].startswith('Скрытый')


async def test_untrusted_provider_error_body_is_not_copied_to_logs(caplog):
    sensitive = 'private-person@example.test'
    client = _build_client(
        lambda _request: _sse_response(
            [{'error': {'code': sensitive, 'message': sensitive}}]
        ),
        request_retries=1,
    )

    with caplog.at_level(logging.WARNING, logger='vera_agent_service'):
        with pytest.raises(LlmApiRequestError, match='provider_error'):
            await client.generate_final_answer(
                [HumanMessage(content='Вопрос')],
                node_name='generate_direct',
            )

    assert sensitive not in caplog.text


async def test_message_serialization_failure_does_not_log_original_exception(
    monkeypatch,
    caplog,
):
    sensitive = 'private-person@example.test'

    def fail_conversion(_messages):
        raise ValueError(sensitive)

    monkeypatch.setattr(
        'app.clients.polza_final_response.convert_to_openai_messages',
        fail_conversion,
    )
    client = _build_client(lambda _request: pytest.fail('HTTP не должен вызываться'))

    with caplog.at_level(logging.ERROR, logger='vera_agent_service'):
        with pytest.raises(LlmApiRequestError, match='message_serialization_error'):
            await client.generate_final_answer(
                [HumanMessage(content='Вопрос')],
                node_name='generate_direct',
            )

    assert sensitive not in caplog.text


async def test_message_serialization_failure_does_not_leak_into_span_stacktrace(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=False)
    sensitive = 'private-person@example.test'

    def fail_conversion(_messages):
        raise ValueError(sensitive)

    monkeypatch.setattr(
        'app.clients.polza_final_response.convert_to_openai_messages',
        fail_conversion,
    )
    client = _build_client(lambda _request: pytest.fail('HTTP не должен вызываться'))

    with pytest.raises(LlmApiRequestError, match='message_serialization_error'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )

    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    ]
    assert len(spans) == 1
    assert sensitive not in str(spans[0].events)


async def test_invalid_converted_message_shape_uses_safe_serialization_error(
    monkeypatch,
):
    monkeypatch.setattr(
        'app.clients.polza_final_response.convert_to_openai_messages',
        lambda _messages: {'role': 'user', 'content': 'не список'},
    )
    client = _build_client(lambda _request: pytest.fail('HTTP не должен вызываться'))

    with pytest.raises(LlmApiRequestError, match='message_serialization_error'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


@pytest.mark.parametrize(
    'unsupported_value',
    [object(), float('nan')],
)
async def test_nested_non_json_message_value_fails_before_http_retry(
    monkeypatch,
    unsupported_value,
):
    monkeypatch.setattr(
        'app.clients.polza_final_response.convert_to_openai_messages',
        lambda _messages: [
            {
                'role': 'user',
                'content': 'Вопрос',
                'metadata': {'unsupported': unsupported_value},
            }
        ],
    )
    client = _build_client(
        lambda _request: pytest.fail('HTTP не должен вызываться'),
        request_retries=3,
    )

    with pytest.raises(LlmApiRequestError, match='message_serialization_error'):
        await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )


async def test_output_guard_aggregates_are_propagated_to_request_trace():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        answer = (
            'The user is asking: hidden. Rules check: allowed.'
            if calls == 1
            else 'Проверенный ответ.'
        )
        return _sse_response(_answer_events(answer))

    trace_data = AgentRequestTraceData()
    context_token = set_request_trace(trace_data)
    try:
        client = _build_client(handler)
        answer = await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )
    finally:
        reset_request_trace(context_token)

    assert answer == 'Проверенный ответ.'
    assert trace_data.output_guard_status == 'retried'
    assert trace_data.output_guard_reason == 'meta_reasoning'
    assert trace_data.output_guard_retry_count == 1
    assert trace_data.output_guard_raw_char_count > len(answer)
    assert trace_data.output_guard_final_char_count == len(answer)


async def test_blocked_fallback_is_propagated_to_request_trace():
    trace_data = AgentRequestTraceData()
    context_token = set_request_trace(trace_data)
    try:
        client = _build_client(
            lambda _request: _sse_response(
                _answer_events('Reasoning: hidden internal process.')
            )
        )
        answer = await client.generate_final_answer(
            [HumanMessage(content='Вопрос')],
            node_name='generate_direct',
        )
    finally:
        reset_request_trace(context_token)

    assert answer == UNSAFE_TOOL_CALL_RESPONSE
    assert trace_data.output_guard_status == 'blocked'
    assert trace_data.output_guard_reason == 'meta_reasoning'
    assert trace_data.output_guard_retry_count == 1
    assert trace_data.output_guard_final_char_count == len(answer)


@pytest.mark.parametrize('trace_content_enabled', [False, True])
async def test_trace_content_setting_controls_raw_output_and_reasoning_attributes(
    trace_content_enabled,
):
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=trace_content_enabled)
    reasoning = 'Phoenix-only reasoning text'
    client = _build_client(
        lambda _request: _sse_response(_answer_events('Ответ.', reasoning=reasoning)),
        trace_content_enabled=trace_content_enabled,
    )

    await client.generate_final_answer(
        [HumanMessage(content='Вопрос')],
        node_name='generate_direct',
    )

    spans = [span for span in exporter.get_finished_spans() if span.name == 'llm.polza.final_response']
    assert len(spans) == 1
    attributes = dict(spans[0].attributes)
    if trace_content_enabled:
        assert reasoning == attributes['vera.llm.reasoning.content']
        assert 'Ответ.' in attributes['output.value']
        assert 'Вопрос' in attributes['input.value']
    else:
        assert 'vera.llm.reasoning.content' not in attributes
        assert 'output.value' not in attributes
        assert 'input.value' not in attributes
    assert attributes['vera.llm.reasoning.char_count'] == len(reasoning)
