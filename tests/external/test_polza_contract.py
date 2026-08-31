"""Опциональные реальные contract-тесты Polza без RabbitMQ/Redis/PostgreSQL.

Запуск: ``RUN_POLZA_CONTRACT_TESTS=1 pytest -q tests/external``.
Содержимое ответов и ключи тесты не печатают.
"""

import os

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from app.clients.polza_final_response import PolzaFinalResponseClient
from app.core.settings import LlmSettings
from app.graph.output_guard import detect_unsafe_answer
from app.graph.policy import UNSAFE_TOOL_CALL_RESPONSE
from app.graph.prompts.context import NO_SEARCH_PERFORMED_INSTRUCTION
from app.graph.prompts.system import FINAL_RESPONSE_SYSTEM_PROMPT
from app.observability.tracing import reset_for_tests

pytestmark = pytest.mark.external


@pytest.fixture
def polza_settings() -> LlmSettings:
    if os.getenv('RUN_POLZA_CONTRACT_TESTS') != '1':
        pytest.skip('реальный Polza contract-test запускается только явно')
    try:
        settings = LlmSettings()
    except ValidationError:
        pytest.skip('LLM-настройки не заданы')
    key = settings.llm_api_key.get_secret_value()
    if not key or key.startswith(('your_', 'change-me')):
        pytest.skip('реальный LLM_API_KEY не задан')
    return settings


async def test_real_polza_accepts_strict_final_answer_schema(polza_settings):
    async with httpx.AsyncClient() as http_client:
        client = PolzaFinalResponseClient(
            http_client,
            polza_settings,
            request_retries=1,
            output_retries=0,
        )
        answer = await client.generate_final_answer(
            [
                SystemMessage(content=FINAL_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content='Привет!'),
                SystemMessage(content=NO_SEARCH_PERFORMED_INSTRUCTION),
            ],
            node_name='external_contract',
        )

    assert answer.strip()
    assert answer != UNSAFE_TOOL_CALL_RESPONSE
    assert detect_unsafe_answer(answer) is None


async def test_real_polza_reasoning_channel_is_separated_when_present(
    polza_settings,
):
    exporter = InMemorySpanExporter()
    reset_for_tests(exporter, trace_content_enabled=True)
    async with httpx.AsyncClient() as http_client:
        client = PolzaFinalResponseClient(
            http_client,
            polza_settings,
            trace_content_enabled=True,
            request_retries=1,
            output_retries=0,
        )
        answer = await client.generate_final_answer(
            [
                SystemMessage(
                    content=(
                        'Реши задачу самостоятельно. Пользователю сообщи только '
                        'итоговый ответ без описания процесса решения.'
                    )
                ),
                HumanMessage(
                    content=(
                        'Есть три коробки с неверными этикетками: яблоки, '
                        'апельсины и смесь. Какое минимальное число фруктов '
                        'нужно достать, чтобы определить содержимое всех коробок?'
                    )
                ),
            ],
            node_name='external_reasoning_contract',
        )

    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == 'llm.polza.final_response'
    ]
    assert len(spans) == 1
    attributes = spans[0].attributes
    assert answer.strip()
    assert answer != UNSAFE_TOOL_CALL_RESPONSE
    assert detect_unsafe_answer(answer) is None
    reasoning_char_count = attributes['vera.llm.reasoning.char_count']
    assert reasoning_char_count >= 0
    if reasoning_char_count:
        assert attributes['vera.llm.reasoning.source'] in {
            'reasoning',
            'reasoning_details',
        }
        assert attributes['vera.llm.reasoning.format'] in {
            'content_block',
            'delta',
            'field',
        }
        reasoning = attributes['vera.llm.reasoning.content']
        assert len(reasoning) == reasoning_char_count
        assert reasoning != answer
        assert reasoning not in answer
    else:
        assert 'vera.llm.reasoning.content' not in attributes
