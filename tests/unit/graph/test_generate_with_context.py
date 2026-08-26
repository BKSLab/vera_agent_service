import json

import httpx
from langchain_core.messages import HumanMessage

from app.core.settings import GraphContextSettings
from app.graph.nodes.generate_with_context import create_generate_with_context_node
from app.graph.prompts.context import (
    NO_ANSWER_INSTRUCTION,
    NO_SEARCH_PERFORMED_INSTRUCTION,
    SEARCH_UNAVAILABLE_INSTRUCTION,
)
from tests.unit.graph._mock_llm import chat_model_with_handler, stream_response

_CONTEXT_SETTINGS = GraphContextSettings()


def _state(retrieved_chunks, search_unavailable):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': [HumanMessage(content='Какая квота на трудоустройство инвалидов?')],
        'retrieved_chunks': retrieved_chunks,
        'tool_calls': ['vera_rag_kb'],
        'search_unavailable': search_unavailable,
    }


def _system_message_contents(request: httpx.Request) -> list[str]:
    payload = json.loads(request.content)
    return [message['content'] for message in payload['messages'] if message['role'] == 'system']


def _last_system_message_content(request: httpx.Request) -> str:
    return _system_message_contents(request)[-1]


async def test_branch_with_chunks_includes_chunk_text_in_instruction():
    captured = {}

    def handler(request):
        captured['instruction'] = _last_system_message_content(request)
        return stream_response(
            [
                'Квота 2%.\n\n',
                'Основание: статья 21 Федерального закона № 181-ФЗ.',
            ]
        )

    chat_model = chat_model_with_handler(handler)
    node = create_generate_with_context_node(chat_model, _CONTEXT_SETTINGS)
    chunks = [
        {
            'chunk_id': 'c1',
            'source_title': 'Федеральный закон № 181-ФЗ',
            'section_number': '21',
            'section_title': 'Статья 21. Установление квоты',
            'text': 'Квота составляет 2 процента',
        }
    ]

    result = await node(_state(chunks, search_unavailable=False))

    assert result['messages'][0].content == (
        'Квота 2%.\n\n'
        'Основание: статья 21 Федерального закона № 181-ФЗ.'
    )
    assert 'Название источника: Федеральный закон № 181-ФЗ' in captured['instruction']
    assert 'Номер статьи, пункта или раздела: 21' in captured['instruction']
    assert 'Название статьи, пункта или раздела: Статья 21. Установление квоты' in captured['instruction']
    assert 'Квота составляет 2 процента' in captured['instruction']
    assert 'Ответ без абзаца «Основание:» недопустим' in captured['instruction']


async def test_branch_with_chunks_marks_missing_source_details_without_inventing_them():
    captured = {}

    def handler(request):
        captured['instruction'] = _last_system_message_content(request)
        return stream_response(['Ответ.', ' Основание: источник из базы знаний.'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_with_context_node(chat_model, _CONTEXT_SETTINGS)
    chunks = [{'chunk_id': 'c1', 'source_title': 'Разъяснение Роструда', 'text': 'Текст нормы'}]

    await node(_state(chunks, search_unavailable=False))

    assert 'Название источника: Разъяснение Роструда' in captured['instruction']
    assert 'Номер статьи, пункта или раздела: не указано' in captured['instruction']
    assert 'Название статьи, пункта или раздела: не указано' in captured['instruction']
    assert 'Не придумывай отсутствующие номера статей' in captured['instruction']


async def test_branch_no_answer_when_chunks_empty_and_search_available():
    captured = {}

    def handler(request):
        captured['system_messages'] = _system_message_contents(request)
        return stream_response(['В базе знаний нет ответа на этот вопрос.'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_with_context_node(chat_model, _CONTEXT_SETTINGS)

    result = await node(_state([], search_unavailable=False))

    assert captured['system_messages'][-1] == NO_ANSWER_INSTRUCTION
    assert NO_SEARCH_PERFORMED_INSTRUCTION not in captured['system_messages']
    assert 'не нашёл информации' in captured['system_messages'][-1]
    assert 'выдумывай' in captured['system_messages'][-1]
    assert result['messages'][0].content == 'В базе знаний нет ответа на этот вопрос.'


async def test_branch_search_unavailable_differs_from_no_answer_branch():
    captured = {}

    def handler(request):
        captured['system_messages'] = _system_message_contents(request)
        return stream_response(['Поиск сейчас недоступен, попробуйте позже.'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_with_context_node(chat_model, _CONTEXT_SETTINGS)

    result = await node(_state([], search_unavailable=True))

    assert captured['system_messages'][-1] == SEARCH_UNAVAILABLE_INSTRUCTION
    assert NO_SEARCH_PERFORMED_INSTRUCTION not in captured['system_messages']
    assert 'технически недоступен' in captured['system_messages'][-1]
    assert 'не нашёл информации' not in captured['system_messages'][-1]
    assert result['messages'][0].content == 'Поиск сейчас недоступен, попробуйте позже.'
