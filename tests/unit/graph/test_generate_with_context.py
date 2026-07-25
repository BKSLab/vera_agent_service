import json

import httpx
from langchain_core.messages import HumanMessage

from app.graph.nodes.generate_with_context import create_generate_with_context_node
from tests.unit.graph._mock_llm import chat_model_with_handler, stream_response


def _state(retrieved_chunks, search_unavailable):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': [HumanMessage(content='Какая квота на трудоустройство инвалидов?')],
        'retrieved_chunks': retrieved_chunks,
        'tool_calls': ['vera_rag_kb'],
        'search_unavailable': search_unavailable,
    }


def _last_system_message_content(request: httpx.Request) -> str:
    payload = json.loads(request.content)
    system_messages = [message for message in payload['messages'] if message['role'] == 'system']
    return system_messages[-1]['content']


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
    node = create_generate_with_context_node(chat_model)
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
    node = create_generate_with_context_node(chat_model)
    chunks = [{'chunk_id': 'c1', 'source_title': 'Разъяснение Роструда', 'text': 'Текст нормы'}]

    await node(_state(chunks, search_unavailable=False))

    assert 'Название источника: Разъяснение Роструда' in captured['instruction']
    assert 'Номер статьи, пункта или раздела: не указано' in captured['instruction']
    assert 'Название статьи, пункта или раздела: не указано' in captured['instruction']
    assert 'Не придумывай отсутствующие номера статей' in captured['instruction']


async def test_branch_no_answer_when_chunks_empty_and_search_available():
    captured = {}

    def handler(request):
        captured['instruction'] = _last_system_message_content(request)
        return stream_response(['В базе знаний нет ответа на этот вопрос.'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_with_context_node(chat_model)

    result = await node(_state([], search_unavailable=False))

    assert 'не нашёл информации' in captured['instruction']
    assert 'выдумывай' in captured['instruction']
    assert result['messages'][0].content == 'В базе знаний нет ответа на этот вопрос.'


async def test_branch_search_unavailable_differs_from_no_answer_branch():
    captured = {}

    def handler(request):
        captured['instruction'] = _last_system_message_content(request)
        return stream_response(['Поиск сейчас недоступен, попробуйте позже.'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_with_context_node(chat_model)

    result = await node(_state([], search_unavailable=True))

    assert 'технически недоступен' in captured['instruction']
    assert 'не нашёл информации' not in captured['instruction']
    assert result['messages'][0].content == 'Поиск сейчас недоступен, попробуйте позже.'
