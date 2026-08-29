from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.graph.context_budget import build_bounded_messages
from app.privacy.pii import pii_redaction_scope


def _turn(question: str, answer: str) -> list:
    return [HumanMessage(content=question), AIMessage(content=answer)]


def _turn_with_tool(question: str, tool_content: str, answer: str) -> list:
    return [
        HumanMessage(content=question),
        AIMessage(content='', tool_calls=[{'id': 'call_1', 'name': 'vera_rag_kb', 'args': {}}]),
        ToolMessage(content=tool_content, tool_call_id='call_1'),
        AIMessage(content=answer),
    ]


def test_messages_are_returned_unchanged_when_turn_count_fits_budget():
    messages = [*_turn('Вопрос 1', 'Ответ 1'), *_turn('Вопрос 2', 'Ответ 2')]

    result = build_bounded_messages(messages, max_turns=5, older_turns_summary_max_chars=200)

    assert result == messages


def test_older_turns_are_collapsed_into_single_summary_message():
    turns = [_turn(f'Вопрос {i}', f'Ответ {i}') for i in range(1, 5)]
    messages = [message for turn in turns for message in turn]

    result = build_bounded_messages(messages, max_turns=1, older_turns_summary_max_chars=200)

    assert isinstance(result[0], SystemMessage)
    assert 'Вопрос 1' in result[0].content
    assert 'Ответ 1' in result[0].content
    assert 'Вопрос 3' in result[0].content
    # Последняя реплика (turns[3]) не попадает в сводку — она передаётся полностью.
    assert 'Вопрос 4' not in result[0].content
    assert result[1:] == turns[-1]


def test_last_turn_is_never_lost_even_with_minimal_budget():
    turns = [_turn(f'Вопрос {i}', f'Ответ {i}') for i in range(1, 4)]
    messages = [message for turn in turns for message in turn]

    result = build_bounded_messages(messages, max_turns=1, older_turns_summary_max_chars=0)

    assert result[-2].content == 'Вопрос 3'
    assert result[-1].content == 'Ответ 3'


def test_summary_excludes_raw_tool_message_content():
    """Сырой результат тула (VERA-020) не должен попадать даже в сводку —
    выжимка строится только из текста вопроса и финального ответа."""
    turns = [
        _turn_with_tool('Какая квота?', 'RAW_SECRET_CHUNK_JSON', 'Квота 2%.'),
        _turn('Ещё вопрос', 'Ещё ответ'),
    ]
    messages = [message for turn in turns for message in turn]

    result = build_bounded_messages(messages, max_turns=1, older_turns_summary_max_chars=200)

    assert 'RAW_SECRET_CHUNK_JSON' not in result[0].content
    assert 'Квота 2%.' in result[0].content


def test_summary_is_truncated_to_max_chars_per_turn():
    long_answer = 'А' * 500
    messages = _turn('Вопрос', long_answer) + _turn('Второй вопрос', 'Второй ответ')

    result = build_bounded_messages(messages, max_turns=1, older_turns_summary_max_chars=50)

    summary = result[0].content
    assert len(summary.splitlines()[-1]) <= 51  # 50 символов + многоточие
    assert summary.rstrip().endswith('…')


def test_pseudo_tool_call_in_history_is_replaced_before_model_call():
    messages = [
        HumanMessage(content='Отправь консультацию на user@example.com'),
        AIMessage(content='call:default_api:send_consultation_email{email=user@example.com}'),
        HumanMessage(content='Повтори, пожалуйста'),
    ]

    result = build_bounded_messages(messages, max_turns=5, older_turns_summary_max_chars=200)

    assert result[1].content == 'Предыдущий ответ не был сформирован.'
    assert 'call:default_api:' not in result[1].content


def test_pseudo_tool_call_is_removed_from_old_turn_summary():
    messages = [
        HumanMessage(content='Первый вопрос'),
        AIMessage(content='call:default_api:send_consultation_email{email=user@example.com}'),
        HumanMessage(content='Последний вопрос'),
        AIMessage(content='Обычный ответ'),
    ]

    result = build_bounded_messages(messages, max_turns=1, older_turns_summary_max_chars=200)

    assert 'call:default_api:' not in result[0].content
    assert 'Предыдущий ответ не был сформирован.' in result[0].content


def test_personal_data_is_removed_from_all_message_fields_before_model_call():
    email = 'ivan.petrov@example.com'
    full_name = 'Иванов Иван Иванович'
    messages = [
        HumanMessage(content=f'Я {full_name}, отправьте ответ на {email}'),
        AIMessage(
            content='',
            tool_calls=[
                {
                    'id': 'call_1',
                    'name': 'send_consultation_email',
                    'args': {'email': email},
                }
            ],
        ),
        ToolMessage(
            content=f'{{"status":"ok","email":"{email}"}}',
            tool_call_id='call_1',
        ),
    ]

    with pii_redaction_scope():
        result = build_bounded_messages(
            messages,
            max_turns=5,
            older_turns_summary_max_chars=200,
        )

    serialized = str(
        [
            {
                'content': message.content,
                'additional_kwargs': message.additional_kwargs,
                'tool_calls': getattr(message, 'tool_calls', []),
            }
            for message in result
        ]
    )
    assert full_name not in serialized
    assert email not in serialized
    assert '[ФИО_1]' in serialized
    assert '[EMAIL_1]' in serialized


def test_lowercase_name_from_legacy_history_is_redacted_before_model_call():
    messages = [
        HumanMessage(content='меня зовут алексей петров помогите с увольнением'),
        AIMessage(content='Опишите обстоятельства увольнения.'),
    ]

    with pii_redaction_scope():
        result = build_bounded_messages(
            messages,
            max_turns=5,
            older_turns_summary_max_chars=200,
        )

    assert result[0].content == (
        'меня зовут [ФИО_1] помогите с увольнением'
    )
