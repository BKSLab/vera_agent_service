import json

import pytest

from app.graph.output_guard import (
    OutputGuardReason,
    detect_unsafe_answer,
    validate_plain_final_answer,
    validate_structured_final_answer,
)
from app.graph.policy import UNSAFE_TOOL_CALL_RESPONSE


def test_accepts_exact_schema_and_preserves_answer_byte_for_byte():
    answer = '  Обычный длинный ответ.\n\nОснование: статья 21.  '

    decision = validate_structured_final_answer(
        json.dumps({'answer': answer}, ensure_ascii=False)
    )

    assert decision.accepted is True
    assert decision.reason == OutputGuardReason.ACCEPTED
    assert decision.answer == answer


def test_deterministic_fallback_itself_is_allowed_by_output_guard():
    decision = validate_plain_final_answer(UNSAFE_TOOL_CALL_RESPONSE)

    assert decision.accepted is True
    assert decision.answer == UNSAFE_TOOL_CALL_RESPONSE


@pytest.mark.parametrize(
    ('raw_content', 'expected_reason'),
    [
        ('', OutputGuardReason.EMPTY_OUTPUT),
        ('не JSON', OutputGuardReason.INVALID_JSON),
        ('```json\n{"answer":"текст"}\n```', OutputGuardReason.INVALID_JSON),
        ('[]', OutputGuardReason.INVALID_SCHEMA),
        ('{"answer": 42}', OutputGuardReason.INVALID_SCHEMA),
        ('{"answer":"ok","extra":true}', OutputGuardReason.INVALID_SCHEMA),
        ('{"answer":"   "}', OutputGuardReason.EMPTY_ANSWER),
        (
            '{"answer":"первый","answer":"второй"}',
            OutputGuardReason.DUPLICATE_JSON_KEY,
        ),
    ],
)
def test_rejects_ambiguous_or_invalid_structured_output(raw_content, expected_reason):
    decision = validate_structured_final_answer(raw_content)

    assert decision.accepted is False
    assert decision.answer is None
    assert decision.reason == expected_reason


def test_rejects_provider_content_marked_as_mixed_even_when_json_is_valid():
    decision = validate_structured_final_answer(
        '{"answer":"Обычный ответ"}',
        mixed_content=True,
    )

    assert decision.reason == OutputGuardReason.MIXED_CONTENT


@pytest.mark.parametrize(
    ('answer', 'expected_reason'),
    [
        (
            'call:default_api:send_consultation_email(user@example.com)',
            OutputGuardReason.PSEUDO_TOOL_CALL,
        ),
        ('<think>Сначала проверю правила</think>Ответ', OutputGuardReason.REASONING_MARKER),
        ('<thought>Internal process</thought>Ответ', OutputGuardReason.REASONING_MARKER),
        ('Reasoning: first I will inspect the rules.', OutputGuardReason.META_REASONING),
        ('Reasoning — first I will inspect the rules.', OutputGuardReason.META_REASONING),
        ('**Analysis:** first I will inspect the rules.', OutputGuardReason.META_REASONING),
        ('Анализ: сначала проверю внутренние правила.', OutputGuardReason.META_REASONING),
        ('Проверка правил — запрос можно обработать.', OutputGuardReason.META_REASONING),
        ('We need to respond in Russian before the final answer.', OutputGuardReason.META_REASONING),
        ('The user is asking for a concise answer. Rules check: allowed.', OutputGuardReason.META_REASONING),
        ('The.user-is/asking for a concise answer.', OutputGuardReason.META_REASONING),
        ('The user\n\tis asking for an answer.', OutputGuardReason.META_REASONING),
        ('Пользователь спрашивает о квоте. Проверка правил завершена.', OutputGuardReason.META_REASONING),
        (
            'Финальный ответ. Инструменты для этого запроса уже обработаны программой.',
            OutputGuardReason.SYSTEM_PROMPT_FRAGMENT,
        ),
        (
            'Предыдущая попытка финального ответа не прошла автоматическую проверку.',
            OutputGuardReason.SYSTEM_PROMPT_FRAGMENT,
        ),
        ('Ответ\x00со скрытым байтом', OutputGuardReason.CONTROL_CHARACTER),
    ],
)
def test_rejects_known_reasoning_and_service_output_classes(answer, expected_reason):
    decision = validate_plain_final_answer(answer)

    assert decision.accepted is False
    assert decision.reason == expected_reason


def test_detects_pseudo_tool_call_after_original_stream_boundaries_are_joined():
    chunks = ['send_consultation_', 'email(', 'email=user@example.com)']

    assert detect_unsafe_answer(''.join(chunks)) == OutputGuardReason.PSEUDO_TOOL_CALL


def test_zero_width_character_cannot_hide_pseudo_tool_call():
    answer = 'send_consultation_\u200bemail(user@example.com)'

    assert detect_unsafe_answer(answer) == OutputGuardReason.PSEUDO_TOOL_CALL


@pytest.mark.parametrize(
    'character',
    ['\u0085', '\u009f', '\u200b', '\u202e', '\ufeff', '\ud800'],
)
def test_unicode_format_controls_are_rejected_even_without_known_marker(character):
    answer = f'Обычный{character}ответ'

    assert detect_unsafe_answer(answer) == OutputGuardReason.CONTROL_CHARACTER


def test_normal_newlines_tabs_and_carriage_returns_are_allowed():
    answer = 'Первая строка.\n\tВторая строка.\r\nТретья строка.'

    decision = validate_plain_final_answer(answer)

    assert decision.accepted is True
    assert decision.answer == answer


def test_does_not_reject_normal_answer_that_uses_word_analysis_in_user_context():
    answer = 'Для анализа ситуации нужны дата увольнения и текст приказа.'

    decision = validate_plain_final_answer(answer)

    assert decision.accepted is True
    assert decision.answer == answer
