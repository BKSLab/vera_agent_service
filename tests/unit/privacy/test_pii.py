import logging

import pytest

from app.privacy.pii import (
    PiiRedactionContext,
    UnresolvedEmailError,
    neutralize_pii_placeholders,
    pii_redaction_scope,
    redact_pii_text,
    redact_pii_value,
    resolve_email_for_tool,
)


def test_redacts_russian_full_name_with_local_ner():
    text = 'Могут ли уволить сотрудника с инвалидностью? Я Иванов Иван Иванович'

    with pii_redaction_scope():
        redacted = redact_pii_text(text)

    assert redacted == 'Могут ли уволить сотрудника с инвалидностью? Я [ФИО_1]'


def test_redacts_lowercase_russian_full_name_with_patronymic_rule():
    with pii_redaction_scope():
        redacted = redact_pii_text('я иванов иван иванович, помогите')

    assert redacted == 'я [ФИО_1], помогите'


def test_redacts_structured_personal_identifiers():
    text = (
        'Email ivan.petrov@example.com, телефон +7 (927) 123-45-67, '
        'СНИЛС 123-456-789 00, паспорт 63 12 123456.'
    )

    with pii_redaction_scope():
        redacted = redact_pii_text(text)

    assert 'ivan.petrov@example.com' not in redacted
    assert '+7 (927) 123-45-67' not in redacted
    assert '123-456-789 00' not in redacted
    assert '63 12 123456' not in redacted
    assert '[EMAIL_1]' in redacted
    assert '[ТЕЛЕФОН_1]' in redacted
    assert '[СНИЛС_1]' in redacted
    assert '[ПАСПОРТ_1]' in redacted


def test_email_alias_is_stable_and_resolves_only_inside_request_scope():
    email = 'User.Name@example.com'

    with pii_redaction_scope():
        first = redact_pii_text(f'Отправьте на {email}', trusted=True)
        second = redact_pii_text(
            f'Адрес ещё раз: {email.lower()}',
            trusted=True,
        )

        assert first.endswith('[EMAIL_1]')
        assert second.endswith('[EMAIL_1]')
        assert resolve_email_for_tool('[EMAIL_1]') == email
        assert resolve_email_for_tool(email.lower()) == email


def test_untrusted_email_is_redacted_but_not_resolvable_or_exported():
    email = 'reference@example.com'

    with pii_redaction_scope() as context:
        redacted = redact_pii_text(f'Справочный адрес: {email}')

        assert redacted == 'Справочный адрес: [EMAIL_1]'
        assert context.export() == {}
        with pytest.raises(UnresolvedEmailError):
            resolve_email_for_tool('[EMAIL_1]')
        with pytest.raises(UnresolvedEmailError):
            resolve_email_for_tool(email)


def test_user_input_upgrades_existing_untrusted_email_alias():
    untrusted_email = 'INFO@example.com'
    user_email = 'info@example.com'

    with pii_redaction_scope() as context:
        from_reference = redact_pii_text(f'Справка: {untrusted_email}')
        from_user = redact_pii_text(
            f'Моя почта: {user_email}',
            trusted=True,
        )

        assert from_reference == 'Справка: [EMAIL_1]'
        assert from_user == 'Моя почта: [EMAIL_1]'
        assert context.export() == {
            '[EMAIL_1]': ['EMAIL', user_email]
        }
        assert resolve_email_for_tool('[EMAIL_1]') == user_email
        assert resolve_email_for_tool(untrusted_email) == user_email


@pytest.mark.parametrize('value', ['[EMAIL_99]', 'unknown@example.com'])
def test_unknown_email_value_is_rejected_inside_request_scope(value):
    with pii_redaction_scope():
        with pytest.raises(UnresolvedEmailError):
            resolve_email_for_tool(value)


@pytest.mark.parametrize('value', ['user@example.com', '[EMAIL_1]'])
def test_email_resolution_without_scope_is_fail_closed(value):
    with pytest.raises(UnresolvedEmailError, match='Контекст обезличивания'):
        resolve_email_for_tool(value)


@pytest.mark.parametrize('value', [None, '', '   ', 123])
def test_missing_or_non_string_email_is_rejected(value):
    with pytest.raises(UnresolvedEmailError):
        resolve_email_for_tool(value)


def test_recursive_redaction_handles_tuples_and_preserves_non_text_values():
    with pii_redaction_scope():
        redacted = redact_pii_value(
            ('user@example.com', 42, None, {'name': 'Меня зовут Алексей'})
        )

    assert redacted == (
        '[EMAIL_1]',
        42,
        None,
        {'name': 'Меня зовут [ФИО_1]'},
    )


@pytest.mark.parametrize(
    ('placeholder', 'neutral_text', 'label'),
    [
        ('[EMAIL_99]', 'указанный адрес электронной почты', 'EMAIL'),
        ('[ФИО_99]', 'указанное вами лицо', 'ФИО'),
        ('[ТЕЛЕФОН_99]', 'указанный номер телефона', 'ТЕЛЕФОН'),
        ('[СНИЛС_99]', 'указанные данные СНИЛС', 'СНИЛС'),
        ('[ПАСПОРТ_99]', 'указанные паспортные данные', 'ПАСПОРТ'),
    ],
)
def test_neutralizes_each_supported_placeholder_type(
    placeholder,
    neutral_text,
    label,
):
    neutralized, counts = neutralize_pii_placeholders(
        f'До {placeholder} после'
    )

    assert neutralized == f'До {neutral_text} после'
    assert counts == {label: 1}


def test_neutralizes_multiple_and_repeated_placeholders():
    neutralized, counts = neutralize_pii_placeholders(
        '[ФИО_1], [ФИО_99], [ТЕЛЕФОН_2] и снова [ТЕЛЕФОН_2]'
    )

    assert neutralized == (
        'указанное вами лицо, указанное вами лицо, '
        'указанный номер телефона и снова указанный номер телефона'
    )
    assert counts == {'ФИО': 2, 'ТЕЛЕФОН': 2}


@pytest.mark.parametrize(
    'text',
    [
        'ч. 1 ст. 81 ТК РФ',
        'См. [Приложение 1] к договору.',
        'Служебное обозначение [СТАТЬЯ_1] не является PII-маркером.',
        'Обычный текст без маркеров.',
    ],
)
def test_placeholder_neutralization_preserves_unrelated_text(text):
    assert neutralize_pii_placeholders(text) == (text, {})


def test_vera_name_is_not_redacted():
    with pii_redaction_scope():
        assert redact_pii_text('Вера, помоги разобраться') == 'Вера, помоги разобраться'


def test_single_ner_false_positive_does_not_break_user_command():
    with pii_redaction_scope():
        assert (
            redact_pii_text('Объясни предыдущий ответ проще')
            == 'Объясни предыдущий ответ проще'
        )


def test_single_name_is_redacted_after_explicit_self_identification():
    with pii_redaction_scope():
        assert redact_pii_text('Меня зовут Иван') == 'Меня зовут [ФИО_1]'


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('меня зовут алексей', 'меня зовут [ФИО_1]'),
        ('МЕНЯ ЗОВУТ АЛЕКСЕЙ', 'МЕНЯ ЗОВУТ [ФИО_1]'),
        ('Меня зовут: алексей', 'Меня зовут: [ФИО_1]'),
        ('Меня зовут — алексей', 'Меня зовут — [ФИО_1]'),
        ('меня зовут «алексей»', 'меня зовут «[ФИО_1]»'),
        ('меня зовут "алексей"', 'меня зовут "[ФИО_1]"'),
        ('мое имя анна', 'мое имя [ФИО_1]'),
        ('моё имя: анна-мария', 'моё имя: [ФИО_1]'),
        ('Мои ФИО иванов иван иванович', 'Мои ФИО [ФИО_1]'),
        (
            'мое ФИО: петров алексей сергеевич',
            'мое ФИО: [ФИО_1]',
        ),
        ('ФИО: петров алексей сергеевич', 'ФИО: [ФИО_1]'),
        ('Ф.И.О.: петров алексей сергеевич', 'Ф.И.О.: [ФИО_1]'),
        (
            'фамилия, имя, отчество: петров алексей сергеевич',
            'фамилия, имя, отчество: [ФИО_1]',
        ),
        ('я — алексей', 'я — [ФИО_1]'),
        ('я: алексей', 'я: [ФИО_1]'),
        (
            'ок, спасибо! я, кстати, Кирилл инвалид по зрению',
            'ок, спасибо! я, кстати, [ФИО_1] инвалид по зрению',
        ),
        (
            'ок, спасибо! я, кстати, кирилл инвалид по зрению',
            'ок, спасибо! я, кстати, [ФИО_1] инвалид по зрению',
        ),
        ('я вообще-то алексей', 'я вообще-то [ФИО_1]'),
        ('я, между прочим, Анна', 'я, между прочим, [ФИО_1]'),
        ('я если что Иван', 'я если что [ФИО_1]'),
        ('я на самом деле Мария', 'я на самом деле [ФИО_1]'),
        ('меня зовут джон смит', 'меня зовут [ФИО_1]'),
        ('меня зовут john smith', 'меня зовут [ФИО_1]'),
        ('меня зовут саид-али', 'меня зовут [ФИО_1]'),
        ('ФИО: иванов и. и.', 'ФИО: [ФИО_1]'),
        ('меня зовут ли вэй', 'меня зовут [ФИО_1]'),
        # Эовин не имеет Name-граммемы в pymorphy2: срабатывает точечный
        # Title Case fallback через NewsNER, а не слепое правило "любое слово".
        ('меня зовут эовин', 'меня зовут [ФИО_1]'),
        ('Меня зовут Вера', 'Меня зовут [ФИО_1]'),
    ],
)
def test_redacts_person_after_explicit_marker_in_informal_input(
    text: str,
    expected: str,
):
    with pii_redaction_scope():
        assert redact_pii_text(text) == expected


@pytest.mark.parametrize(
    'text',
    [
        'я хочу получить консультацию',
        'я работаю удалённо',
        'я инвалид второй группы',
        'Я, возможно, ошибаюсь',
        'я, кстати, работаю удалённо',
        'я вообще-то хочу получить консультацию',
        'я, между прочим, инвалид второй группы',
        'я если что пользователь без регистрации',
        'я на самом деле ошибаюсь',
        'я — инвалид второй группы',
        'я: пользователь без регистрации',
        'меня зовут на работу',
        'меня зовут помочь коллегам',
        'меня зовут работать в офисе',
        'ФИО нужно указать в заявлении',
        'Что означает ФИО?',
        'Поле «ФИО» обязательно',
        'Ф.И.О. расшифровывается как фамилия, имя и отчество',
        'фамилия, имя, отчество должны быть указаны полностью',
        'мое имя пользователя в приложении',
        'моё имя файла указано в заголовке',
        'Вера, помоги разобраться',
        'Объясни предыдущий ответ проще',
        'Я купил самокат в Пятницу',
        'Любовь к работе важна',
    ],
)
def test_contextual_person_rules_do_not_mask_non_names(text: str):
    with pii_redaction_scope():
        assert redact_pii_text(text) == text


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (
            'меня зовут алексей помогите пожалуйста',
            'меня зовут [ФИО_1] помогите пожалуйста',
        ),
        (
            'меня зовут алексей петров помогите пожалуйста',
            'меня зовут [ФИО_1] помогите пожалуйста',
        ),
        (
            'ФИО: петров алексей сергеевич проживаю в Самаре',
            'ФИО: [ФИО_1] проживаю в Самаре',
        ),
        (
            'ФИО: иванов и. и., отправьте документ',
            'ФИО: [ФИО_1], отправьте документ',
        ),
    ],
)
def test_contextual_person_rule_stops_at_end_of_name(
    text: str,
    expected: str,
):
    with pii_redaction_scope():
        assert redact_pii_text(text) == expected


def test_full_name_rule_wins_when_yargy_recognizes_only_name_prefix():
    # «мир» имеет обычную словарную морфологию, поэтому Yargy завершает
    # контекстный span после «алексей». Более широкий строгий шаблон ФИО
    # должен получить приоритет и не оставить отчество открытым.
    text = 'меня зовут алексей мир сергеевич помогите'

    with pii_redaction_scope():
        redacted = redact_pii_text(text)

    assert redacted == 'меня зовут [ФИО_1] помогите'


def test_contextual_person_rule_uses_exact_spans_without_substring_replacement():
    text = 'Меня зовут Анна. Ванна сломалась, слово «Анна» приведено как пример.'

    with pii_redaction_scope():
        redacted = redact_pii_text(text)

    assert redacted == (
        'Меня зовут [ФИО_1]. Ванна сломалась, '
        'слово «Анна» приведено как пример.'
    )


def test_email_has_priority_over_contextual_person_candidate():
    with pii_redaction_scope():
        redacted = redact_pii_text('Меня зовут user@example.com')

    assert redacted == 'Меня зовут [EMAIL_1]'
    assert '[ФИО_' not in redacted


def test_person_alias_is_stable_across_case_variants():
    with pii_redaction_scope():
        first = redact_pii_text('меня зовут алексей петров')
        second = redact_pii_text('Моё имя Алексей Петров')

    assert first == 'меня зовут [ФИО_1]'
    assert second == 'Моё имя [ФИО_1]'


def test_person_aliases_are_isolated_between_request_scopes():
    with pii_redaction_scope():
        first_scope_first = redact_pii_text('меня зовут алексей')
        first_scope_second = redact_pii_text('моё имя мария')

    with pii_redaction_scope():
        second_scope_first = redact_pii_text('моё имя мария')

    assert first_scope_first.endswith('[ФИО_1]')
    assert first_scope_second.endswith('[ФИО_2]')
    assert second_scope_first.endswith('[ФИО_1]')


def test_context_export_and_hydrate_preserve_aliases_and_counters():
    first_context = PiiRedactionContext()
    first_email = redact_pii_text(
        'Почта first@example.com',
        first_context,
        trusted=True,
    )
    first_person = redact_pii_text(
        'Меня зовут Алексей',
        first_context,
        trusted=True,
    )

    second_context = PiiRedactionContext()
    second_context.hydrate(first_context.export())
    repeated_email = redact_pii_text(
        'Снова FIRST@example.com',
        second_context,
        trusted=True,
    )
    next_email = redact_pii_text(
        'Теперь second@example.com',
        second_context,
        trusted=True,
    )
    repeated_person = redact_pii_text(
        'Моё имя Алексей',
        second_context,
        trusted=True,
    )

    assert first_email == 'Почта [EMAIL_1]'
    assert repeated_email == 'Снова [EMAIL_1]'
    assert next_email == 'Теперь [EMAIL_2]'
    assert first_person == 'Меня зовут [ФИО_1]'
    assert repeated_person == 'Моё имя [ФИО_1]'
    assert second_context.resolve_email('[EMAIL_1]') == 'first@example.com'
    assert second_context.resolve_email('[EMAIL_2]') == 'second@example.com'


def test_reserved_legacy_alias_is_not_reused_without_stored_value():
    context = PiiRedactionContext()
    context.reserve_aliases(
        {
            'content': 'Старые [EMAIL_1], [EMAIL_3] и [ФИО_7]',
            'tool_calls': [{'args': {'email': '[EMAIL_2]'}}],
        }
    )

    redacted_email = redact_pii_text(
        'new@example.com',
        context,
        trusted=True,
    )
    redacted_person = redact_pii_text(
        'Меня зовут Анна',
        context,
        trusted=True,
    )

    assert redacted_email == '[EMAIL_4]'
    assert redacted_person == 'Меня зовут [ФИО_8]'
    with pytest.raises(UnresolvedEmailError):
        context.resolve_email('[EMAIL_1]')
    assert context.resolve_email('[EMAIL_4]') == 'new@example.com'


def test_hydrate_ignores_invalid_or_mismatched_email_records_but_reserves_index():
    context = PiiRedactionContext()
    context.hydrate(
        {
            '[EMAIL_4]': ('PERSON', 'attacker@example.com'),
            '[EMAIL_5]': ('EMAIL', 'not-an-email'),
            '[EMAIL_6]': ['EMAIL', 'valid@example.com'],
            '[UNKNOWN_99]': ('EMAIL', 'other@example.com'),
        }
    )

    assert redact_pii_text('next@example.com', context) == '[EMAIL_7]'
    assert context.resolve_email('[EMAIL_6]') == 'valid@example.com'
    with pytest.raises(UnresolvedEmailError):
        context.resolve_email('[EMAIL_4]')
    with pytest.raises(UnresolvedEmailError):
        context.resolve_email('[EMAIL_5]')


@pytest.mark.parametrize(
    'text',
    [
        'закон Яровой',
        'Закон Яровой',
        'явка обязательна',
        'Явка обязательна',
        'ясли для ребёнка',
        'Ясли для ребёнка',
        'ярмарка вакансий',
        'якорь договора',
        'Яндекс открыл вакансию',
        'язык трудового договора',
        'Ярославский районный суд',
    ],
)
def test_letter_ya_inside_word_does_not_start_person_marker(text):
    with pii_redaction_scope():
        assert redact_pii_text(text) == text


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('я Алексей', 'я [ФИО_1]'),
        ('Я — Алексей', 'Я — [ФИО_1]'),
        ('я, кстати, Кирилл', 'я, кстати, [ФИО_1]'),
    ],
)
def test_standalone_ya_still_starts_person_marker(text, expected):
    with pii_redaction_scope():
        assert redact_pii_text(text) == expected


@pytest.mark.parametrize(
    'text',
    [
        '79991234567',
        '+79991234567',
        '8 (999) 123-45-67',
        '7 999 123 45 67',
        '7(999)1234567',
        'телефон 9991234567',
        'телефон: (999) 123-45-67',
        'номер телефона — 999 123-45-67',
        'мобильный номер 9991234567',
        'контактный номер: 9991234567',
        'тел. 999-123-45-67',
        'звоните 9991234567',
        'позвоните мне по номеру 999 123 45 67',
        'WhatsApp: 9991234567',
        'ватсап 9991234567',
        'Телефон +79991234567. В 2024 году адрес изменился.',
        'телефон 9991234567, добавочный 42',
    ],
)
def test_redacts_supported_phone_formats(text):
    with pii_redaction_scope():
        redacted = redact_pii_text(text)

    assert '[ТЕЛЕФОН_1]' in redacted
    assert sum(character.isdigit() for character in redacted) < 10


@pytest.mark.parametrize(
    'text',
    [
        '9991234567',
        'ИНН 9991234567',
        'номер дела 9991234567',
        'заказ № 9991234567',
        '4276 3800 1234 5678',
        '7999 1234 5678 9012',
        '40702810900000000001',
        '799912345678',
        '+7999123456',
        '899912345678',
        'телефон 8 999 123 45 67 89',
        'мой номер 9991234567',
    ],
)
def test_phone_rules_do_not_mask_ambiguous_or_wrong_length_numbers(text):
    with pii_redaction_scope():
        assert redact_pii_text(text) == text


def test_redaction_logs_pipeline_counts_without_personal_values(caplog):
    text = 'ок, спасибо! я, кстати, Кирилл инвалид по зрению'
    caplog.set_level(logging.INFO, logger='vera_agent_service')

    with pii_redaction_scope():
        redact_pii_text(text)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == 'vera_agent_service'
    ]
    assert messages == [
        '🛡️ Проверка ПДн: строгие правила=0, Yargy=1, '
        'Natasha PER=1, принято фильтром=1, отклонено=0, '
        'заменено=1, типы: ФИО=1.'
    ]
    assert 'Кирилл' not in caplog.text
    assert text not in caplog.text


def test_redaction_logs_rejected_natasha_false_positive_without_value(caplog):
    text = 'Объясни предыдущий ответ проще'
    caplog.set_level(logging.INFO, logger='vera_agent_service')

    with pii_redaction_scope():
        redacted = redact_pii_text(text)

    assert redacted == text
    assert 'Natasha PER=1' in caplog.text
    assert 'принято фильтром=0' in caplog.text
    assert 'отклонено=1' in caplog.text
    assert 'заменено=0' in caplog.text
    assert 'Объясни' not in caplog.text
    assert text not in caplog.text


def test_redaction_does_not_log_when_pipeline_finds_no_pii(caplog):
    caplog.set_level(logging.INFO, logger='vera_agent_service')

    with pii_redaction_scope():
        redacted = redact_pii_text('Когда начинается отпуск?')

    assert redacted == 'Когда начинается отпуск?'
    assert not [
        record
        for record in caplog.records
        if record.name == 'vera_agent_service'
    ]
