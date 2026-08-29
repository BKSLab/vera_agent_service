import pytest

from app.privacy.pii import (
    UnresolvedEmailError,
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
        first = redact_pii_text(f'Отправьте на {email}')
        second = redact_pii_text(f'Адрес ещё раз: {email.lower()}')

        assert first.endswith('[EMAIL_1]')
        assert second.endswith('[EMAIL_1]')
        assert resolve_email_for_tool('[EMAIL_1]') == email
        assert resolve_email_for_tool(email.lower()) == email


def test_unknown_email_alias_is_rejected_inside_request_scope():
    with pii_redaction_scope():
        with pytest.raises(UnresolvedEmailError):
            resolve_email_for_tool('[EMAIL_99]')


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
