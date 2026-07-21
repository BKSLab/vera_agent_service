from app.graph.prompts.system import SYSTEM_PROMPT


def test_prompt_defines_role_as_consultant():
    assert 'Вера' in SYSTEM_PROMPT
    assert 'консультант' in SYSTEM_PROMPT.lower()


def test_prompt_covers_both_audiences():
    assert 'соискател' in SYSTEM_PROMPT.lower()
    assert 'работодател' in SYSTEM_PROMPT.lower()


def test_prompt_forbids_inventing_facts():
    assert 'не выдумывай' in SYSTEM_PROMPT.lower()


def test_prompt_requires_honest_refusal_when_no_data():
    lowered = SYSTEM_PROMPT.lower()
    assert 'честно сообщи' in lowered
    assert 'недоступен' in lowered


def test_prompt_explains_kb_search_audience_argument():
    assert 'vera_rag_kb' in SYSTEM_PROMPT
    assert '"seeker"' in SYSTEM_PROMPT
    assert '"employer"' in SYSTEM_PROMPT
    assert '"both"' in SYSTEM_PROMPT


def test_prompt_suggests_registration_for_unauthenticated_personal_requests():
    lowered = SYSTEM_PROMPT.lower()
    assert 'незалогиненн' in lowered
    assert 'зарегистрироваться' in lowered or 'войти в аккаунт' in lowered


def test_prompt_has_six_explicit_paragraphs_without_internal_line_wrapping():
    paragraphs = SYSTEM_PROMPT.rstrip('\n').split('\n\n')

    assert len(paragraphs) == 6
    assert all('\n' not in paragraph for paragraph in paragraphs)
