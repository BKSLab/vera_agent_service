from app.graph.prompts.description import VERA_DESCRIPTION_PROMPT
from app.graph.prompts.role import VERA_ROLE_PROMPT
from app.graph.prompts.system import SOURCES_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_PARTS, TOOL_USAGE_PROMPT


def test_prompt_defines_role_as_consultant():
    assert 'Вера' in SYSTEM_PROMPT
    assert 'консультант' in SYSTEM_PROMPT.lower()


def test_description_and_role_are_reusable_prompt_parts():
    assert SYSTEM_PROMPT_PARTS[:2] == (VERA_DESCRIPTION_PROMPT, VERA_ROLE_PROMPT)
    assert VERA_DESCRIPTION_PROMPT in SYSTEM_PROMPT
    assert VERA_ROLE_PROMPT in SYSTEM_PROMPT


def test_prompt_covers_two_audiences_including_workers_with_disabilities():
    lowered = VERA_ROLE_PROMPT.lower()
    assert 'двум аудиториям' in lowered
    assert 'соискател' in lowered
    assert 'работник' in lowered
    assert 'работодател' in lowered


def test_prompt_forbids_inventing_facts():
    assert 'не выдумывай' in SYSTEM_PROMPT.lower()


def test_sources_prompt_uses_search_results_without_hardcoded_corpus_inventory():
    lowered = SOURCES_PROMPT.lower()
    assert 'только информацию, полученную через инструмент' in lowered
    assert 'из метаданных полученных чанков' in lowered
    assert 'трудовой кодекс' not in lowered
    assert '181-фз' not in lowered


def test_prompt_requires_honest_refusal_when_no_data():
    lowered = SYSTEM_PROMPT.lower()
    assert 'честно сообщи' in lowered
    assert 'недоступен' in lowered


def test_tool_usage_prompt_requires_a_self_contained_query():
    lowered = TOOL_USAGE_PROMPT.lower()
    assert 'vera_rag_kb' in lowered
    assert 'самодостаточный поисковый запрос' in lowered
    assert 'контекст из истории диалога' in lowered


def test_tool_usage_prompt_forbids_role_based_search_restriction():
    lowered = TOOL_USAGE_PROMPT.lower()
    assert 'по всей базе знаний' in lowered
    assert 'не пытайся ограничивать поиск ролью или аудиторией' in lowered
    assert '"seeker"' not in lowered
    assert '"employer"' not in lowered
    assert '"both"' not in lowered


def test_prompt_suggests_registration_for_unauthenticated_personal_requests():
    lowered = SYSTEM_PROMPT.lower()
    assert 'незалогиненн' in lowered
    assert 'зарегистрироваться' in lowered or 'войти в аккаунт' in lowered


def test_prompt_explains_how_to_answer_in_plain_language_on_request():
    lowered = SYSTEM_PROMPT.lower()
    assert 'если пользователь просит объяснить ответ проще' in lowered
    assert 'сохраняй исходный правовой смысл' in lowered
    assert 'условия, исключения, сроки и ограничения' in lowered
    assert 'не используй детский или снисходительный тон' in lowered
    assert 'не добавляй новых фактов' in lowered


def test_prompt_explicitly_forbids_markdown():
    lowered = SYSTEM_PROMPT.lower()
    assert 'только обычным текстом без markdown-разметки' in lowered
    assert 'markdown-заголовки' in lowered
    assert 'маркированные или нумерованные списки' in lowered
    assert 'markdown-ссылки' in lowered


def test_prompt_is_assembled_from_explicit_parts_without_internal_line_wrapping():
    assert SYSTEM_PROMPT == '\n\n'.join(SYSTEM_PROMPT_PARTS)

    assert all('\n' not in part for part in SYSTEM_PROMPT_PARTS)
