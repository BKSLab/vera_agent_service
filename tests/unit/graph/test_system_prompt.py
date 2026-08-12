from app.graph.prompts.description import VERA_DESCRIPTION_PROMPT
from app.graph.prompts.role import VERA_ROLE_PROMPT
from app.graph.prompts.system import (
    CONSULTATION_EMAIL_PROMPT,
    FINAL_RESPONSE_SYSTEM_PROMPT,
    FINAL_RESPONSE_SYSTEM_PROMPT_PARTS,
    FINAL_WITHOUT_SEARCH_PROMPT,
    SOURCES_PROMPT,
    STYLE_PROMPT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_PARTS,
    TOOL_USAGE_PROMPT,
)


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
    assert 'бери только из текста и метаданных полученных чанков' in lowered
    assert 'трудовой кодекс' not in lowered
    assert '181-фз' not in lowered


def test_sources_prompt_requires_rag_grounded_legal_basis_in_every_answer():
    lowered = SOURCES_PROMPT.lower()
    assert 'каждый ответ, сформированный на основе данных vera_rag_kb' in lowered
    assert 'обязательно заверши отдельным обычным абзацем' in lowered
    assert '«основание:»' in lowered
    assert 'ответ на основе rag без абзаца «основание:» считается неполным' in lowered
    assert 'не восстанавливай по памяти' in lowered
    assert 'если ответ опирается на несколько норм' in lowered
    assert 'простыми словами не отменяет обязательное указание основания' in lowered


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


def test_tool_usage_prompt_routes_follow_up_questions_into_search():
    """Уточняющий вопрос наследует тему предыдущего ответа.

    «Каким образом это сделать?» после заземлённого ответа про отпуск ушло
    прямым ответом: в самой реплике нет ни одного предметного слова, кодовый
    guard VERA-021 её не ловит, а промпт про такие вопросы молчал. Модель
    изложила порядок оформления по памяти и подписала его основанием из
    предыдущей реплики.
    """
    lowered = TOOL_USAGE_PROMPT.lower()
    assert 'уточняющий вопрос' in lowered
    assert 'наследует его тему' in lowered
    assert 'не содержит ни одного предметного слова' in lowered
    assert 'требуют поиска так же' in lowered
    assert 'порядок оформления, перечень документов' in lowered
    assert 'нельзя восстанавливать по памяти' in lowered
    assert 'из темы предыдущего вопроса и сути уточнения' in lowered


def test_final_prompt_forbids_new_norms_without_search_data():
    """Вторая линия защиты: даже при ошибке маршрутизации финальный узел не
    должен излагать нормы и порядок действий по памяти."""
    lowered = FINAL_WITHOUT_SEARCH_PROMPT.lower()
    assert 'опирайся только на то, что уже сказано в этом диалоге' in lowered
    assert 'не добавляй новых норм, номеров статей, сроков' in lowered
    assert 'перечней документов и порядка действий' in lowered

    # Упрощение обязано сохранять «Основание:» предыдущего ответа, поэтому
    # запрет точечный: нельзя составлять новое основание, а не выводить его.
    assert 'переформулировать, сократить, упростить или пояснить' in lowered
    assert 'дословно повторяет основание из предыдущего ответа' in lowered
    assert 'не составляй новое основание' in lowered

    assert FINAL_WITHOUT_SEARCH_PROMPT in FINAL_RESPONSE_SYSTEM_PROMPT
    # Правило нужно обоим финальным узлам, но имён MCP-тулов в нём быть не
    # должно — иначе модель воспроизведёт их текстовым псевдовызовом.
    assert 'vera_rag_kb' not in lowered
    assert 'send_consultation_email' not in lowered


def test_consultation_email_prompt_requires_explicit_safe_request():
    lowered = CONSULTATION_EMAIL_PROMPT.lower()
    assert 'явно попросил' in lowered
    assert 'адрес email' in lowered
    assert 'не придумывай' in lowered
    assert 'не исправляй' in lowered
    assert 'не более одного вызова' in lowered
    assert 'не повторяй' in lowered


def test_consultation_email_prompt_delegates_formatting_and_uses_full_text():
    lowered = CONSULTATION_EMAIL_PROMPT.lower()
    assert 'полный самодостаточный итог консультации' in lowered
    assert 'не форматируй консультацию специально' in lowered
    assert 'не сокращай' in lowered
    assert 'не обрезай' in lowered


def test_consultation_email_prompt_handles_tool_result_truthfully():
    lowered = CONSULTATION_EMAIL_PROMPT.lower()
    assert 'status=ok' in lowered
    assert 'status=error' in lowered
    assert 'не утверждай, что письмо доставлено' in lowered
    assert 'проверить почту' in lowered


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


def test_style_prompt_keeps_every_listed_ground_when_simplifying():
    """Просьба объяснить проще не должна укорачивать сам перечень.

    На практике модель прочитала «проще» как «короче» и выбросила из ответа
    одно из оснований для увольнения за прогул, оговорку «независимо от их
    продолжительности» и норму об исчерпывающем перечне грубых нарушений —
    то есть ровно те сведения, ради которых пользователь и спрашивает.
    """
    lowered = STYLE_PROMPT.lower()
    assert 'упрощение касается формулировок, а не состава ответа' in lowered
    assert 'ни один пункт перечня не должен исчезнуть' in lowered
    assert 'упрощай язык, а не сокращай перечень' in lowered
    assert 'ограничивающие права работодателя и защищающие работника' in lowered
    # «два рабочих дня» превратилось в «2 дня»: общего требования сохранять
    # сроки не хватило, единицу измерения пришлось назвать явно.
    assert 'сроки переноси дословно, вместе с единицей измерения' in lowered
    assert 'рабочие дни нельзя заменять днями' in lowered
    # Правило работает и для кнопки «Объяснить проще», и для набранной
    # вручную просьбы: обе идут через `generate_direct` с этим промптом.
    assert STYLE_PROMPT in FINAL_RESPONSE_SYSTEM_PROMPT


def test_style_prompt_says_how_to_simplify_not_only_what_to_keep():
    """Одни запреты на потерю смысла толкают модель пересказать исходный
    ответ почти дословно. Промпт должен так же явно требовать саму работу по
    упрощению и давать её признак."""
    lowered = STYLE_PROMPT.lower()
    assert 'одна мысль в одном коротком предложении' in lowered
    assert 'бытовые слова вместо юридических терминов' in lowered
    assert 'прямое обращение к человеку вместо безличных оборотов' in lowered
    assert 'оставь его и сразу поясни обычными словами' in lowered
    assert 'заметно легче читаться при том же составе сведений' in lowered
    assert 'почти дословно повторяет предыдущий' in lowered


def test_style_prompt_requires_compact_answers_without_losing_legal_meaning():
    lowered = STYLE_PROMPT.lower()
    assert 'по умолчанию отвечай компактно' in lowered
    assert 'не повторяй вопрос, выводы и одинаковые пояснения' in lowered
    assert 'юридически значимые условия, исключения, сроки' in lowered
    assert 'обязательный абзац «основание:»' in lowered
    assert 'неточным или вводящим в заблуждение' in lowered


def test_style_prompt_adapts_form_without_sacrificing_rag_quality():
    lowered = STYLE_PROMPT.lower()
    assert 'учитывай тон, уровень формальности' in lowered
    assert 'на простой или неформальный вопрос отвечай проще' in lowered
    assert 'на подробный или профессиональный' in lowered
    assert 'не копируй ошибки, грубость' in lowered
    assert 'соответствие данным базы знаний' in lowered
    assert STYLE_PROMPT in FINAL_RESPONSE_SYSTEM_PROMPT


def test_prompt_explicitly_forbids_markdown():
    lowered = SYSTEM_PROMPT.lower()
    assert 'только обычным текстом без markdown-разметки' in lowered
    assert 'markdown-заголовки' in lowered
    assert 'маркированные или нумерованные списки' in lowered
    assert 'markdown-ссылки' in lowered


def test_prompt_is_assembled_from_explicit_parts_without_internal_line_wrapping():
    assert SYSTEM_PROMPT == '\n\n'.join(SYSTEM_PROMPT_PARTS)

    assert all('\n' not in part for part in SYSTEM_PROMPT_PARTS)


def test_final_response_prompt_does_not_include_tool_selection_instructions():
    lowered = FINAL_RESPONSE_SYSTEM_PROMPT.lower()

    assert FINAL_RESPONSE_SYSTEM_PROMPT == '\n\n'.join(FINAL_RESPONSE_SYSTEM_PROMPT_PARTS)
    assert TOOL_USAGE_PROMPT not in FINAL_RESPONSE_SYSTEM_PROMPT
    assert CONSULTATION_EMAIL_PROMPT not in FINAL_RESPONSE_SYSTEM_PROMPT
    assert 'vera_rag_kb' not in lowered
    assert 'send_consultation_email' not in lowered
    assert 'не вызывай инструменты' in lowered
    assert 'не имитируй их вызов текстом' in lowered
