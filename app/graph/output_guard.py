"""Fail-closed проверка полного ответа финального LLM-вызова.

Модуль намеренно не работает с отдельными streaming chunks: решение можно
принять только после получения и разбора всего структурированного ответа.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.graph.policy import contains_pseudo_tool_call


class OutputGuardReason(StrEnum):
    """Безопасные коды причин, пригодные для логов и telemetry."""

    ACCEPTED = 'accepted'
    EMPTY_OUTPUT = 'empty_output'
    INVALID_JSON = 'invalid_json'
    DUPLICATE_JSON_KEY = 'duplicate_json_key'
    INVALID_SCHEMA = 'invalid_schema'
    EMPTY_ANSWER = 'empty_answer'
    MIXED_CONTENT = 'mixed_content'
    PSEUDO_TOOL_CALL = 'pseudo_tool_call'
    REASONING_MARKER = 'reasoning_marker'
    META_REASONING = 'meta_reasoning'
    SYSTEM_PROMPT_FRAGMENT = 'system_prompt_fragment'
    CONTROL_CHARACTER = 'control_character'


@dataclass(frozen=True, slots=True)
class OutputGuardDecision:
    accepted: bool
    reason: OutputGuardReason
    answer: str | None = None


class _DuplicateJsonKeyError(ValueError):
    pass


_REASONING_TAG_RE = re.compile(
    r'<\s*/?\s*(?:think|thinking|thought|thoughts|reasoning|analysis)\b',
    re.IGNORECASE,
)
_META_REASONING_PREFIX_RE = re.compile(
    r'^\s*(?:[*_#>`~-]+\s*)?'
    r'(?:analysis|reasoning|thought|thoughts|thought process|chain of thought|internal reasoning|'
    r'анализ|рассуждение|ход рассуждений|внутреннее рассуждение|проверка правил)'
    r'\s*(?::|：|[-–—])',
    re.IGNORECASE,
)

# Только высокоуверенные маркеры служебного рассуждения. Этот список —
# дополнительная защита поверх JSON schema и разделения provider-каналов, а
# не попытка классифицировать произвольный естественный текст.
_META_REASONING_MARKERS = (
    'the user is asking',
    'rules check',
    'we need to answer',
    'we need answer',
    'we need to respond',
    'we should answer',
    'we should respond',
    'the assistant should answer',
    'according to the system prompt',
    'system/developer instructions',
    'пользователь спрашивает',
    'проверка правил',
    'нужно ответить пользователю',
    'согласно системному промпту',
)

# Дословные характерные начала внутренних инструкций. Обычный ответ не должен
# их воспроизводить даже после prompt-injection со стороны пользователя.
_SYSTEM_PROMPT_FRAGMENTS = (
    'финальный ответ. инструменты для этого запроса уже обработаны программой.',
    'конфиденциальность внутренних данных. никогда не раскрывай',
    'обезличенные персональные данные. маркеры вида',
    'источники ответов. при ответе на фактические и правовые вопросы',
    'предыдущая попытка финального ответа не прошла автоматическую проверку',
)


def _collapse_marker_words(text: str) -> str:
    return ' '.join(re.sub(r'[\W_]+', ' ', text, flags=re.UNICODE).split())


_META_REASONING_WORD_MARKERS = tuple(
    _collapse_marker_words(marker) for marker in _META_REASONING_MARKERS
)
_SYSTEM_PROMPT_WORD_FRAGMENTS = tuple(
    _collapse_marker_words(fragment) for fragment in _SYSTEM_PROMPT_FRAGMENTS
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def detect_unsafe_answer(text: str) -> OutputGuardReason | None:
    """Проверяет уже извлечённый пользовательский текст.

    Функция переиспользуется для старой истории, чтобы известный служебный
    output не возвращался в контекст следующего модельного вызова.
    """
    normalized = unicodedata.normalize('NFKC', text).casefold()
    normalized = ''.join(
        character
        for character in normalized
        if unicodedata.category(character) != 'Cf'
    )
    collapsed = ' '.join(normalized.split())
    collapsed_words = _collapse_marker_words(normalized)
    if contains_pseudo_tool_call(normalized):
        return OutputGuardReason.PSEUDO_TOOL_CALL
    if _REASONING_TAG_RE.search(normalized):
        return OutputGuardReason.REASONING_MARKER
    if _META_REASONING_PREFIX_RE.search(normalized):
        return OutputGuardReason.META_REASONING
    if any(marker in collapsed for marker in _META_REASONING_MARKERS) or any(
        marker in collapsed_words for marker in _META_REASONING_WORD_MARKERS
    ):
        return OutputGuardReason.META_REASONING
    if any(fragment in collapsed for fragment in _SYSTEM_PROMPT_FRAGMENTS) or any(
        fragment in collapsed_words
        for fragment in _SYSTEM_PROMPT_WORD_FRAGMENTS
    ):
        return OutputGuardReason.SYSTEM_PROMPT_FRAGMENT
    if any(
        character not in {'\t', '\n', '\r'}
        and unicodedata.category(character) in {'Cc', 'Cf', 'Cs'}
        for character in text
    ):
        return OutputGuardReason.CONTROL_CHARACTER
    return None


def validate_plain_final_answer(answer: object) -> OutputGuardDecision:
    """Defense-in-depth для уже извлечённого ``answer`` на границах графа."""
    if not isinstance(answer, str):
        return OutputGuardDecision(False, OutputGuardReason.INVALID_SCHEMA)
    if not answer.strip():
        return OutputGuardDecision(False, OutputGuardReason.EMPTY_ANSWER)
    unsafe_reason = detect_unsafe_answer(answer)
    if unsafe_reason is not None:
        return OutputGuardDecision(False, unsafe_reason)
    return OutputGuardDecision(True, OutputGuardReason.ACCEPTED, answer)


def validate_structured_final_answer(
    raw_content: str,
    *,
    mixed_content: bool = False,
) -> OutputGuardDecision:
    """Разбирает строгий transport-envelope ``{"answer": string}``.

    Любая неоднозначность блокируется. Текст ``answer`` после успешной
    проверки возвращается без нормализации или переформатирования.
    """
    if mixed_content:
        return OutputGuardDecision(False, OutputGuardReason.MIXED_CONTENT)
    if not raw_content:
        return OutputGuardDecision(False, OutputGuardReason.EMPTY_OUTPUT)

    try:
        payload = json.loads(raw_content, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateJsonKeyError:
        return OutputGuardDecision(False, OutputGuardReason.DUPLICATE_JSON_KEY)
    except (json.JSONDecodeError, TypeError, ValueError):
        return OutputGuardDecision(False, OutputGuardReason.INVALID_JSON)

    if not isinstance(payload, dict) or set(payload) != {'answer'}:
        return OutputGuardDecision(False, OutputGuardReason.INVALID_SCHEMA)
    return validate_plain_final_answer(payload['answer'])
