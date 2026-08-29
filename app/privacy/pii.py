"""Локальное обезличивание текста перед передачей в LLM.

Реальные значения живут только в request-local ``ContextVar``. LangGraph и
LLM получают стабильные маркеры, а email восстанавливается непосредственно
перед вызовом мутирующего MCP-инструмента.
"""

import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
from yargy import Parser, and_, or_, rule
from yargy.predicates import eq, gram, length_eq
from yargy.predicates import type as token_type

EMAIL_KIND = 'EMAIL'
PERSON_KIND = 'PERSON'
PHONE_KIND = 'PHONE'
SNILS_KIND = 'SNILS'
PASSPORT_KIND = 'PASSPORT'

_PLACEHOLDER_LABELS = {
    EMAIL_KIND: 'EMAIL',
    PERSON_KIND: 'ФИО',
    PHONE_KIND: 'ТЕЛЕФОН',
    SNILS_KIND: 'СНИЛС',
    PASSPORT_KIND: 'ПАСПОРТ',
}

_EMAIL_PATTERN = re.compile(
    r"(?<![\w@])[\w.!#$%&'*+/=?^`{|}~-]+@(?:[\w-]+\.)+[\w-]{2,63}(?![\w@])",
    re.UNICODE,
)
_PHONE_PATTERN = re.compile(r'(?<!\d)(?:\+7|8)(?:[\s().-]*\d){10}(?!\d)')
_SNILS_PATTERN = re.compile(r'(?<!\d)\d{3}-\d{3}-\d{3}[ -]\d{2}(?!\d)')
_PASSPORT_PATTERN = re.compile(
    r'(?i)(?:паспорт(?:а)?|серия(?:\s+и\s+номер)?)\s*[:№]?\s*'
    r'(?P<value>\d{2}\s?\d{2}\s?\d{6})(?!\d)'
)
_RUSSIAN_FULL_NAME_PATTERN = re.compile(
    r'(?i)(?<![а-яё-])(?P<value>'
    r'[а-яё-]{2,}\s+[а-яё-]{2,}\s+'
    r'[а-яё-]*(?:ович|евич|ильич|овна|евна|ична)'
    r')(?![а-яё-])'
)
_EXPLICIT_PERSON_MARKER_PATTERN = re.compile(
    r'(?ix)(?:'
    r'\b(?:меня\s+зовут|мо[её]\s+имя|мо(?:е|ё|и)\s+фио)\b'
    r'|(?<!\w)ф\s*\.\s*и\s*\.\s*о\s*\.?'
    r'|\bфио\b(?=\s*[:—-])'
    r'|\bфамилия\s*,?\s*имя\s*,?\s*отчество\b(?=\s*[:—-])'
    r')\s*(?::|[-—])?\s*'
)
_SELF_IDENTIFICATION_ASIDE_PATTERN = (
    r'(?:кстати|вообще(?:-то)?|между\s+прочим|если\s+что|на\s+самом\s+деле)'
)
_I_AM_PERSON_MARKER_PATTERN = re.compile(
    rf'(?ix)\bя'
    rf'(?:\s*,?\s*{_SELF_IDENTIFICATION_ASIDE_PATTERN}\s*,?)?'
    r'\s*(?::|[-—])?\s*'
)
_NAME_OPENING_DELIMITER_PATTERN = re.compile(r'''[\s«„“"'(]*''')
_FALLBACK_PERSON_COMPONENT_PATTERN = re.compile(
    r'(?:[А-ЯЁа-яёA-Za-z]{2,}'
    r'(?:[-‑\'’][А-ЯЁа-яёA-Za-z]{2,})*|[А-ЯЁа-яёA-Za-z]\.)'
    r'(?![\w\'’‑-])'
)
_SAFE_PERSON_ENTITIES = frozenset({'вера'})
_SELF_IDENTIFICATION_PREFIX = re.compile(
    rf'(?ix)(?:'
    rf'\bя(?:\s*,?\s*{_SELF_IDENTIFICATION_ASIDE_PATTERN}\s*,?)?'
    r'|\bменя\s+зовут'
    r'|\bмо[её]\s+имя'
    r')\s*(?:[-—:]\s*)?$'
)

# Yargy использует морфологию, поэтому распознаёт имена, фамилии и отчества
# даже в нижнем регистре. UNKN/LATIN оставлены только внутри явного маркера:
# это позволяет скрывать редкие и иностранные имена, не применяя правило ко
# всему сообщению и не превращая обычные слова в ложные ФИО.
_KNOWN_PERSON_TOKEN = rule(
    and_(token_type('RU'), or_(gram('Name'), gram('Surn'), gram('Patr')))
)
_UNKNOWN_PERSON_TOKEN = rule(
    or_(and_(token_type('RU'), gram('UNKN')), token_type('LATIN'))
)
_PERSON_TOKEN = or_(_KNOWN_PERSON_TOKEN, _UNKNOWN_PERSON_TOKEN)
_PERSON_INITIAL = rule(and_(token_type('RU'), length_eq(1)), eq('.'))
_HYPHENATED_PERSON_TOKEN = rule(_PERSON_TOKEN, eq('-'), _PERSON_TOKEN)
_CONTEXTUAL_PERSON_COMPONENT = or_(
    _HYPHENATED_PERSON_TOKEN,
    _PERSON_INITIAL,
    _KNOWN_PERSON_TOKEN,
    _UNKNOWN_PERSON_TOKEN,
)
_CONTEXTUAL_PERSON_RULE = rule(
    _CONTEXTUAL_PERSON_COMPONENT,
    _CONTEXTUAL_PERSON_COMPONENT.optional(),
    _CONTEXTUAL_PERSON_COMPONENT.optional(),
)


class UnresolvedEmailError(ValueError):
    """Модель вернула email, которого не было в обезличенном контексте."""


@dataclass(frozen=True)
class _DetectedPii:
    start: int
    end: int
    kind: str
    value: str
    priority: int


@dataclass
class PiiRedactionContext:
    """Request-local хранилище соответствий маркеров реальным значениям."""

    _aliases_by_value: dict[tuple[str, str], str] = field(default_factory=dict)
    _values_by_alias: dict[str, tuple[str, str]] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _value_key(kind: str, value: str) -> tuple[str, str]:
        normalized = value.casefold() if kind in {EMAIL_KIND, PERSON_KIND} else value
        return kind, normalized

    def alias_for(self, kind: str, value: str) -> str:
        key = self._value_key(kind, value)
        if alias := self._aliases_by_value.get(key):
            return alias

        index = self._counters.get(kind, 0) + 1
        self._counters[kind] = index
        alias = f'[{_PLACEHOLDER_LABELS[kind]}_{index}]'
        self._aliases_by_value[key] = alias
        self._values_by_alias[alias] = (kind, value)
        return alias

    def resolve_email(self, value: str) -> str:
        stored = self._values_by_alias.get(value)
        if stored is not None and stored[0] == EMAIL_KIND:
            return stored[1]

        alias = self._aliases_by_value.get(self._value_key(EMAIL_KIND, value))
        if alias is not None:
            return self._values_by_alias[alias][1]

        raise UnresolvedEmailError('Email отсутствует в пользовательском контексте')


class PiiRedactor:
    """Каскад строгих правил, морфологии Yargy и локального Natasha NER."""

    def __init__(self) -> None:
        self._segmenter = Segmenter()
        self._ner_tagger = NewsNERTagger(NewsEmbedding())
        # pymorphy2 внутри Yargy предупреждает о deprecated pkg_resources.
        # Совместимая версия setuptools зафиксирована в requirements.txt;
        # локально подавляем только это стороннее предупреждение при создании.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'pkg_resources is deprecated as an API.*',
                category=UserWarning,
                module=r'pymorphy2\.analyzer',
            )
            self._contextual_person_parser = Parser(_CONTEXTUAL_PERSON_RULE)

    @staticmethod
    def _regex_entities(text: str) -> list[_DetectedPii]:
        entities = [
            *(
                _DetectedPii(match.start(), match.end(), EMAIL_KIND, match.group(), 100)
                for match in _EMAIL_PATTERN.finditer(text)
            ),
            *(
                _DetectedPii(match.start(), match.end(), PHONE_KIND, match.group(), 90)
                for match in _PHONE_PATTERN.finditer(text)
            ),
            *(
                _DetectedPii(match.start(), match.end(), SNILS_KIND, match.group(), 90)
                for match in _SNILS_PATTERN.finditer(text)
            ),
        ]
        entities.extend(
            _DetectedPii(
                match.start('value'),
                match.end('value'),
                PASSPORT_KIND,
                match.group('value'),
                90,
            )
            for match in _PASSPORT_PATTERN.finditer(text)
        )
        entities.extend(
            _DetectedPii(
                match.start('value'),
                match.end('value'),
                PERSON_KIND,
                match.group('value'),
                40,
            )
            for match in _RUSSIAN_FULL_NAME_PATTERN.finditer(text)
        )
        return entities

    def _person_entities(self, text: str) -> list[_DetectedPii]:
        document = Doc(text)
        document.segment(self._segmenter)
        document.tag_ner(self._ner_tagger)
        return [
            _DetectedPii(span.start, span.stop, PERSON_KIND, span.text, 10)
            for span in document.spans
            if (
                span.type == 'PER'
                and span.text.casefold() not in _SAFE_PERSON_ENTITIES
                and (
                    len(span.text.split()) >= 2
                    or _SELF_IDENTIFICATION_PREFIX.search(text[:span.start]) is not None
                )
            )
        ]

    def _is_standalone_person(self, value: str) -> bool:
        """Проверяет редкое имя, которое морфологический словарь не знает."""

        title_cased = value.title()
        document = Doc(title_cased)
        document.segment(self._segmenter)
        document.tag_ner(self._ner_tagger)
        return any(
            span.type == 'PER' and span.start == 0 and span.stop == len(title_cased)
            for span in document.spans
        )

    def _contextual_person_entities(self, text: str) -> list[_DetectedPii]:
        """Извлекает ФИО после узких маркеров до общего NER-прохода.

        Yargy принимает только морфологически похожие на ФИО токены (либо
        неизвестные/латинские слова непосредственно после маркера). Поэтому
        конструкции ``я хочу`` и ``меня зовут на работу`` не маскируются.
        Между ``я`` и именем допускаются короткие вводные обороты вроде
        ``кстати``: они не должны отменять уже распознанное имя.
        """

        entities: list[_DetectedPii] = []
        for marker_pattern in (
            _EXPLICIT_PERSON_MARKER_PATTERN,
            _I_AM_PERSON_MARKER_PATTERN,
        ):
            for marker in marker_pattern.finditer(text):
                opening = _NAME_OPENING_DELIMITER_PATTERN.match(text, marker.end())
                value_start = opening.end() if opening is not None else marker.end()
                suffix = text[value_start:]

                match = self._contextual_person_parser.find(suffix)
                if match is not None and match.span.start == 0:
                    value_end = value_start + match.span.stop
                else:
                    # NewsNER на исходном lower-case тексте может пропустить
                    # редкое имя. Проверяем только первый токен после явного
                    # маркера в Title Case, не меняя остальное сообщение.
                    fallback = _FALLBACK_PERSON_COMPONENT_PATTERN.match(
                        text,
                        value_start,
                    )
                    if fallback is None or not self._is_standalone_person(
                        fallback.group()
                    ):
                        continue
                    value_end = fallback.end()

                entities.append(
                    _DetectedPii(
                        value_start,
                        value_end,
                        PERSON_KIND,
                        text[value_start:value_end],
                        30,
                    )
                )
        return entities

    @staticmethod
    def _without_overlaps(entities: list[_DetectedPii]) -> list[_DetectedPii]:
        selected: list[_DetectedPii] = []
        for entity in sorted(
            entities,
            key=lambda item: (-item.priority, -(item.end - item.start), item.start),
        ):
            if any(
                entity.start < existing.end and existing.start < entity.end
                for existing in selected
            ):
                continue
            selected.append(entity)
        return sorted(selected, key=lambda item: item.start)

    def redact(self, text: str, context: PiiRedactionContext) -> str:
        if not text:
            return text
        entities = self._without_overlaps(
            [
                *self._regex_entities(text),
                *self._contextual_person_entities(text),
                *self._person_entities(text),
            ]
        )
        if not entities:
            return text

        parts: list[str] = []
        cursor = 0
        for entity in entities:
            parts.append(text[cursor:entity.start])
            parts.append(context.alias_for(entity.kind, entity.value))
            cursor = entity.end
        parts.append(text[cursor:])
        return ''.join(parts)


_current_context: ContextVar[PiiRedactionContext | None] = ContextVar(
    'pii_redaction_context',
    default=None,
)
_redactor = PiiRedactor()


@contextmanager
def pii_redaction_scope() -> Iterator[PiiRedactionContext]:
    """Создаёт изолированное хранилище маркеров на одну обработку запроса."""

    context = PiiRedactionContext()
    token = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)


def redact_pii_text(text: str) -> str:
    """Возвращает обезличенный текст, не выполняя внешних вызовов."""

    context = _current_context.get() or PiiRedactionContext()
    return _redactor.redact(text, context)


def redact_pii_value(value: Any) -> Any:
    """Рекурсивно очищает строковые значения message/tool-call структур."""

    if isinstance(value, str):
        return redact_pii_text(value)
    if isinstance(value, dict):
        return {key: redact_pii_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_pii_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_pii_value(item) for item in value)
    return value


def resolve_email_for_tool(value: Any) -> str:
    """Восстанавливает подтверждённый email перед локальным MCP-вызовом.

    Без активного scope сохраняется обратная совместимость прямых вызовов
    графа и unit-тестов. Production consumer всегда активирует scope.
    """

    if not isinstance(value, str) or not value.strip():
        raise UnresolvedEmailError('Email не указан')
    context = _current_context.get()
    if context is None:
        return value
    return context.resolve_email(value)
