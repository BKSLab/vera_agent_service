"""Локальное обезличивание текста перед передачей в LLM.

Во время обработки реальные значения доступны через request-local
``ContextVar``; между репликами соответствия из пользовательских сообщений
сохраняются в сессионном Redis-checkpoint. LLM получает только маркеры, а
доверенный пользовательский email восстанавливается непосредственно перед
вызовом мутирующего MCP-инструмента. Значения из AI/tool/system-сообщений
маскируются, но не сохраняются и не разрешаются для отправки.
"""

import logging
import re
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
from yargy import Parser, and_, or_, rule
from yargy.predicates import eq, gram, length_eq
from yargy.predicates import type as token_type

logger = logging.getLogger('vera_agent_service')

EMAIL_KIND = 'EMAIL'
PERSON_KIND = 'PERSON'
PHONE_KIND = 'PHONE'
SNILS_KIND = 'SNILS'
PASSPORT_KIND = 'PASSPORT'


@dataclass(frozen=True)
class _PiiPlaceholderSpec:
    """Единое описание маркера и его безопасного текста для документов."""

    label: str
    neutral_text: str


_PII_PLACEHOLDER_SPECS = {
    EMAIL_KIND: _PiiPlaceholderSpec(
        label='EMAIL',
        neutral_text='указанный адрес электронной почты',
    ),
    PERSON_KIND: _PiiPlaceholderSpec(
        label='ФИО',
        neutral_text='указанное вами лицо',
    ),
    PHONE_KIND: _PiiPlaceholderSpec(
        label='ТЕЛЕФОН',
        neutral_text='указанный номер телефона',
    ),
    SNILS_KIND: _PiiPlaceholderSpec(
        label='СНИЛС',
        neutral_text='указанные данные СНИЛС',
    ),
    PASSPORT_KIND: _PiiPlaceholderSpec(
        label='ПАСПОРТ',
        neutral_text='указанные паспортные данные',
    ),
}
_PLACEHOLDER_LABELS = {
    kind: spec.label for kind, spec in _PII_PLACEHOLDER_SPECS.items()
}

_EMAIL_PATTERN = re.compile(
    r"(?<![\w@])[\w.!#$%&'*+/=?^`{|}~-]+@(?:[\w-]+\.)+[\w-]{2,63}(?![\w@])",
    re.UNICODE,
)
_PHONE_PATTERN = re.compile(
    r'(?<!\d)(?:\+?7|8)(?:[\s().-]*\d){10}(?![\s()-]*\d)'
)
_CONTEXTUAL_PHONE_PATTERN = re.compile(
    r'(?ix)(?:'
    r'\b(?:телефон(?:а|у|ом|е)?|номер\s+телефона|'
    r'мобильный\s+номер|контактный\s+номер)\b'
    r'|(?<!\w)тел\.\s*'
    r'|\b(?:позвоните|звоните|позвонить|звонить)\b'
    r'(?:\s+(?:мне\s+)?по\s+номеру)?'
    r'|\b(?:whatsapp|whats\s*app|ватсап|вотсап)\b'
    r')\s*(?::|№|[-—])?\s*'
    r'(?P<value>(?<!\d)\(?\d(?:[\s().-]*\d){9})(?![\s()-]*\d)'
)
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
    rf'(?ix)\bя\b'
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
    rf'\bя\b(?:\s*,?\s*{_SELF_IDENTIFICATION_ASIDE_PATTERN}\s*,?)?'
    r'|\bменя\s+зовут'
    r'|\bмо[её]\s+имя'
    r')\s*(?:[-—:]\s*)?$'
)

_PLACEHOLDER_KINDS_BY_LABEL = {
    label: kind for kind, label in _PLACEHOLDER_LABELS.items()
}
_PLACEHOLDER_PATTERN = re.compile(
    rf'\[(?P<label>{"|".join(map(re.escape, _PLACEHOLDER_KINDS_BY_LABEL))})_'
    r'(?P<index>[1-9]\d*)\]'
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
    """Хранилище соответствий маркеров в рамках обрабатываемой сессии.

    Сам объект остаётся request-local в ``ContextVar``. Между запросами
    consumer явно восстанавливает его из Redis-checkpoint и сохраняет
    обновлённую копию обратно в состояние LangGraph.
    """

    _aliases_by_value: dict[tuple[str, str], str] = field(default_factory=dict)
    _values_by_alias: dict[str, tuple[str, str]] = field(default_factory=dict)
    _trusted_aliases: set[str] = field(default_factory=set)
    _counters: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _value_key(kind: str, value: str) -> tuple[str, str]:
        normalized = value.casefold() if kind in {EMAIL_KIND, PERSON_KIND} else value
        return kind, normalized

    def alias_for(self, kind: str, value: str, *, trusted: bool = False) -> str:
        """Возвращает alias и при необходимости отмечает пользовательский источник.

        ``trusted`` разрешён только для данных из ``HumanMessage``. Такие
        значения сохраняются между репликами, а email можно восстановить перед
        вызовом инструмента. Данные модели, инструментов и системных сообщений
        тоже маскируются, но не становятся разрешёнными адресами отправки.
        """

        key = self._value_key(kind, value)
        if alias := self._aliases_by_value.get(key):
            if trusted and alias not in self._trusted_aliases:
                # Если значение сначала пришло из недоверенного источника, а
                # затем его явно указал пользователь, сохраняем именно
                # пользовательское написание (важно для local-part email).
                self._values_by_alias[alias] = (kind, value)
                self._trusted_aliases.add(alias)
            return alias

        index = self._counters.get(kind, 0) + 1
        self._counters[kind] = index
        alias = f'[{_PLACEHOLDER_LABELS[kind]}_{index}]'
        self._aliases_by_value[key] = alias
        self._values_by_alias[alias] = (kind, value)
        if trusted:
            self._trusted_aliases.add(alias)
        return alias

    @staticmethod
    def _alias_metadata(alias: str) -> tuple[str, int] | None:
        match = _PLACEHOLDER_PATTERN.fullmatch(alias)
        if match is None:
            return None
        kind = _PLACEHOLDER_KINDS_BY_LABEL[match.group('label')]
        return kind, int(match.group('index'))

    def reserve_aliases(self, value: Any) -> None:
        """Не переиспользует номера маркеров, уже встречавшиеся в истории.

        Это нужно для checkpoint'ов, созданных до появления сессионного
        mapping: значение старого маркера восстановить нельзя, но новый адрес
        не должен получить тот же alias.
        """

        if isinstance(value, str):
            for match in _PLACEHOLDER_PATTERN.finditer(value):
                kind = _PLACEHOLDER_KINDS_BY_LABEL[match.group('label')]
                index = int(match.group('index'))
                self._counters[kind] = max(self._counters.get(kind, 0), index)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                self.reserve_aliases(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self.reserve_aliases(item)

    def hydrate(self, stored: Any) -> None:
        """Безопасно объединяет сохранённые alias → (kind, value)."""

        if not isinstance(stored, Mapping):
            return

        for alias, stored_value in stored.items():
            if not isinstance(alias, str):
                continue
            metadata = self._alias_metadata(alias)
            if metadata is None:
                continue
            alias_kind, index = metadata
            self._counters[alias_kind] = max(
                self._counters.get(alias_kind, 0),
                index,
            )

            if (
                not isinstance(stored_value, (list, tuple))
                or len(stored_value) != 2
            ):
                continue
            kind, value = stored_value
            if kind != alias_kind or not isinstance(value, str) or not value:
                continue
            if kind == EMAIL_KIND and _EMAIL_PATTERN.fullmatch(value) is None:
                continue

            key = self._value_key(kind, value)
            existing_alias = self._aliases_by_value.get(key)
            existing_value = self._values_by_alias.get(alias)
            if (
                (existing_alias is not None and existing_alias != alias)
                or (
                    existing_value is not None
                    and existing_value != (kind, value)
                )
            ):
                continue
            self._aliases_by_value[key] = alias
            self._values_by_alias[alias] = (kind, value)
            # В checkpoint экспортируются только значения из пользовательских
            # сообщений, поэтому успешно загруженная запись доверенная.
            self._trusted_aliases.add(alias)

    def export(self) -> dict[str, list[str]]:
        """Возвращает только доверенный mapping для Redis-checkpoint."""

        return {
            alias: [kind, value]
            for alias, (kind, value) in self._values_by_alias.items()
            if alias in self._trusted_aliases
        }

    def resolve_email(self, value: str) -> str:
        stored = self._values_by_alias.get(value)
        if (
            value in self._trusted_aliases
            and stored is not None
            and stored[0] == EMAIL_KIND
        ):
            return stored[1]

        alias = self._aliases_by_value.get(self._value_key(EMAIL_KIND, value))
        if alias is not None and alias in self._trusted_aliases:
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
                PHONE_KIND,
                match.group('value'),
                90,
            )
            for match in _CONTEXTUAL_PHONE_PATTERN.finditer(text)
        )
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

    def _person_entities(
        self,
        text: str,
    ) -> tuple[list[_DetectedPii], int]:
        document = Doc(text)
        document.segment(self._segmenter)
        document.tag_ner(self._ner_tagger)
        candidates = [span for span in document.spans if span.type == 'PER']
        entities = [
            _DetectedPii(span.start, span.stop, PERSON_KIND, span.text, 10)
            for span in candidates
            if (
                span.text.casefold() not in _SAFE_PERSON_ENTITIES
                and (
                    len(span.text.split()) >= 2
                    or _SELF_IDENTIFICATION_PREFIX.search(text[:span.start]) is not None
                )
            )
        ]
        return entities, len(candidates)

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

    def redact(
        self,
        text: str,
        context: PiiRedactionContext,
        *,
        trusted: bool = False,
    ) -> str:
        if not text:
            return text
        regex_entities = self._regex_entities(text)
        contextual_entities = self._contextual_person_entities(text)
        person_entities, natasha_candidate_count = self._person_entities(text)
        entities = self._without_overlaps(
            [
                *regex_entities,
                *contextual_entities,
                *person_entities,
            ]
        )
        if regex_entities or contextual_entities or natasha_candidate_count:
            type_counts: dict[str, int] = {}
            for entity in entities:
                label = _PLACEHOLDER_LABELS[entity.kind]
                type_counts[label] = type_counts.get(label, 0) + 1
            types_summary = ', '.join(
                f'{label}={count}' for label, count in sorted(type_counts.items())
            ) or 'нет'
            logger.info(
                '🛡️ Проверка ПДн: строгие правила=%d, Yargy=%d, '
                'Natasha PER=%d, принято фильтром=%d, отклонено=%d, '
                'заменено=%d, типы: %s.',
                len(regex_entities),
                len(contextual_entities),
                natasha_candidate_count,
                len(person_entities),
                natasha_candidate_count - len(person_entities),
                len(entities),
                types_summary,
            )
        if not entities:
            return text

        parts: list[str] = []
        cursor = 0
        for entity in entities:
            parts.append(text[cursor:entity.start])
            parts.append(
                context.alias_for(
                    entity.kind,
                    entity.value,
                    trusted=trusted,
                )
            )
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
    """Создаёт изолированный рабочий контекст одной обработки запроса."""

    context = PiiRedactionContext()
    token = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)


def redact_pii_text(
    text: str,
    context: PiiRedactionContext | None = None,
    *,
    trusted: bool = False,
) -> str:
    """Возвращает обезличенный текст, не выполняя внешних вызовов.

    По умолчанию найденные значения не разрешаются для мутирующих
    инструментов. ``trusted=True`` допустим только для пользовательского ввода.
    """

    active_context = context or _current_context.get() or PiiRedactionContext()
    return _redactor.redact(text, active_context, trusted=trusted)


def redact_pii_value(value: Any, *, trusted: bool = False) -> Any:
    """Рекурсивно очищает строковые значения message/tool-call структур."""

    if isinstance(value, str):
        return redact_pii_text(value, trusted=trusted)
    if isinstance(value, dict):
        return {
            key: redact_pii_value(item, trusted=trusted)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_pii_value(item, trusted=trusted) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_pii_value(item, trusted=trusted) for item in value)
    return value


def neutralize_pii_placeholders(text: str) -> tuple[str, dict[str, int]]:
    """Убирает служебные PII-маркеры из пользовательского документа.

    Реальные значения намеренно не восстанавливаются: ``trusted`` означает
    только происхождение из ``HumanMessage``, но не принадлежность получателю
    письма. Возвращаемые счётчики содержат лишь безопасные типы маркеров и
    нужны для наблюдаемости страховочного срабатывания.
    """

    type_counts: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        label = match.group('label')
        kind = _PLACEHOLDER_KINDS_BY_LABEL[label]
        type_counts[label] = type_counts.get(label, 0) + 1
        return _PII_PLACEHOLDER_SPECS[kind].neutral_text

    return _PLACEHOLDER_PATTERN.sub(replace, text), type_counts


def resolve_email_for_tool(value: Any) -> str:
    """Восстанавливает подтверждённый email перед локальным MCP-вызовом.

    Без активного scope разрешение запрещено: потеря ``ContextVar`` не должна
    превращать проверку адреса в fail-open перед мутирующим MCP-вызовом.
    """

    if not isinstance(value, str) or not value.strip():
        raise UnresolvedEmailError('Email не указан')
    context = _current_context.get()
    if context is None:
        raise UnresolvedEmailError('Контекст обезличивания недоступен')
    return context.resolve_email(value)
