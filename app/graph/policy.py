"""Кодовые policy-проверки графа (VERA-021)."""

import re

from langchain_core.messages import BaseMessage, HumanMessage

UNSAFE_TOOL_CALL_RESPONSE = (
    'Не удалось сформировать безопасный ответ. Попробуйте повторить запрос позже.'
)
"""Безопасный fallback, если служебный tool-синтаксис всё же попал в текст."""

PSEUDO_TOOL_CALL_MARKERS = (
    'call:default_api:',
    'send_consultation_email{',
    'send_consultation_email(',
    'vera_rag_kb{',
    'vera_rag_kb(',
)


def contains_pseudo_tool_call(text: str) -> bool:
    """Распознаёт служебный синтаксис MCP, утёкший в обычный текст."""
    normalized = ''.join(text.lower().split())
    return any(marker in normalized for marker in PSEUDO_TOOL_CALL_MARKERS)


_FEDERAL_LAW_REFERENCE_RE = re.compile(
    r'(?<![\w№])(?:№\s*)?\d{1,4}\s*[-‐‑–—]\s*фз\b|'
    r'(?<!\w)фз\s*(?:№\s*|[-‐‑–—]\s*)\d{1,4}\b',
    re.IGNORECASE,
)

_ARTICLE_WITH_CODE_REFERENCE_RE = re.compile(
    r'(?:'
    r'(?<!\w)(?:ст\.?\s*|стать(?:я|и|е|ю|ей|ёй)\s+)\d{1,4}(?:\.\d+)*\b'
    r'[\s,;:()«»"\'/\-]{0,32}'
    r'(?<!\w)(?:тк|гк|гпк|апк|коап|ук|упк|нк|ск|жк|зк|бк|кас)\s*рф\b'
    r'|'
    r'(?<!\w)(?:тк|гк|гпк|апк|коап|ук|упк|нк|ск|жк|зк|бк|кас)\s*рф\b'
    r'[\s,;:()«»"\'/\-]{0,32}'
    r'(?<!\w)(?:ст\.?\s*|стать(?:я|и|е|ю|ей|ёй)\s+)\d{1,4}(?:\.\d+)*\b'
    r')',
    re.IGNORECASE,
)

_ARTICLE_WORD_RE = re.compile(
    r'(?<!\w)стать(?:я|и|е|ю|ей|ёй)(?!\w)',
    re.IGNORECASE,
)

_REFERENCE_NUMBER_PATTERN = r'\d{1,4}(?:\.\d+)*'
_LEGAL_CODE_PATTERN = r'(?:тк|гк|гпк|апк|коап|ук|упк|нк|ск|жк|зк|бк|кас)\s*рф'
_ARTICLE_LABEL_PATTERN = r'(?:ст\.?|стать(?:я|и|е|ю|ей|ёй))'
_SUBDIVISION_PATTERN = (
    rf'(?:п\.?|пункт(?:а|е|у|ом)?|ч\.?|част(?:ь|и|ью))\s*{_REFERENCE_NUMBER_PATTERN}'
)
_CODE_REFERENCE_ONLY_RE = re.compile(
    rf'(?:{_SUBDIVISION_PATTERN}[\s,;]*)*'
    rf'(?:'
    rf'{_ARTICLE_LABEL_PATTERN}\s*{_REFERENCE_NUMBER_PATTERN}'
    rf'[\s,;:()/\-]{{0,32}}{_LEGAL_CODE_PATTERN}'
    rf'|'
    rf'{_LEGAL_CODE_PATTERN}[\s,;:()/\-]{{0,32}}'
    rf'{_ARTICLE_LABEL_PATTERN}\s*{_REFERENCE_NUMBER_PATTERN}'
    rf')'
    rf'[\s.!?]*',
    re.IGNORECASE,
)
_FEDERAL_LAW_NUMBER_PATTERN = (
    r'(?:(?:№\s*)?\d{1,4}\s*[-‐‑–—]\s*фз|фз\s*(?:№\s*|[-‐‑–—]\s*)\d{1,4})'
)
_FEDERAL_LAW_LABEL_PATTERN = r'федеральн(?:ый|ого|ому|ым|ом)\s+закон(?:а|е|у|ом)?'
_LEGAL_DATE_PATTERN = r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}'
_FEDERAL_LAW_REFERENCE_ONLY_RE = re.compile(
    rf'(?:{_ARTICLE_LABEL_PATTERN}\s*{_REFERENCE_NUMBER_PATTERN}[\s,]*)?'
    rf'(?:{_FEDERAL_LAW_LABEL_PATTERN}(?:\s+от\s+{_LEGAL_DATE_PATTERN})?[\s,]*)?'
    rf'{_FEDERAL_LAW_NUMBER_PATTERN}[\s.!?]*',
    re.IGNORECASE,
)


def contains_legal_reference(text: str) -> bool:
    """Есть ли в тексте сильные реквизиты правовой нормы.

    Номер федерального закона распознаётся самостоятельно. Ссылка на статью
    считается правовой только рядом с обозначением кодекса, чтобы бытовые
    фразы про статью, часть или пункт не форсировали поиск.
    """
    return bool(
        _FEDERAL_LAW_REFERENCE_RE.search(text)
        or _ARTICLE_WITH_CODE_REFERENCE_RE.search(text)
    )


def is_reference_only(text: str) -> bool:
    """Состоит ли вся реплика только из реквизитов правовой нормы.

    Предикат намеренно консервативен: ложный отрицательный результат лишь
    оставит обычный LLM-маршрут, а ложный положительный мог бы пропустить
    выбор инструмента и декомпозицию содержательного вопроса. Исходный текст
    не нормализуется для tool query; `strip()` применяется только к проверке.
    """
    stripped = text.strip()
    if not stripped:
        return False
    return bool(
        _CODE_REFERENCE_ONLY_RE.fullmatch(stripped)
        or _FEDERAL_LAW_REFERENCE_ONLY_RE.fullmatch(stripped)
    )


_FACTUAL_LEGAL_KEYWORDS = (
    'закон',
    'кодекс',
    'квот',
    'право',
    'права',
    'льгот',
    'пособ',
    'выплат',
    'обязан',
    'увольнен',
    'приказ',
    'постановлен',
    'указ',
    'вакансия',
    'вакансии',
    'трудоустрой',
    'инвалидност',
    'условия труда',
    'больничн',
    'отпуск',
    'договор',
    'мрот',
    'пенси',
)
"""Ключевые слова предметной области (трудоустройство, трудовое право,
льготы для людей с инвалидностью), по которым код грубо отличает
фактический/правовой вопрос от общей реплики (приветствие, благодарность и
т.п.). Список неполон по построению — это не замена классификации моделью,
а страховка на случай, если модель решила ответить напрямую на вопрос,
явно относящийся к базе знаний (VERA-021). Формы слова «статья» проверяются
отдельным regex с границами слова; «часть» и «пункт» самостоятельно не
считаются достаточным признаком."""

_FACTUAL_LEGAL_KEYWORD_PATTERNS = tuple(
    re.compile(rf'(?<!\w){re.escape(keyword)}', re.IGNORECASE)
    for keyword in _FACTUAL_LEGAL_KEYWORDS
)


def is_probably_factual_or_legal_question(text: str) -> bool:
    """Похож ли вопрос пользователя на фактический/правовой запрос,
    прямой ответ на который мимо базы знаний недопустим (VERA-021)."""
    return bool(
        any(pattern.search(text) for pattern in _FACTUAL_LEGAL_KEYWORD_PATTERNS)
        or _ARTICLE_WORD_RE.search(text)
    )


SIMPLIFY_ANSWER_REQUEST = 'Объясни предыдущий ответ проще'
"""Текст, который сайт подставляет по кнопке «Объяснить проще» под ответом,
построенным на базе знаний.

Агент не отличает эту реплику от набранной вручную: кнопка отправляет её
обычным пользовательским сообщением по общему маршруту, поэтому отдельной
ветки графа, поля в контракте очереди и правок системного промпта здесь нет —
сценарий уже описан в `STYLE_PROMPT`.

Формулировка не случайна и не должна меняться без проверки: ни одно слово из
`_FACTUAL_LEGAL_KEYWORDS` в ней встречаться не может. Иначе VERA-021 сочтёт
нажатие кнопки новым правовым вопросом, принудительно уведёт его в
`vera_rag_kb` и вернёт второй ответ по базе знаний вместо переформулировки
предыдущего. Инвариант закреплён тестом; фронтенд обязан отправлять ровно эту
строку (`frontend/src/hooks/useVeraChat.ts`)."""


def find_last_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    """Последнее сообщение пользователя в истории — сообщение текущего
    turn'а (см. `app/graph/state.py`: `messages` дополняется, а не
    затирается, поэтому оно всегда последнее из `HumanMessage`)."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None
