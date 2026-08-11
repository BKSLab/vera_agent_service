"""Кодовые policy-проверки графа (VERA-021)."""

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

_FACTUAL_LEGAL_KEYWORDS = (
    'закон',
    'кодекс',
    'статья',
    'статьи',
    'квота',
    'квоты',
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
явно относящийся к базе знаний (VERA-021)."""


def is_probably_factual_or_legal_question(text: str) -> bool:
    """Похож ли вопрос пользователя на фактический/правовой запрос,
    прямой ответ на который мимо базы знаний недопустим (VERA-021)."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in _FACTUAL_LEGAL_KEYWORDS)


def find_last_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    """Последнее сообщение пользователя в истории — сообщение текущего
    turn'а (см. `app/graph/state.py`: `messages` дополняется, а не
    затирается, поэтому оно всегда последнее из `HumanMessage`)."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None
