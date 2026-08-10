"""Кодовые policy-проверки графа (VERA-021).

Обе проверки в этом модуле намеренно читают только сохранённую историю
диалога и явные аргументы инструмента — не текст, сгенерированный моделью, и
не намерение, которое модель "объявила" через выбор тула. Мутирующий вызов
и обход базы знаний на фактических/правовых вопросах не должны зависеть
только от того, что решила модель за один шаг.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

CONFIRMATION_REQUEST_MARKER = '?'
"""Грубый, но кодовый признак того, что предыдущий ответ ассистента
запрашивал у пользователя подтверждение или уточнение (наличие вопроса).
Не заменяет собой полноценный confirmation token (см. карточку VERA-021,
"вне скоупа") — минимальное проверяемое условие для этого прогона."""

_EMAIL_SEND_INTENT_MARKERS = (
    'отправ',
    'высл',
    'перешл',
    'пришл',
    'направ',
    'пошл',
    'скин',
)

CONSULTATION_EMAIL_GUARD_NOTICE = (
    'Подтвердите, пожалуйста, адрес электронной почты и отправку консультации?'
)
"""Детерминированный ответ для неразрешённой попытки email-вызова."""

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


def _human_message_positions(messages: list[BaseMessage]) -> list[int]:
    return [
        position
        for position, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]


def _contains_email_in_human_history(messages: list[BaseMessage], email: str) -> bool:
    email_lower = email.lower()
    return any(
        isinstance(message.content, str) and email_lower in message.content.lower()
        for message in messages
        if isinstance(message, HumanMessage)
    )


def _has_send_intent(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _EMAIL_SEND_INTENT_MARKERS)


def consultation_email_send_is_confirmed(messages: list[BaseMessage], email: str) -> bool:
    """Проверяет, что адрес и намерение отправки пришли от пользователя.

    Явная первая просьба с email разрешается без искусственного предыдущего
    вопроса ассистента. Для сценария «Вера спросила адрес -> пользователь
    ответил коротко» сохраняется подтверждение по предыдущему ответу и
    адресу, который уже был введён человеком в этой сессии.
    """
    if not email:
        return False

    human_positions = _human_message_positions(messages)
    if not human_positions:
        return False

    last_human_position = human_positions[-1]
    last_human = messages[last_human_position]
    current_text = last_human.content if isinstance(last_human.content, str) else ''
    email_in_current_message = email.lower() in current_text.lower()

    previous_ai = next(
        (
            message
            for message in reversed(messages[:last_human_position])
            if isinstance(message, AIMessage)
        ),
        None,
    )
    previous_ai_requested_confirmation = bool(
        previous_ai is not None
        and isinstance(previous_ai.content, str)
        and CONFIRMATION_REQUEST_MARKER in previous_ai.content
    )

    if email_in_current_message:
        return previous_ai_requested_confirmation or _has_send_intent(current_text)

    return previous_ai_requested_confirmation and _contains_email_in_human_history(
        messages[:last_human_position], email
    )
