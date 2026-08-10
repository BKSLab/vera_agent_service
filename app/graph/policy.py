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


def consultation_email_send_is_confirmed(messages: list[BaseMessage], email: str) -> bool:
    """Проверяет по сохранённой истории два факта из карточки VERA-021,
    вместо того чтобы доверять только намерению, распознанному моделью:
    адрес получателя присутствует в тексте сообщения пользователя текущего
    turn'а, и предыдущий ответ ассистента запрашивал подтверждение.

    Возвращает `False`, если в истории нет ни одного сообщения пользователя
    или ни одного предшествующего ответа ассистента — первая реплика
    диалога не может быть кодово подтверждённой отправкой.
    """
    if not email:
        return False

    last_human_position = None
    for position in range(len(messages) - 1, -1, -1):
        if isinstance(messages[position], HumanMessage):
            last_human_position = position
            break
    if last_human_position is None:
        return False

    last_human = messages[last_human_position]
    if not isinstance(last_human.content, str) or email.lower() not in last_human.content.lower():
        return False

    previous_ai = next(
        (
            message
            for message in reversed(messages[:last_human_position])
            if isinstance(message, AIMessage)
        ),
        None,
    )
    if previous_ai is None or not isinstance(previous_ai.content, str):
        return False
    return CONFIRMATION_REQUEST_MARKER in previous_ai.content
