"""Ограничение контекста, реально передаваемого модели (VERA-020).

`state['messages']` (и Redis-checkpoint под ним) продолжает копить всю
историю сессии без ограничения — это остаётся ответственностью
LangGraph-reducer'а (`add_messages`, см. `app/graph/state.py`) и не
меняется здесь. Полная история также независимо сохраняется в PostgreSQL.
Этот модуль отвечает только за то, что **уходит в конкретный вызов LLM**:
последние `max_turns` реплик целиком плюс одна безопасная текстовая
выжимка более старых.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.graph.policy import contains_pseudo_tool_call

UNSAFE_HISTORY_ANSWER = 'Предыдущий ответ не был сформирован.'
"""Нейтральная замена старого ответа со служебным псевдовызовом."""


def _sanitize_message_for_model(message: BaseMessage) -> BaseMessage | None:
    """Не даёт старому псевдовызову MCP повторно заразить контекст модели.

    Такие AIMessage могли быть сохранены до включения фильтрации SSE. Их
    нельзя передавать обратно в LLM: локальные модели часто копируют этот
    синтаксис в новый ответ. Нативный tool-call при этом сохраняется, если
    провайдер прислал его вместе с текстом.
    """
    if not isinstance(message, AIMessage) or not isinstance(message.content, str):
        return message
    if not contains_pseudo_tool_call(message.content):
        return message
    if message.tool_calls:
        return message.model_copy(update={'content': UNSAFE_HISTORY_ANSWER})
    return AIMessage(content=UNSAFE_HISTORY_ANSWER)


def _sanitize_messages_for_model(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [
        sanitized
        for message in messages
        if (sanitized := _sanitize_message_for_model(message)) is not None
    ]


def _split_into_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """Группирует плоский список сообщений в реплики: каждый `HumanMessage`
    начинает новую реплику, всё до следующего `HumanMessage` (ответ
    ассистента, промежуточные tool-сообщения) относится к предыдущей.

    Сообщения до первого `HumanMessage` (в норме такого не бывает) относятся
    к первой реплике, чтобы ни одно сообщение не терялось.
    """
    turns: list[list[BaseMessage]] = []
    for message in messages:
        if isinstance(message, HumanMessage) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def _summarize_turn(turn: list[BaseMessage], max_chars: int) -> str:
    """Безопасная текстовая выжимка одной старой реплики: только
    человекочитаемый вопрос и финальный ответ, без сырых tool-результатов и
    служебных полей."""
    human = next((m for m in turn if isinstance(m, HumanMessage)), None)
    final_ai = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
    question = human.content if human and isinstance(human.content, str) else ''
    answer = final_ai.content if final_ai and isinstance(final_ai.content, str) else ''
    if contains_pseudo_tool_call(answer):
        answer = UNSAFE_HISTORY_ANSWER
    summary = f'Вопрос: {question.strip()}\nОтвет: {answer.strip()}'.strip()
    if max_chars > 0 and len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + '…'
    return summary


def build_bounded_messages(
    messages: list[BaseMessage],
    max_turns: int,
    older_turns_summary_max_chars: int,
) -> list[BaseMessage]:
    """Возвращает сообщения, реально передаваемые модели в текущем вызове.

    Реплик не больше `max_turns` — передаются полностью, включая
    tool-сообщения. Более старые реплики схлопываются в одну
    `SystemMessage` с короткой выжимкой каждой ("Вопрос: ... Ответ: ...",
    без сырых tool-результатов). `state['messages']` не изменяется — это
    только представление для конкретного вызова LLM (см. docstring модуля).
    """
    turns = _split_into_turns(messages)
    if len(turns) <= max_turns:
        return _sanitize_messages_for_model(messages)

    older_turns, recent_turns = turns[:-max_turns], turns[-max_turns:]
    summaries = [_summarize_turn(turn, older_turns_summary_max_chars) for turn in older_turns]
    summary_message = SystemMessage(
        content=(
            'Краткая выжимка более ранних реплик диалога, не вошедших в '
            'полный контекст из-за ограничения бюджета:\n\n' + '\n\n'.join(summaries)
        )
    )
    bounded: list[BaseMessage] = [summary_message]
    for turn in recent_turns:
        bounded.extend(turn)
    return _sanitize_messages_for_model(bounded)
