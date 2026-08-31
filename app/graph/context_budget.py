"""Ограничение контекста, реально передаваемого модели (VERA-020).

`state['messages']` (и Redis-checkpoint под ним) продолжает копить всю
историю сессии без ограничения — это остаётся ответственностью
LangGraph-reducer'а (`add_messages`, см. `app/graph/state.py`) и не
меняется этим модулем. Исходная история независимо сохраняется в PostgreSQL,
а новые пользовательские сообщения попадают в LangGraph уже обезличенными.
Этот модуль отвечает только за то, что **уходит в конкретный вызов LLM**:
последние `max_turns` реплик целиком плюс одна безопасная текстовая
выжимка более старых.
"""

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.graph.output_guard import detect_unsafe_answer
from app.privacy.pii import redact_pii_value

logger = logging.getLogger('vera_agent_service')

UNSAFE_HISTORY_ANSWER = 'Предыдущий ответ не был сформирован.'
"""Нейтральная замена старого ответа со служебным псевдовызовом."""

_INTERNAL_AI_FIELD_NAMES = frozenset(
    {
        'analysis',
        'think',
        'thinking',
        'thought',
        'thoughts',
    }
)


def _is_internal_ai_field(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.strip().casefold().replace('-', '_')
    return normalized.startswith('reasoning') or normalized in _INTERNAL_AI_FIELD_NAMES


def _contains_internal_ai_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            _is_internal_ai_field(key) or _contains_internal_ai_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_internal_ai_field(item) for item in value)
    return False


def _strip_internal_ai_fields(value: object) -> tuple[object, int]:
    if isinstance(value, dict):
        cleaned: dict = {}
        removed_count = 0
        for key, item in value.items():
            if _is_internal_ai_field(key):
                removed_count += 1
                continue
            cleaned_item, nested_removed_count = _strip_internal_ai_fields(item)
            cleaned[key] = cleaned_item
            removed_count += nested_removed_count
        return cleaned, removed_count
    if isinstance(value, list):
        cleaned_items = []
        removed_count = 0
        for item in value:
            cleaned_item, nested_removed_count = _strip_internal_ai_fields(item)
            cleaned_items.append(cleaned_item)
            removed_count += nested_removed_count
        return cleaned_items, removed_count
    if isinstance(value, tuple):
        cleaned_items = []
        removed_count = 0
        for item in value:
            cleaned_item, nested_removed_count = _strip_internal_ai_fields(item)
            cleaned_items.append(cleaned_item)
            removed_count += nested_removed_count
        return tuple(cleaned_items), removed_count
    return value, 0


def _text_from_history_block(block: object) -> str | None:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return None
    if _contains_internal_ai_field(block):
        return None
    if block.get('type') not in ('text', 'output_text'):
        return None
    text = block.get('text')
    if isinstance(text, str):
        return text
    if isinstance(text, dict) and isinstance(text.get('value'), str):
        return text['value']
    return None


def _history_ai_content_is_unsafe(content: object) -> bool:
    if isinstance(content, str):
        return detect_unsafe_answer(content) is not None
    if not isinstance(content, list) or not content:
        return True
    visible_parts: list[str] = []
    for block in content:
        text = _text_from_history_block(block)
        if text is None:
            # Reasoning, image или неизвестный typed block не переносится в
            # текстовый финальный генератор: границу нельзя угадать надёжно.
            return True
        visible_parts.append(text)
    visible_text = ''.join(visible_parts)
    return not visible_text.strip() or detect_unsafe_answer(visible_text) is not None


def _history_answer_for_summary(content: object) -> tuple[str, bool]:
    if isinstance(content, str):
        if detect_unsafe_answer(content) is None:
            return content, False
        return UNSAFE_HISTORY_ANSWER, True
    if isinstance(content, list) and not _history_ai_content_is_unsafe(content):
        return ''.join(
            text
            for block in content
            if (text := _text_from_history_block(block)) is not None
        ), False
    return UNSAFE_HISTORY_ANSWER, True


def _sanitize_message_for_model(message: BaseMessage) -> BaseMessage | None:
    """Очищает PII и старый небезопасный output перед отправкой модели.

    Такие AIMessage могли быть сохранены до включения фильтрации SSE. Их
    нельзя передавать обратно в LLM: модель может скопировать reasoning или
    служебный синтаксис в новый ответ. Нативный tool-call при этом сохраняется.
    """
    updates: dict = {}
    trusted = isinstance(message, HumanMessage)
    redacted_content = redact_pii_value(message.content, trusted=trusted)
    if redacted_content != message.content:
        updates['content'] = redacted_content

    redacted_additional_kwargs = redact_pii_value(
        message.additional_kwargs,
        trusted=trusted,
    )
    if redacted_additional_kwargs != message.additional_kwargs:
        updates['additional_kwargs'] = redacted_additional_kwargs

    redacted_response_metadata = redact_pii_value(
        message.response_metadata,
        # Response metadata формируется библиотекой/провайдером, а не
        # пользователем, даже у HumanMessage. Она не должна авторизовать email.
        trusted=False,
    )
    if redacted_response_metadata != message.response_metadata:
        updates['response_metadata'] = redacted_response_metadata

    if isinstance(message, AIMessage):
        cleaned_additional_kwargs, additional_removed_count = (
            _strip_internal_ai_fields(redacted_additional_kwargs)
        )
        cleaned_response_metadata, response_removed_count = (
            _strip_internal_ai_fields(redacted_response_metadata)
        )
        removed_count = additional_removed_count + response_removed_count
        if cleaned_additional_kwargs != redacted_additional_kwargs:
            updates['additional_kwargs'] = cleaned_additional_kwargs
        if cleaned_response_metadata != redacted_response_metadata:
            updates['response_metadata'] = cleaned_response_metadata
        if removed_count:
            logger.warning(
                '🛡️ Reasoning-поля удалены из старого AIMessage перед LLM '
                '(removed_fields=%d)',
                removed_count,
            )

        redacted_tool_calls = redact_pii_value(message.tool_calls)
        if redacted_tool_calls != message.tool_calls:
            updates['tool_calls'] = redacted_tool_calls

        content = redacted_content
        if _history_ai_content_is_unsafe(content):
            updates['content'] = UNSAFE_HISTORY_ANSWER
            logger.warning(
                '🛡️ Небезопасный старый AIMessage заменён перед LLM '
                '(content_kind=%s)',
                'text' if isinstance(content, str) else 'typed_blocks',
            )

    return message.model_copy(update=updates) if updates else message


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
    answer, answer_replaced = (
        _history_answer_for_summary(final_ai.content)
        if final_ai is not None
        else ('', False)
    )
    if answer_replaced:
        logger.warning(
            '🛡️ Небезопасный старый AIMessage заменён в summary перед LLM '
            '(content_kind=%s)',
            'text' if isinstance(final_ai.content, str) else 'typed_blocks',
        )
    # Выжимка станет SystemMessage, поэтому очищаем её части до склейки, пока
    # известен источник: email вопроса остаётся пользовательским, а email из
    # ответа модели только маскируется и не разрешается для отправки.
    question = redact_pii_value(question, trusted=True)
    answer = redact_pii_value(answer)
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
    без сырых tool-результатов). `state['messages']` дополнительно не
    изменяется — это только представление для конкретного вызова LLM
    (см. docstring модуля).
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
