from pydantic import BaseModel, ConfigDict, Field


class KbSearchChunkResult(BaseModel):
    """Один чанк в ответе `vera_rag_kb` (VERA-021).

    `extra='allow'` — RAG-сервис может вернуть дополнительные метаданные
    чанка (например, `score`), которые узлам графа не нужны для генерации,
    но не должны отбрасываться на этапе валидации.
    """

    model_config = ConfigDict(extra='allow')

    chunk_id: str | None = None
    source_title: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    text: str = ''


class KbSearchToolResult(BaseModel):
    """Результат тула `vera_rag_kb`, проверяемый по схеме вместо
    `json.loads` без проверки формы (VERA-021)."""

    model_config = ConfigDict(extra='allow')

    chunks: list[KbSearchChunkResult] = Field(default_factory=list)


class ConsultationEmailToolResult(BaseModel):
    """Результат тула `send_consultation_email` (VERA-021).

    Поля намеренно не строже `str | None`: разбор "ожидаемого ли значения
    `status`" (`ok`/`error`) — доменная проверка узла графа
    (`app/graph/nodes/call_consultation_email.py`), а не забота схемы
    формата ответа MCP. Здесь отвергается только структурно некорректный
    ответ (не тот тип поля, несколько content-блоков и т.п.).
    """

    model_config = ConfigDict(extra='allow')

    status: str | None = None
    code: str | None = None
    message: str | None = None
    email: str | None = None
    document_name: str | None = None
