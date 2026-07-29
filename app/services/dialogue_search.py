from dataclasses import dataclass
from datetime import datetime

from app.exceptions.chat_turn import ChatTurnRepositoryError
from app.exceptions.dialogue_search import DialogueSearchServiceError
from app.exceptions.session_feedback import SessionFeedbackRepositoryError
from app.repositories.chat_turn import ChatTurnRepository, RankedChatTurn
from app.repositories.session_feedback import (
    RankedSessionFeedback,
    SessionFeedbackRepository,
)


@dataclass(frozen=True)
class DialogueSearchResult:
    """Результаты поиска по репликам и комментариям."""

    turns: list[RankedChatTurn]
    comments: list[RankedSessionFeedback]


class DialogueSearchService:
    """Выполняет единый административный поиск по сохранённым диалогам."""

    RESULTS_LIMIT = 50

    def __init__(
        self,
        chat_turn_repository: ChatTurnRepository,
        session_feedback_repository: SessionFeedbackRepository,
    ):
        self.chat_turn_repository = chat_turn_repository
        self.session_feedback_repository = session_feedback_repository

    async def search(
        self,
        search_query: str | None,
        turn_status: str | None,
        rating: str | None,
        audience: str | None,
        created_from: datetime | None,
        created_to: datetime | None,
    ) -> DialogueSearchResult:
        """Возвращает FTS-совпадения с единым набором фильтров."""
        try:
            turns = await self.chat_turn_repository.search(
                search_query=search_query,
                turn_status=turn_status,
                rating=rating,
                audience=audience,
                created_from=created_from,
                created_to=created_to,
                limit=self.RESULTS_LIMIT,
            )
            comments = await self.session_feedback_repository.search_comments(
                search_query=search_query,
                turn_status=turn_status,
                rating=rating,
                audience=audience,
                created_from=created_from,
                created_to=created_to,
                limit=self.RESULTS_LIMIT,
            )
            return DialogueSearchResult(turns=turns, comments=comments)
        except (ChatTurnRepositoryError, SessionFeedbackRepositoryError) as error:
            raise DialogueSearchServiceError from error
