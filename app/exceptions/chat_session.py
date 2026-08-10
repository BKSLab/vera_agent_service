from fastapi import status


class ChatSessionRepositoryError(Exception):
    """Ошибка repository постоянных сессий."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Ошибка базы данных при обработке сессии.'


class ChatSessionServiceError(Exception):
    """Ошибка сервиса постоянных сессий."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Не удалось обработать сессию диалога.'


class ChatSessionNotFoundError(Exception):
    """Сессия не найдена в постоянном хранилище."""

    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(session_id)

    @property
    def detail(self) -> str:
        return f'Сессия {self.session_id} не найдена.'


class ChatSessionAccessDeniedError(Exception):
    """Запрашивающий пользователь не владеет сессией."""

    status_code = status.HTTP_403_FORBIDDEN
    detail = 'Нет доступа к этой сессии.'


class ChatSessionInactiveError(ChatSessionAccessDeniedError):
    """Новая реплика направлена в закрытую или устаревшую сессию."""

    status_code = status.HTTP_409_CONFLICT
    detail = 'Сессия закрыта или истекла. Начните новый диалог.'


class ChatSessionResolutionConflictError(Exception):
    """Повтор lifecycle-запроса не совпал с сохранённым successor."""

    status_code = status.HTTP_409_CONFLICT
    detail = 'Для истёкшей сессии уже создан другой новый диалог.'
