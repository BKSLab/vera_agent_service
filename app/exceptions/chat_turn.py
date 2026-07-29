from fastapi import status


class ChatTurnRepositoryError(Exception):
    """Ошибка repository постоянных реплик."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Ошибка базы данных при обработке реплики.'


class ChatTurnAlreadyExistsError(Exception):
    """Реплика с таким request_id уже существует."""

    status_code = status.HTTP_409_CONFLICT
    detail = 'Реплика с таким request_id уже существует.'


class ChatPersistenceServiceError(Exception):
    """Ошибка сервиса сохранения диалога."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Не удалось сохранить реплику диалога.'


class ChatTurnNotFoundError(Exception):
    """Реплика не найдена в постоянном хранилище."""

    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, request_id: str):
        self.request_id = request_id
        super().__init__(request_id)

    @property
    def detail(self) -> str:
        return f'Реплика {self.request_id} не найдена.'


class ChatTurnSessionMismatchError(Exception):
    """Реплика не относится к указанной сессии."""

    status_code = status.HTTP_409_CONFLICT
    detail = 'Реплика не относится к указанной сессии.'


class ChatTurnNotCompletedError(Exception):
    """Оценивать можно только завершённую реплику."""

    status_code = status.HTTP_409_CONFLICT
    detail = 'Оценивать можно только завершённую реплику.'
