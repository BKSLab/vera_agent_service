from fastapi import status


class MessageFeedbackRepositoryError(Exception):
    """Ошибка repository оценок ответов."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Ошибка базы данных при сохранении оценки.'


class MessageFeedbackServiceError(Exception):
    """Ошибка сервиса оценок ответов."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Не удалось сохранить оценку ответа.'
