from fastapi import status


class SessionFeedbackRepositoryError(Exception):
    """Ошибка repository развёрнутых отзывов."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Ошибка базы данных при сохранении отзыва.'


class SessionFeedbackAlreadyExistsError(Exception):
    """Отзыв с таким submission_id уже существует."""

    status_code = status.HTTP_409_CONFLICT
    detail = 'Отзыв с таким submission_id уже существует.'


class SessionFeedbackSubmissionMismatchError(Exception):
    """Ключ идемпотентности уже относится к другой сессии."""

    status_code = status.HTTP_409_CONFLICT
    detail = 'submission_id уже относится к другой сессии.'


class SessionFeedbackServiceError(Exception):
    """Ошибка сервиса развёрнутых отзывов."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Не удалось сохранить отзыв.'
