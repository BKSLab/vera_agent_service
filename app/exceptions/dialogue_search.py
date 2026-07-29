from fastapi import status


class DialogueSearchServiceError(Exception):
    """Ошибка полнотекстового поиска по сохранённым диалогам."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = 'Не удалось выполнить поиск по диалогам.'
