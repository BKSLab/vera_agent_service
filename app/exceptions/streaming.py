class InvalidStreamTicketError(Exception):
    """Ticket SSE-потока отсутствует, повреждён или больше не действует."""


class SessionAlreadySubscribedError(Exception):
    """Для request_id уже существует активный SSE-подписчик."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        super().__init__(request_id)
