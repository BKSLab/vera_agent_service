import hmac

from sqladmin.authentication import AuthenticationBackend
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request

from app.core.rate_limit import limiter
from app.core.settings import get_settings


class AdminLoginAuth(AuthenticationBackend):
    """Отдельная cookie-аутентификация административной панели."""

    def __init__(self, secret_key: str, https_only: bool = False):
        super().__init__(secret_key=secret_key)
        self.middlewares = [
            Middleware(
                SessionMiddleware,
                secret_key=secret_key,
                https_only=https_only,
            )
        ]

    @limiter.limit('5/minute')
    async def login(self, request: Request) -> bool:
        """Проверяет административный логин и пароль из Settings."""
        settings = get_settings()
        form = await request.form()
        username = form.get('username', '')
        password = form.get('password', '')
        username_valid = hmac.compare_digest(username, settings.app.admin_login)
        password_valid = hmac.compare_digest(
            password,
            settings.app.admin_password.get_secret_value(),
        )
        if username_valid and password_valid:
            request.session['admin_authenticated'] = True
            return True
        return False

    async def logout(self, request: Request) -> bool:
        """Очищает административную сессию."""
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        """Проверяет наличие признака входа в подписанной сессии."""
        return request.session.get('admin_authenticated', False)
