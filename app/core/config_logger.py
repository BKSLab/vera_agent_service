import logging
import logging.config

from app.core.settings import get_settings

logging.config.fileConfig(
    get_settings().app.logging_config_path,
    disable_existing_loggers=False,
)

logger = logging.getLogger('vera_agent_service')
