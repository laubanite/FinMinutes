from finminutes.core.config_manager import ConfigManager
from finminutes.core.logging_setup import setup_logging

config_manager = ConfigManager()
setup_logging(config_manager.config)
