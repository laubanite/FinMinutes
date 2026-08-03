class FinMinutesError(Exception):
    pass


class ConfigError(FinMinutesError):
    pass


class LLMError(FinMinutesError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMAuthenticationError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass
