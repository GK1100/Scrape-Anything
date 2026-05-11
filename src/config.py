"""
Configuration management — loads .env and exposes typed settings.
Single Responsibility: only handles configuration.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralised application settings, read from environment."""

    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    MAX_URLS: int = int(os.getenv("MAX_URLS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "20"))

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing required keys."""
        missing = []
        if not cls.SERPER_API_KEY:
            missing.append("SERPER_API_KEY")
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        return missing


settings = Settings()
