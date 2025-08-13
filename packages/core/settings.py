"""Application configuration utilities.

This module exposes a `Settings` class backed by `pydantic_settings` and a
`get_settings` function cached with `lru_cache` to avoid repetitive reads of
the environment file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Container for environment variables.

    Attributes mirror the keys used across the project. Values are loaded
    from `.env` if present and can be overridden by real environment
    variables at runtime.
    """

    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None
    sandbox: bool = True
    data_dir: str = "data/parquet"
    recv_window: int = 5000
    ws_public_url: str | None = None
    ws_private_url: str | None = None
    rest_base_url: str | None = None
    account_mode: str = "oneway"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance."""

    return Settings()
