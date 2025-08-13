# Application settings

Configuration is centralized in `packages/core/settings.py`.

```python
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
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
    return Settings()
```

`get_settings()` caches values with `lru_cache`, ensuring that all modules
share the same configuration loaded from the `.env` file. A template is
provided in `.env.example`.
