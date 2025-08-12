from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    bybit_api_key: str | None = os.getenv("BYBIT_API_KEY")
    bybit_api_secret: str | None = os.getenv("BYBIT_API_SECRET")
    sandbox: bool = os.getenv("SANDBOX", "true").lower() in ("1","true","yes")
    data_dir: str = os.getenv("DATA_DIR", "data/parquet")

SETTINGS = Settings()
