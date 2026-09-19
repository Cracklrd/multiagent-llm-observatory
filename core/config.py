from pathlib import Path

from dotenv import load_dotenv
import os


BASE_DIR = Path(__file__).resolve().parent.parent


def load_settings() -> dict[str, str | bool | Path]:
    """Load simple local settings used across scripts and tests."""
    load_dotenv(BASE_DIR / ".env")

    database_path = Path(os.getenv("DATABASE_PATH", "data/observability.db"))
    if not database_path.is_absolute():
        database_path = BASE_DIR / database_path

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    placeholder_keys = {"ollama", "dummy", "test", "changeme"}

    return {
        "demo_mode": os.getenv("DEMO_MODE", "true").lower() == "true",
        "database_path": database_path,
        "openai_model": os.getenv("OBSERVATORY_OPENAI_MODEL", "gpt-4.1-mini"),
        "openai_base_url": os.getenv(
            "OBSERVATORY_OPENAI_BASE_URL", "https://api.openai.com/v1"
        ),
        "openai_api_key_available": bool(openai_api_key)
        and openai_api_key.casefold() not in placeholder_keys,
    }
