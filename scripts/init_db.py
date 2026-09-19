import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_settings
from core.database import initialize_database


def main() -> None:
    settings = load_settings()
    database_path = settings["database_path"]
    initialize_database(database_path)
    print(f"SQLite database initialized at {database_path}")


if __name__ == "__main__":
    main()
