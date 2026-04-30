from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sql_generator.database import DB_PATH, initialize_database  # noqa: E402


if __name__ == "__main__":
    path = initialize_database()
    print(f"Created sample database at {path.relative_to(ROOT)}")
