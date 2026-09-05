"""Create a verified online SQLite backup without stopping the application."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings
from app.infra.backup import backup_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db-path', type=Path, default=Path(settings.DB_PATH))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(backup_database(args.db_path, args.output))


if __name__ == '__main__':
    main()
