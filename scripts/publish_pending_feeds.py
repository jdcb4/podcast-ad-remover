#!/usr/bin/env python3
"""Retry pending RSS publication without reprocessing or making LLM requests."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from app.core.config import settings
    if not Path(settings.DB_PATH).is_file():
        raise SystemExit('Database not found; set DATA_DIR to the existing application data directory.')
    from app.infra.database import init_db, get_db_connection
    from app.core.processor import Processor
    init_db()
    asyncio.run(Processor().publish_pending_feeds())
    with get_db_connection() as conn:
        pending = conn.execute('SELECT COUNT(*) FROM episodes WHERE publication_pending=1').fetchone()[0]
    print(f'Pending feed publications: {pending}')
    return 1 if pending else 0


if __name__ == '__main__':
    raise SystemExit(main())
