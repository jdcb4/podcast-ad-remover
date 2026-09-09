"""The queue payload shared by the web and v1 queue endpoints.

app/web/router.py's /api/queue/status and app/api/v1/router.py's
/api/v1/queue return the same three-key body. It is built here, once, so
the two cannot drift apart on timestamp normalization the way they
already have twice.
"""
from app.core.system_status import get_operation_status
from app.core.time_utils import QUEUE_ROW_TIMESTAMP_KEYS, with_utc_timestamps
from app.infra.repository import EpisodeRepository

_ep_repo = EpisodeRepository()


def get_queue_payload() -> dict:
    """Queue, recent history and operation status, timestamps Z-suffixed."""
    return {
        "queue": [
            with_utc_timestamps(row, QUEUE_ROW_TIMESTAMP_KEYS)
            for row in _ep_repo.get_queue()
        ],
        "recently_processed": [
            with_utc_timestamps(row)
            for row in _ep_repo.get_recently_processed()
        ],
        "operation_status": get_operation_status(),
    }
