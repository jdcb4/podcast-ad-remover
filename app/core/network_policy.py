"""Which network targets a feed, media or artwork fetch may reach, and for whom."""
from contextlib import contextmanager
from contextvars import ContextVar

from app.core.config import settings

_public_only = ContextVar('public_network_only', default=False)


def private_targets_allowed() -> bool:
    return settings.ALLOW_PRIVATE_FEEDS and not _public_only.get()


@contextmanager
def user_network_scope(user_id: int | None):
    """Keep fetches made for a non-admin on the public internet.

    None means no account is involved: login is off, a legacy unlinked API token,
    or an unowned podcast. Those keep the ALLOW_PRIVATE_FEEDS policy. A deleted
    owner is treated like a non-admin, since nothing vouches for that source now.
    """
    restricted = False
    if user_id is not None:
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            row = conn.execute('SELECT is_admin FROM users WHERE id = ?', (user_id,)).fetchone()
        restricted = not (row and row['is_admin'])
    if not restricted:
        yield
        return
    token = _public_only.set(True)
    try:
        yield
    finally:
        _public_only.reset(token)
