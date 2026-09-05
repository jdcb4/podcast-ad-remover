"""Shared ownership policy; transport-specific authentication stays at the edge."""


def can_manage_subscription(user, sub) -> bool:
    if not user or not sub:
        return False
    user_id = getattr(user, 'id', None)
    return bool(getattr(user, 'is_admin', False) or (
        user_id and user_id > 0 and getattr(sub, 'owner_user_id', None) == user_id
    ))
