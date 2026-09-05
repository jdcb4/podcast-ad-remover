from fastapi import HTTPException

from app.core.permissions import can_manage_subscription
from app.infra.repository import EpisodeRepository, SubscriptionRepository


def require_episode_management(user, episode_id: int):
    episode = EpisodeRepository().get_by_id(episode_id)
    if episode is None:
        raise HTTPException(404, 'Episode not found')
    sub = SubscriptionRepository().get_by_id(episode.subscription_id)
    if not can_manage_subscription(user, sub):
        raise HTTPException(403, 'Only admins and the podcast owner can manage this episode')
    return episode
