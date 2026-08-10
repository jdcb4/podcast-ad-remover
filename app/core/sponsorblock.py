"""Read-only SponsorBlock segment lookup for YouTube episodes."""

from __future__ import annotations

import hashlib
import json
import logging

import httpx


logger = logging.getLogger(__name__)
SPONSORBLOCK_API_BASE = "https://sponsor.ajay.app"


def categories_for_subscription(subscription) -> list[str]:
    categories: list[str] = []
    if subscription.remove_ads:
        categories.append("sponsor")
    if subscription.remove_promos:
        categories.extend(["selfpromo", "interaction"])
    if subscription.remove_intros:
        categories.append("intro")
    if subscription.remove_outros:
        categories.append("outro")
    return categories


class SponsorBlockClient:
    def __init__(self, base_url: str = SPONSORBLOCK_API_BASE, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_segments(self, video_id: str, categories: list[str]) -> list[dict]:
        """Fetch accepted skip segments through SponsorBlock's hash-prefix endpoint."""
        if not categories:
            return []
        prefix = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:4]
        params = {
            "service": "YouTube",
            "categories": json.dumps(categories, separators=(",", ":")),
            "actionTypes": json.dumps(["skip"], separators=(",", ":")),
        }
        try:
            response = httpx.get(
                f"{self.base_url}/api/skipSegments/{prefix}",
                params=params,
                timeout=self.timeout,
                follow_redirects=False,
            )
            if response.status_code == 404:
                return []
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("SponsorBlock response was not a list")
        except Exception as exc:
            logger.warning("SponsorBlock lookup failed open for %s: %s", video_id, exc)
            return []

        allowed = set(categories)
        normalized: list[dict] = []
        for result in payload:
            if not isinstance(result, dict) or result.get("videoID") != video_id:
                continue
            for raw in result.get("segments") or []:
                if not isinstance(raw, dict):
                    continue
                category = str(raw.get("category") or "")
                interval = raw.get("segment")
                if category not in allowed or raw.get("actionType", "skip") != "skip":
                    continue
                if not isinstance(interval, list) or len(interval) != 2:
                    continue
                try:
                    start, end = float(interval[0]), float(interval[1])
                except (TypeError, ValueError):
                    continue
                if end <= start:
                    continue
                evidence = {
                    "source": "sponsorblock",
                    "category": category,
                    "uuid": raw.get("UUID"),
                    "votes": raw.get("votes"),
                    "locked": raw.get("locked"),
                    "action_type": raw.get("actionType", "skip"),
                }
                normalized.append({
                    "start": start,
                    "end": end,
                    "label": f"SponsorBlock: {category}",
                    "reason": f"Crowdsourced SponsorBlock {category} segment",
                    "source": "sponsorblock",
                    "category": category,
                    "evidence": [evidence],
                })
        return normalized
