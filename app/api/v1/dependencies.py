import logging
from collections.abc import Iterable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.v1.schemas import ApiPrincipal
from app.infra.database import get_db_connection
from app.infra.repository import ApiRateLimitRepository, ApiTokenRepository
from app.web.auth_utils import get_client_ip

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)
token_repo = ApiTokenRepository()
rate_repo = ApiRateLimitRepository()


def get_api_settings() -> dict:
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT ai_api_enabled,
                   ai_api_default_requests_per_minute,
                   ai_api_default_requests_per_day,
                   ai_api_unauth_requests_per_minute
            FROM app_settings
            WHERE id = 1
            """
        ).fetchone()
        return dict(row) if row else {}


def ensure_ai_api_enabled() -> dict:
    settings = get_api_settings()
    if not settings.get("ai_api_enabled"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API not enabled")
    return settings


def _rate_limit(bucket_key: str, limit: int, window_seconds: int, window_name: str) -> None:
    allowed, retry_after = rate_repo.check_and_increment(bucket_key, limit, window_seconds, window_name)
    if allowed:
        return

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded",
        headers={"Retry-After": str(max(1, retry_after))},
    )


def check_unauthenticated_rate_limit(request: Request, settings: dict) -> None:
    client_ip = get_client_ip(request)
    limit = int(settings.get("ai_api_unauth_requests_per_minute") or 10)
    _rate_limit(f"ip:{client_ip}", limit, 60, "minute")


def get_api_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> ApiPrincipal:
    settings = ensure_ai_api_enabled()

    if not credentials or credentials.scheme.lower() != "bearer":
        check_unauthenticated_rate_limit(request, settings)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = token_repo.validate(credentials.credentials)
    if not token:
        check_unauthenticated_rate_limit(request, settings)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")

    principal = ApiPrincipal(
        token_id=token["id"],
        token_prefix=token["token_prefix"],
        name=token["name"],
        scopes={scope.strip() for scope in token["scopes"].split(",") if scope.strip()},
        user_id=token["user_id"],
        username=token["username"],
        is_admin=bool(token["is_admin"]),
        requests_per_minute=token["requests_per_minute"],
        requests_per_day=token["requests_per_day"],
    )

    minute_limit = int(principal.requests_per_minute or settings.get("ai_api_default_requests_per_minute") or 60)
    day_limit = int(principal.requests_per_day or settings.get("ai_api_default_requests_per_day") or 1000)
    _rate_limit(f"token:{principal.token_id}", minute_limit, 60, "minute")
    _rate_limit(f"token:{principal.token_id}", day_limit, 24 * 60 * 60, "day")
    return principal


def require_scopes(required_scopes: Iterable[str]):
    required = set(required_scopes)

    def dependency(principal: ApiPrincipal = Depends(get_api_principal)) -> ApiPrincipal:
        if not required.issubset(principal.scopes):
            logger.warning(
                "AI API scope denied: token=%s required=%s actual=%s",
                principal.token_prefix,
                sorted(required),
                sorted(principal.scopes),
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient API token scope")
        return principal

    dependency.required_scopes = sorted(required)
    return dependency
