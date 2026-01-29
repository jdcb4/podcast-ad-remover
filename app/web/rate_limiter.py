"""
Rate limiting middleware to protect against brute force attacks.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, Tuple
from datetime import datetime, timedelta
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    In-memory rate limiter for tracking login attempts by IP address.
    
    Configuration:
    - Max 5 failed attempts per IP within 15 minutes
    - 15 minute lockout after exceeding attempts
    - Automatic cleanup of old entries every 5 minutes
    """
    
    def __init__(
        self,
        max_attempts: int = 5,
        window_minutes: int = 15,
        lockout_minutes: int = 15
    ):
        self.max_attempts = max_attempts
        self.window_seconds = window_minutes * 60
        self.lockout_seconds = lockout_minutes * 60
        
        # Track failed attempts: {ip: [(timestamp, attempt_count), ...]}
        self.failed_attempts: Dict[str, list] = defaultdict(list)
        
        # Track locked IPs: {ip: lockout_expiry_timestamp}
        self.locked_ips: Dict[str, float] = {}
        
        # Last cleanup time
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def _cleanup_old_entries(self):
        """Remove expired entries to prevent memory bloat."""
        current_time = time.time()
        
        # Only cleanup every 5 minutes
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        # Clean up old failed attempts
        cutoff_time = current_time - self.window_seconds
        for ip in list(self.failed_attempts.keys()):
            self.failed_attempts[ip] = [
                (ts, count) for ts, count in self.failed_attempts[ip]
                if ts > cutoff_time
            ]
            # Remove IP if no recent attempts
            if not self.failed_attempts[ip]:
                del self.failed_attempts[ip]
        
        # Clean up expired lockouts
        for ip in list(self.locked_ips.keys()):
            if current_time > self.locked_ips[ip]:
                del self.locked_ips[ip]
                logger.info(f"Rate limit lockout expired for IP: {ip}")
        
        self.last_cleanup = current_time
    
    def is_locked(self, ip: str) -> Tuple[bool, int]:
        """
        Check if an IP is currently locked out.
        
        Returns:
            Tuple of (is_locked, seconds_remaining)
        """
        self._cleanup_old_entries()
        
        if ip in self.locked_ips:
            remaining = int(self.locked_ips[ip] - time.time())
            if remaining > 0:
                return True, remaining
            else:
                # Lockout expired
                del self.locked_ips[ip]
                return False, 0
        
        return False, 0
    
    def record_failed_attempt(self, ip: str):
        """Record a failed login attempt and lock if threshold exceeded."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Get recent attempts within the time window
        recent_attempts = [
            (ts, count) for ts, count in self.failed_attempts[ip]
            if ts > cutoff_time
        ]
        
        # Count total attempts
        total_attempts = sum(count for _, count in recent_attempts) + 1
        
        # Add current attempt
        recent_attempts.append((current_time, 1))
        self.failed_attempts[ip] = recent_attempts
        
        # Check if threshold exceeded
        if total_attempts >= self.max_attempts:
            lockout_until = current_time + self.lockout_seconds
            self.locked_ips[ip] = lockout_until
            logger.warning(
                f"IP {ip} locked out for {self.lockout_minutes} minutes "
                f"after {total_attempts} failed login attempts"
            )
            return True  # Locked
        else:
            remaining = self.max_attempts - total_attempts
            logger.info(
                f"Failed login attempt from {ip}. "
                f"{remaining} attempts remaining before lockout."
            )
            return False  # Not locked yet
    
    def record_successful_login(self, ip: str):
        """Clear failed attempts after successful login."""
        if ip in self.failed_attempts:
            del self.failed_attempts[ip]
        if ip in self.locked_ips:
            del self.locked_ips[ip]
        logger.info(f"Successful login from {ip}, rate limit cleared")
    
    def get_stats_for_ip(self, ip: str) -> dict:
        """Get current rate limiting stats for an IP."""
        locked, remaining_seconds = self.is_locked(ip)
        
        if locked:
            return {
                "locked": True,
                "remaining_seconds": remaining_seconds,
                "attempts_remaining": 0
            }
        
        # Count recent attempts
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        recent_attempts = sum(
            count for ts, count in self.failed_attempts.get(ip, [])
            if ts > cutoff_time
        )
        
        return {
            "locked": False,
            "remaining_seconds": 0,
            "attempts_remaining": max(0, self.max_attempts - recent_attempts),
            "recent_attempts": recent_attempts
        }


# Global rate limiter instance
login_rate_limiter = RateLimiter(
    max_attempts=5,
    window_minutes=15,
    lockout_minutes=15
)


def check_rate_limit(ip: str) -> None:
    """
    Check if an IP is rate limited and raise HTTPException if locked.
    
    Args:
        ip: Client IP address
        
    Raises:
        HTTPException: 429 Too Many Requests if IP is locked
    """
    locked, remaining_seconds = login_rate_limiter.is_locked(ip)
    
    if locked:
        remaining_minutes = (remaining_seconds + 59) // 60  # Round up
        logger.warning(f"Blocked login attempt from locked IP: {ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Please try again in {remaining_minutes} minute(s)."
        )
