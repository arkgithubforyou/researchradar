"""Simple in-memory rate limiter for protecting LLM endpoints.

Uses a per-IP sliding window approach. No external dependencies.
"""

import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request


class RateLimiter:
    """Token-bucket-style rate limiter keyed by client IP."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For behind ALB."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> None:
        """Raise 429 if the client has exceeded the rate limit."""
        ip = self._client_ip(request)
        now = time.monotonic()

        with self._lock:
            # Prune old hits outside the window
            hits = self._hits[ip]
            cutoff = now - self.window
            self._hits[ip] = [t for t in hits if t > cutoff]
            hits = self._hits[ip]

            if len(hits) >= self.max_requests:
                retry_after = int(self.window - (now - hits[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit exceeded. Max {self.max_requests} searches "
                        f"per {self.window}s. Try again in {retry_after}s."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )

            hits.append(now)


# Shared instance — 10 search requests per minute per IP
search_limiter = RateLimiter(max_requests=10, window_seconds=60)
