"""Shared retry policy for provider generate() calls.

Retries transient errors (timeouts, connection failures, rate limits,
server errors) with exponential backoff and jitter.  Non-retriable
errors (auth, bad request, not found) propagate immediately.
"""

from __future__ import annotations

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transient exception types per provider SDK
# ---------------------------------------------------------------------------
_TRANSIENT: list[type] = [ConnectionError, TimeoutError]

try:
    from openai import (
        APIConnectionError as OAIConnectionError,
        APITimeoutError as OAITimeoutError,
        InternalServerError as OAIInternalServerError,
        RateLimitError as OAIRateLimitError,
    )

    _TRANSIENT += [
        OAITimeoutError,
        OAIConnectionError,
        OAIRateLimitError,
        OAIInternalServerError,
    ]
except ImportError:  # pragma: no cover
    pass

try:
    from anthropic import (
        APIConnectionError as AntConnectionError,
        APITimeoutError as AntTimeoutError,
        InternalServerError as AntInternalServerError,
        RateLimitError as AntRateLimitError,
    )

    _TRANSIENT += [
        AntTimeoutError,
        AntConnectionError,
        AntRateLimitError,
        AntInternalServerError,
    ]
except ImportError:  # pragma: no cover
    pass

try:
    from google.api_core.exceptions import (
        DeadlineExceeded as GDeadlineExceeded,
        ServiceUnavailable as GServiceUnavailable,
    )

    _TRANSIENT += [GServiceUnavailable, GDeadlineExceeded]  # pragma: no cover
except ImportError:  # pragma: no cover
    pass

TRANSIENT_EXCEPTIONS: tuple[type, ...] = tuple(dict.fromkeys(_TRANSIENT))


def provider_retry():
    """Return a tenacity retry decorator for provider generate() calls."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=20),
        retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
