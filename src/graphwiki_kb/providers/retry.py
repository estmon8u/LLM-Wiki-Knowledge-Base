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
    retry_if_exception,
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
    )
    from openai import (
        APITimeoutError as OAITimeoutError,
    )
    from openai import (
        InternalServerError as OAIInternalServerError,
    )
    from openai import (
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
    )
    from anthropic import (
        APITimeoutError as AntTimeoutError,
    )
    from anthropic import (
        InternalServerError as AntInternalServerError,
    )
    from anthropic import (
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
    )
    from google.api_core.exceptions import (
        ServiceUnavailable as GServiceUnavailable,
    )

    _TRANSIENT += [GServiceUnavailable, GDeadlineExceeded]  # pragma: no cover
except ImportError:  # pragma: no cover
    pass

TRANSIENT_EXCEPTIONS: tuple[type, ...] = tuple(dict.fromkeys(_TRANSIENT))
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def provider_retry():
    """Return a tenacity retry decorator for provider generate() calls."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=20),
        retry=retry_if_exception(_is_retryable_provider_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def _is_retryable_provider_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in RETRYABLE_STATUS_CODES
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in RETRYABLE_STATUS_CODES
    return isinstance(exc, TRANSIENT_EXCEPTIONS)
