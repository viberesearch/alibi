"""Retry mechanism with exponential backoff."""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    exponential: bool = True,
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Base delay in seconds between attempts (default: 1.0)
        exceptions: Tuple of exception types to catch and retry (default: all)
        exponential: Whether to use exponential backoff (default: True)

    Returns:
        Decorated function that retries on specified exceptions

    Example:
        @with_retry(max_attempts=3, base_delay=2.0, exceptions=(TimeoutError,))
        def fetch_data():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = (
                            base_delay * (2 ** (attempt - 1))
                            if exponential
                            else base_delay
                        )
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): "
                            f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: "
                            f"{type(e).__name__}: {e}"
                        )

            if last_exception is not None:
                raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = (
                            base_delay * (2 ** (attempt - 1))
                            if exponential
                            else base_delay
                        )
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): "
                            f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: "
                            f"{type(e).__name__}: {e}"
                        )

            if last_exception is not None:
                raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
