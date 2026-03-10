"""Tests for retry mechanism with exponential backoff."""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alibi.utils.retry import with_retry


class TestRetryBasicFunctionality:
    """Tests for basic retry functionality."""

    def test_retry_succeeds_first_attempt(self) -> None:
        """Test that function succeeds without retry when no exceptions occur."""
        call_count = 0

        @with_retry(max_attempts=3)
        def successful_function() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()

        assert result == "success"
        assert call_count == 1

    def test_retry_succeeds_after_failures(self) -> None:
        """Test that function succeeds after failing twice then succeeding."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        def eventually_successful() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        result = eventually_successful()

        assert result == "success"
        assert call_count == 3

    def test_retry_exhausts_attempts(self) -> None:
        """Test that function raises exception after all retry attempts fail."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Attempt {call_count} failed")

        with pytest.raises(ValueError, match="Attempt 3 failed"):
            always_fails()

        assert call_count == 3


class TestRetryBackoff:
    """Tests for retry backoff behavior."""

    def test_retry_exponential_backoff(self) -> None:
        """Test that delays increase exponentially between retry attempts."""
        call_count = 0
        sleep_durations: list[float] = []

        @with_retry(max_attempts=4, base_delay=1.0, exponential=True)
        def failing_function() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                failing_function()

            # Extract sleep durations from mock calls
            sleep_durations = [call[0][0] for call in mock_sleep.call_args_list]

        # Should have 3 sleep calls for 4 attempts (no sleep after last attempt)
        assert len(sleep_durations) == 3
        # Exponential backoff: 1.0, 2.0, 4.0
        assert sleep_durations[0] == 1.0
        assert sleep_durations[1] == 2.0
        assert sleep_durations[2] == 4.0
        assert call_count == 4

    def test_retry_linear_backoff(self) -> None:
        """Test that delays remain constant when exponential=False."""
        call_count = 0
        sleep_durations: list[float] = []

        @with_retry(max_attempts=4, base_delay=1.5, exponential=False)
        def failing_function() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                failing_function()

            sleep_durations = [call[0][0] for call in mock_sleep.call_args_list]

        # Should have 3 sleep calls for 4 attempts
        assert len(sleep_durations) == 3
        # Linear backoff: all delays should be base_delay
        assert all(duration == 1.5 for duration in sleep_durations)
        assert call_count == 4

    def test_retry_custom_base_delay(self) -> None:
        """Test that custom base_delay is used correctly."""

        @with_retry(max_attempts=2, base_delay=0.5)
        def failing_function() -> None:
            raise ValueError("Test error")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                failing_function()

            # Should sleep once with base_delay
            mock_sleep.assert_called_once_with(0.5)


class TestRetryExceptionHandling:
    """Tests for exception handling in retry logic."""

    def test_retry_only_catches_specified_exceptions(self) -> None:
        """Test that only specified exceptions are caught and retried."""
        call_count = 0

        @with_retry(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        def raises_different_exception() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("This should be retried")
            raise TypeError("This should propagate immediately")

        with pytest.raises(TypeError, match="This should propagate immediately"):
            raises_different_exception()

        # Should have tried twice: once with ValueError, once with TypeError
        assert call_count == 2

    def test_retry_multiple_exception_types(self) -> None:
        """Test that multiple exception types can be caught."""
        call_count = 0

        @with_retry(
            max_attempts=4,
            exceptions=(ValueError, TypeError, KeyError),
            base_delay=0.01,
        )
        def raises_various_exceptions() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First error")
            if call_count == 2:
                raise TypeError("Second error")
            if call_count == 3:
                raise KeyError("Third error")
            return "success"

        result = raises_various_exceptions()

        assert result == "success"
        assert call_count == 4

    def test_retry_unspecified_exception_propagates_immediately(self) -> None:
        """Test that exceptions not in the exceptions tuple propagate immediately."""
        call_count = 0

        @with_retry(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        def raises_runtime_error() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("This should not be retried")

        with pytest.raises(RuntimeError, match="This should not be retried"):
            raises_runtime_error()

        # Should only try once since RuntimeError is not in exceptions tuple
        assert call_count == 1


class TestRetryLogging:
    """Tests for retry logging behavior."""

    def test_retry_logs_warning_on_retry(self) -> None:
        """Test that warnings are logged on each retry attempt."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        def failing_function() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        with patch("alibi.utils.retry.logger") as mock_logger:
            with pytest.raises(ValueError):
                failing_function()

            # Should log warning twice (for first 2 failures) and error once (final)
            assert mock_logger.warning.call_count == 2
            assert mock_logger.error.call_count == 1

    def test_retry_logs_error_on_final_failure(self) -> None:
        """Test that error is logged when all attempts are exhausted."""

        @with_retry(max_attempts=2, base_delay=0.01)
        def failing_function() -> None:
            raise ValueError("Test error")

        with patch("alibi.utils.retry.logger") as mock_logger:
            with pytest.raises(ValueError):
                failing_function()

            # Verify error log contains appropriate message
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "failing_function" in error_call
            assert "failed after 2 attempts" in error_call
            assert "ValueError" in error_call

    def test_retry_warning_includes_delay_info(self) -> None:
        """Test that warning log includes delay information."""

        @with_retry(max_attempts=2, base_delay=1.0)
        def failing_function() -> None:
            raise ValueError("Test error")

        with patch("alibi.utils.retry.logger") as mock_logger:
            with patch("time.sleep"):
                with pytest.raises(ValueError):
                    failing_function()

                # Check warning message format
                warning_call = mock_logger.warning.call_args[0][0]
                assert "failing_function" in warning_call
                assert "attempt 1/2" in warning_call
                assert "ValueError" in warning_call
                assert "Retrying in 1.0s" in warning_call


class TestRetryAsyncFunctions:
    """Tests for retry with async functions."""

    @pytest.mark.asyncio
    async def test_retry_async_function(self) -> None:
        """Test that async functions work correctly with retry decorator."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        async def async_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "async success"

        result = await async_function()

        assert result == "async success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_exhausts_attempts(self) -> None:
        """Test that async function raises exception after all attempts fail."""
        call_count = 0

        @with_retry(max_attempts=2, base_delay=0.01)
        async def always_fails_async() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Async attempt {call_count} failed")

        with pytest.raises(ValueError, match="Async attempt 2 failed"):
            await always_fails_async()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_async_exponential_backoff(self) -> None:
        """Test that async functions use exponential backoff correctly."""
        call_count = 0

        @with_retry(max_attempts=4, base_delay=1.0, exponential=True)
        async def failing_async() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        async def mock_async_sleep(delay: float) -> None:
            pass

        with patch(
            "alibi.utils.retry.asyncio.sleep", side_effect=mock_async_sleep
        ) as mock_sleep:
            with pytest.raises(ValueError):
                await failing_async()

            # Extract sleep durations
            sleep_durations = [call[0][0] for call in mock_sleep.call_args_list]

        # Should have 3 sleep calls for 4 attempts
        assert len(sleep_durations) == 3
        # Exponential backoff: 1.0, 2.0, 4.0
        assert sleep_durations[0] == 1.0
        assert sleep_durations[1] == 2.0
        assert sleep_durations[2] == 4.0
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_retry_async_succeeds_first_attempt(self) -> None:
        """Test that async function succeeds without retry on first attempt."""
        call_count = 0

        @with_retry(max_attempts=3)
        async def successful_async() -> str:
            nonlocal call_count
            call_count += 1
            return "async success"

        result = await successful_async()

        assert result == "async success"
        assert call_count == 1


class TestRetryEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_retry_with_return_value(self) -> None:
        """Test that return values are properly passed through."""

        @with_retry(max_attempts=2)
        def returns_complex_value() -> dict[str, Any]:
            return {"status": "success", "data": [1, 2, 3], "count": 42}

        result = returns_complex_value()

        assert result == {"status": "success", "data": [1, 2, 3], "count": 42}

    def test_retry_with_arguments(self) -> None:
        """Test that function arguments are properly passed through."""
        call_count = 0

        @with_retry(max_attempts=2, base_delay=0.01)
        def function_with_args(a: int, b: str, c: bool = False) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return f"{a}-{b}-{c}"

        result = function_with_args(42, "test", c=True)

        assert result == "42-test-True"
        assert call_count == 2

    def test_retry_preserves_function_metadata(self) -> None:
        """Test that decorator preserves original function metadata."""

        @with_retry(max_attempts=3)
        def documented_function() -> str:
            """This is a documented function."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."

    def test_retry_with_max_attempts_one(self) -> None:
        """Test that max_attempts=1 means no retries."""
        call_count = 0

        @with_retry(max_attempts=1)
        def fails_immediately() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Immediate failure")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                fails_immediately()

            # Should not sleep at all with only 1 attempt
            mock_sleep.assert_not_called()

        assert call_count == 1
