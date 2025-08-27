import pytest
import logging
import io
from unittest.mock import Mock
from typing import List

from app.utils.exception_logging import (
    log_exception_with_details,
    format_exception_message,
)


class BrokenStrException(Exception):
    """An exception that breaks when __str__ is called."""

    def __str__(self):
        raise RuntimeError("Cannot convert to string!")

    def __repr__(self):
        return "BrokenStrException(cannot convert to string)"


class BrokenReprException(Exception):
    """An exception that breaks when both __str__ and __repr__ are called."""

    def __str__(self):
        raise RuntimeError("Cannot convert to string!")

    def __repr__(self):
        raise RuntimeError("Cannot convert to repr!")


class CircularRefException(Exception):
    """An exception with circular references."""

    def __init__(self, message="Circular reference exception"):
        super().__init__(message)
        self.self_ref = self
        self.parent = None
        self.children = [self]


class MockExceptionGroup:
    """Mock implementation of an exception group for testing."""

    def __init__(self, message: str, exceptions: List[Exception]):
        self.message = message
        self.exceptions = exceptions

    def __str__(self):
        return f"{self.message} ({len(self.exceptions)} sub-exceptions)"


class MockBrokenExceptionGroup:
    """Mock exception group that breaks when accessing exceptions."""

    def __init__(self, message: str):
        self.message = message

    @property
    def exceptions(self):
        raise RuntimeError("Cannot access exceptions!")

    def __str__(self):
        return self.message


class UnicodeException(Exception):
    """Exception with unicode characters."""

    def __init__(self):
        super().__init__("Unicode: 🚫💥🔥 Error with émojis and áccénts")


class VeryLongException(Exception):
    """Exception with an extremely long message."""

    def __init__(self):
        # Create a very long message (100K characters)
        long_message = "x" * 100000
        super().__init__(f"Very long exception: {long_message}")


class TestLogExceptionWithDetails:
    """Test cases for log_exception_with_details function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.log_stream = io.StringIO()

    def test_normal_exception_logging(self):
        """Test logging a normal exception."""
        exception = ValueError("Normal test error")

        log_exception_with_details(self.logger, "[TEST]", exception)

        self.logger.log.assert_called_once_with(
            logging.ERROR,
            "[TEST] Exception: Normal test error",
            exc_info=exception,  # Updated: now passes the actual exception object
        )

    def test_exception_with_custom_level(self):
        """Test logging with custom logging level."""
        exception = ValueError("Warning level error")

        log_exception_with_details(self.logger, "[TEST]", exception, logging.WARNING)

        self.logger.log.assert_called_once_with(
            logging.WARNING,
            "[TEST] Exception: Warning level error",
            exc_info=exception,  # Updated: now passes the actual exception object
        )

    def test_exception_group_logging(self):
        """Test logging an exception group."""
        sub_exceptions = [
            ValueError("Sub error 1"),
            RuntimeError("Sub error 2"),
            TypeError("Sub error 3"),
        ]
        exception_group = MockExceptionGroup("Multiple errors occurred", sub_exceptions)

        log_exception_with_details(self.logger, "[TEST]", exception_group)

        # Should log the main exception plus each sub-exception
        expected_calls = [
            (
                (
                    logging.ERROR,
                    "[TEST] Exception with 3 sub-exceptions: Multiple errors occurred (3 sub-exceptions)",
                ),
                {},
            ),
            (
                (logging.ERROR, "[TEST] Sub-exception 1: ValueError: Sub error 1"),
                {"exc_info": sub_exceptions[0]},
            ),
            (
                (logging.ERROR, "[TEST] Sub-exception 2: RuntimeError: Sub error 2"),
                {"exc_info": sub_exceptions[1]},
            ),
            (
                (logging.ERROR, "[TEST] Sub-exception 3: TypeError: Sub error 3"),
                {"exc_info": sub_exceptions[2]},
            ),
        ]

        assert self.logger.log.call_count == 4
        for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
            actual_call = self.logger.log.call_args_list[i]
            assert actual_call.args == expected_args
            for key, value in expected_kwargs.items():
                assert actual_call.kwargs[key] == value

    def test_empty_exception_group(self):
        """Test logging an exception group with no sub-exceptions."""
        exception_group = MockExceptionGroup("Empty group", [])

        log_exception_with_details(self.logger, "[TEST]", exception_group)

        # Updated: empty exception groups are now treated as regular exceptions
        self.logger.log.assert_called_once_with(
            logging.ERROR,
            "[TEST] Exception: Empty group (0 sub-exceptions)",
            exc_info=exception_group,
        )

    def test_broken_str_exception(self):
        """Test logging an exception that breaks when converted to string."""
        exception = BrokenStrException()

        # Should not raise an exception even if str() fails
        log_exception_with_details(self.logger, "[TEST]", exception)

        # Should still be called once (the logging framework handles the broken __str__)
        assert self.logger.log.call_count == 1

    def test_broken_exception_group(self):
        """Test logging a broken exception group."""
        broken_group = MockBrokenExceptionGroup("Broken group")

        # Should not raise an exception
        log_exception_with_details(self.logger, "[TEST]", broken_group)

        # Should fall back to normal exception handling
        self.logger.log.assert_called_once_with(
            logging.ERROR,
            "[TEST] Exception: Broken group",
            exc_info=broken_group,  # Updated: now passes the actual exception object
        )

    def test_none_exception(self):
        """Test with None as exception."""
        # Should not break, even with None input
        try:
            log_exception_with_details(self.logger, "[TEST]", None)  # type: ignore
        except Exception as e:
            pytest.fail(f"Should not raise exception, but got: {e}")

    def test_non_exception_object(self):
        """Test with a non-exception object."""
        fake_exception = "Not an exception"

        try:
            log_exception_with_details(self.logger, "[TEST]", fake_exception)  # type: ignore
        except Exception as e:
            pytest.fail(f"Should not raise exception, but got: {e}")

    def test_circular_reference_exception(self):
        """Test exception with circular references."""
        exception = CircularRefException()

        # Should not cause infinite recursion or stack overflow
        log_exception_with_details(self.logger, "[TEST]", exception)

        assert self.logger.log.call_count == 1

    def test_unicode_exception(self):
        """Test exception with unicode characters."""
        exception = UnicodeException()

        log_exception_with_details(self.logger, "[TEST]", exception)

        # Should handle unicode correctly
        expected_message = (
            "[TEST] Exception: Unicode: 🚫💥🔥 Error with émojis and áccénts"
        )
        self.logger.log.assert_called_once_with(
            logging.ERROR,
            expected_message,
            exc_info=exception,  # Updated: now passes the actual exception object
        )

    def test_very_long_exception(self):
        """Test exception with very long message."""
        exception = VeryLongException()

        # Should not cause memory issues or timeouts
        log_exception_with_details(self.logger, "[TEST]", exception)

        assert self.logger.log.call_count == 1

    def test_logger_failure_resilience(self):
        """Test resilience when logger itself fails."""
        broken_logger = Mock(spec=logging.Logger)
        broken_logger.log.side_effect = RuntimeError("Logger is broken!")
        exception = ValueError("Test error")

        # Should not propagate logger exceptions
        try:
            log_exception_with_details(broken_logger, "[TEST]", exception)
        except RuntimeError:
            pytest.fail("Should not propagate logger exceptions")

    def test_exception_group_with_broken_sub_exceptions(self):
        """Test exception group containing broken sub-exceptions."""
        sub_exceptions = [
            BrokenStrException(),
            BrokenReprException(),
            ValueError("Normal exception"),
        ]
        exception_group = MockExceptionGroup("Mixed broken exceptions", sub_exceptions)

        log_exception_with_details(self.logger, "[TEST]", exception_group)

        # Should log all exceptions without failing
        assert self.logger.log.call_count == 4  # 1 main + 3 sub-exceptions

    def test_nested_exception_groups(self):
        """Test nested exception groups."""
        inner_exceptions = [ValueError("Inner 1"), RuntimeError("Inner 2")]
        inner_group = MockExceptionGroup("Inner group", inner_exceptions)

        outer_exceptions = [inner_group, TypeError("Outer exception")]
        outer_group = MockExceptionGroup("Outer group", outer_exceptions)

        log_exception_with_details(self.logger, "[TEST]", outer_group)

        # Should handle nested groups
        assert self.logger.log.call_count == 3  # 1 main + 2 sub-exceptions

    def test_empty_prefix(self):
        """Test with empty prefix."""
        exception = ValueError("Test error")

        log_exception_with_details(self.logger, "", exception)

        self.logger.log.assert_called_once_with(
            logging.ERROR,
            " Exception: Test error",  # Space before "Exception"
            exc_info=exception,  # Updated: now passes the actual exception object
        )

    def test_none_prefix(self):
        """Test with None prefix."""
        exception = ValueError("Test error")

        try:
            log_exception_with_details(self.logger, None, exception)  # type: ignore
        except Exception as e:
            pytest.fail(f"Should not raise exception, but got: {e}")


class TestFormatExceptionMessage:
    """Test cases for format_exception_message function."""

    def test_normal_exception_formatting(self):
        """Test formatting a normal exception."""
        exception = ValueError("Normal test error")

        result = format_exception_message(exception)

        assert result == "Normal test error"

    def test_exception_group_formatting(self):
        """Test formatting an exception group."""
        sub_exceptions = [
            ValueError("Sub error 1"),
            RuntimeError("Sub error 2"),
            TypeError("Sub error 3"),
        ]
        exception_group = MockExceptionGroup("Multiple errors occurred", sub_exceptions)

        result = format_exception_message(exception_group)

        expected = (
            "Multiple errors occurred (3 sub-exceptions) "
            "(Sub-exceptions: ValueError: Sub error 1; RuntimeError: Sub error 2; TypeError: Sub error 3)"
        )
        assert result == expected

    def test_empty_exception_group_formatting(self):
        """Test formatting an exception group with no sub-exceptions."""
        exception_group = MockExceptionGroup("Empty group", [])

        result = format_exception_message(exception_group)

        # Updated: empty exception groups now just return the main string without sub-exceptions text
        expected = "Empty group (0 sub-exceptions)"
        assert result == expected

    def test_broken_str_exception_formatting(self):
        """Test formatting an exception that breaks when converted to string."""
        exception = BrokenStrException()

        # Should not raise an exception
        try:
            result = format_exception_message(exception)
            # Result may vary depending on how Python handles the broken __str__
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Should not raise exception, but got: {e}")

    def test_broken_exception_group_formatting(self):
        """Test formatting a broken exception group."""
        broken_group = MockBrokenExceptionGroup("Broken group")

        # Should not raise an exception and fall back to normal formatting
        result = format_exception_message(broken_group)

        assert result == "Broken group"

    def test_none_exception_formatting(self):
        """Test formatting None as exception."""
        try:
            result = format_exception_message(None)  # type: ignore
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Should not raise exception, but got: {e}")

    def test_non_exception_object_formatting(self):
        """Test formatting a non-exception object."""
        fake_exception = "Not an exception"

        try:
            result = format_exception_message(fake_exception)  # type: ignore
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Should not raise exception, but got: {e}")

    def test_circular_reference_exception_formatting(self):
        """Test formatting exception with circular references."""
        exception = CircularRefException()

        # Should not cause infinite recursion
        result = format_exception_message(exception)

        assert isinstance(result, str)
        assert "Circular reference exception" in result

    def test_unicode_exception_formatting(self):
        """Test formatting exception with unicode characters."""
        exception = UnicodeException()

        result = format_exception_message(exception)

        expected = "Unicode: 🚫💥🔥 Error with émojis and áccénts"
        assert result == expected

    def test_very_long_exception_formatting(self):
        """Test formatting exception with very long message."""
        exception = VeryLongException()

        # Should not cause memory issues or timeouts
        result = format_exception_message(exception)

        assert isinstance(result, str)
        assert result.startswith("Very long exception: x")

    def test_exception_group_with_broken_sub_exceptions_formatting(self):
        """Test formatting exception group containing broken sub-exceptions."""
        sub_exceptions = [
            BrokenStrException(),
            BrokenReprException(),
            ValueError("Normal exception"),
        ]
        exception_group = MockExceptionGroup("Mixed broken exceptions", sub_exceptions)

        # Should not fail even with broken sub-exceptions
        result = format_exception_message(exception_group)

        assert isinstance(result, str)
        assert "Mixed broken exceptions" in result

    def test_nested_exception_groups_formatting(self):
        """Test formatting nested exception groups."""
        inner_exceptions = [ValueError("Inner 1"), RuntimeError("Inner 2")]
        inner_group = MockExceptionGroup("Inner group", inner_exceptions)

        outer_exceptions = [inner_group, TypeError("Outer exception")]
        outer_group = MockExceptionGroup("Outer group", outer_exceptions)

        result = format_exception_message(outer_group)

        assert isinstance(result, str)
        assert "Outer group" in result


class TestEdgeCasesAndStressTests:
    """Additional edge cases and stress tests."""

    def test_memory_pressure_simulation(self):
        """Test behavior under simulated memory pressure."""
        # Create many large exceptions
        exceptions = []
        for i in range(1000):
            try:
                # Create some nested structure to consume memory
                data = {"nested": {"data": "x" * 1000} for _ in range(10)}
                raise ValueError(f"Exception {i}: {data}")
            except ValueError as e:
                exceptions.append(e)

        # Test that we can still log exceptions
        logger = Mock(spec=logging.Logger)

        for exception in exceptions[:10]:  # Test first 10
            try:
                log_exception_with_details(logger, "[STRESS]", exception)
                format_exception_message(exception)
            except Exception as e:
                pytest.fail(f"Failed under memory pressure: {e}")

    def test_deeply_nested_exception_chain(self):
        """Test with deeply nested exception chains."""

        def create_nested_exception(depth):
            if depth == 0:
                return ValueError("Base exception")
            try:
                raise create_nested_exception(depth - 1)
            except Exception as e:
                raise RuntimeError(f"Level {depth}") from e

        try:
            raise create_nested_exception(100)
        except Exception as nested_exception:
            logger = Mock(spec=logging.Logger)

            # Should handle deep nesting without stack overflow
            log_exception_with_details(logger, "[NESTED]", nested_exception)
            result = format_exception_message(nested_exception)

            assert isinstance(result, str)

    def test_concurrent_logging_safety(self):
        """Test that logging is safe under concurrent access."""
        import threading
        import queue

        logger = Mock(spec=logging.Logger)
        exceptions_queue = queue.Queue()
        results = []

        def log_worker():
            while True:
                try:
                    exception = exceptions_queue.get(timeout=1)
                    if exception is None:
                        break
                    log_exception_with_details(logger, "[THREAD]", exception)
                    results.append("success")
                except queue.Empty:
                    break
                except Exception as e:
                    results.append(f"error: {e}")
                finally:
                    exceptions_queue.task_done()

        # Add test exceptions
        for i in range(50):
            exceptions_queue.put(ValueError(f"Concurrent exception {i}"))

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=log_worker)
            thread.start()
            threads.append(thread)

        # Wait for completion
        exceptions_queue.join()

        # Signal threads to stop
        for _ in threads:
            exceptions_queue.put(None)

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert all(result == "success" for result in results)
        assert len(results) == 50


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
