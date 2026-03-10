"""Tests for alibi/errors.py module."""

import pytest
from unittest.mock import Mock, call

from alibi.errors import (
    ActionableError,
    OLLAMA_CONNECTION_ERROR,
    OLLAMA_MODEL_NOT_FOUND,
    VAULT_NOT_FOUND,
    DATABASE_NOT_FOUND,
    UNSUPPORTED_FILE_TYPE,
    NO_INBOX_CONFIGURED,
    IMPORT_FAILED,
    format_error,
)


class TestActionableErrorCreation:
    """Tests for creating ActionableError instances."""

    def test_create_with_message_only(self):
        """Test creating error with just a message."""
        error = ActionableError(message="Something went wrong")
        assert error.message == "Something went wrong"
        assert error.suggestions == []
        assert error.error_code is None

    def test_create_with_suggestions(self):
        """Test creating error with suggestions."""
        suggestions = ["Try this", "Or try that"]
        error = ActionableError(message="Error occurred", suggestions=suggestions)
        assert error.message == "Error occurred"
        assert error.suggestions == suggestions
        assert error.error_code is None

    def test_create_with_error_code(self):
        """Test creating error with error code."""
        error = ActionableError(message="Error occurred", error_code="E999")
        assert error.message == "Error occurred"
        assert error.suggestions == []
        assert error.error_code == "E999"

    def test_create_with_all_fields(self):
        """Test creating error with all fields."""
        error = ActionableError(
            message="Critical error",
            suggestions=["Do this", "Do that"],
            error_code="E001",
        )
        assert error.message == "Critical error"
        assert error.suggestions == ["Do this", "Do that"]
        assert error.error_code == "E001"


class TestActionableErrorDisplay:
    """Tests for ActionableError.display() method."""

    def test_display_message_only(self):
        """Test display with just a message."""
        console = Mock()
        error = ActionableError(message="Test error")

        error.display(console)

        console.print.assert_called_once()
        args = console.print.call_args[0]
        assert len(args) == 1
        # Verify Panel is created with border_style
        panel = args[0]
        assert panel.border_style == "red"

    def test_display_with_suggestions(self):
        """Test display with suggestions."""
        console = Mock()
        error = ActionableError(
            message="Test error",
            suggestions=["Suggestion 1", "Suggestion 2", "Suggestion 3"],
        )

        error.display(console)

        console.print.assert_called_once()
        args = console.print.call_args[0]
        panel = args[0]
        # Panel should contain the rendered text
        assert panel.border_style == "red"

    def test_display_with_empty_suggestions_list(self):
        """Test display with empty suggestions list."""
        console = Mock()
        error = ActionableError(message="Test error", suggestions=[])

        error.display(console)

        console.print.assert_called_once()

    def test_display_console_called_once(self):
        """Test that console.print is called exactly once."""
        console = Mock()
        error = ActionableError(message="Error", suggestions=["Fix 1", "Fix 2"])

        error.display(console)

        assert console.print.call_count == 1


class TestFormatErrorSubstitution:
    """Tests for format_error() function."""

    def test_format_message_substitution(self):
        """Test format_error replaces placeholders in message."""
        error = ActionableError(
            message="Error with {var1} and {var2}",
            suggestions=["No placeholders here"],
            error_code="E001",
        )

        formatted = format_error(error, var1="value1", var2="value2")

        assert formatted.message == "Error with value1 and value2"
        assert formatted.suggestions == ["No placeholders here"]
        assert formatted.error_code == "E001"

    def test_format_suggestions_substitution(self):
        """Test format_error replaces placeholders in suggestions."""
        error = ActionableError(
            message="Error",
            suggestions=["Try {action1}", "Or try {action2}", "Check {location}"],
            error_code="E002",
        )

        formatted = format_error(
            error, action1="restarting", action2="updating", location="/path/to/file"
        )

        assert formatted.suggestions == [
            "Try restarting",
            "Or try updating",
            "Check /path/to/file",
        ]

    def test_format_both_message_and_suggestions(self):
        """Test format_error replaces placeholders in both message and suggestions."""
        error = ActionableError(
            message="Cannot find {item}",
            suggestions=["Check if {item} exists", "Create {item} with: {command}"],
            error_code="E003",
        )

        formatted = format_error(error, item="config.yaml", command="touch config.yaml")

        assert formatted.message == "Cannot find config.yaml"
        assert formatted.suggestions == [
            "Check if config.yaml exists",
            "Create config.yaml with: touch config.yaml",
        ]

    def test_format_with_no_placeholders(self):
        """Test format_error with no placeholders to replace."""
        error = ActionableError(
            message="Static message",
            suggestions=["Static suggestion"],
            error_code="E004",
        )

        formatted = format_error(error)

        assert formatted.message == "Static message"
        assert formatted.suggestions == ["Static suggestion"]
        assert formatted.error_code == "E004"

    def test_format_preserves_original_error(self):
        """Test that format_error doesn't modify original error."""
        original = ActionableError(
            message="Error with {placeholder}",
            suggestions=["Fix {placeholder}"],
            error_code="E005",
        )

        formatted = format_error(original, placeholder="value")

        # Original should be unchanged
        assert original.message == "Error with {placeholder}"
        assert original.suggestions == ["Fix {placeholder}"]
        # Formatted should have substitutions
        assert formatted.message == "Error with value"
        assert formatted.suggestions == ["Fix value"]


class TestPredefinedErrorsHaveSuggestions:
    """Tests verifying all predefined errors have suggestions."""

    def test_ollama_connection_error_has_suggestions(self):
        """Test OLLAMA_CONNECTION_ERROR has suggestions."""
        assert len(OLLAMA_CONNECTION_ERROR.suggestions) > 0
        assert all(isinstance(s, str) for s in OLLAMA_CONNECTION_ERROR.suggestions)

    def test_ollama_model_not_found_has_suggestions(self):
        """Test OLLAMA_MODEL_NOT_FOUND has suggestions."""
        assert len(OLLAMA_MODEL_NOT_FOUND.suggestions) > 0
        assert all(isinstance(s, str) for s in OLLAMA_MODEL_NOT_FOUND.suggestions)

    def test_vault_not_found_has_suggestions(self):
        """Test VAULT_NOT_FOUND has suggestions."""
        assert len(VAULT_NOT_FOUND.suggestions) > 0
        assert all(isinstance(s, str) for s in VAULT_NOT_FOUND.suggestions)

    def test_database_not_found_has_suggestions(self):
        """Test DATABASE_NOT_FOUND has suggestions."""
        assert len(DATABASE_NOT_FOUND.suggestions) > 0
        assert all(isinstance(s, str) for s in DATABASE_NOT_FOUND.suggestions)

    def test_unsupported_file_type_has_suggestions(self):
        """Test UNSUPPORTED_FILE_TYPE has suggestions."""
        assert len(UNSUPPORTED_FILE_TYPE.suggestions) > 0
        assert all(isinstance(s, str) for s in UNSUPPORTED_FILE_TYPE.suggestions)

    def test_no_inbox_configured_has_suggestions(self):
        """Test NO_INBOX_CONFIGURED has suggestions."""
        assert len(NO_INBOX_CONFIGURED.suggestions) > 0
        assert all(isinstance(s, str) for s in NO_INBOX_CONFIGURED.suggestions)

    def test_import_failed_has_suggestions(self):
        """Test IMPORT_FAILED has suggestions."""
        assert len(IMPORT_FAILED.suggestions) > 0
        assert all(isinstance(s, str) for s in IMPORT_FAILED.suggestions)

    def test_all_predefined_errors_have_multiple_suggestions(self):
        """Test that all predefined errors have at least 2 suggestions."""
        errors = [
            OLLAMA_CONNECTION_ERROR,
            OLLAMA_MODEL_NOT_FOUND,
            VAULT_NOT_FOUND,
            DATABASE_NOT_FOUND,
            UNSUPPORTED_FILE_TYPE,
            NO_INBOX_CONFIGURED,
            IMPORT_FAILED,
        ]
        for error in errors:
            assert (
                len(error.suggestions) >= 2
            ), f"Error '{error.message}' should have at least 2 suggestions"


class TestPredefinedErrorsHaveErrorCodes:
    """Tests verifying all predefined errors have error codes."""

    def test_ollama_connection_error_has_code(self):
        """Test OLLAMA_CONNECTION_ERROR has error code."""
        assert OLLAMA_CONNECTION_ERROR.error_code == "E001"

    def test_ollama_model_not_found_has_code(self):
        """Test OLLAMA_MODEL_NOT_FOUND has error code."""
        assert OLLAMA_MODEL_NOT_FOUND.error_code == "E002"

    def test_vault_not_found_has_code(self):
        """Test VAULT_NOT_FOUND has error code."""
        assert VAULT_NOT_FOUND.error_code == "E003"

    def test_database_not_found_has_code(self):
        """Test DATABASE_NOT_FOUND has error code."""
        assert DATABASE_NOT_FOUND.error_code == "E004"

    def test_unsupported_file_type_has_code(self):
        """Test UNSUPPORTED_FILE_TYPE has error code."""
        assert UNSUPPORTED_FILE_TYPE.error_code == "E005"

    def test_no_inbox_configured_has_code(self):
        """Test NO_INBOX_CONFIGURED has error code."""
        assert NO_INBOX_CONFIGURED.error_code == "E006"

    def test_import_failed_has_code(self):
        """Test IMPORT_FAILED has error code."""
        assert IMPORT_FAILED.error_code == "E007"

    def test_all_error_codes_are_unique(self):
        """Test that all predefined errors have unique error codes."""
        errors = [
            OLLAMA_CONNECTION_ERROR,
            OLLAMA_MODEL_NOT_FOUND,
            VAULT_NOT_FOUND,
            DATABASE_NOT_FOUND,
            UNSUPPORTED_FILE_TYPE,
            NO_INBOX_CONFIGURED,
            IMPORT_FAILED,
        ]
        error_codes = [e.error_code for e in errors]
        assert len(error_codes) == len(
            set(error_codes)
        ), "All error codes should be unique"

    def test_all_error_codes_follow_pattern(self):
        """Test that all error codes follow E### pattern."""
        errors = [
            OLLAMA_CONNECTION_ERROR,
            OLLAMA_MODEL_NOT_FOUND,
            VAULT_NOT_FOUND,
            DATABASE_NOT_FOUND,
            UNSUPPORTED_FILE_TYPE,
            NO_INBOX_CONFIGURED,
            IMPORT_FAILED,
        ]
        import re

        pattern = re.compile(r"^E\d{3}$")
        for error in errors:
            assert error.error_code is not None
            assert pattern.match(
                error.error_code
            ), f"Error code '{error.error_code}' should follow E### pattern"


class TestFormatErrorPreservesErrorCode:
    """Tests verifying error_code is preserved by format_error()."""

    def test_format_preserves_error_code(self):
        """Test that format_error preserves the error_code."""
        error = ActionableError(
            message="Error {value}", suggestions=["Fix {value}"], error_code="E999"
        )

        formatted = format_error(error, value="test")

        assert formatted.error_code == "E999"

    def test_format_preserves_none_error_code(self):
        """Test that format_error preserves None error_code."""
        error = ActionableError(
            message="Error {value}", suggestions=["Fix {value}"], error_code=None
        )

        formatted = format_error(error, value="test")

        assert formatted.error_code is None

    def test_format_predefined_error_preserves_code(self):
        """Test that formatting predefined errors preserves their codes."""
        formatted = format_error(OLLAMA_CONNECTION_ERROR, url="http://localhost:11434")

        assert formatted.error_code == "E001"

    def test_all_predefined_errors_preserve_codes_after_format(self):
        """Test that all predefined errors preserve codes after formatting."""
        test_cases = [
            (OLLAMA_CONNECTION_ERROR, {"url": "http://test"}, "E001"),
            (OLLAMA_MODEL_NOT_FOUND, {"model": "llama2"}, "E002"),
            (VAULT_NOT_FOUND, {"path": "/test/path"}, "E003"),
            (DATABASE_NOT_FOUND, {"path": "/test/db"}, "E004"),
            (UNSUPPORTED_FILE_TYPE, {"extension": ".xyz"}, "E005"),
            (NO_INBOX_CONFIGURED, {}, "E006"),
            (IMPORT_FAILED, {"reason": "test reason"}, "E007"),
        ]

        for error, kwargs, expected_code in test_cases:
            formatted = format_error(error, **kwargs)
            assert (
                formatted.error_code == expected_code
            ), f"Error code should be preserved as {expected_code}"
