"""Error types for on-call domain operations."""


class OnCallError(Exception):
    """Base on-call domain exception."""


class OnCallNotFound(OnCallError):
    """Referenced or requested on-call resource is missing."""


class OnCallValidationError(OnCallError):
    """Input data or business rule violation."""


class NoMatchError(OnCallError):
    """No routing rule matched an alert."""

