"""Error types for alert domain operations."""


class AlertError(Exception):
    """Base class for alert domain errors."""


class AlertNotFound(AlertError):
    """Referenced alert is missing."""


class AlertValidationError(AlertError):
    """Validation failure or unsupported input."""


class InvalidAlertState(AlertError):
    """Invalid alert state transition."""


class AlertSilenceError(AlertError):
    """Validation failure in silence rule."""

