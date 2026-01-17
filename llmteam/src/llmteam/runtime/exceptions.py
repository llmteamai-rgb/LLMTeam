"""
Runtime exceptions.
"""


class RuntimeContextError(Exception):
    """Base exception for runtime context errors."""

    pass


class ResourceNotFoundError(RuntimeContextError):
    """Resource not found in registry."""

    pass


class SecretAccessDeniedError(RuntimeContextError):
    """Access to secret denied."""

    pass
