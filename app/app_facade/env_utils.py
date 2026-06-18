import os


def positive_int_env(name: str, default: int) -> int:
    """Read an env var as a positive integer, falling back to default if absent or invalid."""
    raw = os.getenv(name, "")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default
