from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 with millisecond precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
