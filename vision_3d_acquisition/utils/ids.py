from datetime import UTC, datetime

_counter = 0


def generate_take_id() -> str:
    """
    Generate a take id: YYYY-MM-DDTHHMMSS_mmm.

    A three-digit suffix handles multiple takes within the same second.
    """
    global _counter
    now = datetime.now(UTC)
    stamp = now.strftime("%Y-%m-%dT%H%M%S")
    _counter = (_counter + 1) % 1000
    return f"{stamp}_{_counter:03d}"
