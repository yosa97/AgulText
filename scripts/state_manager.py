import json
import os
import redis

STATE_KEY = "state"

# Default TTL (seconds) for state keys.  0 = no expiry.
_DEFAULT_TTL = int(os.getenv("STATE_TTL_SECONDS", "0"))


def _get_redis_client() -> redis.Redis:
    """Build a Redis client from environment variables.

    Supported env vars: REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB.
    """
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD", None)
    db = int(os.getenv("REDIS_DB", 0))

    return redis.Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )


def get_state() -> dict:
    """Read the full state dict from Redis.  Returns {} on miss or parse error."""
    client = _get_redis_client()
    value = client.get(STATE_KEY)

    if value is None:
        return {}

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def set_state(state: dict, ttl: int = _DEFAULT_TTL) -> None:
    """Persist *state* to Redis.  If *ttl* > 0 the key expires after that many seconds."""
    client = _get_redis_client()
    json_value = json.dumps(state)
    if ttl > 0:
        client.set(STATE_KEY, json_value, ex=ttl)
    else:
        client.set(STATE_KEY, json_value)


def delete_state() -> bool:
    """Remove the state key entirely.  Returns True if the key existed."""
    client = _get_redis_client()
    return bool(client.delete(STATE_KEY))


def get_state_field(field: str, default=None):
    """Fetch a single top-level field from the state without re-serialising the whole dict."""
    state = get_state()
    return state.get(field, default)


def set_state_field(field: str, value) -> None:
    """Update a single top-level field and write back the full state atomically."""
    state = get_state()
    state[field] = value
    set_state(state)


def test():
    state = get_state()
    print(json.dumps(state, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    test()
