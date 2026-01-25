import hashlib
import pickle
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def _create_cache_decorator(max_size: int, copy_arrays: bool) -> Callable[[F], F]:
    cache: dict[str, Any] = {}

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{_generate_cache_key(*args, **kwargs)}"

            if cache_key in cache:
                cached_result = cache[cache_key]

                if copy_arrays and isinstance(cached_result, np.ndarray):
                    return cached_result.copy()

                return cached_result

            result = func(*args, **kwargs)

            if len(cache) >= max_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]

            if copy_arrays and isinstance(result, np.ndarray):
                cache[cache_key] = result.copy()
            else:
                cache[cache_key] = result

            return result

        return wrapper  # type: ignore

    return decorator


def image_cache(max_size: int = 64) -> Callable[[F], F]:
    return _create_cache_decorator(max_size, copy_arrays=True)


def ocr_cache(max_size: int = 32) -> Callable[[F], F]:
    return _create_cache_decorator(max_size, copy_arrays=False)


def _generate_cache_key(*args, **kwargs) -> str:
    def _process_value(value: Any, prefix: str = "") -> str:
        if isinstance(value, np.ndarray):
            return f"{prefix}arr_{_hash_array(value)}"
        if hasattr(value, "__dict__"):
            return f"{prefix}cfg_{_hash_config(value)}"
        return f"{prefix}{hash(str(value))}"

    key_parts = [_process_value(arg) for arg in args]
    key_parts.extend(_process_value(v, f"{k}_") for k, v in sorted(kwargs.items()))

    return "_".join(key_parts)


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _hash_config(config: Any) -> str:
    try:
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(config_bytes).hexdigest()[:16]

    except Exception:
        return str(hash(str(config)))
