import hashlib
import pickle
from functools import wraps
from typing import Any, Callable, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def image_cache(max_size: int = 64) -> Callable[[F], F]:
    """Decorator for caching image processing functions"""
    cache: dict[str, Any] = {}

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{_generate_cache_key(*args, **kwargs)}"

            if cache_key in cache:
                cached_result = cache[cache_key]

                if isinstance(cached_result, np.ndarray):
                    return cached_result.copy()

                return cached_result

            result = func(*args, **kwargs)

            if len(cache) >= max_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]

            if isinstance(result, np.ndarray):
                cache[cache_key] = result.copy()
            else:
                cache[cache_key] = result

            return result

        def cache_clear():
            cache.clear()

        def cache_size():
            return len(cache)

        wrapper.cache_clear = cache_clear  # type: ignore
        wrapper.cache_size = cache_size  # type: ignore

        return wrapper  # type: ignore

    return decorator


def ocr_cache(max_size: int = 32) -> Callable[[F], F]:
    """Decorator for caching OCR functions"""
    cache: dict[str, Any] = {}

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{_generate_cache_key(*args, **kwargs)}"

            if cache_key in cache:
                return cache[cache_key]

            result = func(*args, **kwargs)

            if len(cache) >= max_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]

            cache[cache_key] = result
            return result

        def cache_clear():
            cache.clear()

        def cache_size():
            return len(cache)

        wrapper.cache_clear = cache_clear  # type: ignore
        wrapper.cache_size = cache_size  # type: ignore

        return wrapper  # type: ignore

    return decorator


def _generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments"""
    key_parts = []

    for arg in args:
        if isinstance(arg, np.ndarray):
            key_parts.append(f"arr_{_hash_array(arg)}")
        elif hasattr(arg, "__dict__"):
            key_parts.append(f"cfg_{_hash_config(arg)}")
        else:
            key_parts.append(str(hash(str(arg))))

    for k, v in sorted(kwargs.items()):
        if isinstance(v, np.ndarray):
            key_parts.append(f"{k}_arr_{_hash_array(v)}")
        elif hasattr(v, "__dict__"):
            key_parts.append(f"{k}_cfg_{_hash_config(v)}")
        else:
            key_parts.append(f"{k}_{hash(str(v))}")

    return "_".join(key_parts)


def _hash_array(array: np.ndarray) -> str:
    """Generate hash for numpy array"""
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _hash_config(config: Any) -> str:
    """Generate hash for configuration object"""
    try:
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(config_bytes).hexdigest()[:16]

    except Exception:
        return str(hash(str(config)))
