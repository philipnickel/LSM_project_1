"""Timing utilities for performance measurement."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[callable, None, None]:
    """Context manager that measures execution time."""
    start_time = time.time()
    
    def get_elapsed() -> float:
        return time.time() - start_time
    
    try:
        yield get_elapsed
    finally:
        pass  # Time is measured when get_elapsed() is called


def time_function(func, *args, **kwargs) -> tuple:
    """Time a function execution and return (result, duration)."""
    start_time = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start_time
    return result, duration