import functools
import time

from loguru import logger


def time_profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        now = time.time()
        result = func(*args, **kwargs)
        logger.info(f'{func.__name__}: Time -> {time.time() - now}s')
        return result

    return wrapper
