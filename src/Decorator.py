from datetime import datetime
from functools import wraps


def time_indicator(func):

    @wraps(func)
    def wrapper(*args, **kw):
        start_time = datetime.now()
        print(func.__name__, 'start at', start_time)
        res = func(*args, **kw)
        end_time = datetime.now()
        print(func.__name__, 'end at', end_time)
        return res

    return wrapper



