# -*- coding: utf-8 -*-

# @File   : decorator.py
# @Author : Yuvv
# @Date   : 2018/5/5

from functools import wraps

from flask import request


# def cached(timeout=5 * 60, key='view/%s'):
#     def decorator(f):
#         @wraps(f)
#         def decorated_function(*args, **kwargs):
#             cache_key = key % request.path
#             rv = cache.get(cache_key)
#             if rv is not None:
#                 return rv
#             rv = f(*args, **kwargs)
#             cache.set(cache_key, rv, timeout=timeout)
#             return rv
#
#         return decorated_function
#
#     return decorator
