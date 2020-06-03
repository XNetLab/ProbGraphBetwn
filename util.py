#! /usr/bin/python
# coding = utf-8

import time
from functools import wraps


def fn_timer(function):
    """
    Used to output the running time of the function
    :param function: the function to test
    :return:
    """
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
                (function.func_name, str(t1-t0))
                )
        return result
    return function_timer