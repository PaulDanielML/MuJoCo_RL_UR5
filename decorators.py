import functools
import time
import numpy as np
from inspect import signature


def timer(func):
    """Prints the runtime of the decorated function."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"{func.__name__!r} took {run_time:.4f} secs to execute.")
        return value

    return wrapper_timer


def debug(func):
    """Prints the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"########## Debugging {func.__name__} ##########")
        print(f"Calling {func.__name__}({signature}).")
        value = func(*args, **kwargs)
        print(f"{func.__name__} return type: {type(value)!r}")
        if isinstance(value, dict):
            print("Returned dictionary contents:")
            for k, v in value.items():
                print(f"{k}:")
                print(f"{type(v)!r}", end=" ")
                if isinstance(v, np.ndarray):
                    print(f"Array dimensions: {v.shape}")
                elif isinstance(v, int):
                    print(v)
        elif isinstance(value, np.ndarray):
            print(f"Shape: {value.shape}")
            if value.size < 100:
                print(value)
        else:
            print(f"{func.__name__} returned {value!r}.")
        print("#################################")
        return value

    return wrapper_debug


# @size_parameters.setter
def size_parameters(self, new_params):
    """
    Use this setter for changing .
    """

    try:
        assert isinstance(new_params, list), "Size parameters need to be specified in a list."
        assert len(new_params) == 3, "Size parameters list needs to be of length 3."
        # self._size_parameters = new_params
    except Exception as e:
        print(e)
        print(f"Could not set size_parameters.")


def typeassert(*ty_args, **ty_kwargs):
    def decorate(func):
        if not __debug__:
            return func
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError("Argument {} must be {}".format(name, bound_types[name]))
            return func(*args, **kwargs)

        return wrapper

    return decorate


def dict2list(func):
    @functools.wraps(func)
    def wrapper_dict2list(*args, **kwargs):
        output = func(*args, **kwargs)
        if not isinstance(output, dict):
            print("returning original output")
            return output
        else:
            key_list = list(output.keys())
            new_array = output[key_list[0]]
            for key in key_list[1:]:
                if output[key].ndim != output[key_list[0]].ndim:
                    temp = np.expand_dims(output[key], 2)
                    new_array = np.concatenate((new_array, temp), axis=2)
            return new_array

    return wrapper_dict2list


if __name__ == "__main__":
    # @debug
    @typeassert(dict, int, int)
    def test(arg1, arg2, arg3):
        return 17

    test("8329", [1, 2, 3], 98908)
